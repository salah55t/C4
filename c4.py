from flask import Flask, render_template_string
import os

app = Flask(__name__)

# كود HTML المحسن والمدمج
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تقسيم الميزانية الجزائرية - Pro</title>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f1419;
            --panel: #192734;
            --border: #38444d;
            --text: #ffffff;
            --text-secondary: #8899a6;
            --primary: #1da882;
            --danger: #f6465d;
            --warning: #f0b90b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: var(--bg); color: var(--text); font-family: 'Cairo', sans-serif; min-height: 100vh; padding-bottom: 50px; }

        .header {
            background: linear-gradient(135deg, #15202b 0%, #192734 100%);
            padding: 30px 20px;
            border-bottom: 1px solid var(--border);
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .main-input-container {
            max-width: 500px;
            margin: 20px auto 0;
            position: relative;
        }

        .budget-label {
            display: block;
            margin-bottom: 10px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .main-input {
            width: 100%;
            background: #253341;
            border: 2px solid var(--primary);
            color: white;
            padding: 15px;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            border-radius: 15px;
            font-family: 'Cairo', sans-serif;
            transition: all 0.3s ease;
        }

        .main-input:focus {
            outline: none;
            box-shadow: 0 0 15px rgba(29, 168, 130, 0.4);
        }

        .container {
            max-width: 800px;
            margin: 30px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .card {
            background: var(--panel);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            transition: transform 0.2s;
        }

        .card:hover { transform: translateY(-3px); }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }

        .category-icon { font-size: 28px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 12px; }
        
        .category-info h3 { font-size: 18px; margin-bottom: 5px; }
        .category-desc { font-size: 12px; color: var(--text-secondary); line-height: 1.4; background: rgba(0,0,0,0.2); padding: 5px 8px; border-radius: 6px; display: inline-block;}

        .money-stats {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-weight: bold;
            font-size: 14px;
        }

        .allocated { color: var(--primary); }
        .spent-text { color: var(--danger); }

        .progress-bar {
            height: 10px;
            background: #2c3640;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 15px;
        }

        .progress-fill {
            height: 100%;
            background: var(--primary);
            transition: width 0.5s ease, background 0.3s;
        }

        .expense-input-group {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        .mini-input {
            background: var(--bg);
            border: 1px solid var(--border);
            color: white;
            padding: 8px;
            border-radius: 8px;
            width: 100%;
            font-family: 'Cairo';
        }

        .btn-add {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0 15px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
        }
        
        .btn-reset {
            background: transparent;
            color: var(--danger);
            border: 1px solid var(--danger);
            padding: 5px 10px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 12px;
            margin-top: 5px;
        }

        /* Responsive adjustments */
        @media (max-width: 600px) {
            .header h1 { font-size: 20px; }
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <div class="header">
        <h1>📊 مقسم الميزانية الجزائري</h1>
        <div class="main-input-container">
            <label class="budget-label">أدخل الراتب / الميزانية الكلية (دج)</label>
            <input type="number" id="totalBudgetInput" class="main-input" placeholder="مثلاً: 40000" oninput="updateBudget()">
        </div>
        <div style="margin-top: 15px; font-size: 14px; color: var(--text-secondary);">
            المتبقي الصافي: <span id="totalRemaining" style="color: var(--primary); font-weight: bold; font-size: 18px;">0</span> دج
        </div>
    </div>

    <div class="container" id="cardsContainer">
        <!-- Cards will be injected here by JS -->
    </div>

    <div style="text-align: center; margin-top: 20px;">
        <button onclick="resetAllData()" style="background: #333; color: #888; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer;">⚠️ تصفير كل البيانات</button>
    </div>

    <script>
        // تكوين الفئات بالنسب المئوية (مجموعها 100%)
        const CONFIG = [
            { 
                id: 'baby', 
                name: 'احتياجات الرضيع', 
                percent: 0.20, 
                icon: '👶', 
                desc: 'حفاضات + حليب (خط أحمر مقدس ⛔)' 
            },
            { 
                id: 'groceries', 
                name: 'قضيان الشهر', 
                percent: 0.30, 
                icon: '🛒', 
                desc: 'أساسيات + تنظيف (تقليص الكميات بذكاء)' 
            },
            { 
                id: 'market', 
                name: 'الخضر والفواكه', 
                percent: 0.20, 
                icon: '🍎', 
                desc: 'سوق أسبوعي (خيارات موسمية محدودة)' 
            },
            { 
                id: 'meat', 
                name: 'اللحوم والبيض', 
                percent: 0.125, 
                icon: '🍗', 
                desc: 'دجاجة أسبوعياً + بلاطو بيض (وداعاً للحم الأحمر)' 
            },
            { 
                id: 'daily', 
                name: 'الخبز والحليب', 
                percent: 0.10, 
                icon: '🥖', 
                desc: 'للوالدين والطفلة (5 سنوات)' 
            },
            { 
                id: 'bills', 
                name: 'فواتير وطوارئ', 
                percent: 0.075, 
                icon: '💡', 
                desc: 'مبلغ رمزي للكهرباء/الماء/الإنترنت' 
            }
        ];

        // State Management
        let appState = {
            totalBudget: 40000, // الافتراضي
            expenses: {} // لتخزين المصاريف لكل فئة
        };

        // Load data on startup
        function loadData() {
            const savedData = localStorage.getItem('dzBudgetApp');
            if (savedData) {
                appState = JSON.parse(savedData);
            }
            // تعيين قيمة الإدخال
            document.getElementById('totalBudgetInput').value = appState.totalBudget;
            renderCards();
        }

        // Save data
        function saveData() {
            localStorage.setItem('dzBudgetApp', JSON.stringify(appState));
            updateTotalStats();
        }

        // تحديث عند تغيير الميزانية
        function updateBudget() {
            const input = document.getElementById('totalBudgetInput');
            let val = parseFloat(input.value);
            if (isNaN(val) || val < 0) val = 0;
            appState.totalBudget = val;
            saveData();
            renderCards();
        }

        // إضافة مصروف
        function addExpense(catId) {
            const input = document.getElementById(`input-${catId}`);
            const amount = parseFloat(input.value);
            
            if (amount && amount > 0) {
                if (!appState.expenses[catId]) appState.expenses[catId] = 0;
                appState.expenses[catId] += amount;
                input.value = '';
                saveData();
                renderCards();
            }
        }

        // تصفير مصروف فئة معينة
        function resetCategory(catId) {
            if(confirm('هل أنت متأكد من تصفير مصاريف هذا البند؟')) {
                appState.expenses[catId] = 0;
                saveData();
                renderCards();
            }
        }

        // الحسابات والعرض
        function renderCards() {
            const container = document.getElementById('cardsContainer');
            container.innerHTML = '';
            
            let totalSpentGlobal = 0;

            CONFIG.forEach(cat => {
                // الحسابات
                const allocated = Math.round(appState.totalBudget * cat.percent);
                const spent = appState.expenses[cat.id] || 0;
                const remaining = allocated - spent;
                const progress = allocated > 0 ? (spent / allocated) * 100 : 0;
                totalSpentGlobal += spent;

                // تحديد لون الحالة
                let statusColor = 'var(--primary)'; // أخضر
                if (progress > 80) statusColor = 'var(--warning)'; // أصفر
                if (progress >= 100) statusColor = 'var(--danger)'; // أحمر

                // إنشاء البطاقة HTML
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <div class="card-header">
                        <div class="category-icon">${cat.icon}</div>
                        <div class="category-info">
                            <h3>${cat.name}</h3>
                            <div class="category-desc">${cat.desc}</div>
                        </div>
                        <div style="text-align:left">
                            <small style="color:#8899a6">المخصص</small>
                            <div style="font-weight:bold; font-size:18px">${allocated.toLocaleString()} دج</div>
                        </div>
                    </div>

                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${Math.min(progress, 100)}%; background-color: ${statusColor}"></div>
                    </div>

                    <div class="money-stats">
                        <span class="spent-text">صرفت: ${spent.toLocaleString()}</span>
                        <span style="color: ${remaining < 0 ? 'var(--danger)' : 'var(--text)'}">
                            الباقي: ${remaining.toLocaleString()}
                        </span>
                    </div>

                    <div class="expense-input-group">
                        <input type="number" id="input-${cat.id}" class="mini-input" placeholder="أضف مصروف...">
                        <button onclick="addExpense('${cat.id}')" class="btn-add">+</button>
                    </div>
                    <div style="text-align:left">
                         <button onclick="resetCategory('${cat.id}')" class="btn-reset">تصفير</button>
                    </div>
                `;
                container.appendChild(card);
            });

            // تحديث الإجمالي الكلي في الهيدر
            const totalRemaining = appState.totalBudget - totalSpentGlobal;
            const remEl = document.getElementById('totalRemaining');
            remEl.innerText = totalRemaining.toLocaleString();
            remEl.style.color = totalRemaining < 0 ? 'var(--danger)' : 'var(--primary)';
        }

        function resetAllData() {
            if(confirm('هل تريد حقاً مسح جميع البيانات والعودة للصفر؟')) {
                localStorage.removeItem('dzBudgetApp');
                appState = { totalBudget: 40000, expenses: {} };
                loadData();
            }
        }

        // Init
        loadData();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)