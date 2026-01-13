from flask import Flask, render_template_string
import os

app = Flask(__name__)

# كود HTML الخاص بك تم وضعه هنا كقالب
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مصاريف البيت - متتبع الميزانية الذكي</title>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f1419;
            --panel: #192734;
            --panel-hover: #22303c;
            --border: #38444d;
            --text: #ffffff;
            --text-muted: #8899a6;
            --primary: #1da882;
            --primary-glow: #0ecb81;
            --secondary: #f0b90b;
            --danger: #f6465d;
            --warning: #f7931a;
            --accent: #1d9bf0;
            --gradient-primary: linear-gradient(135deg, #1da882 0%, #0ecb81 100%);
            --gradient-gold: linear-gradient(135deg, #f0b90b 0%, #fcd435 100%);
            --gradient-danger: linear-gradient(135deg, #f6465d 0%, #ff6b7a 100%);
            --shadow-glow: 0 0 30px rgba(29, 168, 130, 0.3);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: var(--bg); color: var(--text); font-family: 'Cairo', sans-serif; min-height: 100vh; line-height: 1.6; }

        .header { background: var(--gradient-primary); padding: 30px 20px; border-radius: 0 0 30px 30px; box-shadow: var(--shadow-glow); margin-bottom: -20px; position: relative; z-index: 10; }
        .header-content { max-width: 900px; margin: 0 auto; }
        .header-title { display: flex; align-items: center; gap: 15px; margin-bottom: 25px; }
        .logo-icon { width: 50px; height: 50px; background: rgba(255,255,255,0.2); border-radius: 15px; display: flex; align-items: center; justify-content: center; font-size: 24px; animation: float 3s ease-in-out infinite; }
        @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
        .header h1 { font-size: 28px; font-weight: 900; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        .stat-card { background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 20px 15px; text-align: center; }
        .stat-value { font-size: 22px; font-weight: 900; }
        
        .progress-bar { height: 12px; background: rgba(255,255,255,0.2); border-radius: 10px; overflow: hidden; margin-top: 10px; }
        .progress-fill { height: 100%; transition: width 0.5s ease; }
        .progress-safe { background: #fff; }
        .progress-warning { background: var(--warning); }
        .progress-danger { background: var(--danger); }

        .main-content { max-width: 900px; margin: 0 auto; padding: 40px 20px 30px; }
        .btn { border: none; border-radius: 12px; padding: 12px 20px; font-family: 'Cairo', sans-serif; cursor: pointer; display: flex; align-items: center; gap: 8px; font-size: 14px; }
        .btn-primary { background: var(--gradient-primary); color: white; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 20px; margin-bottom: 15px; }
        .hidden { display: none !important; }
        
        /* Form & Selectors Styles */
        .form-card { background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 25px; margin-bottom: 20px; }
        .form-input { width: 100%; padding: 14px 16px; background: var(--bg); border: 1px solid var(--border); border-radius: 12px; color: var(--text); margin-bottom: 10px; }
        .selector-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }
        .selector-item { width: 48px; height: 48px; border-radius: 12px; border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; cursor: pointer; }
        .selector-item.active { border-color: var(--primary); background: rgba(29, 168, 130, 0.15); }
        .bg-primary { background: var(--gradient-primary); }
        .bg-gold { background: var(--gradient-gold); }
        .bg-blue { background: linear-gradient(135deg, #1d9bf0 0%, #60c5ff 100%); }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="header-title">
                <div class="logo-icon">💰</div>
                <div><h1>مصاريف البيت</h1><p>تتبع ميزانيتك بذكاء</p></div>
            </div>
            <div class="stats-grid">
                <div class="stat-card"><div>الميزانية</div><div class="stat-value" id="totalBudget">0</div></div>
                <div class="stat-card"><div>المصروف</div><div class="stat-value" id="totalSpent">0</div></div>
                <div class="stat-card"><div>المتبقي</div><div class="stat-value" id="remaining">0</div></div>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="overallProgress" style="width: 0%"></div></div>
        </div>
    </header>

    <main class="main-content">
        <div style="display:flex; justify-content:space-between; margin-bottom:20px;">
            <h3>📊 فئات المصاريف</h3>
            <button class="btn btn-primary" onclick="toggleAddCategory()">➕ فئة جديدة</button>
        </div>

        <div id="addCategoryForm" class="form-card hidden">
            <input type="text" id="categoryName" class="form-input" placeholder="اسم الفئة">
            <input type="number" id="categoryBudget" class="form-input" placeholder="الميزانية">
            <div class="selector-grid" id="iconSelector">
                <div class="selector-item active" data-icon="🏠" onclick="selectIcon(this)">🏠</div>
                <div class="selector-item" data-icon="🛒" onclick="selectIcon(this)">🛒</div>
                <div class="selector-item" data-icon="🚗" onclick="selectIcon(this)">🚗</div>
            </div>
            <button class="btn btn-primary" onclick="addCategory()">✅ إضافة</button>
        </div>

        <div id="categoriesContainer"></div>
    </main>

    <script>
        let categories = JSON.parse(localStorage.getItem('expenseCategories')) || [];
        let selectedIcon = '🏠';

        function render() {
            const container = document.getElementById('categoriesContainer');
            const totalBudget = categories.reduce((sum, c) => sum + c.budget, 0);
            const totalSpent = categories.reduce((sum, c) => sum + (c.spent || 0), 0);
            
            document.getElementById('totalBudget').textContent = totalBudget;
            document.getElementById('totalSpent').textContent = totalSpent;
            document.getElementById('remaining').textContent = totalBudget - totalSpent;
            document.getElementById('overallProgress').style.width = (totalBudget > 0 ? (totalSpent/totalBudget)*100 : 0) + '%';

            container.innerHTML = categories.map(c => `
                <div class="card">
                    <div style="display:flex; justify-content:space-between">
                        <strong>${c.icon} ${c.name}</strong>
                        <span>${c.spent || 0} / ${c.budget} ريال</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill bg-primary" style="width:${(c.spent/c.budget)*100}%"></div></div>
                    <div style="margin-top:10px; display:flex; gap:10px">
                        <input type="number" id="exp-${c.id}" placeholder="المبلغ" style="width:80px">
                        <button onclick="addExpense('${c.id}')">إضافة مصروف</button>
                        <button onclick="deleteCategory('${c.id}')" style="color:red">حذف</button>
                    </div>
                </div>
            `).join('');
            localStorage.setItem('expenseCategories', JSON.stringify(categories));
        }

        function toggleAddCategory() { document.getElementById('addCategoryForm').classList.toggle('hidden'); }
        
        function addCategory() {
            const name = document.getElementById('categoryName').value;
            const budget = parseFloat(document.getElementById('categoryBudget').value);
            if(name && budget) {
                categories.push({ id: Date.now().toString(), name, budget, spent: 0, icon: selectedIcon });
                render();
                toggleAddCategory();
            }
        }

        function addExpense(id) {
            const amt = parseFloat(document.getElementById('exp-'+id).value);
            const cat = categories.find(c => c.id === id);
            if(cat && amt) {
                cat.spent = (cat.spent || 0) + amt;
                render();
            }
        }

        function deleteCategory(id) {
            categories = categories.filter(c => c.id !== id);
            render();
        }

        function selectIcon(el) {
            document.querySelectorAll('.selector-item').forEach(i => i.classList.remove('active'));
            el.classList.add('active');
            selectedIcon = el.dataset.icon;
        }

        render();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    # Render يتطلب الاستماع على منفذ ديناميكي
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)