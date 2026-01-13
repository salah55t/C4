from flask import Flask, render_template_string
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة التحكم المالية - Pro</title>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap" rel="stylesheet">
    <!-- استدعاء مكتبة الرسوم البيانية -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            padding: 20px;
            border-bottom: 1px solid var(--border);
            text-align: center;
        }

        .main-input {
            width: 100%; max-width: 400px;
            background: #253341; border: 2px solid var(--primary);
            color: white; padding: 10px; font-size: 20px;
            font-weight: bold; text-align: center; border-radius: 10px;
            font-family: 'Cairo'; margin-top: 10px;
        }

        /* تخطيط الصفحة: قسمين (الرسوم البيانية + البطاقات) */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 350px; /* البطاقات تأخذ مساحة والعرض الجانبي للرسوم */
            gap: 20px;
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 20px;
        }

        @media (max-width: 900px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .charts-panel { order: -1; } /* الرسوم تظهر أولاً في الموبايل */
        }

        .card {
            background: var(--panel); border-radius: 15px;
            padding: 15px; border: 1px solid var(--border);
            margin-bottom: 15px; position: relative;
        }

        .charts-panel {
            background: var(--panel); border-radius: 15px;
            padding: 20px; border: 1px solid var(--border);
            height: fit-content;
            position: sticky; top: 20px;
        }

        .report-box {
            background: rgba(29, 168, 130, 0.1);
            border: 1px solid var(--primary);
            padding: 15px; border-radius: 10px;
            margin-top: 20px; font-size: 14px;
            line-height: 1.6;
        }

        .report-danger { background: rgba(246, 70, 93, 0.1); border-color: var(--danger); }
        .report-warning { background: rgba(240, 185, 11, 0.1); border-color: var(--warning); }

        .progress-bar { height: 8px; background: #2c3640; border-radius: 4px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; transition: width 0.5s ease; }
        
        .btn-action {
            background: var(--primary); color: white; border: none;
            padding: 5px 12px; border-radius: 6px; cursor: pointer;
            font-family: 'Cairo'; margin-left: 5px;
        }
        
        input[type="number"] {
            background: var(--bg); border: 1px solid var(--border);
            color: white; padding: 5px; border-radius: 6px; width: 80px;
        }
    </style>
</head>
<body>

    <div class="header">
        <h1>📊 لوحة القيادة المالية</h1>
        <label style="color:var(--text-secondary)">الميزانية الكلية (دج)</label><br>
        <input type="number" id="totalBudgetInput" class="main-input" placeholder="40000" oninput="updateBudget()">
    </div>

    <div class="dashboard-grid">
        
        <!-- القسم الأيمن: بطاقات المصاريف -->
        <div id="cardsContainer">
            <!-- سيتم توليد البطاقات هنا -->
        </div>

        <!-- القسم الأيسر: الرسوم البيانية والتقرير -->
        <div class="charts-panel">
            <h3 style="margin-bottom: 15px; border-bottom: 1px solid var(--border); padding-bottom: 10px;">📈 التحليل البياني</h3>
            
            <canvas id="budgetChart" width="300" height="300"></canvas>
            
            <div id="smartReport" class="report-box">
                جاري تحليل البيانات...
            </div>

            <div style="margin-top: 20px; text-align: center;">
                <h2 id="totalRemainingDisplay" style="color: var(--primary)">0 دج</h2>
                <small style="color: var(--text-secondary)">المتبقي الصافي</small>
            </div>
            
            <div style="margin-top:20px; text-align:center">
                 <button onclick="resetAllData()" style="background:transparent; color:var(--text-secondary); border:1px solid var(--border); padding:5px 15px; border-radius:15px; cursor:pointer">🗑️ تصفير</button>
            </div>
        </div>

    </div>

    <script>
        // التكوين الأساسي
        const CONFIG = [
            { id: 'baby', name: 'الرضيع', percent: 0.20, color: '#FF6384' },
            { id: 'groceries', name: 'قضيان', percent: 0.30, color: '#36A2EB' },
            { id: 'market', name: 'خضر/فواكه', percent: 0.20, color: '#FFCE56' },
            { id: 'meat', name: 'لحوم', percent: 0.125, color: '#4BC0C0' },
            { id: 'daily', name: 'خبز/حليب', percent: 0.10, color: '#9966FF' },
            { id: 'bills', name: 'فواتير', percent: 0.075, color: '#FF9F40' }
        ];

        let appState = { totalBudget: 40000, expenses: {} };
        let myChart = null; // متغير لتخزين كائن الرسم البياني

        function loadData() {
            const saved = localStorage.getItem('dzBudgetPro_v2');
            if (saved) appState = JSON.parse(saved);
            document.getElementById('totalBudgetInput').value = appState.totalBudget;
            renderAll();
        }

        function saveData() {
            localStorage.setItem('dzBudgetPro_v2', JSON.stringify(appState));
            renderAll();
        }

        function updateBudget() {
            let val = parseFloat(document.getElementById('totalBudgetInput').value);
            appState.totalBudget = (val >= 0) ? val : 0;
            saveData();
        }

        function addExpense(id) {
            const input = document.getElementById(`in-${id}`);
            const val = parseFloat(input.value);
            if (val > 0) {
                appState.expenses[id] = (appState.expenses[id] || 0) + val;
                input.value = '';
                saveData();
            }
        }

        function renderAll() {
            renderCards();
            updateChart();
            generateReport();
        }

        // 1. توليد البطاقات
        function renderCards() {
            const container = document.getElementById('cardsContainer');
            container.innerHTML = '';

            CONFIG.forEach(cat => {
                const allocated = Math.round(appState.totalBudget * cat.percent);
                const spent = appState.expenses[cat.id] || 0;
                const remaining = allocated - spent;
                const progress = allocated > 0 ? (spent / allocated) * 100 : 0;
                
                let barColor = 'var(--primary)';
                if(progress > 80) barColor = 'var(--warning)';
                if(progress > 100) barColor = 'var(--danger)';

                container.innerHTML += `
                    <div class="card">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                            <strong>${cat.name}</strong>
                            <span style="color:${remaining < 0 ? 'var(--danger)' : 'var(--text)'}">
                                ${spent.toLocaleString()} / <small>${allocated.toLocaleString()}</small>
                            </span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width:${Math.min(progress, 100)}%; background:${barColor}"></div>
                        </div>
                        <div style="display:flex; gap:10px; margin-top:5px;">
                            <input type="number" id="in-${cat.id}" placeholder="المبلغ">
                            <button class="btn-action" onclick="addExpense('${cat.id}')">+</button>
                        </div>
                    </div>
                `;
            });
        }

        // 2. تحديث الرسم البياني (Doughnut Chart)
        function updateChart() {
            const ctx = document.getElementById('budgetChart').getContext('2d');
            
            // تحضير البيانات
            const labels = CONFIG.map(c => c.name);
            const spentData = CONFIG.map(c => appState.expenses[c.id] || 0);
            const bgColors = CONFIG.map(c => c.color);

            if (myChart) {
                myChart.data.datasets[0].data = spentData;
                myChart.update();
            } else {
                myChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: spentData,
                            backgroundColor: bgColors,
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { position: 'bottom', labels: { color: '#fff' } },
                            title: { display: true, text: 'توزيع المصاريف الفعلي', color: '#fff' }
                        }
                    }
                });
            }
        }

        // 3. توليد التقرير الذكي
        function generateReport() {
            const totalSpent = Object.values(appState.expenses).reduce((a, b) => a + b, 0);
            const remaining = appState.totalBudget - totalSpent;
            const burnRate = (totalSpent / appState.totalBudget) * 100;
            
            const reportEl = document.getElementById('smartReport');
            const remDisplay = document.getElementById('totalRemainingDisplay');

            remDisplay.innerText = remaining.toLocaleString() + ' دج';
            remDisplay.style.color = remaining < 0 ? 'var(--danger)' : 'var(--primary)';

            let statusHTML = '';
            let alertClass = '';

            if (remaining < 0) {
                statusHTML = `<strong>🚨 حالة طوارئ!</strong> لقد تجاوزت الميزانية بـ ${Math.abs(remaining)} دج. يجب التوقف فوراً عن الشراء.`;
                alertClass = 'report-danger';
            } else if (burnRate > 80) {
                statusHTML = `<strong>⚠️ تحذير:</strong> لقد استهلكت ${burnRate.toFixed(1)}% من الميزانية. تبقى لك مبلغ ضئيل جداً.`;
                alertClass = 'report-warning';
            } else {
                statusHTML = `<strong>✅ الوضع مستقر:</strong> استهلكت ${burnRate.toFixed(1)}% فقط. واصل على هذا المنوال.`;
                alertClass = '';
            }

            // تحليل أعلى فئة استهلاكاً
            let maxCat = CONFIG[0];
            let maxVal = 0;
            CONFIG.forEach(c => {
                let s = appState.expenses[c.id] || 0;
                if(s > maxVal) { maxVal = s; maxCat = c; }
            });

            if(maxVal > 0) {
                statusHTML += `<br><br>👉 أكثر ما يستهلك مالك هو: <strong>${maxCat.name}</strong> (${maxVal} دج).`;
            }

            reportEl.innerHTML = statusHTML;
            reportEl.className = 'report-box ' + alertClass;
        }

        function resetAllData() {
            if(confirm('هل أنت متأكد؟')) {
                localStorage.removeItem('dzBudgetPro_v2');
                appState = { totalBudget: 40000, expenses: {} };
                loadData();
            }
        }

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