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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #0a0f1a;
            --panel: rgba(25, 39, 52, 0.8);
            --panel-solid: #192734;
            --border: rgba(56, 68, 77, 0.5);
            --text: #ffffff;
            --text-secondary: #8899a6;
            --primary: #1da882;
            --primary-glow: rgba(29, 168, 130, 0.3);
            --danger: #f6465d;
            --danger-glow: rgba(246, 70, 93, 0.3);
            --warning: #f0b90b;
            --warning-glow: rgba(240, 185, 11, 0.3);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body { 
            background: linear-gradient(135deg, #0a0f1a 0%, #151c28 50%, #0a0f1a 100%);
            color: var(--text); 
            font-family: 'Cairo', sans-serif; 
            min-height: 100vh; 
            padding-bottom: 50px;
        }

        /* تأثير الخلفية المتحركة */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 20%, rgba(29, 168, 130, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        .header {
            background: linear-gradient(135deg, rgba(21, 32, 43, 0.9) 0%, rgba(25, 39, 52, 0.9) 100%);
            backdrop-filter: blur(20px);
            padding: 30px 20px;
            border-bottom: 1px solid var(--border);
            text-align: center;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--primary) 0%, #4ade80 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .main-input {
            width: 100%; 
            max-width: 400px;
            background: rgba(37, 51, 65, 0.8);
            backdrop-filter: blur(10px);
            border: 2px solid var(--primary);
            box-shadow: 0 0 30px var(--primary-glow);
            color: white; 
            padding: 15px; 
            font-size: 22px;
            font-weight: bold; 
            text-align: center; 
            border-radius: 15px;
            font-family: 'Cairo'; 
            margin-top: 15px;
            transition: all 0.3s ease;
        }

        .main-input:focus {
            outline: none;
            box-shadow: 0 0 40px var(--primary-glow), 0 0 60px rgba(29, 168, 130, 0.2);
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 25px;
            max-width: 1300px;
            margin: 30px auto;
            padding: 0 25px;
        }

        @media (max-width: 900px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .charts-panel { order: -1; }
        }

        .card {
            background: var(--panel);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid var(--border);
            margin-bottom: 20px;
            position: relative;
            transition: all 0.3s ease;
            animation: fadeInUp 0.5s ease forwards;
            opacity: 0;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            border-color: rgba(29, 168, 130, 0.3);
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .charts-panel {
            background: var(--panel);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid var(--border);
            height: fit-content;
            position: sticky; 
            top: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        }

        .charts-panel h3 {
            font-size: 1.2rem;
            background: linear-gradient(135deg, var(--primary) 0%, #4ade80 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .report-box {
            background: rgba(29, 168, 130, 0.1);
            border: 1px solid var(--primary);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            font-size: 14px;
            line-height: 1.8;
            backdrop-filter: blur(10px);
        }

        .report-danger { 
            background: rgba(246, 70, 93, 0.1); 
            border-color: var(--danger);
            box-shadow: 0 0 30px var(--danger-glow);
        }
        
        .report-warning { 
            background: rgba(240, 185, 11, 0.1); 
            border-color: var(--warning);
            box-shadow: 0 0 30px var(--warning-glow);
        }

        .progress-bar { 
            height: 10px; 
            background: rgba(44, 54, 64, 0.8); 
            border-radius: 10px; 
            overflow: hidden; 
            margin: 12px 0;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .progress-fill { 
            height: 100%; 
            transition: width 0.5s ease;
            border-radius: 10px;
            background: linear-gradient(90deg, var(--primary) 0%, #4ade80 100%);
        }

        .progress-fill.warning {
            background: linear-gradient(90deg, var(--warning) 0%, #fcd34d 100%);
        }

        .progress-fill.danger {
            background: linear-gradient(90deg, var(--danger) 0%, #fb7185 100%);
        }
        
        .btn-action {
            background: linear-gradient(135deg, var(--primary) 0%, #16a085 100%);
            color: white; 
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-family: 'Cairo';
            font-weight: bold;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        .btn-action:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--primary-glow);
        }
        
        input[type="number"] {
            background: rgba(10, 15, 26, 0.8);
            border: 1px solid var(--border);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            width: 100px;
            font-family: 'Cairo';
            font-size: 14px;
            transition: all 0.3s ease;
        }

        input[type="number"]:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 20px var(--primary-glow);
        }

        .category-name {
            font-size: 1.1rem;
            font-weight: 700;
        }

        .amount-display {
            font-family: 'Cairo';
            font-weight: 600;
        }

        .remaining-display {
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--primary) 0%, #4ade80 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .remaining-display.danger {
            background: linear-gradient(135deg, var(--danger) 0%, #fb7185 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .reset-btn {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border);
            padding: 10px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-family: 'Cairo';
            transition: all 0.3s ease;
        }

        .reset-btn:hover {
            background: rgba(246, 70, 93, 0.1);
            border-color: var(--danger);
            color: var(--danger);
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .divider {
            border-bottom: 1px solid var(--border);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>

    <div class="header">
        <h1>📊 لوحة القيادة المالية</h1>
        <label class="stat-label">الميزانية الكلية (دج)</label><br>
        <input type="number" id="totalBudgetInput" class="main-input" placeholder="40000" oninput="updateBudget()">
    </div>

    <div class="dashboard-grid">
        
        <div id="cardsContainer"></div>

        <div class="charts-panel">
            <h3 class="divider">📈 التحليل البياني</h3>
            
            <canvas id="budgetChart" width="320" height="320"></canvas>
            
            <div id="smartReport" class="report-box">
                جاري تحليل البيانات...
            </div>

            <div style="margin-top: 25px; text-align: center;">
                <h2 id="totalRemainingDisplay" class="remaining-display">0 دج</h2>
                <small class="stat-label">المتبقي الصافي</small>
            </div>
            
            <div style="margin-top: 25px; text-align: center;">
                <button onclick="resetAllData()" class="reset-btn">🗑️ تصفير البيانات</button>
            </div>
        </div>

    </div>

    <script>
        const CONFIG = [
            { id: 'baby', name: '👶 الرضيع', percent: 0.20, color: '#FF6384' },
            { id: 'groceries', name: '🛒 قضيان', percent: 0.30, color: '#36A2EB' },
            { id: 'market', name: '🥬 خضر/فواكه', percent: 0.20, color: '#FFCE56' },
            { id: 'meat', name: '🥩 لحوم', percent: 0.125, color: '#4BC0C0' },
            { id: 'daily', name: '🥛 خبز/حليب', percent: 0.10, color: '#9966FF' },
            { id: 'bills', name: '📄 فواتير', percent: 0.075, color: '#FF9F40' }
        ];

        let appState = { totalBudget: 40000, expenses: {} };
        let myChart = null;
        let animationDelay = 0;

        function loadData() {
            const saved = localStorage.getItem('dzBudgetPro_v3');
            if (saved) appState = JSON.parse(saved);
            document.getElementById('totalBudgetInput').value = appState.totalBudget;
            renderAll();
        }

        function saveData() {
            localStorage.setItem('dzBudgetPro_v3', JSON.stringify(appState));
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

        function renderCards() {
            const container = document.getElementById('cardsContainer');
            container.innerHTML = '';
            animationDelay = 0;

            CONFIG.forEach((cat, index) => {
                const allocated = Math.round(appState.totalBudget * cat.percent);
                const spent = appState.expenses[cat.id] || 0;
                const remaining = allocated - spent;
                const progress = allocated > 0 ? (spent / allocated) * 100 : 0;
                
                let progressClass = '';
                if(progress > 80) progressClass = 'warning';
                if(progress > 100) progressClass = 'danger';

                const delay = index * 0.1;

                container.innerHTML += `
                    <div class="card" style="animation-delay: ${delay}s">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <span class="category-name">${cat.name}</span>
                            <span class="amount-display" style="color:${remaining < 0 ? 'var(--danger)' : 'var(--text)'}">
                                ${spent.toLocaleString()} / <small style="color:var(--text-secondary)">${allocated.toLocaleString()}</small>
                            </span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill ${progressClass}" style="width:${Math.min(progress, 100)}%"></div>
                        </div>
                        <div style="display:flex; gap:12px; margin-top:15px; align-items:center;">
                            <input type="number" id="in-${cat.id}" placeholder="المبلغ">
                            <button class="btn-action" onclick="addExpense('${cat.id}')">+ إضافة</button>
                        </div>
                    </div>
                `;
            });
        }

        function updateChart() {
            const ctx = document.getElementById('budgetChart').getContext('2d');
            
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
                            borderWidth: 0,
                            hoverOffset: 10
                        }]
                    },
                    options: {
                        responsive: true,
                        cutout: '60%',
                        plugins: {
                            legend: { 
                                position: 'bottom', 
                                labels: { 
                                    color: '#fff',
                                    padding: 15,
                                    font: { family: 'Cairo', size: 12 }
                                } 
                            },
                            title: { 
                                display: true, 
                                text: 'توزيع المصاريف الفعلي', 
                                color: '#fff',
                                font: { family: 'Cairo', size: 14, weight: 'bold' }
                            }
                        }
                    }
                });
            }
        }

        function generateReport() {
            const totalSpent = Object.values(appState.expenses).reduce((a, b) => a + b, 0);
            const remaining = appState.totalBudget - totalSpent;
            const burnRate = appState.totalBudget > 0 ? (totalSpent / appState.totalBudget) * 100 : 0;
            
            const reportEl = document.getElementById('smartReport');
            const remDisplay = document.getElementById('totalRemainingDisplay');

            remDisplay.innerText = remaining.toLocaleString() + ' دج';
            remDisplay.className = remaining < 0 ? 'remaining-display danger' : 'remaining-display';

            let statusHTML = '';
            let alertClass = '';

            if (remaining < 0) {
                statusHTML = `<strong>🚨 حالة طوارئ!</strong><br>لقد تجاوزت الميزانية بـ <strong>${Math.abs(remaining).toLocaleString()}</strong> دج.<br>يجب التوقف فوراً عن الشراء.`;
                alertClass = 'report-danger';
            } else if (burnRate > 80) {
                statusHTML = `<strong>⚠️ تحذير:</strong><br>لقد استهلكت <strong>${burnRate.toFixed(1)}%</strong> من الميزانية.<br>تبقى لك مبلغ ضئيل جداً.`;
                alertClass = 'report-warning';
            } else {
                statusHTML = `<strong>✅ الوضع مستقر:</strong><br>استهلكت <strong>${burnRate.toFixed(1)}%</strong> فقط.<br>واصل على هذا المنوال.`;
                alertClass = '';
            }

            let maxCat = CONFIG[0];
            let maxVal = 0;
            CONFIG.forEach(c => {
                let s = appState.expenses[c.id] || 0;
                if(s > maxVal) { maxVal = s; maxCat = c; }
            });

            if(maxVal > 0) {
                statusHTML += `<br><br>👉 أكثر ما يستهلك مالك: <strong>${maxCat.name}</strong> (${maxVal.toLocaleString()} دج)`;
            }

            reportEl.innerHTML = statusHTML;
            reportEl.className = 'report-box ' + alertClass;
        }

        function resetAllData() {
            if(confirm('هل أنت متأكد من تصفير جميع البيانات؟')) {
                localStorage.removeItem('dzBudgetPro_v3');
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
