import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import random
from threading import Thread, Lock
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- إعدادات عامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot_pro.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Pro')

# --- متغيرات البيئة ---
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ فشل تحميل المتغيرات: {e}")
    exit(1)

# --- المتغيرات العامة ---
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
# هيكل لحفظ حالة السوق لكل فريم
market_state = {
    "score": 50,
    "regime": "sideways",
    "details": {"1h": "neutral", "4h": "neutral", "1d": "neutral"}
}
is_trading_enabled = False
paper_trading_mode = True
usdt_balance = 1000.0 # رصيد افتراضي للبداية

open_signals_cache = {}
live_prices = {}

locks = {
    'signals': Lock(),
    'prices': Lock(),
    'market': Lock(),
    'balance': Lock()
}

# --- قاعدة البيانات ---
conn = None
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, 
                    entry_price DOUBLE PRECISION, stop_loss DOUBLE PRECISION, 
                    target_price_1 DOUBLE PRECISION, target_price_2 DOUBLE PRECISION,
                    quantity DOUBLE PRECISION, strategy_name TEXT, 
                    status TEXT DEFAULT 'open', is_real_trade BOOLEAN DEFAULT FALSE, 
                    closed_at TIMESTAMP, closing_price DOUBLE PRECISION, profit_pct DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        logger.info("✅ قاعدة البيانات جاهزة.")
    except Exception as e:
        logger.error(f"❌ خطأ DB: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- جلب البيانات ---
def fetch_historical_data(client, symbol, interval, limit=100) -> Optional[pd.DataFrame]:
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in df.columns: df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except: return None

# --- تحليل السوق متعدد الفريمات (MTF) ---
def analyze_market_structure_mtf(client):
    """تحليل السوق بناءً على 3 فريمات زمنية للعملات القيادية"""
    global market_state
    
    timeframes = {'1h': 0.2, '4h': 0.3, '1d': 0.5} # الأوزان: اليومي أهم
    tf_scores = {'1h': [], '4h': [], '1d': []}
    
    logger.info("📊 جاري تحليل هيكل السوق (MTF)...")
    
    for sym in LEADING_SYMBOLS:
        for tf in timeframes.keys():
            df = fetch_historical_data(client, sym, tf, limit=200)
            if df is None: continue
            
            # مؤشرات الاتجاه
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            close = df['close'].iloc[-1]
            
            # نظام نقاط بسيط (-1: هابط, 0: محايد, 1: صاعد)
            score = 0
            if close > ema50: score += 1
            else: score -= 1
            
            if close > ema200: score += 1
            else: score -= 1
            
            if ema50 > ema200: score += 1
            else: score -= 1
            
            # تطبيع النتيجة لتكون بين 0 و 100
            # score range is -3 to +3
            normalized_score = ((score + 3) / 6) * 100
            tf_scores[tf].append(normalized_score)
    
    # حساب المتوسط لكل فريم
    avg_tf_scores = {tf: (sum(scores)/len(scores) if scores else 50) for tf, scores in tf_scores.items()}
    
    # حساب المعدل العام الموزون
    weighted_score = (avg_tf_scores['1h'] * 0.2) + (avg_tf_scores['4h'] * 0.3) + (avg_tf_scores['1d'] * 0.5)
    
    # تحديد الحالة النصية لكل فريم
    details = {}
    for tf, sc in avg_tf_scores.items():
        if sc >= 65: details[tf] = "bullish"
        elif sc <= 35: details[tf] = "bearish"
        else: details[tf] = "neutral"
        
    # تحديد النظام العام
    regime = "sideways"
    if weighted_score >= 65: regime = "bullish"
    elif weighted_score <= 35: regime = "bearish"
    
    with locks['market']:
        market_state = {
            "score": weighted_score,
            "regime": regime,
            "details": details
        }
    
    logger.info(f"🌐 حالة السوق: {regime.upper()} (Score: {weighted_score:.1f}) | {details}")

# --- المؤشرات والاستراتيجيات ---
def calculate_features(df):
    df = df.copy()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
    
    # BB & ATR
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_mid'] + (2*std) - df['bb_lower']) / df['bb_mid']
    
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    return df.fillna(0)

def get_signal_from_strategies(symbol, df, regime):
    last = df.iloc[-1]
    
    # 1. Momentum (Bullish Market)
    if regime == "bullish":
        if last['close'] > last['ema50'] and last['macd_hist'] > 0 and 50 < last['rsi'] < 75:
            return "Momentum_Trend"

    # 2. Range Scalp (Sideways Market)
    elif regime == "sideways":
        if last['bb_width'] < 0.15 and last['close'] <= last['bb_lower'] and last['rsi'] < 35:
            return "Range_Scalp"
            
    # 3. Oversold Bounce (Bearish Market - Caution)
    elif regime == "bearish":
        dist = (last['ema200'] - last['close']) / last['ema200']
        if dist > 0.15 and last['rsi'] < 25: # Deep oversold
            return "Deep_Bounce"
            
    return None

def calculate_entry_params(df, strategy):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    if strategy == "Momentum_Trend":
        return close - (atr*2), close + (atr*2), close + (atr*5)
    elif strategy == "Range_Scalp":
        return close - (atr*1.5), last['bb_mid'], close + (atr*3)
    else:
        return close - (atr*2), close + (atr*1.5), close + (atr*3)

# --- إدارة التنفيذ ---
def send_telegram(msg):
    if not TELEGRAM_TOKEN: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

def save_signal(signal_data):
    check_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, quantity, strategy_name, status, is_real_trade)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s) RETURNING id;
            """, (signal_data['symbol'], signal_data['entry_price'], signal_data['stop_loss'], signal_data['target_price_1'], 
                  signal_data['target_price_2'], signal_data['quantity'], signal_data['strategy_name'], not paper_trading_mode))
            return cur.fetchone()['id']
    except Exception as e:
        logger.error(f"DB Save Error: {e}")
        return int(time.time())

def update_closed_signal(symbol, close_price, profit_pct):
    check_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s
                WHERE symbol=%s AND status='open'
            """, (close_price, profit_pct, symbol))
    except Exception as e: logger.error(f"DB Update Error: {e}")

def execute_trade(client, symbol, side, qty):
    if paper_trading_mode: return True
    try:
        client.create_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
        return True
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        return False

# --- مدير الصفقات وإعادة التحليل ---
def trade_manager(client):
    while True:
        try:
            with locks['signals']: signals = list(open_signals_cache.values())
            if not signals: time.sleep(5); continue
            
            for sig in signals:
                sym = sig['symbol']
                df = fetch_historical_data(client, sym, '5m', 50)
                if df is None: continue
                df = calculate_features(df)
                curr = df.iloc[-1]['close']
                with locks['prices']: live_prices[sym] = curr
                
                # Logic
                sl = float(sig['stop_loss'])
                tp2 = float(sig['target_price_2'])
                profit = (curr - sig['entry_price']) / sig['entry_price'] * 100
                
                reason = None
                if curr <= sl: reason = "Stop Loss"
                elif curr >= tp2: reason = "Target Hit"
                
                # Smart Exit
                if not reason:
                    with locks['market']: regime = market_state['regime']
                    if regime == "bearish" and profit < -1.0: reason = "Market Crash Protection"
                    if profit > 1.5 and df.iloc[-1]['rsi'] < 50: 
                        # Tighten SL logic here (omitted for brevity)
                        pass
                
                if reason:
                    execute_trade(client, sym, 'SELL', sig['quantity'])
                    update_closed_signal(sym, curr, profit)
                    with locks['signals']: del open_signals_cache[sym]
                    send_telegram(f"🔴 *Closed {sym}*\nPnL: `{profit:.2f}%`\nReason: {reason}")
            
            time.sleep(3)
        except Exception as e: logger.error(f"Manager Error: {e}"); time.sleep(5)

# --- المحرك الرئيسي ---
def main_engine():
    try: client = Client(API_KEY, API_SECRET)
    except: logger.critical("API Error"); return
    
    Thread(target=trade_manager, args=(client,), daemon=True).start()
    
    # تحميل قائمة الملف
    try:
        with open('crypto_list.txt') as f: 
            file_symbols = [l.strip().upper() for l in f if l.strip()]
            file_symbols = [s if s.endswith('USDT') else s+'USDT' for s in file_symbols]
    except: file_symbols = ['BTCUSDT', 'ETHUSDT']

    while True:
        try:
            if not is_trading_enabled: time.sleep(5); continue
            
            analyze_market_structure_mtf(client)
            
            # Volume Filter
            tickers = client.get_ticker()
            valid = [t for t in tickers if t['symbol'] in file_symbols]
            sorted_tk = sorted(valid, key=lambda x: float(x['quoteVolume']), reverse=True)[:50]
            active_list = [t['symbol'] for t in sorted_tk]
            
            for sym in active_list:
                with locks['signals']: 
                    if sym in open_signals_cache: continue
                
                df = fetch_historical_data(client, sym, '5m', 100)
                if df is None: continue
                df = calculate_features(df)
                
                with locks['market']: regime = market_state['regime']
                strategy = get_signal_from_strategies(sym, df, regime)
                
                if strategy:
                    curr = df.iloc[-1]['close']
                    sl, tp1, tp2 = calculate_entry_params(df, strategy)
                    qty = 15.0 / curr # 15 USDT Fixed
                    
                    if execute_trade(client, sym, 'BUY', qty):
                        sig = {
                            'symbol': sym, 'entry_price': curr, 'stop_loss': sl,
                            'target_price_1': tp1, 'target_price_2': tp2,
                            'quantity': qty, 'strategy_name': strategy, 'status': 'open'
                        }
                        sig['id'] = save_signal(sig)
                        with locks['signals']: open_signals_cache[sym] = sig
                        send_telegram(f"🟢 *Buy {sym}*\nStrat: {strategy}\nPrice: {curr}")
                
                time.sleep(0.2)
            time.sleep(60)
        except Exception as e: logger.error(f"Engine Error: {e}"); time.sleep(30)

# --- تطبيق الويب والرسوم البيانية ---
app = Flask(__name__)
CORS(app)

DASHBOARD_HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>Pro Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --green: #22c55e; --red: #ef4444; --blue: #3b82f6; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
    .card { background: var(--card); padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .stat-val { font-size: 24px; font-weight: bold; }
    .stat-label { color: #94a3b8; font-size: 14px; }
    
    /* Market Lights */
    .lights { display: flex; gap: 10px; justify-content: center; margin-top: 10px; }
    .light { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px; border: 2px solid #334155; opacity: 0.4; }
    .light.active { opacity: 1; box-shadow: 0 0 10px currentColor; }
    .bullish { background: var(--green); color: #fff; }
    .bearish { background: var(--red); color: #fff; }
    .neutral { background: #f59e0b; color: #fff; }
    
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th { text-align: right; color: #94a3b8; padding: 10px; border-bottom: 1px solid #334155; }
    td { padding: 10px; border-bottom: 1px solid #334155; }
    .btn { width: 100%; padding: 12px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 16px; transition: 0.3s; }
    .btn-start { background: var(--blue); color: white; }
    .btn-stop { background: var(--red); color: white; }
</style>
</head>
<body>
    <div class="header">
        <h1>🚀 ProBot Dashboard</h1>
        <div style="text-align: left;">
            <span id="connectionStatus" style="color: var(--green);">● متصل</span>
            <br>
            <small id="tradingMode">ورقي 📝</small>
        </div>
    </div>

    <!-- Market Structure & Stats -->
    <div class="grid">
        <div class="card" style="text-align: center;">
            <h3>🚦 هيكل السوق (MTF)</h3>
            <div class="stat-val" id="marketScore">--</div>
            <div class="lights">
                <div id="light-1h" class="light neutral">1H</div>
                <div id="light-4h" class="light neutral">4H</div>
                <div id="light-1d" class="light neutral">1D</div>
            </div>
            <p style="margin-top: 10px; color: #94a3b8;" id="regimeText">--</p>
        </div>

        <div class="card">
            <h3>📊 إحصائيات الأداء</h3>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div><div class="stat-val" id="totalProfit">0%</div><div class="stat-label">إجمالي الربح</div></div>
                <div><div class="stat-val" id="winRate">0%</div><div class="stat-label">نسبة الفوز</div></div>
                <div><div class="stat-val" id="tradeCount">0</div><div class="stat-label">صفقات مغلقة</div></div>
            </div>
            <button id="toggleBtn" class="btn btn-start" onclick="toggleBot()">تشغيل البوت</button>
        </div>
    </div>

    <!-- Performance Chart -->
    <div class="card">
        <h3>📈 منحنى نمو رأس المال (Equity Curve)</h3>
        <div style="height: 300px;">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <!-- Active Trades -->
    <div class="card">
        <h3>⚡ الصفقات النشطة</h3>
        <div style="overflow-x: auto;">
            <table>
                <thead><tr><th>العملة</th><th>الاستراتيجية</th><th>الدخول</th><th>السعر الحالي</th><th>الربح %</th></tr></thead>
                <tbody id="tradesTable"></tbody>
            </table>
        </div>
    </div>

<script>
    let chartInstance = null;

    function initChart() {
        const ctx = document.getElementById('equityChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Growth %', data: [], borderColor: '#3b82f6', tension: 0.4, fill: true, backgroundColor: 'rgba(59, 130, 246, 0.1)' }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { display: false } }, y: { grid: { color: '#334155' } } }, plugins: { legend: { display: false } } }
        });
    }

    function updateDashboard() {
        fetch('/api/data').then(r => r.json()).then(d => {
            // 1. Market Lights
            document.getElementById('marketScore').innerText = d.market.score.toFixed(0);
            document.getElementById('regimeText').innerText = d.market.regime.toUpperCase();
            
            ['1h', '4h', '1d'].forEach(tf => {
                const el = document.getElementById(`light-${tf}`);
                el.className = `light ${d.market.details[tf] || 'neutral'} active`;
            });

            // 2. Stats
            document.getElementById('totalProfit').innerText = d.stats.total_pnl.toFixed(2) + '%';
            document.getElementById('winRate').innerText = d.stats.win_rate.toFixed(1) + '%';
            document.getElementById('tradeCount').innerText = d.stats.closed_count;
            document.getElementById('tradingMode').innerText = d.is_paper ? "ورقي 📝" : "حقيقي 💵";

            // 3. Button
            const btn = document.getElementById('toggleBtn');
            if(d.is_enabled) { btn.innerText = "إيقاف البوت 🛑"; btn.className = "btn btn-stop"; }
            else { btn.innerText = "تشغيل البوت ▶️"; btn.className = "btn btn-start"; }

            // 4. Table
            const tbody = document.getElementById('tradesTable');
            if(d.signals.length === 0) tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#64748b">لا توجد صفقات نشطة</td></tr>';
            else {
                tbody.innerHTML = d.signals.map(s => {
                    const price = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((price - s.entry_price) / s.entry_price) * 100;
                    return `<tr>
                        <td><b>${s.symbol}</b></td>
                        <td><small>${s.strategy_name}</small></td>
                        <td>${s.entry_price}</td>
                        <td>${price}</td>
                        <td style="color: ${pnl >= 0 ? '#22c55e' : '#ef4444'}"><b>${pnl.toFixed(2)}%</b></td>
                    </tr>`;
                }).join('');
            }

            // 5. Chart
            if(chartInstance) {
                chartInstance.data.labels = d.equity_curve.labels;
                chartInstance.data.datasets[0].data = d.equity_curve.data;
                chartInstance.update();
            }
        });
    }

    function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(updateDashboard); }

    initChart();
    setInterval(updateDashboard, 2000);
    updateDashboard();
</script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/data')
def api_data():
    with locks['market']: mkt = market_state
    with locks['signals']: sigs = list(open_signals_cache.values())
    with locks['prices']: prc = live_prices.copy()
    
    # حساب إحصائيات الأداء من قاعدة البيانات
    stats = {"total_pnl": 0, "win_rate": 0, "closed_count": 0}
    curve = {"labels": [], "data": []}
    
    check_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_pct, closed_at FROM signals WHERE status='closed' ORDER BY closed_at ASC")
            rows = cur.fetchall()
            if rows:
                stats['closed_count'] = len(rows)
                stats['total_pnl'] = sum(r['profit_pct'] for r in rows)
                wins = len([r for r in rows if r['profit_pct'] > 0])
                stats['win_rate'] = (wins / len(rows)) * 100
                
                # بناء المنحنى التراكمي
                cum_pnl = 0
                for r in rows:
                    cum_pnl += r['profit_pct']
                    curve['labels'].append(r['closed_at'].strftime('%m-%d %H:%M'))
                    curve['data'].append(cum_pnl)
            else:
                # بيانات افتراضية للرسم البياني الفارغ
                curve['labels'] = [datetime.now().strftime('%H:%M')]
                curve['data'] = [0]

    except: pass

    return jsonify({
        "market": mkt,
        "signals": sigs,
        "prices": prc,
        "stats": stats,
        "equity_curve": curve,
        "is_enabled": is_trading_enabled,
        "is_paper": paper_trading_mode
    })

@app.route('/api/toggle', methods=['POST'])
def api_toggle():
    global is_trading_enabled
    is_trading_enabled = not is_trading_enabled
    return jsonify({"status": is_trading_enabled})

if __name__ == "__main__":
    print("="*50)
    print("🚀 ProBot V3 Loaded.")
    print("="*50)
    init_db()
    Thread(target=main_engine, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)