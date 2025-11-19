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
from flask import Flask, jsonify, render_template_string, request, redirect, url_for
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- إعدادات عامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot_v5.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_V5')

# --- تحميل متغيرات البيئة ---
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ فشل تحميل المتغيرات: {e}")
    exit(1)

# --- الإعدادات الديناميكية ---
BOT_SETTINGS = {
    "is_trading_enabled": False, # الحالة الافتراضية (يمكن تغييرها من الزر)
    "paper_trading_mode": True,
    "trade_amount_usdt": 15.0,
    "max_open_trades": 5,
    "volume_filter_limit": 50,
    "report_interval_hours": 4 # كل كم ساعة يرسل تقرير
}

# --- المتغيرات العامة ---
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
market_state = {
    "score": 50,
    "regime": "sideways", # الحالة الحالية
    "prev_regime": "sideways", # الحالة السابقة (للمقارنة)
    "details": {"1h": "neutral", "4h": "neutral", "1d": "neutral"},
    "last_update": None
}

open_signals_cache = {}
live_prices = {}

locks = {
    'signals': Lock(),
    'prices': Lock(),
    'market': Lock(),
    'settings': Lock()
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
                    exit_reason TEXT, created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        logger.info("✅ قاعدة البيانات جاهزة.")
    except Exception as e:
        logger.error(f"❌ خطأ DB: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- نظام التنبيهات المتطور (Telegram) ---
def send_telegram(type, data):
    """
    أنواع الرسائل:
    - MARKET_CHANGE: تغير هيكل السوق
    - PERIODIC_REPORT: تقرير دوري
    - BUY/SELL: صفقات
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return

    msg = ""
    if type == "MARKET_CHANGE":
        icon = "🟢" if data['new'] == 'bullish' else ("🔴" if data['new'] == 'bearish' else "🟠")
        msg = (
            f"🔔 *تنبيه تغير السوق*\n"
            f"تغير الهيكل من `{data['old'].upper()}` إلى `{data['new'].upper()}` {icon}\n"
            f"النقاط: `{data['score']}/100`"
        )
    
    elif type == "PERIODIC_REPORT":
        status_icon = "✅ يعمل" if data['enabled'] else "🛑 متوقف"
        msg = (
            f"📊 *تقرير البوت الدوري*\n"
            f"━━━━━━━━━━━━━━\n"
            f"الحالة: {status_icon}\n"
            f"السوق: `{data['regime'].upper()}`\n"
            f"الصفقات المفتوحة: `{data['open_count']}`\n"
            f"إجمالي الربح المحقق: `{data['total_pnl']:.2f}%`\n"
            f"الوقت: `{datetime.now().strftime('%H:%M')}`"
        )

    elif type == "BUY":
        mode = "📝 ورقي" if BOT_SETTINGS['paper_trading_mode'] else "💵 حقيقي"
        msg = (
            f"🟢 *شراء جديد ({data['symbol']})*\n"
            f"الاستراتيجية: `{data['strategy']}`\n"
            f"السعر: `{data['price']}` | الوقف: `{data['sl']}`\n"
            f"الوضع: {mode}"
        )

    elif type == "SELL":
        emoji = "✅" if data['profit'] > 0 else "🔻"
        msg = (
            f"{emoji} *إغلاق ({data['symbol']})*\n"
            f"الربح: `{data['profit']:.2f}%`\n"
            f"السبب: _{data['reason']}_"
        )

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        Thread(target=requests.post, args=(url,), kwargs={'data': payload}).start()
    except: pass

# --- حلقة التقارير الدورية ---
def periodic_report_loop():
    logger.info("🕒 بدء خدمة التقارير الدورية...")
    while True:
        try:
            interval = BOT_SETTINGS['report_interval_hours'] * 3600
            time.sleep(interval)
            
            # تجميع البيانات
            with locks['settings']: enabled = BOT_SETTINGS['is_trading_enabled']
            with locks['market']: regime = market_state['regime']
            with locks['signals']: open_count = len(open_signals_cache)
            
            total_pnl = 0
            check_db()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT SUM(profit_pct) as total FROM signals WHERE status='closed'")
                    res = cur.fetchone()
                    if res and res['total']: total_pnl = res['total']
            except: pass

            data = {
                "enabled": enabled, "regime": regime, 
                "open_count": open_count, "total_pnl": total_pnl
            }
            send_telegram("PERIODIC_REPORT", data)
            
        except Exception as e:
            logger.error(f"Report Loop Error: {e}"); time.sleep(60)

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

# --- تحليل السوق (مع التنبيه عند التغير) ---
def analyze_market_structure(client):
    global market_state
    
    timeframes = {'1h': 0.2, '4h': 0.3, '1d': 0.5}
    tf_scores = {'1h': [], '4h': [], '1d': []}
    
    for sym in LEADING_SYMBOLS:
        for tf in timeframes:
            df = fetch_historical_data(client, sym, tf, limit=200)
            if df is None: continue
            close = df['close'].iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            score = 1 if close > ema200 else 0
            tf_scores[tf].append(score * 100)
            
    avg_scores = {tf: (sum(s)/len(s) if s else 50) for tf, s in tf_scores.items()}
    total_score = sum(avg_scores[tf] * w for tf, w in timeframes.items())
    
    new_regime = "sideways"
    if total_score >= 65: new_regime = "bullish"
    elif total_score <= 35: new_regime = "bearish"
    
    details = {}
    for tf, sc in avg_scores.items():
        details[tf] = "bullish" if sc > 60 else ("bearish" if sc < 40 else "neutral")
    
    # التحقق من التغير لإرسال تنبيه
    with locks['market']:
        old_regime = market_state['regime']
        if new_regime != old_regime:
            logger.info(f"🔔 Market Regime Changed: {old_regime} -> {new_regime}")
            send_telegram("MARKET_CHANGE", {"old": old_regime, "new": new_regime, "score": total_score})
        
        market_state = {
            "score": total_score, "regime": new_regime, "prev_regime": old_regime,
            "details": details, "last_update": datetime.now()
        }
    
    logger.info(f"🌐 Market Analysis: {new_regime.upper()} ({total_score:.1f})")

# --- المؤشرات والاستراتيجيات ---
def calculate_features(df):
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
    
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    return df.fillna(0)

def get_strategy_signal(symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if regime == "bullish":
        # Volatility Breakout
        if prev['bb_width'] < 0.15 and last['close'] > last['bb_upper'] and last['volume'] > prev['volume'] * 1.5:
            return "Volatility_Breakout"
        # Trend Pullback
        if last['close'] > last['ema200'] and last['close'] < last['ema20'] and last['rsi'] < 55 and last['macd_hist'] > prev['macd_hist']:
            return "Trend_Pullback"
            
    if regime == "sideways":
        # Liquidity Grab
        if last['low'] < last['bb_lower'] and last['close'] > last['bb_lower'] and last['close'] > last['open']:
            return "SMC_Liquidity_Grab"

    return None

def calculate_params(df, strategy):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    if strategy == "Volatility_Breakout":
        return close - (atr * 2), close + (atr * 4)
    elif strategy == "SMC_Liquidity_Grab":
        return last['low'] - (atr * 0.5), last['bb_mid']
    else: 
        return close - (atr * 1.5), close + (atr * 3)

# --- إدارة التنفيذ ---
def execute_trade(client, symbol, side, qty):
    with locks['settings']: is_paper = BOT_SETTINGS['paper_trading_mode']
    if is_paper: return True
    try:
        client.create_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
        return True
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        return False

def save_signal(data):
    check_db()
    try:
        with locks['settings']: is_paper = BOT_SETTINGS['paper_trading_mode']
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, quantity, strategy_name, status, is_real_trade)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s) RETURNING id;
            """, (data['symbol'], data['entry_price'], data['sl'], data['tp'], data['tp'], data['qty'], data['strat'], not is_paper))
            return cur.fetchone()['id']
    except: return int(time.time())

def close_signal_db(symbol, price, reason):
    check_db()
    try:
        profit = 0.0
        with locks['signals']:
            if symbol in open_signals_cache:
                entry = open_signals_cache[symbol]['entry_price']
                profit = ((price - entry) / entry) * 100
                del open_signals_cache[symbol]
        
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, exit_reason=%s WHERE symbol=%s AND status='open'", (price, profit, reason, symbol))
        send_telegram("SELL", {"symbol": symbol, "price": price, "profit": profit, "reason": reason})
    except: pass

def trade_manager_loop(client):
    while True:
        try:
            with locks['signals']: signals = list(open_signals_cache.values())
            if not signals: time.sleep(2); continue
            
            for sig in signals:
                sym = sig['symbol']
                df = fetch_historical_data(client, sym, '5m', 50)
                if df is None: continue
                df = calculate_features(df)
                curr = df.iloc[-1]['close']
                with locks['prices']: live_prices[sym] = curr
                
                sl = float(sig['stop_loss'])
                tp = float(sig['target_price_1'])
                
                reason = None
                if curr <= sl: reason = "Stop Loss 🛑"
                elif curr >= tp: reason = "Take Profit 🎯"
                
                if reason:
                    execute_trade(client, sym, 'SELL', sig['quantity'])
                    close_signal_db(sym, curr, reason)
            time.sleep(2)
        except Exception as e: logger.error(f"Manager Error: {e}"); time.sleep(5)

def main_bot_loop():
    try: client = Client(API_KEY, API_SECRET)
    except: logger.critical("API Error"); return
    
    Thread(target=trade_manager_loop, args=(client,), daemon=True).start()
    Thread(target=periodic_report_loop, daemon=True).start() # تشغيل تقارير تلغرام
    
    try:
        with open('crypto_list.txt') as f:
            file_symbols = [l.strip().upper().replace('\n','') for l in f if l.strip()]
            file_symbols = [s if s.endswith('USDT') else s+'USDT' for s in file_symbols]
    except: file_symbols = ['BTCUSDT', 'ETHUSDT']

    while True:
        try:
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                limit = BOT_SETTINGS['volume_filter_limit']
                amount = BOT_SETTINGS['trade_amount_usdt']
                max_trades = BOT_SETTINGS['max_open_trades']

            if not enabled: time.sleep(5); continue
            
            analyze_market_structure(client)
            
            tickers = client.get_ticker()
            valid = [t for t in tickers if t['symbol'] in file_symbols]
            sorted_tk = sorted(valid, key=lambda x: float(x['quoteVolume']), reverse=True)[:limit]
            
            for t in sorted_tk:
                sym = t['symbol']
                with locks['signals']:
                    if sym in open_signals_cache: continue
                    if len(open_signals_cache) >= max_trades: break
                
                df = fetch_historical_data(client, sym, '5m', 100)
                if df is None: continue
                df = calculate_features(df)
                
                with locks['market']: regime = market_state['regime']
                strategy = get_strategy_signal(sym, df, regime)
                
                if strategy:
                    curr = df.iloc[-1]['close']
                    sl, tp = calculate_params(df, strategy)
                    qty = amount / curr
                    
                    if execute_trade(client, sym, 'BUY', qty):
                        sig = {'symbol': sym, 'entry_price': curr, 'sl': sl, 'tp': tp, 'qty': qty, 'strat': strategy}
                        db_id = save_signal(sig)
                        cache = {'id': db_id, 'symbol': sym, 'entry_price': curr, 'stop_loss': sl, 'target_price_1': tp, 'quantity': qty, 'strategy_name': strategy}
                        with locks['signals']: open_signals_cache[sym] = cache
                        send_telegram("BUY", {"symbol": sym, "strategy": strategy, "price": curr, "sl": sl, "tp": tp})
                time.sleep(0.2)
            time.sleep(60)
        except Exception as e: logger.error(f"Main: {e}"); time.sleep(30)

# --- تطبيق الويب ---
app = Flask(__name__)
CORS(app)

DASHBOARD_HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>SmartBot V5</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
<style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #3b82f6; --green: #22c55e; --red: #ef4444; }
    body { background: var(--bg); color: var(--text); font-family: 'Cairo', sans-serif; margin: 0; padding: 20px; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
    .card { background: var(--card); padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    /* Control Panel */
    .control-panel { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .toggle-btn {
        padding: 15px 30px; border: none; border-radius: 10px;
        font-size: 1.1rem; font-weight: bold; cursor: pointer;
        width: 100%; transition: 0.3s; color: white;
    }
    .btn-on { background: var(--green); box-shadow: 0 0 15px rgba(34, 197, 94, 0.4); }
    .btn-off { background: var(--red); opacity: 0.8; }

    table { width: 100%; border-collapse: collapse; }
    td, th { padding: 12px; border-bottom: 1px solid #334155; text-align: right; }
    .nav-link { color: #94a3b8; text-decoration: none; margin-left: 15px; font-size: 1.1rem; }
</style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🚀 SmartBot V5</h1>
            <small id="statusText" style="color: var(--green)">● متصل</small>
        </div>
        <div>
            <a href="/settings" class="nav-link">⚙️ الإعدادات</a>
        </div>
    </div>

    <!-- Control Card (New) -->
    <div class="card" style="margin-bottom: 20px; border: 1px solid #334155;">
        <div class="control-panel">
            <div>
                <h3>حالة البوت: <span id="botStateLabel">--</span></h3>
                <small id="modeLabel" style="color: #94a3b8;">--</small>
            </div>
            <div style="width: 200px;">
                <button id="mainToggleBtn" class="toggle-btn btn-off" onclick="toggleBot()">تشغيل</button>
            </div>
        </div>
    </div>

    <div class="grid">
        <!-- Market -->
        <div class="card">
            <h3>📊 هيكل السوق</h3>
            <div style="font-size: 2rem; font-weight: bold;" id="marketScore">--</div>
            <div id="regimeText" style="font-size: 1.2rem; font-weight: bold; color: #f59e0b;">--</div>
            <div style="margin-top: 10px;">
                <span id="light-1h">1H</span> | <span id="light-4h">4H</span> | <span id="light-1d">1D</span>
            </div>
        </div>

        <!-- Stats -->
        <div class="card">
            <h3>💰 الأداء</h3>
            <div style="font-size: 2rem; font-weight: bold;" id="totalPnl">0.00%</div>
            <div>الصفقات المغلقة: <b id="tradesCount">0</b></div>
            <div>نسبة الفوز: <b id="winRate">0%</b></div>
        </div>
    </div>

    <div class="card">
        <h3>⚡ الصفقات المفتوحة</h3>
        <table>
            <thead><tr><th>الرمز</th><th>الاستراتيجية</th><th>الدخول</th><th>الحالي</th><th>الربح</th></tr></thead>
            <tbody id="tradesTable"></tbody>
        </table>
    </div>

<script>
    function update() {
        fetch('/api/dashboard').then(r=>r.json()).then(d => {
            // 1. Control Button
            const btn = document.getElementById('mainToggleBtn');
            const stateLabel = document.getElementById('botStateLabel');
            
            if (d.settings.enabled) {
                btn.innerText = "إيقاف 🛑";
                btn.className = "toggle-btn btn-on";
                stateLabel.innerText = "يعمل ✅";
                stateLabel.style.color = "var(--green)";
            } else {
                btn.innerText = "تشغيل ▶️";
                btn.className = "toggle-btn btn-off";
                stateLabel.innerText = "متوقف ⏸️";
                stateLabel.style.color = "var(--red)";
            }
            document.getElementById('modeLabel').innerText = d.settings.paper ? "وضع التداول الورقي (آمن)" : "وضع التداول الحقيقي (Real Money)";

            // 2. Market
            document.getElementById('marketScore').innerText = d.market.score.toFixed(0);
            document.getElementById('regimeText').innerText = d.market.regime.toUpperCase();
            
            // 3. Stats
            document.getElementById('totalPnl').innerText = d.stats.pnl.toFixed(2) + "%";
            document.getElementById('tradesCount').innerText = d.stats.count;
            document.getElementById('winRate').innerText = d.stats.win_rate.toFixed(1) + "%";

            // 4. Table
            const tbody = document.getElementById('tradesTable');
            if(d.signals.length === 0) tbody.innerHTML = "<tr><td colspan='5' style='text-align:center;color:#94a3b8;'>لا توجد صفقات</td></tr>";
            else {
                tbody.innerHTML = d.signals.map(s => {
                    const price = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((price - s.entry_price)/s.entry_price)*100;
                    return `<tr>
                        <td><b>${s.symbol}</b></td>
                        <td>${s.strategy_name}</td>
                        <td>${s.entry_price}</td>
                        <td>${price}</td>
                        <td style="color:${pnl>=0?'var(--green)':'var(--red)'}"><b>${pnl.toFixed(2)}%</b></td>
                    </tr>`;
                }).join('');
            }
        });
    }

    function toggleBot() {
        fetch('/api/toggle_trading', {method: 'POST'}).then(update);
    }

    setInterval(update, 2000);
    update();
</script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/settings')
def settings_page():
    # صفحة إعدادات مبسطة (للاختصار في هذا الرد)
    return render_template_string("""
        <html><body><h1>Settings</h1>
        <form action="/api/settings" method="POST">
            Enable Paper Trading: <input type="checkbox" name="paper_mode" checked><br>
            Trade Amount: <input type="number" name="amount" value="15"><br>
            <button type="submit">Save</button>
        </form>
        </body></html>
    """)

@app.route('/api/dashboard')
def api_dashboard():
    with locks['market']: mkt = market_state.copy()
    with locks['signals']: sigs = list(open_signals_cache.values())
    with locks['prices']: prices = live_prices.copy()
    with locks['settings']: 
        enabled = BOT_SETTINGS['is_trading_enabled']
        paper = BOT_SETTINGS['paper_trading_mode']
    
    # Stats from DB
    stats = {'pnl': 0, 'count': 0, 'win_rate': 0}
    check_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_pct FROM signals WHERE status='closed'")
            rows = cur.fetchall()
            if rows:
                stats['count'] = len(rows)
                stats['pnl'] = sum(r['profit_pct'] for r in rows)
                wins = len([r for r in rows if r['profit_pct'] > 0])
                stats['win_rate'] = (wins/len(rows))*100
    except: pass

    return jsonify({
        "market": mkt, "signals": sigs, "prices": prices, "stats": stats,
        "settings": {"enabled": enabled, "paper": paper}
    })

@app.route('/api/toggle_trading', methods=['POST'])
def api_toggle():
    with locks['settings']:
        BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    logger.info(f"Bot Toggled: {BOT_SETTINGS['is_trading_enabled']}")
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("🚀 SmartBot V5 Started...")
    init_db()
    Thread(target=main_bot_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)