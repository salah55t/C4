import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import random
from threading import Thread, Lock
from datetime import datetime, timedelta
from collections import deque
from decouple import config
from typing import List, Dict, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request, redirect, url_for
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- 1. إعدادات النظام ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot_v8.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_V8')

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ Config Error: {e}")
    exit(1)

# --- 2. المتغيرات والإعدادات ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "trade_amount_usdt": 20.0,
    "max_open_trades": 5,
    "stop_loss_atr_multiplier": 2.0, # From video concept
    "trailing_atr_multiplier": 2.3,  # Specific value from video
    "volume_filter_limit": 40
}

LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
market_state = {
    "score": 50, "regime": "sideways", 
    "details": {"1h": 50, "4h": 50, "1d": 50},
    "last_update": None
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=50)
performance_history = []

locks = {
    'signals': Lock(), 'prices': Lock(), 'market': Lock(), 
    'settings': Lock(), 'logs': Lock()
}

# --- 3. قاعدة البيانات ---
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
                    exit_reason TEXT, atr_at_entry DOUBLE PRECISION, created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        logger.info("✅ Database Initialized.")
    except Exception as e: logger.error(f"DB Error: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 4. التنبيهات ---
def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    msg = ""
    if event == "BUY":
        msg = (f"🚀 *New Trade ({payload['symbol']})*\n"
               f"Strategy: `{payload['strategy']}`\n"
               f"Price: `{payload['price']}`\n"
               f"Mode: {'📝 Paper' if BOT_SETTINGS['paper_trading_mode'] else '💵 Real'}")
    elif event == "SELL":
        emoji = "✅" if payload['profit'] > 0 else "🔻"
        msg = (f"{emoji} *Closed ({payload['symbol']})*\nProfit: `{payload['profit']:.2f}%`\nReason: _{payload['reason']}_")
    elif event == "UPDATE":
        msg = (f"🛡️ *Trailing Stop Update ({payload['symbol']})*\nNew SL: `{payload['new_sl']}`")
    
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- 5. التحليل الفني (تم إضافة AO من الفيديو) ---
def fetch_data(client, symbol, interval, limit=100):
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except: return None

def add_indicators(df):
    df = df.copy()
    # Standard Indicators
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # ATR (Video Specific: used for exits)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr35'] = df['tr'].rolling(35).mean() # For Video Strategy Trailing
    
    # Awesome Oscillator (Video Strategy Core)
    median_price = (df['high'] + df['low']) / 2
    df['ao'] = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    # Recent High/Low (For Breakout)
    df['recent_high'] = df['high'].rolling(20).max() # 20 candles lookback
    
    return df.fillna(0)

# --- استراتيجيات التداول (محدثة من الفيديو) ---
def get_strategy_signal(symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الفلتر العام (تأكيد الحجم والاتجاه)
    if last['volume'] < last['vol_ma'] * 0.8: return None, "Low Volume"
    
    # 1. AO Breakout Strategy (مستوحاة من الفيديو)
    # الشروط: AO يعبر الصفر للأعلى + السعر يكسر قمة الـ 20 شمعة السابقة
    if last['ao'] > 0 and prev['ao'] <= 0: # Zero Cross
        if last['close'] > prev['recent_high']: # Breakout Confirmation
            if regime == "bullish" or regime == "sideways":
                return "AO_Breakout", "AO Cross + High Breakout"
    
    # 2. Volatility Squeeze Breakout
    if regime == "bullish":
        if prev['bb_width'] < 0.15 and last['close'] > last['bb_upper']:
            return "Vol_Breakout", "Bollinger Squeeze Break"

    # 3. Trend Pullback
    if regime == "bullish":
        if last['close'] > last['ema200'] and last['rsi'] < 55 and last['rsi'] > 40 and last['close'] > last['ema50']:
             # Simple pullback to EMA50 area
            return "Trend_Pullback", "EMA50 Bounce"

    return None, "No Signal"

def calculate_params(df, strategy):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    # معايير الفيديو: Stop = ATR based
    if strategy == "AO_Breakout":
        sl = close - (atr * 2.0)
        tp = close + (atr * 5.0) # هدف مفتوح للترند
    elif strategy == "Vol_Breakout":
        sl = close - (atr * 2.0)
        tp = close + (atr * 4.0)
    else:
        sl = close - (atr * 1.5)
        tp = close + (atr * 3.0)
        
    return sl, tp

# --- 6. المحرك الذكي (Active Brain) ---
def reanalyze_position(symbol, signal, df, regime):
    """
    تطبيق منطق الفيديو: Trailing Stop باستخدام ATR
    """
    last = df.iloc[-1]
    curr = float(last['close'])
    entry = float(signal['entry_price'])
    sl = float(signal['stop_loss'])
    
    profit_pct = (curr - entry) / entry * 100
    
    # Video Logic: Trailing Stop = 2.3 * ATR(35)
    # نطبقه فقط عندما تكون الصفقة رابحة لضمان حجز الربح
    if profit_pct > 1.0:
        atr35 = last['atr35']
        trailing_dist = atr35 * BOT_SETTINGS['trailing_atr_multiplier']
        
        new_sl = curr - trailing_dist
        
        # تحديث الوقف فقط إذا كان أعلى من الوقف الحالي (لا ننزله أبداً)
        if new_sl > sl:
            return "UPDATE_SL", new_sl, "ATR Trailing Logic"

    # Exit if trend reverses violently below EMA200 in bullish setup
    if regime == "bearish" and curr < last['ema200'] and profit_pct < -1:
        return "CLOSE_NOW", curr, "Trend Reversal (Below EMA200)"

    return "HOLD", 0, ""

# --- 7. التنفيذ ---
def execute_order(client, symbol, side, qty):
    if BOT_SETTINGS['paper_trading_mode']: return True
    try:
        client.create_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
        return True
    except Exception as e:
        logger.error(f"Exec Error: {e}")
        return False

def trade_manager(client):
    logger.info("🛡️ Trade Manager Running...")
    while True:
        try:
            with locks['signals']: signals = list(open_signals_cache.values())
            if not signals: time.sleep(5); continue
            
            for sig in signals:
                sym = sig['symbol']
                df = fetch_data(client, sym, '5m', 60)
                if df is None: continue
                df = add_indicators(df)
                curr = df.iloc[-1]['close']
                with locks['prices']: live_prices[sym] = curr
                
                sl, tp = float(sig['stop_loss']), float(sig['target_price_1'])
                
                reason = None
                if curr <= sl: reason = "Stop Loss 🛑"
                elif curr >= tp: reason = "Take Profit 🎯"
                
                if not reason:
                    with locks['market']: regime = market_state['regime']
                    act, val, note = reanalyze_position(sym, sig, df, regime)
                    
                    if act == "UPDATE_SL":
                        with locks['signals']: open_signals_cache[sym]['stop_loss'] = val
                        # Update DB
                        check_db()
                        with conn.cursor() as cur: cur.execute("UPDATE signals SET stop_loss=%s WHERE id=%s", (val, sig['id']))
                        send_telegram("UPDATE", {"symbol": sym, "new_sl": val})
                    
                    elif act == "CLOSE_NOW":
                        execute_order(client, sym, 'SELL', sig['quantity'])
                        close_signal_db(sym, curr, f"Smart Exit: {note}")
                        continue

                if reason:
                    execute_order(client, sym, 'SELL', sig['quantity'])
                    close_signal_db(sym, curr, reason)
            
            time.sleep(5)
        except: time.sleep(5)

# --- 8. تحليل هيكل السوق ---
def analyze_mtf(client):
    global market_state
    timeframes = ['1h', '4h', '1d']
    tf_scores = {'1h':0, '4h':0, '1d':0}
    
    for sym in LEADING_SYMBOLS:
        for tf in timeframes:
            df = fetch_data(client, sym, tf, 100)
            if df is None: continue
            close = df['close'].iloc[-1]
            ema = df['close'].ewm(span=200).mean().iloc[-1]
            if close > ema: tf_scores[tf] += 25 # Max 100 per TF
            
    final = {k: v for k,v in tf_scores.items()}
    score = (final['1h']*0.2 + final['4h']*0.3 + final['1d']*0.5)
    regime = "bullish" if score >= 60 else ("bearish" if score <= 40 else "sideways")
    
    with locks['market']:
        if market_state['regime'] != regime:
            pass # Could verify change
        market_state = {"score": score, "regime": regime, "details": final, "last_update": datetime.now()}

# --- 9. Main Loop ---
def main_loop():
    try: client = Client(API_KEY, API_SECRET)
    except: return
    
    Thread(target=trade_manager, args=(client,), daemon=True).start()
    
    try:
        with open('crypto_list.txt') as f:
            file_symbols = [l.strip().upper().replace('\n','') for l in f if l.strip()]
            file_symbols = [s if s.endswith('USDT') else s+'USDT' for s in file_symbols]
    except: file_symbols = ['BTCUSDT']

    while True:
        try:
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                limit = BOT_SETTINGS['volume_filter_limit']
                amt = BOT_SETTINGS['trade_amount_usdt']
                max_t = BOT_SETTINGS['max_open_trades']
            
            if not enabled: time.sleep(5); continue
            
            analyze_mtf(client)
            
            # Volume Filter
            tickers = client.get_ticker()
            valid = [t for t in tickers if t['symbol'] in file_symbols]
            sorted_tk = sorted(valid, key=lambda x: float(x['quoteVolume']), reverse=True)[:limit]
            
            for t in sorted_tk:
                sym = t['symbol']
                with locks['signals']:
                    if sym in open_signals_cache: continue
                    if len(open_signals_cache) >= max_t: break
                
                df = fetch_data(client, sym, '5m', 100)
                if df is None: continue
                df = add_indicators(df)
                
                with locks['market']: regime = market_state['regime']
                strat, reason = get_strategy_signal(sym, df, regime)
                
                if strat:
                    curr = df.iloc[-1]['close']
                    sl, tp = calculate_params(df, strat)
                    qty = amt / curr
                    atr = df.iloc[-1]['atr']
                    
                    if execute_order(client, sym, 'BUY', qty):
                        sig = {'symbol': sym, 'entry_price': curr, 'sl': sl, 'tp': tp, 'qty': qty, 'strat': strat, 'atr': atr}
                        db_id = save_signal_db(sig)
                        cache = {'id': db_id, 'symbol': sym, 'entry_price': curr, 'stop_loss': sl, 'target_price_1': tp, 'quantity': qty, 'strategy_name': strat}
                        with locks['signals']: open_signals_cache[sym] = cache
                        
                        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'ACCEPTED', 'r': strat})
                        send_telegram("BUY", {"symbol": sym, "strategy": strat, "price": curr})
                else:
                    if random.random() < 0.05:
                        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'REJECTED', 'r': reason})
                
                time.sleep(0.2)
            time.sleep(60)
        except Exception as e: logger.error(f"Loop: {e}"); time.sleep(30)

# --- DB Utils ---
def save_signal_db(data):
    check_db()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, quantity, strategy_name, status, is_real_trade, atr_at_entry) VALUES (%s,%s,%s,%s,%s,%s,%s,'open',%s,%s) RETURNING id", 
            (data['symbol'], data['entry_price'], data['sl'], data['tp'], data['tp'], data['qty'], data['strat'], not BOT_SETTINGS['paper_trading_mode'], data['atr']))
            return cur.fetchone()['id']
    except: return int(time.time())

def close_signal_db(symbol, price, reason):
    check_db()
    try:
        profit = 0.0
        with locks['signals']:
            if symbol in open_signals_cache:
                profit = ((price - open_signals_cache[symbol]['entry_price']) / open_signals_cache[symbol]['entry_price']) * 100
                del open_signals_cache[symbol]
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, exit_reason=%s WHERE symbol=%s AND status='open'", (price, profit, reason, symbol))
        send_telegram("SELL", {"symbol": symbol, "price": price, "profit": profit, "reason": reason})
    except: pass

# --- Flask App ---
app = Flask(__name__)
CORS(app)

DASHBOARD = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>SmartBot V8 Algo Master</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
<style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --accent: #8b5cf6; --green: #10b981; --red: #ef4444; --orange: #f59e0b; }
    body { background: var(--bg); color: var(--text); font-family: 'Tajawal', sans-serif; margin: 0; padding: 20px; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
    .card { background: var(--card); padding: 20px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
    .btn { padding: 10px 20px; border-radius: 8px; border: none; font-weight: bold; cursor: pointer; color: white; text-decoration: none; }
    .btn-run { background: var(--green); } .btn-stop { background: var(--red); }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9rem; }
    th, td { padding: 10px; border-bottom: 1px solid #334155; text-align: right; }
    .status-ACCEPTED { color: var(--green); } .status-REJECTED { color: var(--orange); }
</style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🤖 SmartBot <span style="color:var(--accent)">V8 Master</span></h1>
            <small style="color:#94a3b8">تطبيق استراتيجيات الفيديو التعليمي</small>
        </div>
        <div>
            <a href="/settings" class="btn" style="background:#334155">⚙️ الإعدادات</a>
        </div>
    </div>

    <div class="grid">
        <!-- Control -->
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <div>
                    <h2 id="statusTxt" style="margin:0">--</h2>
                    <small id="modeTxt">--</small>
                </div>
                <button id="toggleBtn" class="btn btn-stop" onclick="toggle()">--</button>
            </div>
            <hr style="border-color:#334155; margin:15px 0">
            <div>
                <span>المؤشر العام: <b id="mktScore">0</b></span>
                <span id="mktRegime" style="float:left">--</span>
            </div>
        </div>

        <!-- Chart -->
        <div class="card">
            <h3>نمو المحفظة</h3>
            <div style="height:150px"><canvas id="equityChart"></canvas></div>
        </div>
        
        <!-- Radar -->
        <div class="card">
             <h3>قوة الاتجاه (MTF)</h3>
             <div style="height:150px"><canvas id="radarChart"></canvas></div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>⚡ الصفقات النشطة</h3>
            <table>
                <thead><tr><th>العملة</th><th>الاستراتيجية</th><th>السعر</th><th>الربح</th></tr></thead>
                <tbody id="tradesTable"></tbody>
            </table>
        </div>
        <div class="card">
            <h3>📝 سجل الفحص (Live Logs)</h3>
            <div style="max-height:300px; overflow-y:auto">
                <table>
                    <thead><tr><th>الوقت</th><th>الرمز</th><th>النتيجة</th><th>السبب</th></tr></thead>
                    <tbody id="logsTable"></tbody>
                </table>
            </div>
        </div>
    </div>

<script>
    let chart, radar;
    function init() {
        const ctx = document.getElementById('equityChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Equity', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true, tension: 0.3 }] },
            options: { plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { color: '#334155' } } }, maintainAspectRatio: false }
        });
        const ctxR = document.getElementById('radarChart').getContext('2d');
        radar = new Chart(ctxR, {
            type: 'radar',
            data: { labels: ['1H', '4H', '1D'], datasets: [{ label: 'Score', data: [0,0,0], borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.5)' }] },
            options: { scales: { r: { min: 0, max: 100, ticks: { display: false } } }, plugins: { legend: { display: false } }, maintainAspectRatio: false }
        });
    }
    function update() {
        fetch('/api/data').then(r=>r.json()).then(d => {
            // Control
            const btn = document.getElementById('toggleBtn');
            const st = document.getElementById('statusTxt');
            if(d.settings.enabled) { btn.innerText="إيقاف"; btn.className="btn btn-run"; st.innerText="يعمل ✅"; st.style.color="var(--green)"; }
            else { btn.innerText="تشغيل"; btn.className="btn btn-stop"; st.innerText="متوقف 🛑"; st.style.color="var(--red)"; }
            document.getElementById('modeTxt').innerText = d.settings.paper ? "تداول ورقي" : "تداول حقيقي";
            
            // Market
            document.getElementById('mktScore').innerText = d.market.score.toFixed(0);
            document.getElementById('mktRegime').innerText = d.market.regime.toUpperCase();
            
            // Charts
            radar.data.datasets[0].data = [d.market.details['1h'], d.market.details['4h'], d.market.details['1d']];
            radar.update();
            if(d.history.length > 0) {
                chart.data.labels = d.history.map(h=>h.date);
                chart.data.datasets[0].data = d.history.map(h=>h.pnl);
                chart.update();
            }

            // Trades
            const tb = document.getElementById('tradesTable');
            tb.innerHTML = d.signals.length ? d.signals.map(s => {
                const pnl = ((d.prices[s.symbol]-s.entry_price)/s.entry_price)*100;
                return `<tr><td><b>${s.symbol}</b></td><td><small>${s.strategy_name}</small></td><td>${d.prices[s.symbol]}</td><td style="color:${pnl>=0?'var(--green)':'var(--red)'}">${pnl.toFixed(2)}%</td></tr>`;
            }).join('') : "<tr><td colspan='4' style='text-align:center'>لا توجد صفقات</td></tr>";

            // Logs
            document.getElementById('logsTable').innerHTML = d.logs.map(l => `<tr><td>${l.t}</td><td>${l.s}</td><td class="status-${l.st}">${l.st}</td><td>${l.r}</td></tr>`).join('');
        });
    }
    function toggle() { fetch('/api/toggle', {method:'POST'}).then(update); }
    init(); setInterval(update, 1000); update();
</script>
</body>
</html>
"""

SETTINGS_PAGE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head><meta charset="utf-8"><title>الإعدادات</title>
<style>body{background:#0f172a;color:white;font-family:sans-serif;padding:20px;max-width:600px;margin:0 auto;} .card{background:#1e293b;padding:20px;border-radius:10px;} input{width:100%;padding:10px;background:#0f172a;border:1px solid #334155;color:white;margin-bottom:15px;border-radius:5px;} button{width:100%;padding:15px;background:#8b5cf6;color:white;border:none;border-radius:5px;cursor:pointer;font-weight:bold;}</style>
</head>
<body><div class="card"><h1>⚙️ إعدادات البوت</h1><form action="/api/settings" method="POST">
<label>قيمة الصفقة (USDT)</label><input type="number" name="amount" value="{{s.trade_amount_usdt}}">
<label>أقصى عدد صفقات</label><input type="number" name="max_trades" value="{{s.max_open_trades}}">
<label>مضاعف الوقف (ATR Multiplier)</label><input type="number" step="0.1" name="sl_mult" value="{{s.stop_loss_atr_multiplier}}">
<label>مضاعف الوقف المتحرك (Trailing)</label><input type="number" step="0.1" name="trail_mult" value="{{s.trailing_atr_multiplier}}">
<div style="margin-bottom:15px"><input type="checkbox" name="paper" style="width:auto" {% if s.paper_trading_mode %}checked{% endif %}><label> تداول ورقي</label></div>
<button type="submit">حفظ</button></form><a href="/" style="display:block;text-align:center;margin-top:20px;color:#94a3b8">عودة</a></div></body></html>
"""

@app.route('/')
def index(): return render_template_string(DASHBOARD)
@app.route('/settings')
def settings(): 
    with locks['settings']: return render_template_string(SETTINGS_PAGE, s=BOT_SETTINGS)

@app.route('/api/data')
def api_data():
    with locks['market']: m = market_state.copy()
    with locks['signals']: s = list(open_signals_cache.values())
    with locks['prices']: p = live_prices.copy()
    with locks['settings']: st = {'enabled': BOT_SETTINGS['is_trading_enabled'], 'paper': BOT_SETTINGS['paper_trading_mode']}
    with locks['logs']: l = list(scan_logs)
    
    hist = []
    try:
        check_db()
        with conn.cursor() as cur:
            cur.execute("SELECT closed_at, profit_pct FROM signals WHERE status='closed' ORDER BY closed_at ASC")
            rows = cur.fetchall()
            cum = 0
            for r in rows:
                cum += r['profit_pct']
                hist.append({'date': r['closed_at'].strftime('%H:%M'), 'pnl': cum})
    except: pass
    return jsonify({"market": m, "signals": s, "prices": p, "settings": st, "logs": l, "history": hist})

@app.route('/api/toggle', methods=['POST'])
def api_toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify({"status": "ok"})

@app.route('/api/settings', methods=['POST'])
def api_set():
    with locks['settings']:
        BOT_SETTINGS['trade_amount_usdt'] = float(request.form.get('amount'))
        BOT_SETTINGS['max_open_trades'] = int(request.form.get('max_trades'))
        BOT_SETTINGS['stop_loss_atr_multiplier'] = float(request.form.get('sl_mult'))
        BOT_SETTINGS['trailing_atr_multiplier'] = float(request.form.get('trail_mult'))
        BOT_SETTINGS['paper_trading_mode'] = 'paper' in request.form
    return redirect('/')

if __name__ == "__main__":
    print("🚀 SmartBot V8 Started...")
    init_db()
    Thread(target=main_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)