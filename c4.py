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
from collections import deque
from decouple import config
from typing import List, Dict, Optional, Tuple
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
    handlers=[logging.FileHandler('smart_bot_v6.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_V6')

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
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "trade_amount_usdt": 15.0,
    "max_open_trades": 5,
    "volume_filter_limit": 30, # فحص أفضل 30 عملة فقط للسرعة
    "report_interval_hours": 4
}

# --- المتغيرات العامة ---
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
market_state = {
    "score": 50, "regime": "sideways", "details": {}, "last_update": None
}

open_signals_cache = {}
live_prices = {}
# 🆕 سجل الفحص المباشر (آخر 50 عملية)
scan_logs_cache = deque(maxlen=50)

locks = {
    'signals': Lock(),
    'prices': Lock(),
    'market': Lock(),
    'settings': Lock(),
    'logs': Lock() # قفل جديد للسجلات
}

# --- دالة تسجيل الفحص (Log Scanner) ---
def log_scan(symbol, status, reason, strategy=None):
    """
    status: 'CHECKING', 'REJECTED', 'ACCEPTED', 'ERROR'
    """
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "symbol": symbol,
        "status": status,
        "reason": reason,
        "strategy": strategy or "-"
    }
    with locks['logs']:
        scan_logs_cache.appendleft(entry)
    
    # طباعة في الكونسول فقط للمقبول والأخطاء لتخفيف الضجيج
    if status in ['ACCEPTED', 'ERROR']:
        logger.info(f"Scan Log [{symbol}]: {status} - {reason}")

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

# --- تليجرام ---
def send_telegram(type, data):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    msg = ""
    if type == "BUY":
        mode = "📝 ورقي" if BOT_SETTINGS['paper_trading_mode'] else "💵 حقيقي"
        msg = (f"🟢 *شراء ({data['symbol']})*\nالاستراتيجية: `{data['strategy']}`\n"
               f"السعر: `{data['price']}`\nالوضع: {mode}")
    elif type == "SELL":
        emoji = "✅" if data['profit'] > 0 else "🔻"
        msg = (f"{emoji} *إغلاق ({data['symbol']})*\nالربح: `{data['profit']:.2f}%`\nالسبب: _{data['reason']}_")
    elif type == "MARKET":
        msg = (f"🔔 *تغير السوق*\nالهيكل الجديد: `{data['new'].upper()}`")
    
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- تحليل البيانات ---
def fetch_historical_data(client, symbol, interval, limit=100) -> Optional[pd.DataFrame]:
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in df.columns: df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        log_scan(symbol, "ERROR", f"Data Fetch: {str(e)}")
        return None

# --- تحليل السوق ---
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
    
    with locks['market']:
        old_regime = market_state['regime']
        if new_regime != old_regime:
            send_telegram("MARKET", {"new": new_regime})
        market_state = {"score": total_score, "regime": new_regime, "details": details}

# --- المؤشرات والاستراتيجيات (مع أسباب الرفض) ---
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
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min() # Simplified ATR
    return df.fillna(0)

def get_strategy_signal_with_reason(symbol, df, regime) -> Tuple[Optional[str], str]:
    """
    ترجع: (اسم الاستراتيجية، السبب)
    السبب هنا إما سبب النجاح أو سبب الرفض التفصيلي
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. Volatility Breakout (Bullish)
    if regime == "bullish":
        if not (prev['bb_width'] < 0.20):
            return None, "BB Width not tight enough"
        if not (last['close'] > last['bb_upper']):
            return None, "Price not breaking BB Upper"
        if not (last['volume'] > prev['volume'] * 1.2):
            return None, "Weak Volume Breakout"
        return "Volatility_Breakout", "Strong breakout detected"

    # 2. SMC Liquidity Grab (Sideways)
    if regime == "sideways":
        if not (last['low'] < last['bb_lower']):
            return None, "Price didn't grab liquidity (Low > BB Lower)"
        if not (last['close'] > last['bb_lower']):
            return None, "Price didn't close back inside range"
        if not (last['close'] > last['open']):
            return None, "Candle is not bullish (Red Candle)"
        return "SMC_Liquidity_Grab", "Liquidity sweep confirmed"

    # 3. Trend Pullback (Bullish)
    if regime == "bullish":
        if not (last['close'] > last['ema200']):
            return None, "Price below EMA200 (Bearish Trend)"
        if not (last['close'] < last['ema20']):
            return None, "Price not pulling back (Above EMA20)"
        if not (last['rsi'] < 60):
            return None, "RSI too high for pullback entry"
        return "Trend_Pullback", "Healthy pullback in uptrend"
    
    # 4. Oversold (Bearish)
    if regime == "bearish":
        if last['rsi'] < 25:
            return "Oversold_Bounce", "Extreme oversold (Risky)"
        return None, f"Bearish market - RSI is {last['rsi']:.1f}"

    return None, f"No strategy for {regime}"

def calculate_params(df, strategy):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    if strategy == "Volatility_Breakout": return close-(atr*2), close+(atr*4)
    elif strategy == "SMC_Liquidity_Grab": return last['low']-(atr*0.5), last['bb_mid']
    else: return close-(atr*1.5), close+(atr*3)

# --- التنفيذ ---
def execute_trade(client, symbol, side, qty):
    with locks['settings']: is_paper = BOT_SETTINGS['paper_trading_mode']
    if is_paper: return True
    try:
        client.create_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
        return True
    except Exception as e:
        log_scan(symbol, "ERROR", f"Execution Failed: {e}")
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

# --- الحلقات الرئيسية ---
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
                
                sl, tp = float(sig['stop_loss']), float(sig['target_price_1'])
                reason = "Stop Loss 🛑" if curr <= sl else ("Take Profit 🎯" if curr >= tp else None)
                
                if not reason:
                    # Trailing Stop
                    profit = (curr - sig['entry_price'])/sig['entry_price']*100
                    if profit > 1.5:
                        new_sl = curr * 0.995
                        if new_sl > sl:
                            with locks['signals']: open_signals_cache[sym]['stop_loss'] = new_sl
                            # Log trailing update if needed
                
                if reason:
                    execute_trade(client, sym, 'SELL', sig['quantity'])
                    close_signal_db(sym, curr, reason)
            time.sleep(2)
        except: time.sleep(5)

def main_bot_loop():
    try: client = Client(API_KEY, API_SECRET)
    except: logger.critical("API Error"); return
    
    Thread(target=trade_manager_loop, args=(client,), daemon=True).start()
    
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
            active_symbols = [t['symbol'] for t in sorted_tk]
            
            # --- 🆕 LOGGING: Start Scan Cycle ---
            log_scan("SYSTEM", "INFO", f"Scanning top {len(active_symbols)} symbols based on volume...")

            for sym in active_symbols:
                with locks['signals']:
                    if sym in open_signals_cache:
                        log_scan(sym, "SKIP", "Trade already open")
                        continue
                    if len(open_signals_cache) >= max_trades:
                        log_scan("SYSTEM", "WARNING", "Max open trades reached")
                        break
                
                df = fetch_historical_data(client, sym, '5m', 100)
                if df is None: continue
                df = calculate_features(df)
                
                with locks['market']: regime = market_state['regime']
                
                # --- 🆕 استخدام الدالة الجديدة مع سبب الرفض ---
                strategy, reason = get_strategy_signal_with_reason(sym, df, regime)
                
                if strategy:
                    curr = df.iloc[-1]['close']
                    sl, tp = calculate_params(df, strategy)
                    qty = amount / curr
                    
                    if execute_trade(client, sym, 'BUY', qty):
                        sig_data = {'symbol': sym, 'entry_price': curr, 'sl': sl, 'tp': tp, 'qty': qty, 'strat': strategy}
                        db_id = save_signal(sig_data)
                        
                        cache = {'id': db_id, 'symbol': sym, 'entry_price': curr, 'stop_loss': sl, 'target_price_1': tp, 'quantity': qty, 'strategy_name': strategy}
                        with locks['signals']: open_signals_cache[sym] = cache
                        
                        log_scan(sym, "ACCEPTED", f"Strategy: {strategy}", strategy)
                        send_telegram("BUY", {"symbol": sym, "strategy": strategy, "price": curr})
                else:
                    # --- 🆕 تسجيل سبب الرفض ---
                    log_scan(sym, "REJECTED", reason)
                
                time.sleep(0.2)
            time.sleep(60)
        except Exception as e: logger.error(f"Main: {e}"); time.sleep(30)

# --- واجهة الويب ---
app = Flask(__name__)
CORS(app)

DASHBOARD_HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>SmartBot Spy</title>
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
<style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #3b82f6; --green: #22c55e; --red: #ef4444; --orange: #f59e0b; }
    body { background: var(--bg); color: var(--text); font-family: 'Cairo', sans-serif; margin: 0; padding: 20px; }
    .header { display: flex; justify-content: space-between; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
    .card { background: var(--card); padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .toggle-btn { width: 100%; padding: 15px; border: none; border-radius: 10px; font-weight: bold; cursor: pointer; color: white; font-size: 1.1rem; }
    .btn-on { background: var(--green); } .btn-off { background: var(--red); }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { padding: 10px; border-bottom: 1px solid #334155; text-align: right; }
    
    /* Logs Table Styling */
    .log-table-container { max-height: 300px; overflow-y: auto; }
    .status-ACCEPTED { color: var(--green); font-weight: bold; }
    .status-REJECTED { color: var(--orange); }
    .status-ERROR { color: var(--red); }
    .status-SKIP { color: #64748b; }
    
    .strategy-tag { background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; }
</style>
</head>
<body>
    <div class="header">
        <h1>🕵️ SmartBot Spy Dashboard</h1>
        <div>
            <small id="connection" style="color:var(--green)">● متصل</small>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>التحكم والحالة</h3>
            <div style="display:flex; justify-content:space-between; margin-bottom:10px">
                <span id="statusLabel">--</span>
                <span id="modeLabel" style="color:#94a3b8">--</span>
            </div>
            <button id="toggleBtn" class="toggle-btn btn-off" onclick="toggleBot()">--</button>
        </div>
        
        <div class="card">
            <h3>هيكل السوق</h3>
            <div style="font-size:1.5rem; font-weight:bold" id="marketScore">--</div>
            <div id="regimeText" style="color:var(--orange)">--</div>
        </div>
    </div>

    <!-- Live Scan Logs (The Spy Section) -->
    <div class="card">
        <h3>📝 سجل الفحص المباشر (Live Inspection)</h3>
        <div class="log-table-container">
            <table>
                <thead><tr><th>الوقت</th><th>الرمز</th><th>الحالة</th><th>السبب / التفاصيل</th></tr></thead>
                <tbody id="logsTable"></tbody>
            </table>
        </div>
    </div>

    <div class="card" style="margin-top:20px">
        <h3>⚡ الصفقات المفتوحة</h3>
        <table>
            <thead><tr><th>الرمز</th><th>الاستراتيجية</th><th>السعر</th><th>الربح %</th></tr></thead>
            <tbody id="tradesTable"></tbody>
        </table>
    </div>

<script>
    function update() {
        fetch('/api/dashboard').then(r=>r.json()).then(d => {
            // Control
            const btn = document.getElementById('toggleBtn');
            const lbl = document.getElementById('statusLabel');
            if (d.settings.enabled) {
                btn.innerText = "إيقاف النظام 🛑"; btn.className = "toggle-btn btn-on";
                lbl.innerText = "النظام يعمل ✅"; lbl.style.color = "var(--green)";
            } else {
                btn.innerText = "تشغيل النظام ▶️"; btn.className = "toggle-btn btn-off";
                lbl.innerText = "النظام متوقف ⏸️"; lbl.style.color = "var(--red)";
            }
            document.getElementById('modeLabel').innerText = d.settings.paper ? "تداول ورقي" : "تداول حقيقي";

            // Market
            document.getElementById('marketScore').innerText = d.market.score.toFixed(0);
            document.getElementById('regimeText').innerText = d.market.regime.toUpperCase();

            // Logs (The Spy)
            const logsBody = document.getElementById('logsTable');
            if (d.logs.length === 0) logsBody.innerHTML = "<tr><td colspan='4' style='text-align:center'>لا توجد سجلات بعد...</td></tr>";
            else {
                logsBody.innerHTML = d.logs.map(l => `
                    <tr>
                        <td>${l.time}</td>
                        <td><b>${l.symbol}</b></td>
                        <td class="status-${l.status}">${l.status}</td>
                        <td>${l.reason} ${l.strategy !== '-' ? `<span class="strategy-tag">${l.strategy}</span>` : ''}</td>
                    </tr>
                `).join('');
            }

            // Trades
            const tradesBody = document.getElementById('tradesTable');
            if (d.signals.length === 0) tradesBody.innerHTML = "<tr><td colspan='4' style='text-align:center'>لا توجد صفقات مفتوحة</td></tr>";
            else {
                tradesBody.innerHTML = d.signals.map(s => {
                    const price = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((price - s.entry_price)/s.entry_price)*100;
                    return `<tr>
                        <td>${s.symbol}</td>
                        <td>${s.strategy_name}</td>
                        <td>${price}</td>
                        <td style="color:${pnl>=0?'var(--green)':'var(--red)'}"><b>${pnl.toFixed(2)}%</b></td>
                    </tr>`;
                }).join('');
            }
        });
    }
    function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(update); }
    setInterval(update, 1000); // تحديث سريع (ثانية واحدة) لمراقبة السجل
    update();
</script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/dashboard')
def api_dashboard():
    with locks['market']: mkt = market_state.copy()
    with locks['signals']: sigs = list(open_signals_cache.values())
    with locks['prices']: prices = live_prices.copy()
    with locks['settings']: sett = {'enabled': BOT_SETTINGS['is_trading_enabled'], 'paper': BOT_SETTINGS['paper_trading_mode']}
    with locks['logs']: logs = list(scan_logs_cache) # جلب السجلات
    
    return jsonify({"market": mkt, "signals": sigs, "prices": prices, "settings": sett, "logs": logs})

@app.route('/api/toggle', methods=['POST'])
def api_toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("🚀 SmartBot Spy V6 Started...")
    init_db()
    Thread(target=main_bot_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)