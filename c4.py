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
from typing import List, Dict, Optional, Union
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
    handlers=[logging.FileHandler('smart_bot.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Ultimate')

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

# --- الإعدادات الديناميكية (قابلة للتغيير من واجهة الويب) ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "trade_amount_usdt": 15.0,  # حجم الصفقة الثابت بالدولار
    "max_open_trades": 5,       # أقصى عدد صفقات مفتوحة
    "default_stop_loss_pct": 2.0, # وقف الخسارة الافتراضي %
    "trailing_stop_trigger": 1.5, # تفعيل الوقف المتحرك بعد ربح %
    "volume_filter_limit": 50   # عدد العملات للفحص (الأعلى سيولة)
}

# --- المتغيرات العامة (Global State) ---
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
market_state = {
    "score": 50,
    "regime": "sideways",
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

# --- نظام التنبيهات (Telegram) المطور ---
def send_telegram(type, data):
    """
    إرسال تنبيهات منسقة حسب نوع الحدث
    type: BUY, SELL, INFO, ERROR, UPDATE
    data: قاموس يحتوي على التفاصيل
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return

    msg = ""
    if type == "BUY":
        mode = "📝 ورقي" if BOT_SETTINGS['paper_trading_mode'] else "💵 حقيقي"
        msg = (
            f"🟢 *شراء جديد ({data['symbol']})*\n"
            f"━━━━━━━━━━━━━━\n"
            f"🎯 *الاستراتيجية:* `{data['strategy']}`\n"
            f"💰 *السعر:* `{data['price']}`\n"
            f"🛑 *الوقف:* `{data['sl']}`\n"
            f"🏁 *الهدف:* `{data['tp']}`\n"
            f"📊 *الوضع:* {mode}\n"
            f"📉 *حالة السوق:* `{data['regime']}`"
        )
    elif type == "SELL":
        emoji = "✅" if data['profit'] > 0 else "🔻"
        msg = (
            f"{emoji} *إغلاق صفقة ({data['symbol']})*\n"
            f"━━━━━━━━━━━━━━\n"
            f"💰 *سعر الخروج:* `{data['price']}`\n"
            f"📈 *الربح/الخسارة:* `{data['profit']:.2f}%`\n"
            f"📝 *السبب:* _{data['reason']}_"
        )
    elif type == "UPDATE":
        msg = f"🛡️ *تحديث ({data['symbol']})*\nتم تحريك وقف الخسارة إلى `{data['new_sl']}` لحجز الأرباح."
    elif type == "ERROR":
        msg = f"⚠️ *خطأ في النظام*\n`{data['error']}`"

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        Thread(target=requests.post, args=(url,), kwargs={'data': payload}).start()
    except: pass

# --- تحليل البيانات والمؤشرات ---
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

def calculate_features(df):
    df = df.copy()
    # المتوسطات
    df['ema20'] = df['close'].ewm(span=20).mean()
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
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    return df.fillna(0)

# --- تحليل هيكل السوق (MTF) ---
def analyze_market_structure(client):
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
    
    regime = "sideways"
    if total_score >= 65: regime = "bullish"
    elif total_score <= 35: regime = "bearish"
    
    details = {}
    for tf, sc in avg_scores.items():
        details[tf] = "bullish" if sc > 60 else ("bearish" if sc < 40 else "neutral")
        
    with locks['market']:
        global market_state
        market_state = {"score": total_score, "regime": regime, "details": details, "last_update": datetime.now()}
    
    logger.info(f"🌐 Market Analysis: {regime.upper()} ({total_score:.1f})")

# --- الاستراتيجيات المتقدمة ---
def get_strategy_signal(symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. Volatility Breakout (انفجار سعري) - للسوق الصاعد
    # الشروط: البولنجر كان ضيق ثم انفرج السعر للأعلى بقوة
    if regime == "bullish":
        if prev['bb_width'] < 0.15 and last['close'] > last['bb_upper'] and last['volume'] > prev['volume'] * 1.5:
            return "Volatility_Breakout"
            
    # 2. SMC Liquidity Grab (صيد السيولة) - للسوق الجانبي
    # الشروط: السعر نزل تحت الحد السفلي (ذيل شمعة) ثم أغلق فوقه (شمعة همر)
    if regime == "sideways":
        if last['low'] < last['bb_lower'] and last['close'] > last['bb_lower'] and last['rsi'] < 40:
             # تأكد أن جسم الشمعة إيجابي
            if last['close'] > last['open']:
                return "SMC_Liquidity_Grab"

    # 3. Trend Pullback (تصحيح في الاتجاه)
    if regime == "bullish":
        if last['close'] > last['ema200'] and last['close'] < last['ema20'] and last['rsi'] < 55 and last['macd_hist'] > prev['macd_hist']:
            return "Trend_Pullback"
            
    return None

def calculate_params(df, strategy):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    if strategy == "Volatility_Breakout":
        sl = close - (atr * 2)
        tp = close + (atr * 4)
    elif strategy == "SMC_Liquidity_Grab":
        sl = last['low'] - (atr * 0.5) # وقف ضيق تحت الذيل
        tp = last['bb_mid']
    else: # Trend Pullback
        sl = close - (atr * 1.5)
        tp = close + (atr * 3)
        
    return sl, tp

# --- التنفيذ وإدارة الصفقات ---
def execute_trade(client, symbol, side, qty):
    with locks['settings']: is_paper = BOT_SETTINGS['paper_trading_mode']
    if is_paper: return True
    try:
        client.create_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
        return True
    except Exception as e:
        logger.error(f"❌ Execution Error: {e}")
        send_telegram("ERROR", {"error": str(e)})
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
            cur.execute("""
                UPDATE signals SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, exit_reason=%s
                WHERE symbol=%s AND status='open'
            """, (price, profit, reason, symbol))
            
        send_telegram("SELL", {"symbol": symbol, "price": price, "profit": profit, "reason": reason})
    except Exception as e: logger.error(f"Close DB Error: {e}")

def trade_manager_loop(client):
    logger.info("🛡️ Trade Manager Started")
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
                
                sl = float(sig['stop_loss'])
                tp = float(sig['target_price_1'])
                
                # Check Exit
                reason = None
                if curr <= sl: reason = "Stop Loss 🛑"
                elif curr >= tp: reason = "Take Profit 🎯"
                
                # Smart Re-analysis (Trailing Stop)
                if not reason:
                    profit_pct = (curr - sig['entry_price']) / sig['entry_price'] * 100
                    with locks['settings']: trigger = BOT_SETTINGS['trailing_stop_trigger']
                    
                    if profit_pct >= trigger:
                        new_sl = curr * 0.995 # حجز 0.5% تحت السعر الحالي
                        if new_sl > sl:
                            with locks['signals']: open_signals_cache[sym]['stop_loss'] = new_sl
                            # تحديث قاعدة البيانات
                            check_db()
                            with conn.cursor() as cur:
                                cur.execute("UPDATE signals SET stop_loss=%s WHERE id=%s", (new_sl, sig['id']))
                            send_telegram("UPDATE", {"symbol": sym, "new_sl": new_sl})
                
                if reason:
                    execute_trade(client, sym, 'SELL', sig['quantity'])
                    close_signal_db(sym, curr, reason)
            
            time.sleep(2)
        except Exception as e: logger.error(f"Manager Error: {e}"); time.sleep(5)

# --- المحرك الرئيسي ---
def main_bot_loop():
    try: client = Client(API_KEY, API_SECRET)
    except: logger.critical("API Error"); return
    
    Thread(target=trade_manager_loop, args=(client,), daemon=True).start()
    
    # تحميل قائمة العملات من الملف
    try:
        with open('crypto_list.txt') as f:
            file_symbols = [l.strip().upper() for l in f if l.strip()]
            file_symbols = [s if s.endswith('USDT') else s+'USDT' for s in file_symbols]
    except: file_symbols = ['BTCUSDT', 'ETHUSDT']

    while True:
        try:
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                max_trades = BOT_SETTINGS['max_open_trades']
                vol_limit = int(BOT_SETTINGS['volume_filter_limit'])
                trade_amt = BOT_SETTINGS['trade_amount_usdt']

            if not enabled: time.sleep(5); continue
            
            # 1. تحليل السوق
            analyze_market_structure(client)
            
            # 2. فلترة السيولة
            tickers = client.get_ticker()
            valid_tk = [t for t in tickers if t['symbol'] in file_symbols]
            sorted_tk = sorted(valid_tk, key=lambda x: float(x['quoteVolume']), reverse=True)[:vol_limit]
            active_symbols = [t['symbol'] for t in sorted_tk]
            
            # 3. فحص الفرص
            for sym in active_symbols:
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
                    qty = trade_amt / curr
                    
                    # فلترة MinNotional بسيطة (تأكد من القيم الحقيقية في الإنتاج)
                    if qty * curr < 5: continue 

                    if execute_trade(client, sym, 'BUY', qty):
                        sig_data = {
                            'symbol': sym, 'entry_price': curr, 'sl': sl, 'tp': tp,
                            'qty': qty, 'strat': strategy
                        }
                        db_id = save_signal(sig_data)
                        
                        cache_data = {
                            'id': db_id, 'symbol': sym, 'entry_price': curr, 
                            'stop_loss': sl, 'target_price_1': tp, 'quantity': qty, 
                            'strategy_name': strategy
                        }
                        with locks['signals']: open_signals_cache[sym] = cache_data
                        
                        send_telegram("BUY", {
                            "symbol": sym, "strategy": strategy, "price": curr,
                            "sl": sl, "tp": tp, "regime": regime
                        })
                
                time.sleep(0.2)
            time.sleep(60)
        except Exception as e: logger.error(f"Main Loop: {e}"); time.sleep(30)

# --- واجهة الويب (Flask) ---
app = Flask(__name__)
CORS(app)

# HTML القوالب
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>لوحة القيادة - SmartBot</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
<style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #3b82f6; --green: #22c55e; --red: #ef4444; }
    body { background: var(--bg); color: var(--text); font-family: 'Cairo', sans-serif; margin: 0; padding: 20px; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .nav a { color: #94a3b8; text-decoration: none; margin-left: 15px; font-size: 1.1rem; }
    .nav a.active { color: var(--accent); font-weight: bold; }
    
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
    .card { background: var(--card); padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stat-val { font-size: 2rem; font-weight: bold; margin: 10px 0; }
    .stat-label { color: #94a3b8; font-size: 0.9rem; }
    
    table { width: 100%; border-collapse: collapse; }
    th { text-align: right; color: #94a3b8; padding: 12px; border-bottom: 1px solid #334155; }
    td { padding: 12px; border-bottom: 1px solid #334155; }
    
    .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-left: 5px; }
    .green { color: var(--green); } .red { color: var(--red); }
</style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🚀 SmartBot Ultimate</h1>
            <small id="connectionStatus" style="color: var(--green);">● النظام متصل</small>
        </div>
        <div class="nav">
            <a href="/" class="active">الرئيسية</a>
            <a href="/settings">الإعدادات</a>
        </div>
    </div>

    <div class="grid">
        <!-- Market State -->
        <div class="card">
            <div class="stat-label">حالة السوق (MTF)</div>
            <div class="stat-val" id="marketScore">--</div>
            <div id="regimeText" style="font-weight:bold; color: #f59e0b;">--</div>
            <div style="margin-top:10px; display:flex; gap:5px;">
                <span id="tf-1h" style="padding:2px 8px; background:#334155; border-radius:4px; font-size:0.8rem">1H</span>
                <span id="tf-4h" style="padding:2px 8px; background:#334155; border-radius:4px; font-size:0.8rem">4H</span>
                <span id="tf-1d" style="padding:2px 8px; background:#334155; border-radius:4px; font-size:0.8rem">1D</span>
            </div>
        </div>

        <!-- Stats -->
        <div class="card">
            <div class="stat-label">الأرباح المغلقة</div>
            <div class="stat-val" id="totalPnl">0.00%</div>
            <div style="display:flex; justify-content:space-between">
                <span>نسبة الفوز: <b id="winRate">0%</b></span>
                <span>الصفقات: <b id="tradesCount">0</b></span>
            </div>
        </div>

        <!-- Active Trades -->
        <div class="card">
            <div class="stat-label">صفقات مفتوحة</div>
            <div class="stat-val" id="openCount">0</div>
            <small id="tradingModeText">--</small>
        </div>
    </div>

    <!-- Equity Chart -->
    <div class="card" style="margin-bottom: 20px;">
        <h3>📈 نمو المحفظة</h3>
        <div style="height: 250px;">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <!-- Active Trades Table -->
    <div class="card">
        <h3>⚡ الصفقات الحالية</h3>
        <table>
            <thead><tr><th>الرمز</th><th>الاستراتيجية</th><th>الدخول</th><th>الحالي</th><th>الربح %</th></tr></thead>
            <tbody id="tradesTable"></tbody>
        </table>
    </div>

<script>
    let chart;
    function initChart() {
        const ctx = document.getElementById('equityChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'PNL %', data: [], borderColor: '#3b82f6', tension: 0.4, fill: true, backgroundColor: 'rgba(59, 130, 246, 0.1)' }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { color: '#334155' } } } }
        });
    }

    function update() {
        fetch('/api/dashboard').then(r=>r.json()).then(d => {
            // Market
            document.getElementById('marketScore').innerText = d.market.score.toFixed(0);
            const regimeEl = document.getElementById('regimeText');
            regimeEl.innerText = d.market.regime.toUpperCase();
            regimeEl.style.color = d.market.regime == 'bullish' ? 'var(--green)' : (d.market.regime == 'bearish' ? 'var(--red)' : '#f59e0b');
            
            // TF Colors
            ['1h', '4h', '1d'].forEach(tf => {
                const el = document.getElementById(`tf-${tf}`);
                const state = d.market.details[tf];
                el.style.background = state == 'bullish' ? 'var(--green)' : (state == 'bearish' ? 'var(--red)' : '#334155');
            });

            // Stats
            document.getElementById('totalPnl').innerText = d.stats.pnl.toFixed(2) + '%';
            document.getElementById('totalPnl').style.color = d.stats.pnl >= 0 ? 'var(--green)' : 'var(--red)';
            document.getElementById('winRate').innerText = d.stats.win_rate.toFixed(1) + '%';
            document.getElementById('tradesCount').innerText = d.stats.count;
            
            document.getElementById('openCount').innerText = d.signals.length;
            document.getElementById('tradingModeText').innerText = d.settings.paper_mode ? "وضع التداول الورقي 📝" : "وضع التداول الحقيقي 💵";

            // Table
            const tbody = document.getElementById('tradesTable');
            if (d.signals.length === 0) tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#94a3b8; padding:20px">لا توجد صفقات نشطة</td></tr>';
            else {
                tbody.innerHTML = d.signals.map(s => {
                    const price = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((price - s.entry_price) / s.entry_price) * 100;
                    return `<tr>
                        <td><b>${s.symbol}</b></td>
                        <td><span style="background:#334155; padding:3px 8px; border-radius:4px; font-size:0.8rem">${s.strategy_name}</span></td>
                        <td>${s.entry_price}</td>
                        <td>${price}</td>
                        <td style="color: ${pnl >= 0 ? 'var(--green)' : 'var(--red)'}"><b>${pnl.toFixed(2)}%</b></td>
                    </tr>`;
                }).join('');
            }

            // Chart
            if (chart) {
                chart.data.labels = d.chart.labels;
                chart.data.datasets[0].data = d.chart.data;
                chart.update();
            }
        });
    }
    initChart();
    setInterval(update, 2000);
    update();
</script>
</body>
</html>
"""

SETTINGS_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>الإعدادات - SmartBot</title>
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
<style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #3b82f6; }
    body { background: var(--bg); color: var(--text); font-family: 'Cairo', sans-serif; margin: 0; padding: 20px; }
    .container { max-width: 600px; margin: 0 auto; }
    .card { background: var(--card); padding: 30px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1 { margin-bottom: 30px; text-align: center; }
    .form-group { margin-bottom: 20px; }
    label { display: block; margin-bottom: 8px; color: #94a3b8; }
    input[type="number"], input[type="text"] { width: 100%; padding: 12px; background: #0f172a; border: 1px solid #334155; color: white; border-radius: 8px; box-sizing: border-box; }
    .toggle-switch { display: flex; align-items: center; justify-content: space-between; background: #0f172a; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #334155; }
    .btn { width: 100%; padding: 15px; background: var(--accent); color: white; border: none; border-radius: 8px; font-size: 1.1rem; cursor: pointer; font-weight: bold; margin-top: 20px; }
    .btn:hover { opacity: 0.9; }
    .back-link { display: block; text-align: center; margin-top: 20px; color: #94a3b8; text-decoration: none; }
</style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>⚙️ إعدادات البوت</h1>
        <form action="/api/settings" method="POST">
            
            <div class="toggle-switch">
                <span>تشغيل التداول الآلي</span>
                <input type="checkbox" name="is_trading_enabled" {% if settings.is_trading_enabled %}checked{% endif %}>
            </div>

            <div class="toggle-switch">
                <span>وضع التداول الورقي (Paper Trading)</span>
                <input type="checkbox" name="paper_trading_mode" {% if settings.paper_trading_mode %}checked{% endif %}>
            </div>

            <div class="form-group">
                <label>مبلغ الصفقة الثابت (USDT)</label>
                <input type="number" step="0.1" name="trade_amount_usdt" value="{{ settings.trade_amount_usdt }}">
            </div>

            <div class="form-group">
                <label>أقصى عدد صفقات مفتوحة</label>
                <input type="number" name="max_open_trades" value="{{ settings.max_open_trades }}">
            </div>
            
            <div class="form-group">
                <label>تفعيل الوقف المتحرك عند ربح (%)</label>
                <input type="number" step="0.1" name="trailing_stop_trigger" value="{{ settings.trailing_stop_trigger }}">
            </div>

            <div class="form-group">
                <label>عدد العملات للفحص (أعلى سيولة)</label>
                <input type="number" name="volume_filter_limit" value="{{ settings.volume_filter_limit }}">
            </div>

            <button type="submit" class="btn">حفظ الإعدادات</button>
        </form>
        <a href="/" class="back-link">← عودة للوحة التحكم</a>
    </div>
</div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/settings')
def settings_page():
    with locks['settings']: return render_template_string(SETTINGS_TEMPLATE, settings=BOT_SETTINGS)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    with locks['settings']:
        BOT_SETTINGS['is_trading_enabled'] = 'is_trading_enabled' in request.form
        BOT_SETTINGS['paper_trading_mode'] = 'paper_trading_mode' in request.form
        BOT_SETTINGS['trade_amount_usdt'] = float(request.form.get('trade_amount_usdt', 15.0))
        BOT_SETTINGS['max_open_trades'] = int(request.form.get('max_open_trades', 5))
        BOT_SETTINGS['trailing_stop_trigger'] = float(request.form.get('trailing_stop_trigger', 1.5))
        BOT_SETTINGS['volume_filter_limit'] = int(request.form.get('volume_filter_limit', 50))
    return redirect(url_for('settings_page'))

@app.route('/api/dashboard')
def api_dashboard():
    with locks['market']: mkt = market_state.copy()
    with locks['signals']: sigs = list(open_signals_cache.values())
    with locks['prices']: prices = live_prices.copy()
    with locks['settings']: sett = BOT_SETTINGS.copy()
    
    # Stats Calculation
    stats = {'pnl': 0, 'win_rate': 0, 'count': 0}
    chart_data = {'labels': [], 'data': []}
    
    check_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_pct, closed_at FROM signals WHERE status='closed' ORDER BY closed_at ASC")
            rows = cur.fetchall()
            if rows:
                stats['count'] = len(rows)
                stats['pnl'] = sum(r['profit_pct'] for r in rows)
                wins = len([r for r in rows if r['profit_pct'] > 0])
                stats['win_rate'] = (wins / len(rows)) * 100 if len(rows) > 0 else 0
                
                cum = 0
                for r in rows:
                    cum += r['profit_pct']
                    chart_data['labels'].append(r['closed_at'].strftime('%d %H:%M'))
                    chart_data['data'].append(cum)
    except: pass

    return jsonify({
        "market": mkt, "signals": sigs, "prices": prices, 
        "stats": stats, "chart": chart_data, 
        "settings": {"paper_mode": sett['paper_trading_mode']}
    })

# --- التشغيل ---
if __name__ == "__main__":
    print("="*50)
    print("🚀 SMART BOT ULTIMATE STARTED")
    print("="*50)
    init_db()
    Thread(target=main_bot_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)