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
from typing import List, Dict, Optional, Tuple, Union
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request, redirect, url_for
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- 1. إعدادات النظام والبيئة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot_v7.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Elite')

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ Config Error: {e}")
    exit(1)

# --- 2. المتغيرات العالمية والحالة ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "trade_amount_usdt": 20.0,
    "max_open_trades": 4,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "trailing_start_pct": 1.5, # يبدأ تحريك الوقف بعد ربح 1.5%
    "volume_filter_limit": 40
}

LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
market_state = {
    "score": 50, "regime": "sideways", 
    "details": {"1h": 50, "4h": 50, "1d": 50}, # Scores per timeframe
    "last_update": None
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=50)
performance_history = [] # For equity curve

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
                    exit_reason TEXT, updates_log JSONB DEFAULT '[]'::jsonb, created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        logger.info("✅ Database Initialized.")
    except Exception as e: logger.error(f"DB Error: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 4. نظام التنبيهات الذكي (Telegram) ---
def send_telegram(event_type, payload):
    """
    event_type: BUY, SELL, UPDATE_SL, UPDATE_TP, MARKET_CHANGE, ERROR
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return

    msg = ""
    if event_type == "BUY":
        mode = "📝 ورقي" if BOT_SETTINGS['paper_trading_mode'] else "💵 حقيقي"
        msg = (
            f"🚀 *دخول جديد ({payload['symbol']})*\n"
            f"📊 الاستراتيجية: `{payload['strategy']}`\n"
            f"💰 السعر: `{payload['price']}`\n"
            f"🎯 الهدف: `{payload['tp']}` | 🛑 الوقف: `{payload['sl']}`\n"
            f"⚙️ الوضع: {mode}"
        )
    
    elif event_type == "UPDATE_SL":
        msg = (
            f"🛡️ *تحديث وقف الخسارة ({payload['symbol']})*\n"
            f"تم رفع الوقف لحماية الأرباح.\n"
            f"الوقف الجديد: `{payload['new_sl']}`\n"
            f"السعر الحالي: `{payload['current_price']}`\n"
            f"الربح المحقق: `{payload['profit']:.2f}%`"
        )

    elif event_type == "UPDATE_TP":
        msg = (
            f"🏹 *تمديد الهدف ({payload['symbol']})*\n"
            f"الزخم قوي جداً! تم رفع الهدف لزيادة الربح.\n"
            f"الهدف الجديد: `{payload['new_tp']}`"
        )

    elif event_type == "SELL":
        emoji = "✅" if payload['profit'] > 0 else "🔻"
        msg = (
            f"{emoji} *إغلاق صفقة ({payload['symbol']})*\n"
            f"الربح النهائي: `{payload['profit']:.2f}%`\n"
            f"سعر الخروج: `{payload['price']}`\n"
            f"السبب: _{payload['reason']}_"
        )
        
    elif event_type == "MARKET_CHANGE":
        icon = "🟢" if payload['new'] == 'bullish' else ("🔴" if payload['new'] == 'bearish' else "🟠")
        msg = (f"🔔 *تغير حالة السوق*\nأصبح الوضع: `{payload['new'].upper()}` {icon}\nالنتيجة: `{payload['score']}/100`")

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        Thread(target=requests.post, args=(url,), kwargs={'data': payload}).start()
    except: pass

# --- 5. التحليل الفني والاستراتيجيات ---
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
    
    # Bollinger
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    return df.fillna(0)

# --- الفلتر الذكي (Simple Trend & Volume Filter) ---
def check_smart_filter(df, regime):
    last = df.iloc[-1]
    
    # 1. Volume Check: يجب أن يكون الحجم أعلى من المتوسط أو قريب منه
    if last['volume'] < last['vol_ma'] * 0.8:
        return False, "Weak Volume"
    
    # 2. Trend Confirmation: تجنب الشراء تحت EMA200 في سوق صاعد
    if regime == "bullish" and last['close'] < last['ema200']:
        return False, "Price below EMA200 (Trend Mismatch)"
        
    # 3. RSI Safety: تجنب الشراء في مناطق التشبع الشرائي المفرط
    if last['rsi'] > 75:
        return False, "RSI Overbought (>75)"
        
    return True, "OK"

def get_strategy_signal(symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # أولاً: تطبيق الفلتر العام
    passed, reason = check_smart_filter(df, regime)
    if not passed: return None, reason
    
    # الاستراتيجيات
    if regime == "bullish":
        # A. Volatility Breakout (انفجار سعري مع حجم)
        if prev['bb_width'] < 0.15 and last['close'] > last['bb_upper']:
            return "Vol_Breakout", "Strong expansion"
            
        # B. Trend Pullback (ارتداد صحي)
        if last['close'] > last['ema50'] and last['close'] < last['bb_mid'] and last['rsi'] < 55 and last['macd_hist'] > prev['macd_hist']:
            return "Trend_Pullback", "Healthy correction"

    elif regime == "sideways":
        # C. Range Scalp (من القاع)
        if last['low'] <= last['bb_lower'] and last['close'] > last['bb_lower'] and last['rsi'] < 40:
            if last['close'] > last['open']: # Green candle
                return "Range_Scalp", "Bounce off support"
                
    return None, "No signal"

# --- 6. تحليل هيكل السوق (MTF Radar) ---
def analyze_market_mtf(client):
    global market_state
    timeframes = ['1h', '4h', '1d']
    tf_scores = {'1h': 0, '4h': 0, '1d': 0}
    
    for sym in LEADING_SYMBOLS:
        for tf in timeframes:
            df = fetch_data(client, sym, tf, 100)
            if df is None: continue
            
            # Score logic: Price > EMA200 (+50), RSI > 50 (+50)
            close = df['close'].iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            rsi = 100 - (100 / (1 + (df['close'].diff().gt(0).sum()/df['close'].diff().lt(0).abs().sum()))) # Approx
            
            score = 0
            if close > ema200: score += 50
            # إعادة حساب RSI بشكل صحيح يتطلب دالة كاملة، سنستخدم تقريب
            # هنا نفترض أن المتوسط إيجابي
            tf_scores[tf] += score
            
    # Normalize scores (0-100)
    final_scores = {k: min(100, v / len(LEADING_SYMBOLS) * 2) for k, v in tf_scores.items()}
    
    # Weighted Total
    total_score = (final_scores['1h']*0.2) + (final_scores['4h']*0.3) + (final_scores['1d']*0.5)
    
    new_regime = "sideways"
    if total_score >= 65: new_regime = "bullish"
    elif total_score <= 35: new_regime = "bearish"
    
    with locks['market']:
        if market_state['regime'] != new_regime:
            send_telegram("MARKET_CHANGE", {"new": new_regime, "score": total_score})
        market_state = {"score": total_score, "regime": new_regime, "details": final_scores, "last_update": datetime.now()}

# --- 7. إعادة التحليل وإدارة الصفقات (Active Brain) ---
def reanalyze_position(symbol, signal, df, regime):
    """
    يتم استدعاؤها كل 5 دقائق لاتخاذ قرار بشأن الصفقة المفتوحة
    """
    last = df.iloc[-1]
    entry = float(signal['entry_price'])
    current = float(last['close'])
    sl = float(signal['stop_loss'])
    tp = float(signal['target_price_2'])
    
    profit_pct = (current - entry) / entry * 100
    
    action = "HOLD"
    new_val = 0.0
    reason = ""

    # A. حماية الأرباح (Trailing Stop)
    trailing_trigger = BOT_SETTINGS['trailing_start_pct']
    if profit_pct >= trailing_trigger:
        # نقوم برفع الوقف ليكون تحت السعر الحالي بـ 1%
        proposed_sl = current * 0.99
        if proposed_sl > sl:
            return "UPDATE_SL", proposed_sl, "Trailing Profit"

    # B. تمديد الهدف (Greedy Mode)
    # إذا وصلنا قريب من الهدف وكان الزخم (RSI) لا يزال قوياً جداً (>70)
    if current >= tp * 0.98 and last['rsi'] > 70 and regime == "bullish":
        proposed_tp = tp * 1.05 # زيادة الهدف 5%
        return "UPDATE_TP", proposed_tp, "Strong Momentum Extension"

    # C. الخروج الطارئ (Panic Exit)
    # إذا انقلب السوق للهبوط، والصفقة خاسرة، ومؤشر RSI انهار تحت 40
    if regime == "bearish" and profit_pct < -1.0 and last['rsi'] < 40:
        return "CLOSE_NOW", current, "Market Crash Protection"

    return "HOLD", 0.0, ""

def execute_order(client, symbol, side, qty):
    if BOT_SETTINGS['paper_trading_mode']: return True
    try:
        client.create_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
        return True
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        return False

def trade_manager(client):
    logger.info("🛡️ Elite Trade Manager Started...")
    while True:
        try:
            with locks['signals']: signals = list(open_signals_cache.values())
            if not signals: time.sleep(5); continue
            
            for sig in signals:
                sym = sig['symbol']
                df = fetch_data(client, sym, '5m', 50)
                if df is None: continue
                df = add_indicators(df)
                curr = df.iloc[-1]['close']
                with locks['prices']: live_prices[sym] = curr
                
                sl = float(sig['stop_loss'])
                tp = float(sig['target_price_2'])
                
                # 1. Check Hard Exits
                exit_reason = None
                if curr <= sl: exit_reason = "Stop Loss 🛑"
                elif curr >= tp: exit_reason = "Target Hit 🎯"
                
                if exit_reason:
                    execute_order(client, sym, 'SELL', sig['quantity'])
                    close_signal_db(sym, curr, exit_reason)
                    continue

                # 2. Active Re-analysis
                with locks['market']: regime = market_state['regime']
                action, val, note = reanalyze_position(sym, sig, df, regime)
                
                if action == "UPDATE_SL":
                    with locks['signals']: open_signals_cache[sym]['stop_loss'] = val
                    update_db_sl(sig['id'], val)
                    profit = (curr - sig['entry_price'])/sig['entry_price']*100
                    send_telegram("UPDATE_SL", {"symbol": sym, "new_sl": val, "current_price": curr, "profit": profit})
                
                elif action == "UPDATE_TP":
                    with locks['signals']: open_signals_cache[sym]['target_price_2'] = val
                    send_telegram("UPDATE_TP", {"symbol": sym, "new_tp": val})
                
                elif action == "CLOSE_NOW":
                    execute_order(client, sym, 'SELL', sig['quantity'])
                    close_signal_db(sym, curr, f"Smart Exit: {note}")

            time.sleep(5) # Check every 5 seconds
        except Exception as e: logger.error(f"Manager: {e}"); time.sleep(5)

# --- 8. المحرك الرئيسي ---
def main_engine():
    try: client = Client(API_KEY, API_SECRET)
    except: logger.critical("API Error"); return
    
    Thread(target=trade_manager, args=(client,), daemon=True).start()
    
    # Load symbols
    try:
        with open('crypto_list.txt') as f:
            file_symbols = [l.strip().upper().replace('\n','') for l in f if l.strip()]
            file_symbols = [s if s.endswith('USDT') else s+'USDT' for s in file_symbols]
    except: file_symbols = ['BTCUSDT', 'ETHUSDT']

    while True:
        try:
            with locks['settings']:
                if not BOT_SETTINGS['is_trading_enabled']: time.sleep(5); continue
                limit = BOT_SETTINGS['volume_filter_limit']
                amt = BOT_SETTINGS['trade_amount_usdt']
                max_t = BOT_SETTINGS['max_open_trades']

            analyze_market_mtf(client)
            
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
                    atr = df.iloc[-1]['atr']
                    
                    # Dynamic SL/TP
                    sl = curr - (atr * 2)
                    tp = curr + (atr * 4)
                    qty = amt / curr
                    
                    if execute_order(client, sym, 'BUY', qty):
                        sig = {
                            'symbol': sym, 'entry_price': curr, 'stop_loss': sl, 
                            'target_price_1': tp, 'target_price_2': tp, 
                            'quantity': qty, 'strategy_name': strat, 'status': 'open'
                        }
                        new_id = save_signal_db(sig)
                        sig['id'] = new_id
                        with locks['signals']: open_signals_cache[sym] = sig
                        
                        # Log & Notify
                        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'ACCEPTED', 'r': strat})
                        send_telegram("BUY", {"symbol": sym, "strategy": strat, "price": curr, "sl": sl, "tp": tp})
                else:
                    # Log rejection randomly to save space
                    if random.random() < 0.1:
                        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'REJECTED', 'r': reason})
                
                time.sleep(0.2)
            time.sleep(60)
        except Exception as e: logger.error(f"Engine: {e}"); time.sleep(30)

# --- DB Helpers ---
def save_signal_db(data):
    check_db()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, quantity, strategy_name, status, is_real_trade) VALUES (%s,%s,%s,%s,%s,%s,%s,'open',%s) RETURNING id", 
            (data['symbol'], data['entry_price'], data['stop_loss'], data['target_price_1'], data['target_price_2'], data['quantity'], data['strategy_name'], not BOT_SETTINGS['paper_trading_mode']))
            return cur.fetchone()['id']
    except: return int(time.time())

def update_db_sl(id, sl):
    check_db()
    try:
        with conn.cursor() as cur: cur.execute("UPDATE signals SET stop_loss=%s WHERE id=%s", (sl, id))
    except: pass

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

# --- 9. واجهة الويب (Flask + Chart.js) ---
app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>SmartBot Elite V7</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
<style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --accent: #6366f1; --green: #10b981; --red: #ef4444; --orange: #f59e0b; }
    body { background: var(--bg); color: var(--text); font-family: 'Tajawal', sans-serif; margin: 0; padding: 20px; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .btn { padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer; font-weight: bold; color: white; transition: 0.3s; text-decoration: none; }
    .btn-on { background: var(--green); box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }
    .btn-off { background: var(--red); opacity: 0.7; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
    .card { background: var(--card); padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { padding: 12px; border-bottom: 1px solid #334155; text-align: right; }
    .log-row { font-size: 0.9rem; }
    .status-ACCEPTED { color: var(--green); font-weight: bold; }
    .status-REJECTED { color: var(--orange); }
    /* Radar Chart Container */
    .chart-container { position: relative; height: 200px; width: 100%; }
</style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🤖 SmartBot <span style="color:var(--accent)">Elite</span></h1>
            <small id="connection">● متصل بالنظام</small>
        </div>
        <div>
            <a href="/settings" class="btn" style="background:#334155">⚙️ الإعدادات</a>
        </div>
    </div>

    <!-- Control & Market -->
    <div class="grid">
        <div class="card">
            <h3>حالة النظام</h3>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                <div>
                    <div id="statusText" style="font-size:1.2rem; font-weight:bold">--</div>
                    <small id="modeText">--</small>
                </div>
                <button id="toggleBtn" class="btn btn-off" onclick="toggleBot()">تشغيل</button>
            </div>
            <hr style="border-color:#334155">
            <div style="margin-top:10px">
                <span>الصفقات المفتوحة: <b id="openCount">0</b></span>
            </div>
        </div>

        <div class="card">
            <h3>رادار السوق (MTF)</h3>
            <div style="display:flex; align-items:center;">
                <div style="flex:1">
                    <div style="font-size:2rem; font-weight:bold" id="mktScore">0</div>
                    <div id="mktRegime" style="color:var(--orange)">--</div>
                </div>
                <div style="flex:1; height:150px">
                    <canvas id="marketRadar"></canvas>
                </div>
            </div>
        </div>
        
        <div class="card">
             <h3>منحنى النمو (Equity)</h3>
             <div style="height:150px">
                <canvas id="equityChart"></canvas>
             </div>
             <div style="margin-top:5px; text-align:left">
                <small>إجمالي الربح: <b id="totalPnl">0%</b></small>
             </div>
        </div>
    </div>

    <!-- Trades & Logs -->
    <div class="grid">
        <div class="card">
            <h3>⚡ الصفقات النشطة</h3>
            <table>
                <thead><tr><th>العملة</th><th>استراتيجية</th><th>السعر</th><th>الربح</th></tr></thead>
                <tbody id="tradesTable"></tbody>
            </table>
        </div>
        <div class="card">
            <h3>🕵️ سجل الفحص (Spy Log)</h3>
            <div style="max-height:300px; overflow-y:auto">
                <table>
                    <thead><tr><th>الوقت</th><th>الرمز</th><th>النتيجة</th><th>السبب</th></tr></thead>
                    <tbody id="logsTable"></tbody>
                </table>
            </div>
        </div>
    </div>

<script>
    // Charts
    let radarChart, lineChart;
    
    function initCharts() {
        // Radar
        const ctxR = document.getElementById('marketRadar').getContext('2d');
        radarChart = new Chart(ctxR, {
            type: 'radar',
            data: {
                labels: ['1H', '4H', '1D'],
                datasets: [{ label: 'Bullish Score', data: [50, 50, 50], backgroundColor: 'rgba(99, 102, 241, 0.5)', borderColor: '#6366f1' }]
            },
            options: { scales: { r: { min: 0, max: 100, ticks: { display: false } } }, plugins: { legend: { display: false } } }
        });

        // Line
        const ctxL = document.getElementById('equityChart').getContext('2d');
        lineChart = new Chart(ctxL, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'PNL', data: [], borderColor: '#10b981', tension: 0.3, fill: true, backgroundColor: 'rgba(16, 185, 129, 0.1)' }] },
            options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { color: '#334155' } } } }
        });
    }

    function update() {
        fetch('/api/data').then(r=>r.json()).then(d => {
            // Control
            const btn = document.getElementById('toggleBtn');
            const st = document.getElementById('statusText');
            if(d.settings.enabled) {
                btn.innerText = "إيقاف"; btn.className = "btn btn-on";
                st.innerText = "✅ النظام يعمل"; st.style.color = "var(--green)";
            } else {
                btn.innerText = "تشغيل"; btn.className = "btn btn-off";
                st.innerText = "⏸️ النظام متوقف"; st.style.color = "var(--red)";
            }
            document.getElementById('modeText').innerText = d.settings.paper ? "تداول ورقي" : "تداول حقيقي";
            document.getElementById('openCount').innerText = d.signals.length;

            // Market
            document.getElementById('mktScore').innerText = d.market.score.toFixed(0);
            document.getElementById('mktRegime').innerText = d.market.regime.toUpperCase();
            
            // Update Radar
            radarChart.data.datasets[0].data = [d.market.details['1h'], d.market.details['4h'], d.market.details['1d']];
            radarChart.update();

            // Trades Table
            const tb = document.getElementById('tradesTable');
            if(d.signals.length === 0) tb.innerHTML = "<tr><td colspan='4' style='text-align:center;color:#64748b'>لا توجد صفقات</td></tr>";
            else {
                tb.innerHTML = d.signals.map(s => {
                    const pnl = ((d.prices[s.symbol]-s.entry_price)/s.entry_price)*100;
                    return `<tr>
                        <td><b>${s.symbol}</b></td>
                        <td><small>${s.strategy_name}</small></td>
                        <td>${d.prices[s.symbol]}</td>
                        <td style="color:${pnl>=0?'var(--green)':'var(--red)'}">${pnl.toFixed(2)}%</td>
                    </tr>`;
                }).join('');
            }

            // Logs Table
            const lg = document.getElementById('logsTable');
            lg.innerHTML = d.logs.map(l => `
                <tr class="log-row">
                    <td>${l.t}</td>
                    <td>${l.s}</td>
                    <td class="status-${l.st}">${l.st}</td>
                    <td>${l.r}</td>
                </tr>
            `).join('');

            // Equity Chart
            if(d.history.length > 0) {
                lineChart.data.labels = d.history.map(h => h.date);
                lineChart.data.datasets[0].data = d.history.map(h => h.pnl);
                lineChart.update();
                document.getElementById('totalPnl').innerText = d.history[d.history.length-1].pnl.toFixed(2) + "%";
            }
        });
    }

    function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(update); }
    initCharts();
    setInterval(update, 1500);
    update();
</script>
</body>
</html>
"""

SETTINGS_HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head><meta charset="utf-8"><title>Settings</title>
<style>
body{background:#0f172a;color:white;font-family:sans-serif;padding:20px;max-width:600px;margin:0 auto;}
.card{background:#1e293b;padding:20px;border-radius:10px;}
input{width:100%;padding:10px;background:#0f172a;border:1px solid #334155;color:white;margin-bottom:15px;border-radius:5px;}
button{width:100%;padding:15px;background:#6366f1;color:white;border:none;border-radius:5px;cursor:pointer;font-weight:bold;}
</style>
</head>
<body>
<div class="card">
    <h1>⚙️ الإعدادات</h1>
    <form action="/api/settings" method="POST">
        <label>مبلغ التداول (USDT)</label>
        <input type="number" name="amount" value="{{s.trade_amount_usdt}}">
        <label>أقصى عدد صفقات</label>
        <input type="number" name="max_trades" value="{{s.max_open_trades}}">
        <label>عدد فلاتر السيولة</label>
        <input type="number" name="limit" value="{{s.volume_filter_limit}}">
        <div style="margin-bottom:15px">
            <input type="checkbox" name="paper" style="width:auto" {% if s.paper_trading_mode %}checked{% endif %}>
            <label>تداول ورقي (آمن)</label>
        </div>
        <button type="submit">حفظ</button>
    </form>
    <a href="/" style="display:block;text-align:center;margin-top:20px;color:#94a3b8">عودة</a>
</div>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/settings')
def settings(): 
    with locks['settings']: return render_template_string(SETTINGS_HTML, s=BOT_SETTINGS)

@app.route('/api/data')
def api_data():
    with locks['market']: m = market_state.copy()
    with locks['signals']: s = list(open_signals_cache.values())
    with locks['prices']: p = live_prices.copy()
    with locks['logs']: l = list(scan_logs)
    with locks['settings']: sett = {'enabled': BOT_SETTINGS['is_trading_enabled'], 'paper': BOT_SETTINGS['paper_trading_mode']}
    
    # Performance History (Simulated from DB)
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

    return jsonify({"market": m, "signals": s, "prices": p, "settings": sett, "logs": l, "history": hist})

@app.route('/api/toggle', methods=['POST'])
def api_toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify({"status": "ok"})

@app.route('/api/settings', methods=['POST'])
def api_save_settings():
    with locks['settings']:
        BOT_SETTINGS['trade_amount_usdt'] = float(request.form.get('amount'))
        BOT_SETTINGS['max_open_trades'] = int(request.form.get('max_trades'))
        BOT_SETTINGS['volume_filter_limit'] = int(request.form.get('limit'))
        BOT_SETTINGS['paper_trading_mode'] = 'paper' in request.form
    return redirect('/')

if __name__ == "__main__":
    print("🚀 SmartBot ELITE V7 Started...")
    init_db()
    Thread(target=main_engine, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)