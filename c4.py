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
from decimal import Decimal
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime
from decouple import config
from typing import List, Dict, Optional
import warnings

# --- إعدادات عامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Complete')

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
current_market_regime = "sideways"
market_score = 50
is_trading_enabled = False
paper_trading_mode = True # غيرها لـ False للتداول الحقيقي
usdt_balance = 0.0

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
                    closed_at TIMESTAMP
                );
            """)
        logger.info("✅ قاعدة البيانات جاهزة.")
    except Exception as e:
        logger.error(f"❌ خطأ DB: {e}")

# --- الدوال المساعدة (المفقودة سابقاً) ---
def fetch_historical_data(client, symbol, interval, days) -> Optional[pd.DataFrame]:
    """جلب البيانات التاريخية من Binance"""
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines: return None
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'q_vol', 'n_trades', 'tb_base', 'tb_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"خطأ في جلب بيانات {symbol}: {e}")
        return None

def analyze_market_structure(client):
    """تحليل حالة السوق بناءً على العملات القيادية"""
    global current_market_regime, market_score
    scores = []
    
    for sym in LEADING_SYMBOLS:
        df = fetch_historical_data(client, sym, '1h', 2)
        if df is None: continue
        
        # مؤشرات بسيطة للاتجاه
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        ema200 = df['close'].ewm(span=200).mean().iloc[-1]
        close = df['close'].iloc[-1]
        
        score = 50
        if close > ema50: score += 15
        if close > ema200: score += 15
        if ema50 > ema200: score += 10
        scores.append(score)

    if scores:
        avg = sum(scores) / len(scores)
        with locks['market']:
            market_score = avg
            if avg >= 70: current_market_regime = "bullish"
            elif avg <= 35: current_market_regime = "bearish"
            else: current_market_regime = "sideways"
        logger.info(f"🌐 حالة السوق: {current_market_regime.upper()} (Score: {avg:.0f})")

# --- الجزء الثاني: المؤشرات والاستراتيجيات ---
def calculate_features(df):
    df = df.copy()
    # المتوسطات
    df['ema21'] = df['close'].ewm(span=21).mean()
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
    df['macd_sig'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']
    
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

# الاستراتيجيات
def strategy_momentum(df):
    last = df.iloc[-1]
    if (last['close'] > last['ema21'] > last['ema50']) and (50 < last['rsi'] < 70) and (last['macd_hist'] > 0):
        return True
    return False

def strategy_range(df):
    last = df.iloc[-1]
    if (last['bb_width'] < 0.10) and (last['close'] <= last['bb_lower'] * 1.01) and (last['rsi'] < 40):
        return True
    return False

def strategy_oversold(df):
    last = df.iloc[-1]
    dist = (last['ema50'] - last['close']) / last['ema50']
    if dist > 0.06 and last['rsi'] < 25:
        return True
    return False

def get_signal_from_strategies(symbol, df, regime):
    if regime == "bullish":
        if strategy_momentum(df): return "Momentum_Bullish"
    elif regime == "sideways":
        if strategy_range(df): return "Range_Scalp"
        if strategy_momentum(df): return "Momentum_Breakout"
    elif regime == "bearish":
        if strategy_oversold(df): return "Oversold_Bounce"
    return None

def calculate_entry_params(df, strategy):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    if strategy == "Momentum_Bullish":
        return close - (atr * 1.5), close + (atr * 2), close + (atr * 5)
    elif strategy == "Range_Scalp":
        return close - (atr * 1.2), last['bb_mid'], last['bb_upper']
    else:
        return close - (atr * 2), close + (atr * 2), close + (atr * 3)

# --- الجزء الثالث: التنفيذ وإدارة الصفقات ---
def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        Thread(target=requests.post, args=(url,), kwargs={'data': {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}}).start()
    except: pass

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ ملف {filename} غير موجود. جاري إنشاء ملف افتراضي.")
            default_coins = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
            with open(file_path, 'w') as f: f.write("\n".join(default_coins))
            return default_coins
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = {l.strip().upper() for l in f if l.strip() and not l.startswith('#')}
        return [s if s.endswith('USDT') else s+'USDT' for s in raw]
    except: return ['BTCUSDT']

def get_top_volume_symbols(client, file_symbols, limit=50):
    try:
        all_tickers = client.get_ticker()
        valid = [t for t in all_tickers if t['symbol'] in file_symbols]
        sorted_tickers = sorted(valid, key=lambda x: float(x['quoteVolume']), reverse=True)
        return [t['symbol'] for t in sorted_tickers[:limit]]
    except: return file_symbols[:limit]

def save_signal_to_db(signal_data):
    try:
        if conn is None or conn.closed != 0: init_db()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, quantity, strategy_name, status, is_real_trade, closed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s, NULL) RETURNING id;
            """, (signal_data['symbol'], signal_data['entry_price'], signal_data['stop_loss'], signal_data['target_price_1'], 
                  signal_data['target_price_2'], signal_data['quantity'], signal_data['strategy_name'], not paper_trading_mode))
            return cur.fetchone()['id']
    except Exception as e:
        logger.error(f"❌ DB Error: {e}")
        return int(time.time())

def execute_order(client, symbol, side, quantity):
    if paper_trading_mode:
        logger.info(f"📝 تنفيذ ورقي: {side} {symbol}")
        return {"status": "FILLED"}
    else:
        try:
            return client.create_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
        except Exception as e:
            logger.error(f"❌ فشل التنفيذ الحقيقي: {e}")
            return None

def reanalyze_open_position(symbol, signal_data, df, market_regime):
    last = df.iloc[-1]
    entry = float(signal_data['entry_price'])
    curr = float(last['close'])
    profit = (curr - entry) / entry * 100
    
    if market_regime == "bearish" and profit < -0.5: return "EXIT_NOW", "انقلاب السوق"
    if profit > 1.0 and last['rsi'] < 45: return "TIGHTEN_SL", "ضعف الزخم"
    if profit > 3.5: return "EXTEND_TP", "زخم قوي"
    return "HOLD", ""

def close_trade(client, sig, price, reason):
    execute_order(client, sig['symbol'], 'SELL', float(sig['quantity']))
    with locks['signals']:
        if sig['symbol'] in open_signals_cache: del open_signals_cache[sig['symbol']]
    send_telegram_alert(f"🔴 *إغلاق {sig['symbol']}*\nالنتيجة: `{(price-sig['entry_price'])/sig['entry_price']*100:.2f}%`\nالسبب: {reason}")

def trade_management_loop(client):
    logger.info("🛡️ بدء مدير الصفقات...")
    while True:
        try:
            with locks['signals']: signals = list(open_signals_cache.values())
            if not signals: time.sleep(2); continue

            for sig in signals:
                symbol = sig['symbol']
                df = fetch_historical_data(client, symbol, '5m', 1)
                if df is None: continue
                df = calculate_features(df)
                curr = df.iloc[-1]['close']
                with locks['prices']: live_prices[symbol] = curr

                sl = float(sig['stop_loss'])
                tp2 = float(sig['target_price_2'])
                
                if curr <= sl: close_trade(client, sig, curr, "Stop Loss")
                elif curr >= tp2: close_trade(client, sig, curr, "Target Hit")
                else:
                    with locks['market']: regime = current_market_regime
                    action, reason = reanalyze_open_position(symbol, sig, df, regime)
                    if action == "EXIT_NOW": close_trade(client, sig, curr, reason)
                    elif action == "TIGHTEN_SL":
                        new_sl = curr * 0.995
                        if new_sl > sl:
                            with locks['signals']: open_signals_cache[symbol]['stop_loss'] = new_sl
                            logger.info(f"🔧 رفع الوقف لـ {symbol}")
            time.sleep(3)
        except Exception as e: logger.error(f"Trade Error: {e}"); time.sleep(5)

def main_bot_loop():
    logger.info("🚀 بدء المحرك الرئيسي...")
    try:
        client = Client(API_KEY, API_SECRET)
    except:
        logger.critical("فشل الاتصال بـ Binance"); return

    Thread(target=trade_management_loop, args=(client,), daemon=True).start()
    base_file_symbols = get_validated_symbols('crypto_list.txt')
    
    while True:
        try:
            if not is_trading_enabled: time.sleep(5); continue
            
            analyze_market_structure(client)
            active_symbols = get_top_volume_symbols(client, base_file_symbols, limit=50)
            logger.info(f"🔍 فحص {len(active_symbols)} عملة (الأعلى سيولة من الملف)...")

            for symbol in active_symbols:
                with locks['signals']:
                    if symbol in open_signals_cache: continue
                
                df = fetch_historical_data(client, symbol, '5m', 2)
                if df is None: continue
                df = calculate_features(df)
                
                with locks['market']: regime = current_market_regime
                strategy = get_signal_from_strategies(symbol, df, regime)
                
                if strategy:
                    logger.info(f"✨ إشارة ({strategy}) لـ {symbol}")
                    curr = df.iloc[-1]['close']
                    sl, tp1, tp2 = calculate_entry_params(df, strategy)
                    qty = 15.0 / curr
                    
                    order = execute_order(client, symbol, 'BUY', qty)
                    if order:
                        sig_data = {
                            'symbol': symbol, 'entry_price': curr, 'stop_loss': sl,
                            'target_price_1': tp1, 'target_price_2': tp2, 'quantity': qty,
                            'strategy_name': strategy, 'status': 'open'
                        }
                        new_id = save_signal_to_db(sig_data)
                        sig_data['id'] = new_id
                        with locks['signals']: open_signals_cache[symbol] = sig_data
                        send_telegram_alert(f"🟢 *شراء {symbol}*\nالاستراتيجية: {strategy}\nالسعر: {curr}")
                
                time.sleep(0.2)
            time.sleep(60)
        except Exception as e: logger.error(f"Main Loop Error: {e}"); time.sleep(30)

# --- الجزء الرابع: واجهة المستخدم ---
DASHBOARD_HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>لوحة التحكم الذكية</title>
<style>
body{background:#0f172a;color:#f1f5f9;font-family:sans-serif;padding:20px;}
.card{background:#1e293b;padding:20px;border-radius:12px;margin-bottom:20px;}
.btn{background:#3b82f6;color:#fff;border:none;padding:10px 20px;cursor:pointer;width:100%;}
.btn.stop{background:#ef4444;}
.green{color:#22c55e;} .red{color:#ef4444;}
</style>
</head>
<body>
    <div class="card">
        <h1>🤖 Smart Bot Dashboard</h1>
        <div>السوق: <b id="marketScore">--</b> (<span id="marketText">--</span>)</div>
        <button id="toggleBtn" class="btn" onclick="toggleBot()">تشغيل</button>
    </div>
    <div class="card" id="tradesList"></div>
<script>
function update() {
    fetch('/api/status').then(r=>r.json()).then(d=>{
        document.getElementById('marketScore').innerText = d.market_score.toFixed(0);
        document.getElementById('marketText').innerText = d.market_regime;
        const btn = document.getElementById('toggleBtn');
        btn.innerText = d.is_enabled ? "إيقاف 🛑" : "تشغيل ▶️";
        btn.className = d.is_enabled ? "btn stop" : "btn";
        
        const list = document.getElementById('tradesList');
        list.innerHTML = d.signals.length ? d.signals.map(s=>`
            <div style="border-bottom:1px solid #333;padding:10px;display:flex;justify-content:space-between;">
                <b>${s.symbol} (${s.strategy_name})</b>
                <span class="${(d.prices[s.symbol]-s.entry_price)>=0?'green':'red'}">
                    ${(((d.prices[s.symbol]||s.entry_price)-s.entry_price)/s.entry_price*100).toFixed(2)}%
                </span>
            </div>
        `).join('') : 'لا توجد صفقات نشطة';
    });
}
function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(update); }
setInterval(update, 2000); update();
</script></body></html>
"""

app = Flask(__name__)
CORS(app)

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def api_status():
    with locks['market']: regime = current_market_regime; score = market_score
    with locks['signals']: sigs = list(open_signals_cache.values())
    with locks['prices']: prices = live_prices.copy()
    return jsonify({'market_regime': regime, 'market_score': score, 'signals': sigs, 'prices': prices, 'is_enabled': is_trading_enabled})

@app.route('/api/toggle', methods=['POST'])
def api_toggle():
    global is_trading_enabled
    is_trading_enabled = not is_trading_enabled
    return jsonify({'status': is_trading_enabled})

if __name__ == "__main__":
    init_db()
    Thread(target=main_bot_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)