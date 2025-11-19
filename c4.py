# smart_bot_part1.py
# --- الجزء الأول: الإعدادات، قاعدة البيانات، وتحليل هيكل السوق ---

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

# إعدادات التجاهل واللوجر
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Final')

# --- متغيرات البيئة ---
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    REDIS_URL = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ فشل تحميل المتغيرات: {e}")
    exit(1)

# --- المتغيرات العامة (Global State) ---
# الرموز القيادية لتحديد اتجاه السوق العام
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

current_market_regime = "sideways" # bullish, bearish, sideways
market_score = 50 # 0-100
is_trading_enabled = False
paper_trading_mode = True # اجعلها False للتداول الحقيقي
usdt_balance = 0.0

# الكاش المباشر
open_signals_cache = {}
live_prices = {}

# الأقفال لتنظيم العمليات المتوازية
locks = {
    'signals': Lock(),
    'prices': Lock(),
    'market': Lock(),
    'balance': Lock()
}

# --- اتصال قاعدة البيانات ---
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

# --- تحليل هيكل السوق (Market Structure) ---
def analyze_market_structure(client):
    global current_market_regime, market_score
    scores = []
    
    for sym in LEADING_SYMBOLS:
        try:
            klines = client.get_historical_klines(sym, '1h', "2 day ago UTC")
            if not klines: continue
            df = pd.DataFrame(klines, columns=['t','o','h','l','c','v','x','y','z','a','b','d'])
            df['close'] = pd.to_numeric(df['c'])
            
            # مؤشرات بسيطة للاتجاه
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            close = df['close'].iloc[-1]
            
            score = 50
            if close > ema50: score += 15
            if close > ema200: score += 15
            if ema50 > ema200: score += 10
            scores.append(score)
        except: pass

    if scores:
        avg = sum(scores) / len(scores)
        with locks['market']:
            market_score = avg
            if avg >= 70: current_market_regime = "bullish"
            elif avg <= 35: current_market_regime = "bearish"
            else: current_market_regime = "sideways"
        logger.info(f"🌐 حالة السوق: {current_market_regime.upper()} (Score: {avg:.0f})")
        # smart_bot_part2.py
# --- الجزء الثاني: المؤشرات والاستراتيجيات ---

def calculate_features(df):
    df = df.copy()
    # تحويل البيانات لأرقام
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
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

# --- الاستراتيجيات ---

# 1. استراتيجية الزخم (للسوق الصاعد)
def strategy_momentum(df):
    last = df.iloc[-1]
    # السعر فوق المتوسطات + RSI قوي + MACD إيجابي
    if (last['close'] > last['ema21'] > last['ema50']) and \
       (50 < last['rsi'] < 70) and \
       (last['macd_hist'] > 0):
        return True
    return False

# 2. استراتيجية النطاق (للسوق الجانبي)
def strategy_range(df):
    last = df.iloc[-1]
    # بولنجر باند ضيق + السعر يلمس الحد السفلي + RSI منخفض
    if (last['bb_width'] < 0.08) and \
       (last['close'] <= last['bb_lower'] * 1.01) and \
       (last['rsi'] < 40):
        return True
    return False

# 3. استراتيجية الارتداد (للسوق الهابط/التصحيح)
def strategy_oversold(df):
    last = df.iloc[-1]
    # ابتعاد كبير عن المتوسط + تشبع بيعي حاد
    dist = (last['ema50'] - last['close']) / last['ema50']
    if dist > 0.06 and last['rsi'] < 25:
        return True
    return False

# --- الموجه الرئيسي للاستراتيجيات ---
def get_signal_from_strategies(symbol, df, regime):
    """اختيار الاستراتيجية بناءً على حالة السوق"""
    if regime == "bullish":
        if strategy_momentum(df): return "Momentum_Bullish"
    
    elif regime == "sideways":
        if strategy_range(df): return "Range_Scalp"
        if strategy_momentum(df): return "Momentum_Breakout" # أحياناً ينجح الاختراق
    
    elif regime == "bearish":
        if strategy_oversold(df): return "Oversold_Bounce"
    
    return None

# --- حساب الأهداف ---
def calculate_entry_params(df, strategy):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    if strategy == "Momentum_Bullish":
        sl = close - (atr * 1.5)
        tp1 = close + (atr * 2)
        tp2 = close + (atr * 5)
    
    elif strategy == "Range_Scalp":
        sl = close - (atr * 1.2)
        tp1 = last['bb_mid']
        tp2 = last['bb_upper']
    
    else: # Oversold / Default
        sl = close - (atr * 2)
        tp1 = close + (atr * 2)
        tp2 = close + (atr * 3)
        
    return sl, tp1, tp2
    # smart_bot_part3_volume.py
# --- الجزء الثالث (معدل): الفلترة حسب الحجم من قائمة الملف ---

import os

# --- إعدادات تلغرام ---
TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')

def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        Thread(target=requests.post, args=(url,), kwargs={'data': {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}}).start()
    except: pass

# --- دالة قراءة الرموز من الملف ---
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """قراءة الرموز من ملف نصي"""
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ ملف الرموز '{filename}' غير موجود! جاري إنشاء ملف افتراضي.")
            default_coins = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "TRXUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT"]
            with open(file_path, 'w') as f: f.write("\n".join(default_coins))
            return default_coins
            
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = [s if s.endswith('USDT') else s + 'USDT' for s in raw_symbols]
        return list(set(formatted))
    except Exception as e:
        logger.error(f"❌ خطأ في قراءة الملف: {e}")
        return ['BTCUSDT']

# --- دالة حفظ في قاعدة البيانات ---
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

# --- دالة التنفيذ ---
def execute_order(client, symbol, side, quantity):
    if paper_trading_mode:
        logger.info(f"📝 تنفيذ ورقي: {side} {symbol}")
        return {"status": "FILLED", "executedQty": quantity}
    else:
        try:
            # تنبيه: تأكد من ضبط stepSize للكمية في الإنتاج
            return client.create_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
        except Exception as e:
            logger.error(f"❌ فشل التنفيذ الحقيقي لـ {symbol}: {e}")
            return None

# --- آلية إعادة التحليل ---
def reanalyze_open_position(symbol, signal_data, df, market_regime):
    last = df.iloc[-1]
    entry = float(signal_data['entry_price'])
    curr = float(last['close'])
    profit = (curr - entry) / entry * 100
    
    if market_regime == "bearish" and profit < -0.5: return "EXIT_NOW", "انقلاب السوق 📉"
    if profit > 1.0 and last['rsi'] < 45: return "TIGHTEN_SL", "ضعف الزخم ⚠️"
    if profit > 3.5: return "EXTEND_TP", "زخم قوي 🚀"
    
    return "HOLD", ""

# --- إدارة الصفقات ---
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
                
                if curr <= sl: close_trade(client, sig, curr, "Stop Loss 🛑")
                elif curr >= tp2: close_trade(client, sig, curr, "Target Hit 🎯")
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
        except Exception as e: logger.error(f"Trade Loop Error: {e}"); time.sleep(5)

def close_trade(client, sig, price, reason):
    execute_order(client, sig['symbol'], 'SELL', float(sig['quantity']))
    with locks['signals']:
        if sig['symbol'] in open_signals_cache: del open_signals_cache[sig['symbol']]
    send_telegram_alert(f"🔴 *إغلاق {sig['symbol']}*\nالنتيجة: `{(price-sig['entry_price'])/sig['entry_price']*100:.2f}%`\nالسبب: {reason}")

# --- دالة ترتيب الرموز حسب الحجم ---
def get_top_volume_symbols(client, file_symbols, limit=50):
    """جلب بيانات السوق وترتيب الرموز الموجودة في الملف حسب حجم التداول"""
    try:
        # جلب بيانات الـ 24 ساعة لكل السوق (طلب واحد سريع)
        all_tickers = client.get_ticker()
        
        valid_tickers = []
        for t in all_tickers:
            # 1. نتحقق أن الرمز موجود في قائمتنا المفضلة
            if t['symbol'] in file_symbols:
                valid_tickers.append(t)
        
        # 2. الترتيب تنازلياً حسب حجم التداول بالدولار (quoteVolume)
        sorted_tickers = sorted(valid_tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
        
        # 3. أخذ أفضل 50
        top_symbols = [t['symbol'] for t in sorted_tickers[:limit]]
        
        logger.info(f"📊 تم اختيار أفضل {len(top_symbols)} عملة من الملف حسب السيولة.")
        return top_symbols
    except Exception as e:
        logger.error(f"❌ خطأ في فلترة السيولة: {e}")
        return file_symbols[:limit] # العودة للقائمة العادية عند الخطأ

# --- الحلقة الرئيسية ---
def main_bot_loop():
    logger.info("🚀 بدء المحرك الرئيسي (أفضل 50 سيولة من الملف)...")
    client = Client(API_KEY, API_SECRET)
    Thread(target=trade_management_loop, args=(client,), daemon=True).start()
    
    # تحميل القائمة الأساسية من الملف مرة واحدة
    base_file_symbols = get_validated_symbols('crypto_list.txt')
    
    while True:
        try:
            if not is_trading_enabled: time.sleep(5); continue
            
            analyze_market_structure(client)
            
            # --- الخطوة الجديدة: الفلترة حسب الحجم ---
            # نقوم بتحديث القائمة النشطة بناءً على السيولة الحالية
            active_symbols = get_top_volume_symbols(client, base_file_symbols, limit=50)
            
            logger.info(f"🔍 بدء المسح على {len(active_symbols)} عملة...")

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
                
                time.sleep(0.2) # تسريع المسح قليلاً
            
            logger.info("💤 استراحة دقيقة قبل تحديث قائمة السيولة...")
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Main Loop Error: {e}"); time.sleep(30)
            # smart_bot_part4.py
# --- الجزء الرابع: واجهة المستخدم وتشغيل النظام ---

# --- قالب HTML للوحة التحكم ---
DASHBOARD_HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم الذكية</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--text:#f1f5f9;--accent:#3b82f6;--green:#22c55e;--red:#ef4444;--orange:#f97316;}
body{background:var(--bg);color:var(--text);font-family:sans-serif;margin:0;padding:20px;}
.container{max-width:1200px;margin:0 auto;}
.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;border-bottom:1px solid #334155;padding-bottom:15px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));gap:20px;}
.card{background:var(--card);padding:20px;border-radius:12px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);}
.score-box{text-align:center;margin:15px 0;}
.score-val{font-size:3rem;font-weight:bold;}
.badge{padding:5px 10px;border-radius:99px;font-size:0.8rem;font-weight:bold;}
.btn{background:var(--accent);color:#fff;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:1rem;width:100%;}
.btn.stop{background:var(--red);}

.trade-item{border-bottom:1px solid #334155;padding:10px 0;display:flex;justify-content:space-between;align-items:center;}
.trade-info small{color:#94a3b8;display:block;}
.pnl{font-weight:bold;}
.pos{color:var(--green);} .neg{color:var(--red);}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🤖 Smart Bot Dashboard</h1>
        <span class="badge" style="background:#334155" id="statusBadge">جاري الاتصال...</span>
    </div>

    <div class="grid">
        <!-- حالة السوق -->
        <div class="card">
            <h3>📊 مؤشر السوق (Market Regime)</h3>
            <div class="score-box">
                <div class="score-val" id="marketScore">--</div>
                <div id="marketText" style="font-size:1.2rem;margin-top:5px;">...</div>
            </div>
            <p style="color:#94a3b8;font-size:0.9rem;">يعتمد على تحليل BTC, ETH, SOL, BNB.</p>
        </div>

        <!-- التحكم -->
        <div class="card">
            <h3>⚙️ التحكم</h3>
            <div style="display:flex;justify-content:space-between;margin-bottom:15px;">
                <span>الصفقات: <b id="openCount">0</b></span>
                <span>الوضع: <b id="tradingMode">--</b></span>
            </div>
            <button id="toggleBtn" class="btn" onclick="toggleBot()">تشغيل البوت</button>
        </div>
    </div>

    <div class="card" style="margin-top:20px;">
        <h3>⚡ الصفقات النشطة (Active Signals)</h3>
        <div id="tradesList">
            <p style="color:#94a3b8;text-align:center;">لا توجد صفقات نشطة حالياً.</p>
        </div>
    </div>
</div>

<script>
function update() {
    fetch('/api/status')
    .then(r => r.json())
    .then(data => {
        // السوق
        const score = data.market_score;
        const scoreEl = document.getElementById('marketScore');
        scoreEl.innerText = score.toFixed(0);
        scoreEl.style.color = score > 65 ? 'var(--green)' : (score < 35 ? 'var(--red)' : 'var(--orange)');
        document.getElementById('marketText').innerText = data.market_regime.toUpperCase();
        
        // التحكم
        const btn = document.getElementById('toggleBtn');
        if(data.is_enabled) {
            btn.innerText = "إيقاف البوت 🛑";
            btn.className = "btn stop";
            document.getElementById('statusBadge').innerText = "متصل • يعمل";
            document.getElementById('statusBadge').style.background = "var(--green)";
        } else {
            btn.innerText = "تشغيل البوت ▶️";
            btn.className = "btn";
            document.getElementById('statusBadge').innerText = "متصل • متوقف";
            document.getElementById('statusBadge').style.background = "var(--red)";
        }
        
        document.getElementById('openCount').innerText = data.signals.length;
        document.getElementById('tradingMode').innerText = data.paper_mode ? "ورقي 📝" : "حقيقي 💵";

        // الصفقات
        const list = document.getElementById('tradesList');
        if(data.signals.length === 0) {
            list.innerHTML = '<p style="color:#94a3b8;text-align:center;">لا توجد صفقات نشطة حالياً.</p>';
        } else {
            list.innerHTML = data.signals.map(s => {
                const price = data.prices[s.symbol] || s.entry_price;
                const pnl = ((price - s.entry_price) / s.entry_price) * 100;
                return `
                <div class="trade-item">
                    <div class="trade-info">
                        <b style="font-size:1.1rem">${s.symbol}</b>
                        <small>${s.strategy_name}</small>
                    </div>
                    <div style="text-align:right">
                        <div class="pnl ${pnl >= 0 ? 'pos' : 'neg'}">${pnl.toFixed(2)}%</div>
                        <small>Price: ${price}</small>
                    </div>
                </div>`;
            }).join('');
        }
    });
}
function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(update); }
setInterval(update, 2000);
update();
</script>
</body>
</html>
"""

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def api_status():
    with locks['market']: regime = current_market_regime; score = market_score
    with locks['signals']: sigs = list(open_signals_cache.values())
    with locks['prices']: prices = live_prices.copy()
    
    return jsonify({
        'market_regime': regime,
        'market_score': score,
        'signals': sigs,
        'prices': prices,
        'is_enabled': is_trading_enabled,
        'paper_mode': paper_trading_mode
    })

@app.route('/api/toggle', methods=['POST'])
def api_toggle():
    global is_trading_enabled
    is_trading_enabled = not is_trading_enabled
    return jsonify({'status': is_trading_enabled})

# --- نقطة الدخول الرئيسية (Main Entry Point) ---
if __name__ == "__main__":
    print("="*50)
    print("🚀 جاري تشغيل Smart Bot V2 (Volume Filter Edition)...")
    print("="*50)
    
    # 1. تهيئة قاعدة البيانات
    init_db()
    
    # 2. تشغيل حلقة التداول الرئيسية في الخلفية (من الجزء الثالث)
    # ملاحظة: تأكد من أن دالة main_bot_loop موجودة ومستوردة من الجزء 3
    bot_thread = Thread(target=main_bot_loop, daemon=True)
    bot_thread.start()
    
    # 3. تشغيل خادم الويب (Flask)
    logger.info("🌐 الخادم يعمل على http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)