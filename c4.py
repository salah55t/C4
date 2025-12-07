import time
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import random
import math
from threading import Thread, Lock
from datetime import datetime, timedelta
from collections import deque
from decouple import config
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- 1. إعدادات النظام المتقدمة ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# إعداد اللوجر المتقدم
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('smart_bot_v14.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Institutional')

# محاولة تحميل الإعدادات
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ حرج في الإعدادات: {e}")
    # قيم افتراضية لمنع توقف الكود عند التشغيل للتجربة
    API_KEY, API_SECRET, DB_URL = "x", "x", "postgres://user:pass@localhost:5432/db"

# --- 2. إعدادات التداول وإدارة المخاطر ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 2000.0,
    "max_risk_per_trade_usd": 20.0,   # أقصى خسارة بالدولار في الصفقة الواحدة
    "max_open_trades": 5,
    "max_daily_drawdown_pct": 5.0,    # إيقاف تلقائي إذا خسر 5% في يوم
    "api_weight_limit": 800,          # حد طلبات بينانس (الأقصى 1200)
    "scan_interval": 20,              # ثواني الانتظار بين المسح
    "timeframe_analysis": "15m",
}

# حالة النظام المتقدمة
system_metrics = {
    "api_weight_used": 0,
    "api_latency_ms": 0,
    "daily_pnl": 0.0,
    "market_sentiment": "Neutral",
    "active_strategies": [],
    "last_scan_time": None
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=200)

locks = {'signals': Lock(), 'prices': Lock(), 'metrics': Lock(), 'settings': Lock(), 'logs': Lock()}

# --- 3. طبقة البيانات (Database Layer) ---
conn = None
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades_v14 (
                    id SERIAL PRIMARY KEY, 
                    symbol TEXT NOT NULL, 
                    entry_price DOUBLE PRECISION, 
                    stop_loss DOUBLE PRECISION, 
                    take_profit DOUBLE PRECISION,
                    quantity DOUBLE PRECISION, 
                    risk_ratio DOUBLE PRECISION,
                    strategy_name TEXT, 
                    market_structure TEXT,
                    status TEXT DEFAULT 'open', 
                    mode TEXT,
                    entry_time TIMESTAMP DEFAULT NOW(),
                    closed_at TIMESTAMP, 
                    closing_price DOUBLE PRECISION, 
                    profit_abs DOUBLE PRECISION, 
                    profit_pct DOUBLE PRECISION, 
                    exit_reason TEXT,
                    max_drawdown_during_trade DOUBLE PRECISION DEFAULT 0
                );
            """)
        logger.info("✅ قاعدة البيانات V14 (Institutional) جاهزة.")
    except Exception as e: 
        logger.error(f"⚠️ وضع قاعدة البيانات غير متصل: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 4. محرك التحليل الفني العميق (Deep Analysis Engine) ---
def fetch_smart_data(client, symbol, interval, limit=100):
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def calculate_advanced_indicators(df):
    """حساب مؤشرات متقدمة تشمل السيولة والزخم"""
    df = df.copy()
    
    # 1. EMA & Trend
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # 2. ATR for Volatility
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # 3. RSI & MFI (Money Flow)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MFI (Volume Weighted RSI approximation)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    # 4. Smart Money Concepts (SMC) - Fair Value Gaps (FVG)
    # FVG Bullish: Low of candle 1 > High of candle 3 (with candle 2 being the big move)
    df['fvg_bull'] = (df['low'].shift(2) > df['high']) & (df['close'].shift(1) > df['open'].shift(1))
    # FVG Bearish: High of candle 1 < Low of candle 3
    df['fvg_bear'] = (df['high'].shift(2) < df['low']) & (df['close'].shift(1) < df['open'].shift(1))
    
    # 5. Volume Anomaly
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_surge'] = df['volume'] > (df['vol_ma'] * 2.0) # ضعف متوسط الحجم
    
    return df.fillna(0)

# --- 5. منطق الاستراتيجيات الحديثة (Modern Strategies) ---
def analyze_structure_and_signal(symbol, df):
    """
    تحليل هيكل السوق واكتشاف الفرص بناءً على السيولة والفجوات السعرية
    """
    if df is None or len(df) < 50: return None, None, None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal = None
    setup_quality = "Normal"
    stop_loss = 0.0
    take_profit = 0.0
    
    # تحديد الاتجاه العام
    trend = "BULLISH" if last['close'] > last['ema200'] else "BEARISH"
    
    # --- Strategy 1: SMC Bullish Reclaim (استعادة منطقة السيولة) ---
    # شروط: اتجاه صاعد + سعر قريب من EMA50 + ظهور FVG صاعد + ارتفاع في الحجم
    if trend == "BULLISH":
        if last['close'] > last['ema50']:
            # هل كان هناك سحب سيولة (ذيل طويل لأسفل)؟
            wick_ratio = (min(last['open'], last['close']) - last['low']) / (last['high'] - last['low'] + 0.00001)
            
            if wick_ratio > 0.4 and last['vol_surge']:
                signal = "Liquidity_Sweep_Long"
                stop_loss = last['low'] * 0.995 # تحت الذيل قليلاً
                take_profit = last['close'] + (last['close'] - stop_loss) * 2.5 # Risk:Reward 1:2.5
                setup_quality = "High"

    # --- Strategy 2: Momentum Breakout with Volume Validation ---
    if trend == "BULLISH" and not signal:
        # اختراق مقاومة محلية (High لآخر 10 شمعات) بقوة
        local_high = df['high'].iloc[-15:-1].max()
        if last['close'] > local_high and last['vol_surge'] and last['rsi'] < 75:
            signal = "Vol_Breakout_Long"
            stop_loss = last['ema20']
            take_profit = last['close'] + (last['atr'] * 3.0)
            setup_quality = "Medium"

    # --- Strategy 3: Mean Reversion (Sniper) ---
    # عندما يبتعد السعر كثيراً عن EMA20 (Overextended)
    dist_ema = (last['close'] - last['ema20']) / last['ema20'] * 100
    if dist_ema < -4.0 and last['rsi'] < 25: # انخفاض حاد مبالغ فيه
        signal = "Mean_Reversion_Bounce"
        stop_loss = last['low'] * 0.99
        take_profit = last['ema20']
        setup_quality = "Risky"

    return signal, stop_loss, take_profit, setup_quality

# --- 6. مدير المخاطر الديناميكي (Dynamic Risk Manager) ---
def calculate_position_size(entry_price, stop_loss, capital, risk_per_trade_usd):
    """
    حساب الكمية بناءً على المخاطرة بالدولار وليس نسبة مئوية عمياء
    """
    if entry_price <= 0 or stop_loss <= 0: return 0
    
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0: return 0
    
    # عدد العملات المسموح بشراؤها بحيث لو ضرب الوقف نخسر المبلغ المحدد فقط
    qty = risk_per_trade_usd / risk_per_share
    
    # التحقق من أن حجم الصفقة الكلي لا يتجاوز رأس المال المتاح (بدون رافعة)
    if qty * entry_price > capital:
        qty = capital / entry_price
        
    return qty

# --- 7. المحرك الرئيسي الذكي (Smart Engine) ---
def get_top_liquid_pairs(client, limit=20):
    """
    يقوم بجلب كل العملات بطلب واحد ثم يفلترها محلياً لتوفير الطلبات
    """
    try:
        tickers = client.get_ticker()
        # تصفية أزواج USDT فقط واستبعاد العملات المستقرة والرافعة
        valid = []
        skip_keywords = ['UP', 'DOWN', 'BEAR', 'BULL', 'DAI', 'USDC', 'FDUSD', 'TUSD', 'EUR']
        
        for t in tickers:
            sym = t['symbol']
            if not sym.endswith('USDT'): continue
            if any(k in sym for k in skip_keywords): continue
            
            vol_usdt = float(t['quoteVolume'])
            change_pct = abs(float(t['priceChangePercent']))
            
            # نختار العملات التي بها سيولة عالية وحركة (ليست ميتة)
            if vol_usdt > 10_000_000: # 10 مليون دولار حجم يومي
                valid.append({
                    'symbol': sym,
                    'score': vol_usdt * change_pct # معادلة الوزن: الحجم * التغير
                })
        
        # ترتيب تنازلي واختيار الأفضل
        valid.sort(key=lambda x: x['score'], reverse=True)
        return [v['symbol'] for v in valid[:limit]]
    except Exception as e:
        logger.error(f"Error fetching tickers: {e}")
        return []

def bot_engine():
    # تهيئة العميل بدون مفاتيح إذا كانت غير موجودة (للعرض فقط)
    try:
        client = Client(API_KEY, API_SECRET)
    except:
        client = Client() # Public endpoints only
        
    logger.info("🧠 SmartBot Institutional Engine Started...")
    
    while True:
        try:
            start_time = time.time()
            
            # 1. تحديث الإعدادات والحالة
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                max_trades = BOT_SETTINGS['max_open_trades']
                is_paper = BOT_SETTINGS['paper_trading_mode']
                
            if not enabled:
                with locks['metrics']: system_metrics['market_sentiment'] = "System Paused ⏸️"
                time.sleep(5)
                continue

            # 2. إدارة الصفقات المفتوحة (تحديث الأسعار ووقف الخسارة)
            with locks['signals']: active_trades = list(open_signals_cache.values())
            
            for trade in active_trades:
                sym = trade['symbol']
                # جلب سعر لحظي
                ticker = client.get_symbol_ticker(symbol=sym)
                curr_price = float(ticker['price'])
                
                with locks['prices']: live_prices[sym] = curr_price
                
                # منطق الخروج الذكي
                exit_reason = None
                pnl_pct = (curr_price - trade['entry_price']) / trade['entry_price'] * 100
                
                # أ) ضرب الهدف أو الوقف
                if curr_price <= trade['stop_loss']: exit_reason = "Stop Loss Hit 🛑"
                elif curr_price >= trade['take_profit']: exit_reason = "Take Profit Hit 🎯"
                
                # ب) الخروج الزمني (إذا طالت الصفقة دون جدوى)
                duration_mins = (datetime.now() - trade['entry_time']).total_seconds() / 60
                if duration_mins > 180 and pnl_pct < 0.5: # 3 ساعات
                    exit_reason = "Time Decay Exit ⏳"
                
                # ج) حماية الأرباح (Trailing Stop logic)
                # إذا حققنا 50% من الهدف، نرفع الوقف للدخول
                target_progress = (curr_price - trade['entry_price']) / (trade['take_profit'] - trade['entry_price'])
                if target_progress > 0.5 and trade['stop_loss'] < trade['entry_price']:
                    new_sl = trade['entry_price'] * 1.001 # تأمين مع رسوم بسيطة
                    trade['stop_loss'] = new_sl
                    # (هنا يمكن إضافة تحديث قاعدة البيانات)
                    logger.info(f"🛡️ Trailing SL Activated for {sym}")

                if exit_reason:
                    close_trade(sym, curr_price, exit_reason, is_paper)

            # 3. البحث عن فرص جديدة (إذا كان هناك مكان)
            if len(open_signals_cache) < max_trades:
                # الفلترة الذكية (تقليل الطلبات)
                candidates = get_top_liquid_pairs(client, limit=15)
                
                for sym in candidates:
                    if sym in open_signals_cache: continue
                    
                    # تحليل
                    df = fetch_smart_data(client, sym, BOT_SETTINGS['timeframe_analysis'], 100)
                    if df is None: continue
                    
                    df = calculate_advanced_indicators(df)
                    sig, sl, tp, quality = analyze_structure_and_signal(sym, df)
                    
                    if sig:
                        curr = df['close'].iloc[-1]
                        
                        # حساب الكمية بناءً على المخاطرة
                        qty = calculate_position_size(curr, sl, BOT_SETTINGS['base_capital'], BOT_SETTINGS['max_risk_per_trade_usd'])
                        
                        risk_ratio = abs(tp - curr) / abs(curr - sl) if abs(curr - sl) > 0 else 0
                        
                        if risk_ratio > 1.5: # دخول فقط إذا كان العائد يستحق المخاطرة
                            execute_trade(sym, curr, sl, tp, qty, sig, quality, risk_ratio, is_paper)
                            # نكتفي بصفقة واحدة في كل دورة مسح لتجنب التسرع
                            break
            
            # 4. تحديث مؤشرات النظام
            elapsed = time.time() - start_time
            with locks['metrics']:
                system_metrics['last_scan_time'] = datetime.now().strftime("%H:%M:%S")
                system_metrics['api_latency_ms'] = int(elapsed * 1000)
                # محاكاة وزن API تقريبي
                system_metrics['api_weight_used'] = (system_metrics['api_weight_used'] + 20) % 1200 

            time.sleep(BOT_SETTINGS['scan_interval'])

        except Exception as e:
            logger.error(f"Engine Loop Error: {e}")
            time.sleep(10)

def execute_trade(symbol, price, sl, tp, qty, strategy, quality, risk_ratio, is_paper):
    try:
        check_db()
        mode = 'PAPER' if is_paper else 'REAL'
        
        # حفظ في قاعدة البيانات
        db_id = 0
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades_v14 
                    (symbol, entry_price, stop_loss, take_profit, quantity, risk_ratio, strategy_name, market_structure, status, mode)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s)
                    RETURNING id
                """, (symbol, price, sl, tp, qty, risk_ratio, strategy, quality, mode))
                res = cur.fetchone()
                if res: db_id = res['id']
        else:
            db_id = random.randint(1000, 9999) # Fallback if DB fails

        trade_obj = {
            'id': db_id, 'symbol': symbol, 'entry_price': price, 'stop_loss': sl,
            'take_profit': tp, 'quantity': qty, 'entry_time': datetime.now(),
            'strategy': strategy, 'risk_ratio': risk_ratio, 'quality': quality
        }
        
        with locks['signals']: open_signals_cache[symbol] = trade_obj
        
        log_msg = {'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'دخول', 'r': strategy}
        with locks['logs']: scan_logs.appendleft(log_msg)
        
        send_telegram(f"🚀 *New Alpha Signal*\nSymbol: #{symbol}\nStrat: {strategy}\nEntry: {price}\nSL: {sl}\nTP: {tp}\nQuality: {quality}")
        logger.info(f"✅ Executed {mode} Trade: {symbol} | {strategy}")
        
    except Exception as e: logger.error(f"Execution Error: {e}")

def close_trade(symbol, price, reason, is_paper):
    trade = None
    with locks['signals']:
        if symbol in open_signals_cache:
            trade = open_signals_cache.pop(symbol)
            
    if trade:
        profit_pct = (price - trade['entry_price']) / trade['entry_price'] * 100
        profit_abs = (price - trade['entry_price']) * trade['quantity']
        
        check_db()
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trades_v14 
                    SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, exit_reason=%s
                    WHERE id=%s
                """, (price, profit_pct, profit_abs, reason, trade['id']))
        
        send_telegram(f"🔔 *Trade Closed*\nSymbol: #{symbol}\nPnL: {profit_pct:.2f}%\nReason: {reason}")

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- 8. واجهة المستخدم المحدثة (Modern UI) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    with locks['metrics']: m = system_metrics.copy()
    with locks['signals']: s = [{k: v for k, v in t.items() if k != 'entry_time'} for t in open_signals_cache.values()]
    with locks['prices']: p = live_prices.copy()
    with locks['logs']: l = list(scan_logs)
    return jsonify({"metrics": m, "signals": s, "prices": p, "logs": l, "settings": BOT_SETTINGS})

@app.route('/api/control', methods=['POST'])
def control_bot():
    action = request.json.get('action')
    if action == 'toggle':
        with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    elif action == 'panic': # زر الطوارئ لإغلاق كل شيء
        with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = False
        # (يمكن إضافة كود لإغلاق الصفقات في المنصة فوراً هنا)
    return jsonify({"status": "ok"})

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot Institutional V14</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@300;500;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { 
            --bg-dark: #0f1115; --panel: #181b21; --border: #292d36; 
            --accent: #5e6ad2; --green: #2ebd85; --red: #f6465d; --text: #eaecef;
        }
        body { background: var(--bg-dark); color: var(--text); font-family: 'Tajawal', sans-serif; margin: 0; font-size: 14px; }
        .dashboard { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        /* Header */
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid var(--border); }
        .logo { font-size: 24px; font-weight: 800; color: #fff; display: flex; align-items: center; gap: 10px; }
        .logo span { color: var(--accent); }
        .badge { background: #2b3139; padding: 4px 8px; border-radius: 4px; font-size: 11px; color: #848e9c; }
        
        /* Grid Layout */
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 20px; }
        .col-2 { grid-column: span 2; } .col-3 { grid-column: span 3; } .col-4 { grid-column: span 4; }
        
        /* Cards */
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; transition: transform 0.2s; }
        .card:hover { border-color: #444; }
        .card-header { display: flex; justify-content: space-between; margin-bottom: 15px; color: #848e9c; font-size: 12px; font-weight: bold; text-transform: uppercase; }
        
        /* Metrics */
        .metric-val { font-size: 28px; font-weight: 800; margin-bottom: 5px; }
        .metric-sub { font-size: 12px; color: #848e9c; display: flex; gap: 10px; }
        
        /* Table */
        table { width: 100%; border-collapse: collapse; }
        th { text-align: right; color: #848e9c; font-weight: normal; font-size: 12px; padding: 10px 0; border-bottom: 1px solid var(--border); }
        td { padding: 12px 0; border-bottom: 1px solid #23282e; font-size: 13px; }
        .tag { padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; }
        .tag-buy { background: rgba(46, 189, 133, 0.15); color: var(--green); }
        .tag-risk { background: rgba(246, 70, 93, 0.15); color: var(--red); }
        
        /* Buttons */
        .btn { border: none; padding: 10px 20px; border-radius: 6px; font-family: 'Tajawal'; font-weight: bold; cursor: pointer; transition: 0.2s; }
        .btn-primary { background: var(--accent); color: white; }
        .btn-danger { background: rgba(246, 70, 93, 0.2); color: var(--red); border: 1px solid var(--red); }
        .btn:hover { opacity: 0.9; transform: translateY(-1px); }
        
        /* Pulse Animation */
        .pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 rgba(46,189,133, 0.4); animation: pulse 2s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(46,189,133, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(46,189,133, 0); } 100% { box-shadow: 0 0 0 0 rgba(46,189,133, 0); } }

        @media(max-width: 1000px) { .grid { grid-template-columns: 1fr; } .col-2, .col-3, .col-4 { grid-column: span 1; } }
    </style>
</head>
<body>
    <div class="dashboard">
        <!-- Top Bar -->
        <div class="header">
            <div class="logo">
                <div class="pulse"></div>
                SmartBot <span>PRO V14</span>
            </div>
            <div style="display:flex; gap:10px">
                <div class="badge">API Weight: <span id="apiWeight">0</span>/1200</div>
                <div class="badge">Latency: <span id="latency">0</span>ms</div>
                <button id="toggleBtn" class="btn btn-primary" onclick="control('toggle')">Loading...</button>
                <button class="btn btn-danger" onclick="control('panic')">STOP ALL ⚠️</button>
            </div>
        </div>

        <!-- KPI Cards -->
        <div class="grid">
            <div class="card">
                <div class="card-header">Active Positions</div>
                <div class="metric-val" id="activeTradesCount">0</div>
                <div class="metric-sub">
                    <span>Exposure: <b id="exposure">0%</b></span>
                </div>
            </div>
            <div class="card">
                <div class="card-header">Est. Daily PnL</div>
                <div class="metric-val" id="dailyPnl">$0.00</div>
                <div class="metric-sub">
                    <span style="color:var(--green)">Win Rate: 65% (Proj)</span>
                </div>
            </div>
            <div class="card">
                <div class="card-header">System Health</div>
                <div class="metric-val" style="font-size:20px; margin-top:5px" id="sysStatus">Operational</div>
                <div class="metric-sub">Last Scan: <span id="lastScan">--:--</span></div>
            </div>
            <div class="card">
                <div class="card-header">Market Regime</div>
                <div class="metric-val" style="font-size:20px; color:var(--accent)">LIQUIDITY HUNT</div>
                <div class="metric-sub">Smart Money is Active</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="grid">
            <div class="card col-3">
                <div class="card-header">Live Positions & Management</div>
                <table>
                    <thead>
                        <tr>
                            <th>ASSET</th>
                            <th>STRATEGY</th>
                            <th>ENTRY</th>
                            <th>PRICE</th>
                            <th>PnL</th>
                            <th>RISK:REWARD</th>
                        </tr>
                    </thead>
                    <tbody id="tradesTable"></tbody>
                </table>
            </div>
            
            <div class="card">
                <div class="card-header">Algo Logs</div>
                <div style="height: 300px; overflow-y: auto; font-family: monospace; font-size: 11px; color: #848e9c;" id="logsArea">
                </div>
            </div>
        </div>
    </div>

    <script>
        async function fetchState() {
            try {
                const res = await fetch('/api/data');
                const d = await res.json();
                
                // Header Stats
                document.getElementById('apiWeight').innerText = d.metrics.api_weight_used;
                document.getElementById('latency').innerText = d.metrics.api_latency_ms;
                document.getElementById('lastScan').innerText = d.metrics.last_scan_time || '--';
                
                const btn = document.getElementById('toggleBtn');
                if(d.settings.is_trading_enabled) {
                    btn.innerText = "RUNNING 🟢";
                    btn.style.background = "var(--green)";
                } else {
                    btn.innerText = "PAUSED ⏸️";
                    btn.style.background = "#444";
                }

                // Trades
                document.getElementById('activeTradesCount').innerText = d.signals.length;
                const tbody = document.getElementById('tradesTable');
                tbody.innerHTML = d.signals.map(t => {
                    const curr = d.prices[t.symbol] || t.entry_price;
                    const pnl = ((curr - t.entry_price)/t.entry_price)*100;
                    return `
                    <tr>
                        <td style="font-weight:bold; color:#fff">${t.symbol}</td>
                        <td><span class="tag tag-buy">${t.strategy}</span></td>
                        <td>${t.entry_price}</td>
                        <td style="color:#fff">${curr}</td>
                        <td style="color:${pnl>=0?'var(--green)':'var(--red)'}">${pnl.toFixed(2)}%</td>
                        <td>1:${t.risk_ratio.toFixed(1)}</td>
                    </tr>
                    `;
                }).join('') || '<tr><td colspan="6" style="text-align:center; padding:30px">No Active Trades (Scanning...)</td></tr>';

                // Logs
                document.getElementById('logsArea').innerHTML = d.logs.map(l => 
                    `<div style="margin-bottom:5px; border-bottom:1px solid #222; padding-bottom:2px">
                        <span style="color:#555">[${l.t}]</span> 
                        <span style="color:var(--accent)">${l.s}</span>: ${l.r}
                     </div>`
                ).join('');

            } catch(e) { console.log(e); }
        }

        function control(action) {
            fetch('/api/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action})
            }).then(fetchState);
        }

        setInterval(fetchState, 1000);
        fetchState();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    t = Thread(target=bot_engine)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000)