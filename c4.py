import time
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import json
import warnings
from threading import Thread
from datetime import datetime
from decouple import config
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor

# --- 1. إعدادات النظام ---
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('smart_bot_redis.log', encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SmartBot_Arab_Redis')

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    REDIS_URL = config('REDIS_URL', default='redis://localhost:6379/0')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ في الإعدادات البيئية: {e}")
    exit(1)

# إعداد اتصال Redis
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("✅ تم الاتصال بـ Redis بنجاح.")
except Exception as e:
    logger.critical(f"❌ فشل الاتصال بـ Redis: {e}")
    exit(1)

# --- 2. إعدادات التداول الافتراضية ---
DEFAULT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,
    "risk_per_trade_pct": 2.0,
    "max_open_trades": 5,
    "max_drawdown_protect": 10.0,
    "timeframe_analysis": "15m",
    "request_delay": 0.5
}

# تخزين الإعدادات في Redis إذا لم تكن موجودة
if not redis_client.exists("bot_settings"):
    redis_client.set("bot_settings", json.dumps(DEFAULT_SETTINGS))

LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']

# --- 3. أدوات مساعدة لـ Redis ---
def get_settings():
    try:
        data = redis_client.get("bot_settings")
        return json.loads(data) if data else DEFAULT_SETTINGS
    except: return DEFAULT_SETTINGS

def update_settings(new_settings):
    redis_client.set("bot_settings", json.dumps(new_settings))

def add_log(symbol, status, reason):
    """إضافة سجل بصيغة تتوافق مع الواجهة القديمة"""
    log_entry = {
        "t": datetime.now().strftime('%H:%M'), # تنسيق الوقت كما في القديم
        "s": symbol,
        "st": status,
        "r": reason
    }
    redis_client.lpush("bot_logs", json.dumps(log_entry))
    redis_client.ltrim("bot_logs", 0, 199)

def update_market_state(regime, score, adx, volatility):
    state = {
        "market_regime": regime,
        "global_score": score,
        "trend_strength": adx,
        "volatility_index": volatility,
        "last_update": datetime.now().strftime('%H:%M')
    }
    redis_client.set("market_state", json.dumps(state))

def get_market_state():
    data = redis_client.get("market_state")
    if data: return json.loads(data)
    return {"market_regime": "Neutral", "global_score": 0, "trend_strength": 0}

# --- 4. قاعدة البيانات (PostgreSQL) ---
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
                    tp1 DOUBLE PRECISION,
                    tp2 DOUBLE PRECISION,
                    quantity DOUBLE PRECISION, 
                    strategy_name TEXT, 
                    market_regime TEXT,
                    status TEXT DEFAULT 'open', 
                    mode TEXT,
                    entry_time TIMESTAMP DEFAULT NOW(),
                    closed_at TIMESTAMP, 
                    closing_price DOUBLE PRECISION, 
                    profit_abs DOUBLE PRECISION, 
                    profit_pct DOUBLE PRECISION, 
                    exit_reason TEXT
                );
            """)
        logger.info("✅ قاعدة البيانات جاهزة (V14).")
    except Exception as e: logger.error(f"خطأ قاعدة البيانات: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 5. نظام التنبيهات ---
def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    mode_icon = "🧪 تجريبي" if payload.get('is_paper') else "💰 حقيقي"
    msg = ""
    if event == "BUY":
        msg = (f"🚀 *تنفيذ دخول استراتيجي | {payload['symbol']}*\n"
               f"📊 الاستراتيجية: `{payload['strategy']}`\n"
               f"💵 السعر: `{payload['entry_price']}`\n"
               f"🛑 الوقف: `{payload['stop_loss']}`\n"
               f"🕹️ الوضع: {mode_icon}")
    elif event == "SELL":
        emoji = "✅ ربح" if payload['profit'] > 0 else "🔻 خسارة"
        msg = (f"{emoji} *إغلاق مركز | {payload['symbol']}*\n"
               f"💰 الصافي: `{payload['profit']:.2f}%`\n"
               f"📝 السبب: _{payload['reason']}_")

    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- 6. التحليل الفني ---
def fetch_data(client, symbol, interval, limit=100):
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except: return None

def calculate_technical_indicators(df):
    df = df.copy()
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    df['plus_di'] = 100 * (pd.Series(plus_dm).rolling(14).mean() / df['atr'])
    df['minus_di'] = 100 * (pd.Series(minus_dm).rolling(14).mean() / df['atr'])
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(14).mean()

    df['bb_upper'] = df['ema20'] + (2 * df['close'].rolling(20).std())
    df['bb_lower'] = df['ema20'] - (2 * df['close'].rolling(20).std())
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ema20']
    
    df['vol_ma'] = df['volume'].rolling(20).mean()
    return df.fillna(0)

# --- 7. تحليل حالة السوق ---
def analyze_market_regime(client):
    settings = get_settings()
    total_score = 0
    analyzed_count = 0
    total_adx = 0
    
    for symbol in LEADING_SYMBOLS:
        try:
            time.sleep(settings.get('request_delay', 0.5))
            klines_1h = fetch_data(client, symbol, '1h', 60)
            if klines_1h is None: continue
            df_1h = calculate_technical_indicators(klines_1h).iloc[-1]
            
            score = 0
            if df_1h['close'] > df_1h['ema50']: score += 1
            if df_1h['macd_hist'] > 0: score += 1
            if df_1h['close'] < df_1h['ema50']: score -= 1
            
            total_score += score
            analyzed_count += 1
            total_adx += df_1h['adx']
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    if analyzed_count == 0: return

    avg_score = total_score / analyzed_count
    avg_adx = total_adx / analyzed_count
    
    regime = "Neutral"
    if avg_score >= 1.5 and avg_adx > 25: regime = "Bull_Trend_Strong"
    elif avg_score >= 0.5: regime = "Bull_Accumulation"
    elif avg_score <= -1.0: regime = "Bear_Trend_Strong"
    else: regime = "Ranging"

    update_market_state(regime, round(avg_score, 2), int(avg_adx), "Normal")

# --- 8. الاستراتيجية الذكية (مع أسباب الرفض) ---
def get_smart_signal(symbol, df, regime):
    last = df.iloc[-1]
    
    # 1. فلتر السيولة
    if last['volume'] < last['vol_ma'] * 0.5:
        return None, "سيولة ضعيفة"

    # 2. الاستراتيجيات
    if regime == "Bull_Trend_Strong":
        if last['close'] > last['ema20']:
            if last['macd_hist'] > 0 and last['adx'] > 25:
                return "Momentum_Breakout", "اختراق زخم قوي"
            else: return None, "الزخم (ADX/MACD) غير كافٍ"
        else: return None, "السعر تحت المتوسط المتحرك"

    elif regime == "Bull_Accumulation":
        if last['close'] > last['ema200']:
            if last['rsi'] < 60:
                dist = abs(last['close'] - last['ema50']) / last['close']
                if dist < 0.02: return "Trend_Pullback", "تصحيح للدعم"
                else: return None, "بعيد عن منطقة الشراء"
            else: return None, "تشبع شرائي (RSI عالي)"
        else: return None, "اتجاه عام هابط"

    elif regime == "Ranging":
        if last['bb_width'] < 0.15:
            if last['rsi'] < 35 and last['close'] > last['bb_lower']:
                return "Sniper_Reversion", "ارتداد من قاع النطاق"
            else: return None, "لم يلمس القاع أو RSI ليس منخفضاً"
        else: return None, "النطاق واسع جداً"

    return None, f"لا توجد فرصة مناسبة ({regime})"

# --- 9. المحرك الرئيسي ---
def bot_engine():
    logger.info("🚀 SmartBot Redis Engine Started")
    client = None
    while not client:
        try:
            client = Client(API_KEY, API_SECRET)
            client.get_system_status()
        except: time.sleep(10)

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT']

    while True:
        settings = get_settings()
        if not settings['is_trading_enabled']:
            update_market_state("Bot Paused", 0, 0, "None")
            time.sleep(5)
            continue

        try:
            analyze_market_regime(client)
            market = get_market_state()
            regime = market['market_regime']
            
            check_db()
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM trades_v14 WHERE status='open'")
                active_trades = cur.fetchall()

            for trade in active_trades:
                sym = trade['symbol']
                df = fetch_data(client, sym, '5m', 30)
                if df is None: continue
                
                curr = df['close'].iloc[-1]
                redis_client.hset("live_prices", sym, curr) # تحديث السعر للواجهة

                # إدارة الخروج
                exit_reason = None
                if curr <= trade['stop_loss']: exit_reason = "ضرب وقف الخسارة 🛑"
                elif curr >= trade['tp2']: exit_reason = "تحقيق الهدف الثاني 🎯"
                
                if exit_reason:
                    pnl_pct = (curr - trade['entry_price']) / trade['entry_price'] * 100
                    pnl_abs = (curr - trade['entry_price']) * trade['quantity']
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE trades_v14 
                            SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, exit_reason=%s 
                            WHERE id=%s
                        """, (curr, pnl_pct, pnl_abs, exit_reason, trade['id']))
                    send_telegram("SELL", {'symbol': sym, 'profit': pnl_pct, 'reason': exit_reason})
                    add_log(sym, "خروج", f"{exit_reason} | {pnl_pct:.2f}%")

            if len(active_trades) < settings['max_open_trades']:
                for sym in symbols:
                    if any(t['symbol'] == sym for t in active_trades): continue
                    time.sleep(settings.get('request_delay', 0.5))

                    df = fetch_data(client, sym, settings['timeframe_analysis'], 60)
                    if df is None: continue
                    
                    df = calculate_technical_indicators(df)
                    strat, reason = get_smart_signal(sym, df, regime)
                    
                    if strat:
                        curr = df['close'].iloc[-1]
                        atr = df['atr'].iloc[-1]
                        sl = curr - (atr * 2.0)
                        tp1 = curr + (atr * 2.0)
                        tp2 = curr + (atr * 4.0)
                        
                        risk_amt = settings['base_capital'] * (settings['risk_per_trade_pct'] / 100)
                        qty = risk_amt / (curr - sl) if (curr - sl) > 0 else 0
                        
                        mode = 'PAPER' if settings['paper_trading_mode'] else 'REAL'
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO trades_v14 
                                (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, mode)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (sym, curr, sl, tp1, tp2, qty, strat, regime, mode))
                        
                        add_log(sym, "دخول", f"{strat}")
                        send_telegram("BUY", {'symbol': sym, 'strategy': strat, 'entry_price': curr, 'stop_loss': sl, 'is_paper': True})
                        break 
                    else:
                        # تسجيل سبب الرفض في اللوڨ ليظهر في اللوحة
                        add_log(sym, "فحص", f"{reason}")

            time.sleep(10)

        except Exception as e:
            logger.error(f"Loop Error: {e}")
            time.sleep(5)

# --- 10. واجهة Flask (متوافقة مع اللوحة القديمة) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/analytics')
def analytics():
    # تجهيز البيانات بنفس الهيكلية التي يتوقعها الكود القديم
    settings = get_settings()
    market = get_market_state()
    
    # 1. السجلات
    logs_raw = redis_client.lrange("bot_logs", 0, 199)
    logs = [json.loads(l) for l in logs_raw]
    
    # 2. الصفقات والأسعار
    signals = []
    prices = {}
    stats = {'win_rate': 0, 'profit_factor': 0, 'total_pnl_usd': 0, 'trade_count': 0, 'history': []}
    
    try:
        # جلب الأسعار الحية من Redis
        prices_raw = redis_client.hgetall("live_prices")
        prices = {k: float(v) for k, v in prices_raw.items()}
        
        check_db()
        with conn.cursor() as cur:
            # الصفقات المفتوحة
            cur.execute("SELECT * FROM trades_v14 WHERE status='open' ORDER BY entry_time DESC")
            open_trades = cur.fetchall()
            for t in open_trades:
                signals.append({
                    'symbol': t['symbol'],
                    'strategy': t['strategy_name'],
                    'entry_price': t['entry_price'],
                    'tp1': t['tp1'],
                    'tp2': t['tp2'],
                    'stop_loss': t['stop_loss']
                })
            
            # الإحصائيات
            cur.execute("SELECT closed_at, profit_pct, profit_abs FROM trades_v14 WHERE status='closed' ORDER BY closed_at ASC")
            closed_trades = cur.fetchall()
            
            wins = 0
            gross_profit = 0
            gross_loss = 0
            cum_pnl = 0
            
            for r in closed_trades:
                val = r['profit_abs'] if r['profit_abs'] else 0
                pct = r['profit_pct'] if r['profit_pct'] else 0
                
                if pct > 0: 
                    wins += 1
                    gross_profit += val
                else: 
                    gross_loss += abs(val)
                
                cum_pnl += pct
                stats['history'].append({'t': r['closed_at'].strftime('%d %H:%M'), 'v': cum_pnl})
            
            total_trades = len(closed_trades)
            stats['trade_count'] = total_trades
            stats['total_pnl_usd'] = gross_profit - gross_loss
            stats['win_rate'] = (wins / total_trades * 100) if total_trades > 0 else 0
            stats['profit_factor'] = (gross_profit / gross_loss) if gross_loss > 0 else 0

    except Exception as e: logger.error(f"API Error: {e}")

    # هيكل JSON المطابق للوحة القديمة
    return jsonify({
        "market": market,
        "signals": signals,
        "prices": prices,
        "stats": stats,
        "logs": logs,
        "settings": settings
    })

@app.route('/api/toggle', methods=['POST'])
def toggle():
    s = get_settings()
    s['is_trading_enabled'] = not s['is_trading_enabled']
    update_settings(s)
    return jsonify("OK")

# --- اللوحة القديمة ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot V13 - لوحة التحكم</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --panel: #151a1e; --border: #2b3139; --text: #eaecef; --green: #0ecb81; --red: #f6465d; --accent: #f0b90b; }
        * { box-sizing: border-box; }
        body { background: var(--bg); color: var(--text); font-family: 'Tajawal', sans-serif; margin: 0; padding: 20px; font-size: 14px; text-align: right; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid var(--border); }
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; margin-bottom: 20px; }
        .col-3 { grid-column: span 3; } .col-4 { grid-column: span 4; } .col-6 { grid-column: span 6; } .col-8 { grid-column: span 8; } .col-12 { grid-column: span 12; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; position: relative; }
        .card h3 { margin: 0 0 15px 0; color: #848e9c; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
        .big-num { font-size: 28px; font-weight: 900; color: var(--text); }
        .sub-text { color: #848e9c; font-size: 12px; }
        .status-dot { height: 10px; width: 10px; background-color: #555; border-radius: 50%; display: inline-block; margin-left: 5px; }
        .dot-green { background-color: var(--green); box-shadow: 0 0 10px var(--green); }
        .btn { background: var(--accent); color: #000; border: none; padding: 8px 20px; border-radius: 4px; font-weight: bold; cursor: pointer; transition: 0.2s; font-family: 'Tajawal'; }
        .btn:hover { opacity: 0.9; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: right; padding: 12px; border-bottom: 1px solid var(--border); }
        th { color: #848e9c; font-size: 12px; }
        .pnl-g { color: var(--green); } .pnl-r { color: var(--red); }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
        
        @media(max-width: 768px) { .col-3, .col-4, .col-6, .col-8 { grid-column: span 12; } }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size:24px">SmartBot <span style="color:var(--accent)">V13 AR</span></h1>
            <span style="font-size:12px; color:#848e9c">نظام إدارة المحفظة الذكي - النسخة العربية</span>
        </div>
        <div style="display:flex; gap:15px; align-items:center">
            <div style="text-align:left; margin-left:15px">
                <span id="connectionStatus" class="status-dot"></span>
                <span style="font-size:12px">متصل</span>
            </div>
            <button id="powerBtn" class="btn" onclick="toggleBot()">جاري التحميل...</button>
        </div>
    </div>

    <!-- مؤشرات الأداء -->
    <div class="grid">
        <div class="card col-3">
            <h3>حالة السوق</h3>
            <div id="regime" class="big-num" style="color:var(--accent); font-size:20px">--</div>
            <div class="sub-text">قوة الاتجاه: <span id="trendStr">0</span></div>
        </div>
        <div class="card col-3">
            <h3>نسبة النجاح</h3>
            <div class="big-num"><span id="winRate">0</span><small>%</small></div>
            <div class="sub-text">عدد الصفقات: <span id="tradeCount">0</span></div>
        </div>
        <div class="card col-3">
            <h3>صافي الأرباح (USDT)</h3>
            <div id="totalPnl" class="big-num">$0.00</div>
            <div class="sub-text">عامل الربح: <span id="profFact">0</span></div>
        </div>
        <div class="card col-3">
            <h3>المخاطرة الحالية</h3>
            <div class="big-num"><span id="openRisk">0</span><small>%</small></div>
            <div class="sub-text">صفقات مفتوحة: <span id="activeCount">0</span></div>
        </div>
    </div>

    <!-- الرسوم البيانية -->
    <div class="grid">
        <div class="card col-8">
            <h3>نمو المحفظة (Equity Curve)</h3>
            <div style="height: 250px;"><canvas id="equityChart"></canvas></div>
        </div>
        <div class="card col-4">
            <h3>توزيع الصفقات</h3>
            <div style="height: 250px; position:relative">
                <canvas id="statsChart"></canvas>
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); text-align:center">
                    <span style="font-size:20px; font-weight:bold" id="winRateCenter">0%</span><br>
                    <span style="font-size:10px; color:#888">فوز</span>
                </div>
            </div>
        </div>
    </div>

    <!-- الجداول -->
    <div class="grid">
        <div class="card col-8">
            <h3>المحفظة النشطة (Active Trades)</h3>
            <table>
                <thead>
                    <tr>
                        <th>العملة</th>
                        <th>الاستراتيجية</th>
                        <th>الدخول</th>
                        <th>السعر</th>
                        <th>الربح %</th>
                        <th>الأهداف</th>
                    </tr>
                </thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>
        <div class="card col-4">
            <h3>سجل النظام (Logs)</h3>
            <div style="height: 300px; overflow-y: auto;">
                <table style="font-size:12px">
                    <tbody id="logsBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let equityChart, statsChart;
        Chart.defaults.color = '#848e9c';
        Chart.defaults.borderColor = '#2b3139';
        Chart.defaults.font.family = 'Tajawal';

        // قاموس الترجمة لحالات السوق
        const regimeMap = {
            "Bull_Trend_Strong": "اتجاه صاعد قوي 🐂",
            "Bull_Accumulation": "تجميع صاعد 📈",
            "Bear_Trend_Strong": "اتجاه هابط قوي 🐻",
            "High_Volatility_Choppy": "تذبذب عالي ⚡",
            "Ranging": "عرضي مستقر 🦀",
            "Neutral": "محايد ⚖️"
        };

        // قاموس الترجمة للاستراتيجيات
        const stratMap = {
            "Trend_Pullback": "إعادة دخول (ترند)",
            "Momentum_Breakout": "اختراق زخم",
            "Sniper_Reversion": "قناص مرتد",
            "Deep_Value_Scalp": "خطف سيولة",
            "Golden_Cross": "تقاطع ذهبي"
        };

        function initCharts() {
            const ctx1 = document.getElementById('equityChart').getContext('2d');
            const gradient = ctx1.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, 'rgba(240, 185, 11, 0.2)');
            gradient.addColorStop(1, 'rgba(240, 185, 11, 0)');

            equityChart = new Chart(ctx1, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'النمو %', data: [], borderColor: '#f0b90b', backgroundColor: gradient, borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { borderDash: [5, 5] } } } }
            });

            const ctx2 = document.getElementById('statsChart').getContext('2d');
            statsChart = new Chart(ctx2, {
                type: 'doughnut',
                data: { labels: ['ربح', 'خسارة'], datasets: [{ data: [50, 50], backgroundColor: ['#0ecb81', '#f6465d'], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { position: 'bottom' } } }
            });
        }

        async function updateData() {
            try {
                const res = await fetch('/api/analytics');
                const d = await res.json();

                // 1. تحديث الأزرار
                const btn = document.getElementById('powerBtn');
                document.getElementById('connectionStatus').className = "status-dot dot-green";
                if(d.settings.is_trading_enabled) {
                    btn.innerText = "إيقاف البوت 🛑";
                    btn.style.background = "var(--red)";
                    btn.style.color = "#fff";
                } else {
                    btn.innerText = "تشغيل البوت 🚀";
                    btn.style.background = "var(--green)";
                    btn.style.color = "#fff";
                }

                // 2. المؤشرات
                const regKey = d.market.market_regime;
                document.getElementById('regime').innerText = regimeMap[regKey] || regKey;
                document.getElementById('trendStr').innerText = d.market.trend_strength;
                
                document.getElementById('winRate').innerText = d.stats.win_rate.toFixed(1);
                document.getElementById('winRateCenter').innerText = d.stats.win_rate.toFixed(1) + "%";
                document.getElementById('tradeCount').innerText = d.stats.trade_count;
                
                const pnl = d.stats.total_pnl_usd;
                const pnlEl = document.getElementById('totalPnl');
                pnlEl.innerText = "$" + pnl.toFixed(2);
                pnlEl.style.color = pnl >= 0 ? "var(--green)" : "var(--red)";
                document.getElementById('profFact').innerText = d.stats.profit_factor.toFixed(2);

                document.getElementById('activeCount').innerText = d.signals.length;
                document.getElementById('openRisk').innerText = (d.signals.length * 2).toFixed(1); 

                // 3. تحديث الشارت
                if(d.stats.history.length > 0) {
                    equityChart.data.labels = d.stats.history.map(h => h.t);
                    equityChart.data.datasets[0].data = d.stats.history.map(h => h.v);
                    equityChart.update();
                    
                    statsChart.data.datasets[0].data = [d.stats.win_rate, 100 - d.stats.win_rate];
                    statsChart.update();
                }

                // 4. جدول الصفقات
                const tb = document.getElementById('tradesBody');
                tb.innerHTML = d.signals.length ? d.signals.map(s => {
                    const curr = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                    const stratName = stratMap[s.strategy] || s.strategy;
                    return `
                    <tr>
                        <td style="font-weight:bold; color:var(--text)">${s.symbol}</td>
                        <td><span style="background:#2b3139; padding:2px 6px; border-radius:4px; font-size:11px">${stratName}</span></td>
                        <td>${s.entry_price}</td>
                        <td>${curr}</td>
                        <td class="${pnl>=0?'pnl-g':'pnl-r'}">${pnl.toFixed(2)}%</td>
                        <td style="font-size:11px; color:#848e9c">${s.tp1} ➔ ${s.tp2}</td>
                    </tr>`;
                }).join('') : "<tr><td colspan='6' style='text-align:center; padding:20px; color:#444'>لا توجد صفقات نشطة حالياً</td></tr>";

                // 5. السجل
                document.getElementById('logsBody').innerHTML = d.logs.map(l => `
                    <tr>
                        <td style="color:#666">${l.t}</td>
                        <td style="font-weight:bold">${l.s}</td>
                        <td style="color:${l.st==='دخول'?'var(--green)':'#848e9c'}">${l.st}</td>
                        <td>${l.r}</td>
                    </tr>
                `).join('');

            } catch(e) { console.error(e); }
        }

        function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(updateData); }

        initCharts();
        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    Thread(target=bot_engine, daemon=True).start()
    logger.info("✅ Web Server Started on 5000")
    app.run(host='0.0.0.0', port=5000)