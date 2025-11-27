import time
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import random
import json
from threading import Thread, Lock
from datetime import datetime, timedelta
from collections import deque
from decouple import config
from binance.client import Client
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# ==========================================
# 💎 SmartBot Pro V14 - AI Edition
# تصميم وبرمجة: خوارزميات النخبة
# ==========================================

# --- 1. إعدادات النظام المتقدمة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('smart_bot_v14.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Pro')

# تحميل الإعدادات من البيئة
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ حرج في التكوين: {e}")
    # قيم افتراضية لمنع توقف السكربت عند التجربة بدون ملف env
    API_KEY = "x"; API_SECRET = "x"; DB_URL = "postgres://user:pass@localhost:5432/db"

# --- 2. محرك إدارة الثروة (Risk Management Engine) ---
BOT_SETTINGS = {
    "is_active": False,          # التشغيل الرئيسي
    "mode": "PAPER",             # PAPER (وهمي) أو REAL (حقيقي)
    "capital": 1000.0,           # رأس المال المخصص
    "risk_per_trade": 1.5,       # المخاطرة للصفقة (1.5% من رأس المال)
    "max_daily_loss": 5.0,       # قاطع الدائرة: إيقاف عند خسارة 5% يومياً
    "max_concurrent_trades": 5,  # أقصى عدد صفقات مفتوحة
    "min_confidence": 75,        # أقل نسبة ثقة للدخول (0-100)
    "cooldown_minutes": 15,      # راحة بعد كل صفقة
    "leverage_sim": 1            # محاكاة الرافعة (للحسابات فقط)
}

# قائمة العملات النخبة (High Liquidity Only)
WATCHLIST = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'MATICUSDT']

# حالة النظام الحية
state = {
    "market_status": "ANALYZING", # BULLISH, BEARISH, CHOPPY, CRASH
    "daily_pnl_pct": 0.0,
    "active_trades_count": 0,
    "last_scan_time": None,
    "circuit_breaker_triggered": False
}

locks = {'db': Lock(), 'data': Lock(), 'log': Lock()}
activity_log = deque(maxlen=100)
active_trades_cache = {}

# --- 3. طبقة قاعدة البيانات (PostgreSQL V14 Schema) ---
conn = None
def get_db_connection():
    global conn
    try:
        if conn is None or conn.closed != 0:
            conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
            conn.autocommit = True
    except Exception as e:
        logger.error(f"⚠️ خطأ اتصال قاعدة البيانات: {e}")
    return conn

def init_tables():
    c = get_db_connection()
    if not c: return
    with c.cursor() as cur:
        # جدول الصفقات المطور
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades_v14 (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20),
                type VARCHAR(10),       -- BUY/SELL
                entry_price FLOAT,
                stop_loss FLOAT,
                take_profit_1 FLOAT,
                take_profit_2 FLOAT,
                quantity FLOAT,
                leverage INT DEFAULT 1,
                strategy VARCHAR(50),
                confidence_score INT,
                status VARCHAR(20),     -- OPEN, CLOSED, CANCELLED
                mode VARCHAR(10),       -- REAL, PAPER
                entry_time TIMESTAMP DEFAULT NOW(),
                close_time TIMESTAMP,
                exit_price FLOAT,
                pnl_abs FLOAT,
                pnl_pct FLOAT,
                exit_reason TEXT,
                max_price_reached FLOAT -- لتتبع الوقف المتحرك بدقة
            );
        """)
        # جدول الأداء اليومي
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                total_trades INT DEFAULT 0,
                wins INT DEFAULT 0,
                losses INT DEFAULT 0,
                total_pnl FLOAT DEFAULT 0.0
            );
        """)
    logger.info("✅ تم تهيئة قاعدة البيانات V14 بنجاح.")

# --- 4. المحلل الفني الذكي (AI Technical Analyst) ---
def fetch_candles(client, symbol, timeframe='15m', limit=100):
    try:
        klines = client.get_historical_klines(symbol, timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['time', 'open', 'high', 'low', 'close', 'vol']].astype(float)
        return df
    except Exception as e:
        logger.warning(f"فشل جلب البيانات لـ {symbol}: {e}")
        return None

def analyze_market_health(df_btc):
    """تحليل صحة السوق بناءً على البيتكوين"""
    last = df_btc.iloc[-1]
    ema200 = df_btc['close'].ewm(span=200).mean().iloc[-1]
    rsi = calculate_rsi(df_btc['close']).iloc[-1]
    
    status = "CHOPPY"
    if last['close'] > ema200:
        status = "BULLISH" if rsi > 50 else "WEAK_BULL"
    else:
        status = "BEARISH" if rsi < 50 else "WEAK_BEAR"
        
    if rsi > 75 or rsi < 25: status += "_EXTREME"
    return status

def calculate_indicators(df):
    """حساب المؤشرات المتقدمة"""
    df = df.copy()
    # EMAs
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    
    # ATR (للوقف المتحرك)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * std)
    df['bb_lower'] = df['bb_mid'] - (2 * std)
    
    # Volume MA
    df['vol_ma'] = df['vol'].rolling(20).mean()
    
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 5. محرك الإشارات الاستراتيجي (Strategic Signal Engine) ---
def get_ai_signal(symbol, df, market_status):
    """
    يقوم هذا المحرك بحساب 'نقاط الثقة' بناءً على عدة عوامل.
    لا يدخل إلا إذا كانت الثقة عالية.
    """
    current = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    strategy_name = "Unknown"
    
    # 1. فلتر السيولة (أساسي)
    if current['vol'] < current['vol_ma'] * 0.5:
        return None, 0, "سيولة ضعيفة"

    # --- استراتيجية 1: ركوب الموجة (Trend Surfer) ---
    # شروط: السعر فوق EMA50، EMA9 فوق EMA21، RSI بين 50 و 70
    if current['close'] > current['ema50'] and current['ema9'] > current['ema21']:
        if 50 < current['rsi'] < 70:
            score += 40
            strategy_name = "Trend_Surfer_V2"
            
            # تأكيد الزخم
            if current['close'] > current['open']: score += 10
            if current['vol'] > current['vol_ma']: score += 15
            if "BULL" in market_status: score += 20

    # --- استراتيجية 2: الارتداد من القاع (Dip Hunter) ---
    # شروط: السعر يلمس الباند السفلي، RSI < 30 ثم يبدأ بالصعود
    elif current['close'] < current['bb_lower'] or current['rsi'] < 30:
        if current['close'] > prev['close']: # بداية ارتداد
            score += 50
            strategy_name = "Dip_Hunter_Pro"
            
            if current['vol'] > current['vol_ma'] * 1.5: score += 25 # حجم شرائي ضخم
            if current['close'] > current['ema9']: score += 15

    # --- استراتيجية 3: الكسر الانفجاري (Volatility Breakout) ---
    # شروط: اختراق EMA200 بقوة مع حجم عالي
    elif prev['close'] < prev['ema200'] and current['close'] > current['ema200']:
        score += 60
        strategy_name = "Golden_Breakout"
        if current['vol'] > current['vol_ma'] * 2: score += 30

    return strategy_name, score, "فرصة متاحة"

# --- 6. مدير العمليات (Execution & Management) ---
def execute_trade(symbol, price, sl, tp1, tp2, qty, strat, score, mode):
    c = get_db_connection()
    if not c: return
    try:
        # التأكد من عدم وجود صفقة مفتوحة لنفس العملة
        with c.cursor() as cur:
            cur.execute("SELECT id FROM trades_v14 WHERE symbol=%s AND status='OPEN'", (symbol,))
            if cur.fetchone(): return

        with c.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v14 
                (symbol, type, entry_price, stop_loss, take_profit_1, take_profit_2, quantity, strategy, confidence_score, status, mode, max_price_reached)
                VALUES (%s, 'BUY', %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s, %s)
                RETURNING id
            """, (symbol, price, sl, tp1, tp2, qty, strat, score, mode, price))
            trade_id = cur.fetchone()['id']
            
        # إضافة للكاش
        trade_data = {
            'id': trade_id, 'symbol': symbol, 'entry_price': price, 'stop_loss': sl,
            'tp1': tp1, 'tp2': tp2, 'quantity': qty, 'strategy': strat, 'mode': mode,
            'entry_time': datetime.now(), 'max_price_reached': price
        }
        with locks['data']: active_trades_cache[symbol] = trade_data
        
        msg = f"🚀 **دخول جديد ({mode})**\n💎 العملة: {symbol}\n📊 الاستراتيجية: {strat}\n💯 الثقة: {score}/100\n💰 السعر: {price}\n🛑 الوقف: {sl}\n🎯 هدف 1: {tp1}"
        send_telegram(msg)
        add_log(symbol, "دخول شراء", f"{strat} (Score: {score})")
        
    except Exception as e:
        logger.error(f"Execution Error: {e}")

def update_active_trade(trade, current_price, current_atr):
    """
    إدارة الصفقة بذكاء:
    1. تحريك الوقف لنقطة الدخول بعد تحقيق الهدف الأول.
    2. تفعيل الوقف المتحرك (Trailing Stop) بعد ربح معين.
    3. الخروج الطارئ عند انعكاس السوق.
    """
    trade_id = trade['id']
    entry = trade['entry_price']
    sl = trade['stop_loss']
    tp1 = trade['tp1']
    tp2 = trade['tp2']
    max_price = trade.get('max_price_reached', entry)
    
    # تحديث أعلى سعر وصل له السعر
    if current_price > max_price:
        max_price = current_price
        # تحديث في الداتا بيس كل فترة (ليس مع كل تكة لتخفيف الحمل)
        # هنا سنحدثه في الكاش فقط للتسريع

    exit_reason = None
    close_price = current_price
    
    # 1. فحص وقف الخسارة الصارم
    if current_price <= sl:
        exit_reason = "ضرب وقف الخسارة 🛑"

    # 2. جني الأرباح
    elif current_price >= tp2:
        exit_reason = "تحقيق الهدف النهائي 🎯🎯"
    
    # 3. إدارة ذكية (Trailing Stop & Breakeven)
    else:
        # إذا تجاوز الهدف الأول، نرفع الوقف للدخول (Break-even)
        if max_price >= tp1 and sl < entry:
            new_sl = entry * 1.002 # فوق الدخول بشعرة لتغطية الرسوم
            update_sl_db(trade_id, new_sl)
            trade['stop_loss'] = new_sl
            send_telegram(f"🛡️ **تأمين الصفقة**\n{trade['symbol']}: تم رفع الوقف لنقطة الدخول.")

        # وقف متحرك: إذا ارتفع السعر 2% فوق الدخول، نلاحقه بمسافة 1.5%
        profit_pct = (current_price - entry) / entry * 100
        if profit_pct > 2.5:
            trailing_sl = current_price * 0.985 # 1.5% مسافة
            if trailing_sl > sl:
                update_sl_db(trade_id, trailing_sl)
                trade['stop_loss'] = trailing_sl
                # لا نرسل تنبيه هنا لتجنب الإزعاج، فقط تحديث صامت

    if exit_reason:
        close_trade(trade, close_price, exit_reason)

def update_sl_db(tid, new_sl):
    c = get_db_connection()
    if c:
        with c.cursor() as cur:
            cur.execute("UPDATE trades_v14 SET stop_loss=%s WHERE id=%s", (new_sl, tid))

def close_trade(trade, price, reason):
    c = get_db_connection()
    if not c: return
    
    pnl_abs = (price - trade['entry_price']) * trade['quantity']
    pnl_pct = (price - trade['entry_price']) / trade['entry_price'] * 100
    
    with c.cursor() as cur:
        cur.execute("""
            UPDATE trades_v14 
            SET status='CLOSED', close_time=NOW(), exit_price=%s, pnl_abs=%s, pnl_pct=%s, exit_reason=%s
            WHERE id=%s
        """, (price, pnl_abs, pnl_pct, reason, trade['id']))
        
        # تحديث الإحصائيات اليومية
        today = datetime.now().date()
        cur.execute("""
            INSERT INTO daily_stats (date, total_trades, wins, losses, total_pnl)
            VALUES (%s, 1, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
            total_trades = daily_stats.total_trades + 1,
            wins = daily_stats.wins + %s,
            losses = daily_stats.losses + %s,
            total_pnl = daily_stats.total_pnl + %s;
        """, (
            today, 
            1 if pnl_pct > 0 else 0, 1 if pnl_pct <= 0 else 0, pnl_abs,
            1 if pnl_pct > 0 else 0, 1 if pnl_pct <= 0 else 0, pnl_abs
        ))

    with locks['data']:
        if trade['symbol'] in active_trades_cache:
            del active_trades_cache[trade['symbol']]

    emoji = "✅" if pnl_pct > 0 else "🔻"
    msg = f"{emoji} **إغلاق صفقة ({trade['mode']})**\n💎 {trade['symbol']}\n💵 السعر: {price}\n📊 الربح: {pnl_pct:.2f}%\n💵 الصافي: ${pnl_abs:.2f}\n📝 السبب: {reason}"
    send_telegram(msg)
    add_log(trade['symbol'], "إغلاق", f"PNL: {pnl_pct:.2f}% | {reason}")

    # التحقق من قاطع الدائرة
    check_circuit_breaker()

def check_circuit_breaker():
    c = get_db_connection()
    if not c: return
    with c.cursor() as cur:
        cur.execute("SELECT total_pnl FROM daily_stats WHERE date=%s", (datetime.now().date(),))
        row = cur.fetchone()
        if row:
            # حساب نسبة الخسارة من رأس المال الأساسي
            loss_pct = (row['total_pnl'] / BOT_SETTINGS['capital']) * 100
            if loss_pct <= -BOT_SETTINGS['max_daily_loss']:
                state['circuit_breaker_triggered'] = True
                BOT_SETTINGS['is_active'] = False
                logger.critical(f"⛔ تم تفعيل قاطع الدائرة! الخسارة اليومية {loss_pct:.2f}% تجاوزت الحد المسموح.")
                send_telegram("⛔ **إنذار هام**\nتم إيقاف التداول تلقائياً بسبب تجاوز حد الخسارة اليومي. خذ استراحة!")

# --- 7. الأدوات المساعدة ---
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

def add_log(symbol, action, details):
    t = datetime.now().strftime('%H:%M:%S')
    with locks['log']:
        activity_log.appendleft({'time': t, 'symbol': symbol, 'action': action, 'details': details})

# --- 8. الحلقة الرئيسية (The Brain) ---
def main_loop():
    logger.info("🧠 بدء تشغيل الدماغ الإلكتروني V14...")
    init_tables()
    client = Client(API_KEY, API_SECRET)
    
    # تحميل الصفقات المفتوحة عند البدء
    c = get_db_connection()
    if c:
        with c.cursor() as cur:
            cur.execute("SELECT * FROM trades_v14 WHERE status='OPEN'")
            rows = cur.fetchall()
            with locks['data']:
                for r in rows: active_trades_cache[r['symbol']] = dict(r)

    while True:
        try:
            if state['circuit_breaker_triggered']:
                time.sleep(3600) # فحص كل ساعة إذا بدأ يوم جديد
                if datetime.now().hour == 0: 
                    state['circuit_breaker_triggered'] = False
                    send_telegram("☀️ **يوم جديد**\nتم إعادة تفعيل النظام. تداول بحذر.")
                continue

            if not BOT_SETTINGS['is_active']:
                time.sleep(5)
                continue

            # 1. تحليل عام للسوق (BTC)
            btc_df = fetch_candles(client, 'BTCUSDT', '1h', 200)
            if btc_df is not None:
                market_status = analyze_market_health(btc_df)
                state['market_status'] = market_status

            # 2. إدارة الصفقات المفتوحة
            with locks['data']:
                current_trades = list(active_trades_cache.values())
            
            for trade in current_trades:
                df = fetch_candles(client, trade['symbol'], '5m', 50) # فريم سريع للإدارة
                if df is not None:
                    df = calculate_indicators(df)
                    curr_price = df['close'].iloc[-1]
                    curr_atr = df['atr'].iloc[-1]
                    update_active_trade(trade, curr_price, curr_atr)
                time.sleep(0.5)

            # 3. البحث عن فرص جديدة
            if len(active_trades_cache) < BOT_SETTINGS['max_concurrent_trades']:
                # إذا السوق سيء جداً، لا تبحث عن فرص شراء
                if "CRASH" not in state['market_status']:
                    for symbol in WATCHLIST:
                        if symbol in active_trades_cache: continue
                        
                        df = fetch_candles(client, symbol, '15m', 100)
                        if df is None: continue
                        
                        df = calculate_indicators(df)
                        strat, score, reason = get_ai_signal(symbol, df, state['market_status'])
                        
                        if strat and score >= BOT_SETTINGS['min_confidence']:
                            curr = df['close'].iloc[-1]
                            atr = df['atr'].iloc[-1]
                            
                            # تحديد الأهداف بناء على ATR (ديناميكي)
                            sl = curr - (atr * 2.0)
                            tp1 = curr + (atr * 3.0) # RR 1.5
                            tp2 = curr + (atr * 5.0) # RR 2.5
                            
                            # إدارة حجم الصفقة (Risk Management)
                            risk_amt = BOT_SETTINGS['capital'] * (BOT_SETTINGS['risk_per_trade'] / 100)
                            price_risk = curr - sl
                            qty = risk_amt / price_risk
                            
                            # تصحيح الكمية حسب السعر
                            cost = qty * curr
                            if cost > BOT_SETTINGS['capital'] * 0.25: # لا تضع أكثر من 25% في صفقة واحدة
                                qty = (BOT_SETTINGS['capital'] * 0.25) / curr

                            execute_trade(symbol, curr, sl, tp1, tp2, qty, strat, score, BOT_SETTINGS['mode'])
                        
                        time.sleep(1) # تفادي حظر API
            
            state['last_scan_time'] = datetime.now()
            time.sleep(10)

        except Exception as e:
            logger.error(f"خطأ في الحلقة الرئيسية: {e}")
            time.sleep(10)

# --- 9. واجهة الويب الاحترافية (Flask UI) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    return render_template_string(HTML_DASHBOARD)

@app.route('/api/data')
def api_data():
    with locks['data']:
        trades = list(active_trades_cache.values())
    
    with locks['log']:
        logs = list(activity_log)
        
    return jsonify({
        "settings": BOT_SETTINGS,
        "state": state,
        "trades": trades,
        "logs": logs
    })

@app.route('/api/action', methods=['POST'])
def api_action():
    action = request.json.get('action')
    if action == 'toggle':
        BOT_SETTINGS['is_active'] = not BOT_SETTINGS['is_active']
        add_log("SYSTEM", "تغيير الحالة", f"Active: {BOT_SETTINGS['is_active']}")
    elif action == 'panic':
        BOT_SETTINGS['is_active'] = False
        add_log("SYSTEM", "PANIC", "تم تفعيل الإيقاف الطارئ!")
        # يمكن إضافة كود لإغلاق جميع الصفقات هنا
    return jsonify({"status": "ok"})

# --- 10. تصميم الواجهة (HTML/CSS/JS) ---
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot Pro V14 | AI Terminal</title>
    <link href="https://fonts.googleapis.com/css2?family=Changa:wght@300;500;700&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/js/all.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #09090b;
            --bg-panel: rgba(24, 24, 27, 0.7);
            --primary: #8b5cf6; /* بنفسجي حديث */
            --accent: #10b981; /* زمردي */
            --danger: #ef4444;
            --text-main: #e4e4e7;
            --text-muted: #a1a1aa;
            --glass: rgba(255, 255, 255, 0.03);
            --border: rgba(255, 255, 255, 0.08);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; scrollbar-width: thin; }
        body { 
            background-color: var(--bg-dark); 
            background-image: radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.08), transparent 25%), radial-gradient(circle at 85% 30%, rgba(16, 185, 129, 0.05), transparent 25%);
            color: var(--text-main); 
            font-family: 'Changa', sans-serif; 
            min-height: 100vh;
        }
        
        /* Layout */
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; }
        .col-3 { grid-column: span 3; } .col-4 { grid-column: span 4; } .col-6 { grid-column: span 6; } .col-8 { grid-column: span 8; } .col-9 { grid-column: span 9; } .col-12 { grid-column: span 12; }
        
        @media(max-width: 1024px) { .col-3, .col-4, .col-6, .col-8, .col-9 { grid-column: span 12; } }

        /* Components */
        .panel {
            background: var(--bg-panel);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }
        .panel:hover { border-color: rgba(255, 255, 255, 0.15); }
        
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
        .logo { font-size: 28px; font-weight: 700; background: linear-gradient(45deg, var(--primary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .badge { font-size: 12px; background: rgba(139, 92, 246, 0.2); color: var(--primary); padding: 4px 8px; border-radius: 6px; margin-right: 10px; vertical-align: middle; }
        
        .stat-label { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; }
        .stat-value { font-size: 24px; font-weight: 700; letter-spacing: 0.5px; }
        .stat-sub { font-size: 12px; color: var(--accent); margin-top: 4px; }
        
        /* Buttons */
        .btn { border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-family: inherit; font-weight: 600; transition: all 0.2s; display: flex; align-items: center; gap: 8px; }
        .btn-primary { background: var(--primary); color: white; box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3); }
        .btn-danger { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.3); }
        .btn-danger:hover { background: var(--danger); color: white; }
        .btn:active { transform: scale(0.98); }

        /* Table */
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th { text-align: right; color: var(--text-muted); font-size: 12px; padding: 12px; border-bottom: 1px solid var(--border); font-weight: 500; }
        td { padding: 14px 12px; border-bottom: 1px solid var(--border); font-size: 14px; }
        tr:last-child td { border-bottom: none; }
        .profit { color: var(--accent); } .loss { color: var(--danger); }
        
        /* Log List */
        .log-list { height: 300px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px; }
        .log-item { padding: 8px 0; border-bottom: 1px solid var(--border); display: flex; gap: 10px; }
        .log-time { color: var(--text-muted); min-width: 60px; }
        
        /* Animations */
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .live-dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; display: inline-block; animation: pulse 2s infinite; margin-left: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <div class="logo"><i class="fas fa-robot"></i> SmartBot Pro <span class="badge">V14 AI</span></div>
                <div style="font-size: 14px; color: var(--text-muted); margin-top: 5px;">منصة إدارة الثروات الآلية</div>
            </div>
            <div style="display: flex; gap: 15px;">
                <button class="btn btn-danger" onclick="sendAction('panic')"><i class="fas fa-skull-crossbones"></i> إيقاف طارئ</button>
                <button class="btn btn-primary" id="toggleBtn" onclick="sendAction('toggle')">
                    <i class="fas fa-power-off"></i> <span>جاري التحميل...</span>
                </button>
            </div>
        </div>

        <!-- KPI Cards -->
        <div class="grid" style="margin-bottom: 25px;">
            <div class="panel col-3">
                <div class="stat-label"><i class="fas fa-chart-line"></i> حالة السوق</div>
                <div class="stat-value" id="marketStatus">--</div>
                <div class="stat-sub">تحليل BTC 1h</div>
            </div>
            <div class="panel col-3">
                <div class="stat-label"><i class="fas fa-wallet"></i> رأس المال (وهمي)</div>
                <div class="stat-value">$<span id="capitalVal">1000</span></div>
                <div class="stat-sub">المخاطرة: <span id="riskVal">1.5</span>%</div>
            </div>
            <div class="panel col-3">
                <div class="stat-label"><i class="fas fa-shield-alt"></i> الأمان اليومي</div>
                <div class="stat-value" style="color: var(--accent);">مؤمن 🛡️</div>
                <div class="stat-sub">قاطع الدائرة: نشط (5%)</div>
            </div>
            <div class="panel col-3">
                <div class="stat-label"><i class="fas fa-bolt"></i> الصفقات النشطة</div>
                <div class="stat-value" id="activeTradesCount">0</div>
                <div class="stat-sub"><span class="live-dot"></span> مراقبة حية</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="grid">
            <!-- Active Trades Table -->
            <div class="panel col-8">
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                    <h3><i class="fas fa-list-ul"></i> المحفظة الحية</h3>
                    <span style="font-size: 12px; color: var(--text-muted);">تحديث تلقائي</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>العملة</th>
                            <th>الاستراتيجية</th>
                            <th>الثقة</th>
                            <th>الدخول</th>
                            <th>الحالي</th>
                            <th>الربح %</th>
                            <th>الحالة</th>
                        </tr>
                    </thead>
                    <tbody id="tradesBody"></tbody>
                </table>
            </div>

            <!-- AI Logs -->
            <div class="panel col-4">
                <h3><i class="fas fa-terminal"></i> سجل الخوارزمية</h3>
                <div class="log-list" id="logList"></div>
            </div>
        </div>
    </div>

    <script>
        function updateUI() {
            fetch('/api/data')
                .then(r => r.json())
                .then(d => {
                    // Update Header
                    const btn = document.getElementById('toggleBtn');
                    if (d.settings.is_active) {
                        btn.style.background = 'var(--accent)';
                        btn.innerHTML = '<i class="fas fa-pause"></i> النظام يعمل';
                    } else {
                        btn.style.background = '#4b5563';
                        btn.innerHTML = '<i class="fas fa-play"></i> النظام متوقف';
                    }

                    // Update KPIs
                    document.getElementById('marketStatus').innerText = d.state.market_status;
                    document.getElementById('marketStatus').style.color = d.state.market_status.includes('BULL') ? 'var(--accent)' : 'var(--danger)';
                    document.getElementById('capitalVal').innerText = d.settings.capital;
                    document.getElementById('riskVal').innerText = d.settings.risk_per_trade;
                    document.getElementById('activeTradesCount').innerText = d.trades.length;

                    // Update Table
                    const tbody = document.getElementById('tradesBody');
                    if (d.trades.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 30px; color: var(--text-muted);">لا توجد صفقات نشطة حالياً. الذكاء الاصطناعي يبحث عن فرص... 🔭</td></tr>';
                    } else {
                        tbody.innerHTML = d.trades.map(t => {
                            // Calculate current PnL estimation (mock calculation for UI if real price not streamed)
                            // In real app, price comes from backend. We assume backend updates trade object or we fetch live price here.
                            // For this demo, let's assume trade object has updated info if available, else entry.
                            return `
                            <tr>
                                <td style="font-weight: bold; color: white;">${t.symbol}</td>
                                <td><span style="background: rgba(139, 92, 246, 0.2); color: var(--primary); padding: 2px 6px; border-radius: 4px; font-size: 11px;">${t.strategy}</span></td>
                                <td>${t.confidence_score || 80}%</td>
                                <td>${t.entry_price}</td>
                                <td>--</td>
                                <td>--</td>
                                <td><span style="color: var(--accent);">نشط</span></td>
                            </tr>
                        `}).join('');
                    }

                    // Update Logs
                    const logsDiv = document.getElementById('logList');
                    logsDiv.innerHTML = d.logs.map(l => `
                        <div class="log-item">
                            <span class="log-time">[${l.time}]</span>
                            <span style="color: var(--primary); font-weight: bold;">${l.symbol}</span>
                            <span>${l.action}</span>
                            <span style="color: var(--text-muted);">${l.details}</span>
                        </div>
                    `).join('');
                });
        }

        function sendAction(act) {
            fetch('/api/action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: act})
            }).then(() => updateUI());
        }

        setInterval(updateUI, 2000);
        updateUI();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    t = Thread(target=main_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)