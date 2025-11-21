import time
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
from binance.client import Client
from flask import Flask, jsonify, render_template_string, request, redirect
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- 1. إعدادات النظام ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot_v11.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Ultimate')

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ Config Error: {e}")
    exit(1)

# --- 2. إعدادات التداول المتقدمة ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "trade_amount_usdt": 50.0,
    "max_open_trades": 5,             # زيادة عدد الصفقات المسموحة
    "stop_loss_atr_multiplier": 1.8,  
    "trailing_atr_multiplier": 2.5,
    "volume_filter_limit": 60,        # فحص أكبر 60 عملة
    "breakeven_trigger_pct": 1.2,     # تأمين الصفقة عند ربح 1.2%
    "max_hold_time_hours": 4          # أقصى مدة احتفاظ
}

LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
market_state = {
    "score": 50, "regime": "neutral", 
    "details": {"15m": 0, "1h": 0, "4h": 0},
    "trend_strength": "weak"
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=150)

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
                CREATE TABLE IF NOT EXISTS signals_v11 (
                    id SERIAL PRIMARY KEY, 
                    symbol TEXT NOT NULL, 
                    entry_price DOUBLE PRECISION, 
                    stop_loss DOUBLE PRECISION, 
                    target_price DOUBLE PRECISION,
                    quantity DOUBLE PRECISION, 
                    strategy_name TEXT, 
                    status TEXT DEFAULT 'open', 
                    mode TEXT,
                    entry_time TIMESTAMP DEFAULT NOW(),
                    closed_at TIMESTAMP, 
                    closing_price DOUBLE PRECISION, 
                    profit_pct DOUBLE PRECISION,
                    exit_reason TEXT, 
                    duration_minutes INTEGER,
                    last_analysis TEXT 
                );
            """)
        logger.info("✅ Database V11 Initialized.")
    except Exception as e: logger.error(f"DB Error: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 4. نظام التنبيهات العربي المتطور ---
def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    mode_txt = "تجريبي 📝" if payload.get('is_paper') else "حقيقي 💵"
    msg = ""

    if event == "BUY":
        msg = (
            f"🔵 *إشارة دخول جديدة | {payload['symbol']}*\n"
            f"ــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"استراتيجية: *{payload['strategy']}*\n"
            f"سعر الدخول: `{payload['price']}`\n"
            f"وقف الخسارة: `{payload['sl']}`\n"
            f"الهدف الأول: `{payload['tp']}`\n"
            f"حالة السوق: *{payload['regime']}*\n"
            f"الوضع: {mode_txt}"
        )
    elif event == "SELL":
        pnl = payload['profit']
        emoji = "✅ ربح" if pnl > 0 else "🔴 خسارة"
        msg = (
            f"{emoji} *إغلاق صفقة | {payload['symbol']}*\n"
            f"ــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"سعر الإغلاق: `{payload['price']}`\n"
            f"الصافي: `{pnl:.2f}%`\n"
            f"المدة: {payload['duration']} دقيقة\n"
            f"سبب الخروج: _{payload['reason']}_\n"
            f"الوضع: {mode_txt}"
        )
    elif event == "UPDATE":
        msg = (
            f"🛡️ *تحديث وقف الخسارة | {payload['symbol']}*\n"
            f"الوقف الجديد: `{payload['new_sl']}`\n"
            f"السبب: {payload['reason']}"
        )

    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except Exception as e: logger.error(f"Telegram Error: {e}")

# --- 5. التحليل الفني المتقدم ---
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
    # المتوسطات
    df['ema9'] = df['close'].ewm(span=9).mean()
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
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # ADX (مؤشر قوة الترند)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(14).mean()

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()

    return df.fillna(0)

# --- 6. محرك الاستراتيجيات الذكي ---
def get_strategy_signal(symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # === 1. فلتر السيولة المخفف (Smart Volume Filter) ===
    # خفضنا النسبة لـ 25% لقبول المزيد من العملات
    vol_threshold = last['vol_ma'] * 0.25
    
    # استثناء: إذا كان هناك زخم سعري قوي جداً، نقبل بسيولة أقل قليلاً
    is_high_momentum = last['rsi'] > 60 and last['macd'] > last['macd_signal']
    
    if last['volume'] < vol_threshold and not is_high_momentum:
        return None, "ضعف سيولة"

    # === 2. استراتيجيات السوق الصاعد (Bullish) ===
    if "bull" in regime:
        # A. Trend Sniper (تم تخفيف شرط ADX لـ 15 بدلاً من 20)
        if last['adx'] > 15: 
            if last['close'] > last['ema200'] and last['ema9'] > last['ema21']:
                # شرط الدخول: تقاطع الماكد أو اختراق RSI لـ 50
                if (last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']):
                    return "Trend_Sniper", "تقاطع إيجابي مع ترند قوي"

        # B. Quick Scalp (للسرعة)
        # دخول سريع إذا السعر فوق المتوسطات والـ RSI غير متشبع
        if last['close'] > last['ema50'] and last['rsi'] > 50 and last['rsi'] < 70:
            if last['high'] > prev['high']: # كسر شمعة سابقة
                return "Quick_Scalp", "زخم لحظي سريع"

    # === 3. استراتيجيات السوق العرضي (Sideways) ===
    elif "sideways" in regime:
        # C. Precision Reversion (ارتداد دقيق)
        if last['close'] < last['bb_lower'] and last['rsi'] < 35:
             return "Sniper_Reversion", "ارتداد من قاع القناة (تشبع بيعي)"

    # === 4. استراتيجيات السوق الهابط (Bearish) ===
    elif "bearish" in regime:
        # D. Dead Cat Bounce (حذر جداً)
        if last['rsi'] < 25: # تشبع بيعي حاد
            return "Deep_Dip", "صيد القاع (خطر)"

    # === 5. الجوكر (Golden Cross) ===
    # تعمل دائماً إذا تحقق التقاطع الذهبي لمتوسط 50 مع 200
    if last['ema50'] > last['ema200'] and prev['ema50'] <= prev['ema200']:
         return "Golden_Cross", "التقاطع الذهبي 50/200"

    return None, "لا توجد شروط مطابقة"

# --- 7. مدير الصفقات الدوري والمحلل (The Analyst) ---
def manage_trade(symbol, signal, df, regime):
    last = df.iloc[-1]
    curr = float(last['close'])
    entry = float(signal['entry_price'])
    sl = float(signal['stop_loss'])
    atr = float(last['atr'])
    
    profit_pct = (curr - entry) / entry * 100
    duration = (datetime.now() - signal['entry_time']).total_seconds() / 3600 # بالساعات

    # تحليل صحة الصفقة للتسجيل
    health_status = "مستقر"
    if profit_pct > 1: health_status = "ربح متنامي 🟢"
    elif profit_pct < -1: health_status = "خسارة 🔴"
    elif duration > 2: health_status = "ركود زمني ⚠️"

    # 1. تأمين الصفقة (Breakeven)
    if profit_pct >= BOT_SETTINGS['breakeven_trigger_pct'] and sl < entry:
        new_sl = entry * 1.0015 # فوق الدخول قليلاً لتغطية العمولة
        return "UPDATE_SL", new_sl, "تأمين الصفقة (بدون مخاطرة)", health_status

    # 2. وقف الخسارة الذكي (Smart Trailing)
    # كلما زاد الربح، اقترب الوقف أكثر
    if profit_pct >= 2.5:
        new_sl = curr - (atr * 1.5) # وقف ضيق جداً لحماية الأرباح الكبيرة
        if new_sl > sl:
            return "UPDATE_SL", new_sl, "حجز أرباح متقدم (وقف ضيق)", health_status
    elif profit_pct >= 1.5:
        new_sl = curr - (atr * 2.0) # وقف متوسط
        if new_sl > sl:
            return "UPDATE_SL", new_sl, "حجز أرباح (ATR)", health_status

    # 3. الخروج الزمني (Time Decay)
    # إذا مرت 4 ساعات والربح أقل من 0.5%، نخرج
    if duration > BOT_SETTINGS['max_hold_time_hours'] and profit_pct < 0.5:
        return "CLOSE_NOW", curr, "انتهاء الوقت (ركود السعر)", health_status

    # 4. الخروج عند انعكاس الترند (Panic Exit)
    if regime == "bearish" and curr < last['ema50'] and profit_pct < -1.5:
        return "CLOSE_NOW", curr, "انعكاس الترند العام (حماية)", health_status
    
    # 5. الخروج عند التشبع الشرائي القوي (فقط لصفقات الارتداد)
    if "Reversion" in signal['strategy_name'] and last['rsi'] > 68:
         return "CLOSE_NOW", curr, "وصول للهدف (RSI)", health_status

    return "HOLD", 0, "", health_status

# --- 8. تحليل هيكل السوق ---
def analyze_market_structure(client):
    global market_state
    # أوزان الفريمات: 4 ساعات هو الأهم للاتجاه العام
    tf_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.5}
    total_score = 0
    
    for sym in LEADING_SYMBOLS:
        sym_score = 0
        for tf, weight in tf_weights.items():
            df = fetch_data(client, sym, tf, 60)
            if df is None: continue
            
            close = df['close'].iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            
            # نقاط إيجابية
            if close > ema200: sym_score += 100 * weight
            
        total_score += sym_score
    
    final_avg_score = total_score / len(LEADING_SYMBOLS)
    
    if final_avg_score >= 70: regime = "bull_strong"
    elif final_avg_score >= 50: regime = "bull_weak"
    elif final_avg_score <= 30: regime = "bearish"
    else: regime = "sideways"
    
    with locks['market']:
        market_state = {
            "score": final_avg_score, 
            "regime": regime,
            "last_update": datetime.now()
        }
    
    logger.info(f"📊 تحليل السوق: {final_avg_score:.1f} | الحالة: {regime}")

# --- 9. حلقات التنفيذ الرئيسية ---
def execute_trade(client, symbol, side, qty):
    if BOT_SETTINGS['paper_trading_mode']: return True
    try:
        # هنا يتم وضع كود التنفيذ الحقيقي مع باينانس
        # client.create_order(...) 
        return True
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        return False

def bot_loop():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 SmartBot Ultimate Engine Started")
    
    try:
        with open('crypto_list.txt') as f:
            symbols = [l.strip().upper() for l in f if l.strip()]
            symbols = [s if s.endswith('USDT') else s+'USDT' for s in symbols]
    except: symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']

    while True:
        try:
            # تحديث الإعدادات
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                limit = BOT_SETTINGS['volume_filter_limit']
                amt = BOT_SETTINGS['trade_amount_usdt']
                max_t = BOT_SETTINGS['max_open_trades']
                paper = BOT_SETTINGS['paper_trading_mode']
            
            if not enabled:
                time.sleep(10)
                continue

            # 1. تحليل السوق الدوري
            analyze_market_structure(client)
            with locks['market']: regime = market_state['regime']

            # 2. مراقبة وتحليل الصفقات المفتوحة
            with locks['signals']: open_trades = list(open_signals_cache.values())
            
            for trade in open_trades:
                sym = trade['symbol']
                df = fetch_data(client, sym, '5m', 60)
                if df is None: continue
                df = add_indicators(df)
                
                curr_price = df['close'].iloc[-1]
                with locks['prices']: live_prices[sym] = curr_price
                
                exit_reason = None
                if curr_price <= trade['stop_loss']: exit_reason = "ضرب وقف الخسارة 🛑"
                elif curr_price >= trade['target_price']: exit_reason = "تحقيق الهدف 🎯"
                
                if not exit_reason:
                    action, val, note, health = manage_trade(sym, trade, df, regime)
                    
                    # تحديث التحليل في الكاش للعرض
                    open_signals_cache[sym]['last_analysis'] = f"{health} - {datetime.now().strftime('%H:%M')}"
                    
                    if action == "UPDATE_SL":
                        open_signals_cache[sym]['stop_loss'] = val
                        check_db()
                        with conn.cursor() as cur: 
                            cur.execute("UPDATE signals_v11 SET stop_loss=%s, last_analysis=%s WHERE id=%s", (val, health, trade['id']))
                        send_telegram("UPDATE", {"symbol": sym, "new_sl": val, "reason": note})
                    elif action == "CLOSE_NOW":
                        exit_reason = f"خروج ذكي: {note}"

                if exit_reason:
                    execute_trade(client, sym, 'SELL', trade['quantity'])
                    close_trade_system(sym, curr_price, exit_reason, paper)

            # 3. البحث عن فرص جديدة
            if len(open_signals_cache) < max_t:
                tickers = client.get_ticker()
                valid_tickers = [t for t in tickers if t['symbol'] in symbols]
                # ترتيب حسب السيولة
                valid_tickers.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                top_coins = valid_tickers[:limit]
                
                for t in top_coins:
                    if len(open_signals_cache) >= max_t: break
                    sym = t['symbol']
                    if sym in open_signals_cache: continue
                    
                    # الفحص على فريم 15 دقيقة للدخول
                    df = fetch_data(client, sym, '15m', 100) 
                    if df is None: continue
                    df = add_indicators(df)
                    
                    strat, reason = get_strategy_signal(sym, df, regime)
                    
                    if strat:
                        curr = df['close'].iloc[-1]
                        atr = df['atr'].iloc[-1]
                        
                        # حساب الأهداف
                        sl = curr - (atr * BOT_SETTINGS['stop_loss_atr_multiplier'])
                        tp = curr + (atr * 3.0) 
                        qty = amt / curr
                        
                        if execute_trade(client, sym, 'BUY', qty):
                            record_new_trade(sym, curr, sl, tp, qty, strat, paper, regime)
                            logger.info(f"✅ ENTRY: {sym} | {strat}")
                    else:
                         # تسجيل الرفض بشكل أقل تكراراً لتجنب الازدحام
                         if random.random() < 0.05: 
                            with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'تخطي', 'r': reason})
                    
                    time.sleep(0.3) # تأخير بسيط لتجنب الحظر

            time.sleep(20)

        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            time.sleep(10)

# --- 10. أدوات النظام وقاعدة البيانات ---
def record_new_trade(symbol, price, sl, tp, qty, strat, is_paper, regime):
    check_db()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals_v11 
                (symbol, entry_price, stop_loss, target_price, quantity, strategy_name, status, mode, entry_time, last_analysis)
                VALUES (%s, %s, %s, %s, %s, %s, 'open', %s, NOW(), 'جديد')
                RETURNING id
            """, (symbol, price, sl, tp, qty, strat, mode))
            db_id = cur.fetchone()['id']
        
        trade_obj = {
            'id': db_id, 'symbol': symbol, 'entry_price': price, 
            'stop_loss': sl, 'target_price': tp, 'quantity': qty, 
            'strategy_name': strat, 'entry_time': datetime.now(),
            'is_paper': is_paper, 'last_analysis': 'جديد'
        }
        with locks['signals']: open_signals_cache[symbol] = trade_obj
        
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'دخول', 'r': strat})
        
        send_telegram("BUY", {'symbol': symbol, 'strategy': strat, 'price': price, 'sl': sl, 'tp': tp, 'regime': regime, 'is_paper': is_paper})
        
    except Exception as e: logger.error(f"Record Trade Error: {e}")

def close_trade_system(symbol, price, reason, is_paper):
    check_db()
    try:
        trade = None
        with locks['signals']:
            if symbol in open_signals_cache:
                trade = open_signals_cache[symbol]
                del open_signals_cache[symbol]
        
        if not trade: return

        profit_pct = ((price - trade['entry_price']) / trade['entry_price']) * 100
        duration = int((datetime.now() - trade['entry_time']).total_seconds() / 60)
        
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals_v11 
                SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, exit_reason=%s, duration_minutes=%s
                WHERE id=%s
            """, (price, profit_pct, reason, duration, trade['id']))
            
        send_telegram("SELL", {'symbol': symbol, 'price': price, 'profit': profit_pct, 'reason': reason, 'duration': duration, 'is_paper': is_paper})
        
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'إغلاق', 'r': f"{profit_pct:.2f}%"})

    except Exception as e: logger.error(f"Close Trade Error: {e}")

# --- 11. واجهة التحكم (Web Dashboard) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/data')
def api_data():
    with locks['market']: m = market_state.copy()
    with locks['signals']: s = [{k: v for k, v in t.items() if k != 'entry_time'} for t in open_signals_cache.values()]
    with locks['prices']: p = live_prices.copy()
    with locks['settings']: st = {'enabled': BOT_SETTINGS['is_trading_enabled'], 'paper': BOT_SETTINGS['paper_trading_mode']}
    with locks['logs']: l = list(scan_logs)
    
    hist = []
    stats = {'wins': 0, 'losses': 0, 'total_pnl': 0}
    try:
        check_db()
        with conn.cursor() as cur:
            cur.execute("SELECT closed_at, profit_pct FROM signals_v11 WHERE status='closed' ORDER BY closed_at ASC")
            rows = cur.fetchall()
            cum = 0
            for r in rows:
                cum += r['profit_pct']
                if r['profit_pct'] > 0: stats['wins'] += 1
                else: stats['losses'] += 1
                hist.append({'date': r['closed_at'].strftime('%d %H:%M'), 'pnl': cum})
            stats['total_pnl'] = cum
    except: pass
    
    return jsonify({"market": m, "signals": s, "prices": p, "settings": st, "logs": l, "history": hist, "stats": stats})

@app.route('/api/toggle', methods=['POST'])
def toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify({"status": "ok"})

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>SmartBot Ultimate V11</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --green: #10b981; --red: #ef4444; --accent: #8b5cf6; }
        body { background: var(--bg); color: var(--text); font-family: 'Cairo', sans-serif; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border-bottom: 1px solid #334155; padding-bottom: 15px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        .btn { padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; border: none; font-family: inherit; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 10px; }
        th, td { padding: 12px; text-align: right; border-bottom: 1px solid #334155; }
        th { color: #94a3b8; font-weight: normal; }
        .val-green { color: var(--green); } .val-red { color: var(--red); }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🚀 SmartBot <span style="color:var(--accent)">Ultimate V11</span></h1>
            <small style="color:#94a3b8">نظام إدارة المحفظة الذكي مع تعريب كامل</small>
        </div>
        <div>
            <button id="toggleBtn" class="btn" onclick="toggleBot()">جاري التحميل...</button>
        </div>
    </div>

    <div class="grid" style="margin-bottom: 20px;">
        <div class="card">
            <h3>📊 هيكل السوق</h3>
            <h1 id="regime" style="color:var(--accent)">--</h1>
            <p>مؤشر القوة: <b id="score">0</b>/100</p>
        </div>
        <div class="card">
            <h3>💰 الأداء العام</h3>
            <div style="display:flex; justify-content:space-between; margin-bottom:10px">
                <span>✅ رابحة: <b id="wins" style="color:var(--green)">0</b></span>
                <span>🔴 خاسرة: <b id="losses" style="color:var(--red)">0</b></span>
            </div>
            <h2 id="totalPnl" dir="ltr">0.00%</h2>
        </div>
        <div class="card">
            <h3>📈 منحنى النمو (Equity)</h3>
            <div style="height:100px"><canvas id="pnlChart"></canvas></div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>⚡ الصفقات النشطة (تحليل دوري)</h3>
            <table>
                <thead><tr><th>العملة</th><th>الاستراتيجية</th><th>السعر</th><th>الربح %</th><th>الوضع الصحي</th></tr></thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>
        <div class="card">
            <h3>📝 سجل الفحص المباشر</h3>
            <div style="height:300px; overflow-y:auto;">
                <table>
                    <thead><tr><th>الوقت</th><th>العملة</th><th>الحدث</th><th>التفاصيل</th></tr></thead>
                    <tbody id="logsBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let chart;
        function initChart() {
            const ctx = document.getElementById('pnlChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'PNL %', data: [], borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.1)', fill: true, tension: 0.4 }] },
                options: { plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } }, maintainAspectRatio: false }
            });
        }

        async function update() {
            try {
                const res = await fetch('/api/data');
                const d = await res.json();

                const btn = document.getElementById('toggleBtn');
                if (d.settings.enabled) {
                    btn.innerText = "إيقاف البوت 🛑";
                    btn.style.background = "var(--red)";
                    btn.style.color = "white";
                } else {
                    btn.innerText = "تشغيل البوت 🚀";
                    btn.style.background = "var(--green)";
                    btn.style.color = "white";
                }

                document.getElementById('regime').innerText = d.market.regime === 'bull_strong' ? 'صاعد قوي 🐂' : 
                                                            d.market.regime === 'bull_weak' ? 'صاعد ضعيف 📈' :
                                                            d.market.regime === 'bearish' ? 'هابط 🐻' : 'عرضي 🦀';
                document.getElementById('score').innerText = d.market.score.toFixed(1);

                document.getElementById('wins').innerText = d.stats.wins;
                document.getElementById('losses').innerText = d.stats.losses;
                const tpnl = document.getElementById('totalPnl');
                tpnl.innerText = d.stats.total_pnl.toFixed(2) + "%";
                tpnl.className = d.stats.total_pnl >= 0 ? "val-green" : "val-red";

                if(d.history.length > 0) {
                    chart.data.labels = d.history.map(h => h.date);
                    chart.data.datasets[0].data = d.history.map(h => h.pnl);
                    chart.update();
                }

                const tb = document.getElementById('tradesBody');
                tb.innerHTML = d.signals.length ? d.signals.map(s => {
                    const curr = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                    return `<tr>
                        <td><b>${s.symbol}</b></td>
                        <td><small>${s.strategy_name}</small></td>
                        <td>${curr}</td>
                        <td class="${pnl>=0?'val-green':'val-red'}">${pnl.toFixed(2)}%</td>
                        <td><small>${s.last_analysis || 'جديد'}</small></td>
                    </tr>`;
                }).join('') : "<tr><td colspan='5' style='text-align:center'>لا توجد صفقات نشطة</td></tr>";

                document.getElementById('logsBody').innerHTML = d.logs.map(l => 
                    `<tr><td>${l.t}</td><td>${l.s}</td><td>${l.st}</td><td><small>${l.r}</small></td></tr>`
                ).join('');
            } catch(e) { console.log(e); }
        }

        async function toggleBot() { await fetch('/api/toggle', {method:'POST'}); update(); }
        
        initChart();
        setInterval(update, 2000);
        update();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    Thread(target=bot_loop, daemon=True).start()
    logger.info("🖥️ Web Dashboard running on port 5000")
    app.run(host='0.0.0.0', port=5000)