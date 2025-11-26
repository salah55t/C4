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
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- 1. إعدادات النظام ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('smart_bot_v14.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Arab_V14')

try:
    API_KEY = config('BINANCE_API_KEY', default='')
    API_SECRET = config('BINANCE_API_SECRET', default='')
    DB_URL = config('DATABASE_URL', default='')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.warning(f"⚠️ تحذير الإعدادات: {e}")

# --- 2. إعدادات البوت الافتراضية ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,
    "risk_per_trade_pct": 2.0,
    "max_open_trades": 5,
    "timeframe_analysis": "15m",
    "commission_rate": 0.1,    # عمولة المنصة 0.1%
    "min_score_entry": 60      # أقل درجة جودة للدخول
}

# حالة النظام
system_state = {
    "market_regime": "Neutral",
    "trend_strength": 0,
    "volatility_index": "Low",
    "last_update": None
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=200)

locks = {
    'signals': Lock(), 'prices': Lock(), 'market': Lock(), 
    'settings': Lock(), 'logs': Lock()
}

# --- 3. قاعدة البيانات ---
conn = None
def init_db():
    global conn
    if not DB_URL: return
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
                    commission_paid DOUBLE PRECISION DEFAULT 0.0,
                    exit_reason TEXT
                );
            """)
        logger.info("✅ قاعدة البيانات جاهزة (V14 Final).")
    except Exception as e: logger.error(f"DB Error: {e}")

def check_db():
    global conn
    if conn is None or (conn and conn.closed != 0): init_db()

# --- 4. التحليل الفني المتقدم (الخوارزميات الجديدة) ---
def fetch_data(client, symbol, interval, limit=100):
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except: return None

# أ) تحليل السيولة المتقدم
def advanced_volume_analysis(df, period=20):
    df = df.copy()
    vol_mean = df['volume'].rolling(period).mean()
    df['volume_ratio'] = df['volume'] / vol_mean.replace(0, 1)
    df['volume_spike'] = df['volume_ratio'] > 2.0
    
    # Accumulation/Distribution
    chl = (df['close'] - df['low']) - (df['high'] - df['close'])
    h_l = (df['high'] - df['low']).replace(0, 0.000001)
    df['adl'] = (chl / h_l) * df['volume']
    return df

def volume_confirmation(df, min_ratio=1.2):
    last = df.iloc[-1]
    return last['volume_ratio'] >= min_ratio or last['volume_spike']

# ب) المؤشرات القيادية والهيكل
def add_leading_indicators(df):
    # TR & ATR
    tr = np.maximum(df['high'] - df['low'], 
           np.maximum(abs(df['high'] - df['close'].shift(1)), 
                      abs(df['low'] - df['close'].shift(1))))
    df['tr'] = tr
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Vortex
    df['vm+'] = abs(df['high'] - df['low'].shift(1))
    df['vm-'] = abs(df['low'] - df['high'].shift(1))
    tr14 = df['tr'].rolling(14).sum().replace(0, 1)
    df['vortex+'] = df['vm+'].rolling(14).sum() / tr14
    df['vortex-'] = df['vm-'].rolling(14).sum() / tr14
    
    # Supertrend
    atr_mul = 3
    hl2 = (df['high'] + df['low']) / 2
    df['supertrend_upper'] = hl2 + (atr_mul * df['atr'])
    
    return df

def price_structure_analysis(df, lookback=50):
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    resistance = highs.rolling(5).max().iloc[-1]
    support = lows.rolling(5).min().iloc[-1]
    
    df['higher_high'] = df['high'] > df['high'].shift(1)
    uptrend = df['higher_high'].tail(3).sum() >= 2
    
    last = df['close'].iloc[-1]
    return {
        'resistance': resistance,
        'support': support,
        'trend': 'uptrend' if uptrend else 'ranging',
        'dist_res': (resistance - last) / last * 100,
        'dist_sup': (last - support) / last * 100
    }

# ج) دمج كل المؤشرات
def calculate_full_technical_indicators(df):
    df = df.copy()
    # المتوسطات القديمة (مهمة للوحة التحكم)
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # RSI & Stoch
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.0001)
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    
    min_r = df['rsi'].rolling(14).min()
    max_r = df['rsi'].rolling(14).max()
    df['stoch_k'] = ((df['rsi'] - min_r) / (max_r - min_r).replace(0, 0.001)) * 100
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # ADX (Standard)
    df = add_leading_indicators(df) # Adds ATR
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = df['low'].diff().clip(lower=0)
    tr_safe = df['atr'].replace(0, 1)
    df['adx'] = (100 * abs((plus_dm/tr_safe) - (minus_dm/tr_safe))).rolling(14).mean()
    
    # New Volume Logic
    df = advanced_volume_analysis(df)
    
    return df.fillna(0)

# د) نظام التنقيط وإدارة المخاطر
def weighted_signal_scoring(symbol, df, regime, struct):
    score = 0
    last = df.iloc[-1]
    
    # 1. تقني (40)
    if last['ema50'] > last['ema200']: score += 8
    if last['close'] > last['ema50']: score += 7
    if 30 < last['rsi'] < 70: score += 5
    if last['macd'] > last['macd_signal']: score += 5
    if 20 < last['stoch_k'] < 80: score += 5
    if last['adx'] > 25: score += 10
    
    # 2. سيولة (20)
    if volume_confirmation(df): score += 20
    
    # 3. هيكل (25)
    if struct['trend'] == 'uptrend': score += 10
    if struct['dist_res'] > 2: score += 10
    if struct['dist_sup'] < 3: score += 5
    
    # 4. سوق (15)
    if "Bull" in regime: score += 15
    elif "Ranging" in regime: score += 8
    
    grade = 'A' if score >= 80 else 'B' if score >= 65 else 'C' if score >= 50 else 'D'
    return {'total': score, 'grade': grade}

def adaptive_risk_management(df, regime, score):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    # مضاعفات ديناميكية
    if "High_Volatility" in regime:
        mul_sl, mul_tp, risk_f = 1.5, 2.5, 0.7
    elif "Bull_Trend" in regime:
        mul_sl, mul_tp, risk_f = 1.8, 3.0, 1.2
    else:
        mul_sl, mul_tp, risk_f = 2.0, 3.0, 1.0
        
    if score['grade'] == 'A': risk_f *= 1.2
    elif score['grade'] == 'D': risk_f *= 0.5
    
    sl = close - (atr * mul_sl)
    tp1 = close + (atr * mul_tp)
    tp2 = close + (atr * (mul_tp + 1.5))
    
    return sl, tp1, tp2, risk_f

def get_enhanced_signal(symbol, df, regime):
    struct = price_structure_analysis(df)
    score = weighted_signal_scoring(symbol, df, regime, struct)
    
    if score['total'] < BOT_SETTINGS['min_score_entry']:
        return None, None, f"نقاط ضعيفة: {score['total']}"
    
    last = df.iloc[-1]
    strat = None
    
    # المنطق المطور
    if score['total'] >= 75 and last['vortex+'] > last['vortex-']:
        strat = "Enhanced_Trend"
    elif struct['dist_sup'] < 1.5 and volume_confirmation(df):
        strat = "Enhanced_Reversion"
    elif struct['dist_res'] < 1.0 and last['adx'] > 25:
        strat = "Enhanced_Breakout"
        
    if strat:
        return strat, score, f"{strat} (Score: {score['total']})"
    return None, None, "لا يوجد تطابق"

# --- 5. المحرك الرئيسي ---
def analyze_market(client):
    try:
        # استخدام BTC لتحديد حالة السوق
        df = fetch_data(client, 'BTCUSDT', '4h', 100)
        if df is None: return
        df = calculate_full_technical_indicators(df)
        last = df.iloc[-1]
        
        adx = last['adx']
        trend = "Bull" if last['close'] > last['ema200'] else "Bear"
        
        regime = "Neutral"
        if trend == "Bull" and adx > 25: regime = "Bull_Trend_Strong"
        elif trend == "Bear" and adx > 25: regime = "Bear_Trend_Strong"
        elif (last['atr']/last['close'])*100 > 2.0: regime = "High_Volatility"
        else: regime = "Ranging"
        
        with locks['market']:
            system_state['market_regime'] = regime
            system_state['trend_strength'] = int(adx)
    except Exception as e: logger.error(f"Market Error: {e}")

def bot_engine():
    client = None
    while not client:
        try: client = Client(API_KEY, API_SECRET)
        except: time.sleep(5)
    
    logger.info("🚀 SmartBot Engine Started (Safe Mode)")
    
    while True:
        try:
            with locks['settings']:
                if not BOT_SETTINGS['is_trading_enabled']:
                    time.sleep(5)
                    continue
                paper = BOT_SETTINGS['paper_trading_mode']
                max_t = BOT_SETTINGS['max_open_trades']

            # 1. تحديث السوق
            analyze_market(client)
            with locks['market']: regime = system_state['market_regime']

            # 2. إدارة الصفقات المفتوحة
            with locks['signals']: trades = list(open_signals_cache.values())
            for t in trades:
                sym = t['symbol']
                df = fetch_data(client, sym, '5m', 60)
                if df is not None:
                    df = calculate_full_technical_indicators(df)
                    curr = df['close'].iloc[-1]
                    with locks['prices']: live_prices[sym] = curr
                    
                    # منطق الخروج
                    sl = t['stop_loss']
                    if curr <= sl:
                        close_trade(sym, curr, "ضرب وقف الخسارة 🛑", paper)
                    elif curr >= t['tp2'] and sl < t['tp1']:
                        update_sl(sym, t['tp1'], "تأمين هدف 1")
                    elif curr >= t['tp1'] and sl < t['entry_price']:
                        update_sl(sym, t['entry_price']*1.002, "تأمين دخول")
                
                time.sleep(1) # هام جداً لتجنب الحظر

            # 3. البحث (Rate Limited)
            if len(trades) < max_t:
                tickers = client.get_ticker()
                # نأخذ أعلى عملات في السيولة لتجنب العملات الميتة
                valid = [x for x in tickers if x['symbol'].endswith('USDT') and float(x['quoteVolume']) > 15000000]
                valid.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                
                # نفحص فقط 10 عملات في كل دورة لتخفيف الضغط
                for item in valid[:10]:
                    sym = item['symbol']
                    if sym in open_signals_cache: continue
                    
                    df = fetch_data(client, sym, BOT_SETTINGS['timeframe_analysis'], 100)
                    if df is not None:
                        df = calculate_full_technical_indicators(df)
                        strat, score, reason = get_enhanced_signal(sym, df, regime)
                        
                        if strat:
                            sl, tp1, tp2, risk_f = adaptive_risk_management(df, regime, score)
                            curr = df['close'].iloc[-1]
                            
                            # حساب الكمية
                            risk_usd = BOT_SETTINGS['base_capital'] * (BOT_SETTINGS['risk_per_trade_pct']/100) * risk_f
                            dist = curr - sl
                            qty = risk_usd / dist if dist > 0 else 0
                            
                            # حماية: لا تتجاوز 25% من المحفظة للصفقة
                            if qty * curr > BOT_SETTINGS['base_capital'] * 0.25:
                                qty = (BOT_SETTINGS['base_capital'] * 0.25) / curr
                                
                            open_trade(sym, curr, sl, tp1, tp2, qty, strat, regime, paper)
                        else:
                             # تسجيل لوج بسيط
                             if random.random() < 0.1:
                                 with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'فحص', 'r': reason})
                    
                    time.sleep(1.5) # تأخير إلزامي

            time.sleep(20) # راحة للمحرك

        except Exception as e:
            logger.error(f"Engine Loop: {e}")
            time.sleep(30)

# --- 6. التنفيذ والعمولات ---
def update_sl(sym, new_sl, reason):
    with locks['signals']:
        if sym in open_signals_cache:
            open_signals_cache[sym]['stop_loss'] = float(new_sl)
            check_db()
            with conn.cursor() as cur:
                cur.execute("UPDATE trades_v14 SET stop_loss=%s WHERE symbol=%s AND status='open'", (float(new_sl), sym))
            send_telegram("UPDATE", {'symbol': sym, 'new_sl': new_sl, 'reason': reason})

def open_trade(sym, price, sl, tp1, tp2, qty, strat, regime, paper):
    check_db()
    try:
        mode = 'PAPER' if paper else 'REAL'
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v14 (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, status, mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s) RETURNING id
            """, (sym, float(price), float(sl), float(tp1), float(tp2), float(qty), strat, regime, mode))
            tid = cur.fetchone()['id']
        
        trade = {
            'id': tid, 'symbol': sym, 'entry_price': price, 'stop_loss': sl,
            'tp1': tp1, 'tp2': tp2, 'quantity': qty, 'strategy': strat, 'entry_time': datetime.now(), 'is_paper': paper
        }
        with locks['signals']: open_signals_cache[sym] = trade
        send_telegram("BUY", {**trade, 'price': price, 'sl': sl})
        
    except Exception as e: logger.error(f"Open Error: {e}")

def close_trade(sym, price, reason, paper):
    check_db()
    try:
        trade = None
        with locks['signals']:
            if sym in open_signals_cache:
                trade = open_signals_cache[sym]
                del open_signals_cache[sym]
        if not trade: return

        price = float(price)
        qty = float(trade['quantity'])
        entry = float(trade['entry_price'])
        
        # حساب العمولات والربح
        rate = BOT_SETTINGS['commission_rate'] / 100
        comm_entry = entry * qty * rate
        comm_exit = price * qty * rate
        total_comm = comm_entry + comm_exit
        
        gross_pnl = (price - entry) * qty
        net_pnl = gross_pnl - total_comm
        pct = ((price - entry) / entry) * 100
        
        if paper:
            with locks['settings']: BOT_SETTINGS['base_capital'] += net_pnl

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades_v14 
                SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, commission_paid=%s, exit_reason=%s
                WHERE id=%s
            """, (price, pct, net_pnl, total_comm, reason, trade['id']))
            
        send_telegram("SELL", {'symbol': sym, 'price': price, 'profit': pct, 'net': net_pnl, 'reason': reason})
        
    except Exception as e: logger.error(f"Close Error: {e}")

def send_telegram(event, payload):
    if not TELEGRAM_TOKEN: return
    msg = ""
    if event == "BUY":
        msg = f"🚀 *دخول جديد | {payload['symbol']}*\nاستراتيجية: {payload['strategy']}\nالسعر: {payload['price']}\nوقف: {payload['sl']}"
    elif event == "SELL":
        icon = "✅" if payload['net'] > 0 else "🔻"
        msg = f"{icon} *إغلاق | {payload['symbol']}*\nالسعر: {payload['price']}\nالصافي: {payload['net']:.2f}$"
    elif event == "UPDATE":
        msg = f"🛡️ *تحديث وقف | {payload['symbol']}*\nالجديد: {payload['new_sl']}"
        
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- 7. API & Dashboard ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/analytics')
def analytics():
    with locks['market']: m = system_state.copy()
    with locks['signals']: s = [{k: v for k, v in t.items() if k != 'entry_time'} for t in open_signals_cache.values()]
    with locks['prices']: p = live_prices.copy()
    with locks['logs']: l = list(scan_logs)
    with locks['settings']: sett = BOT_SETTINGS.copy()
    
    stats = {'win': 0, 'pf': 0, 'pnl': 0, 'comm': 0, 'count': 0, 'hist': []}
    
    try:
        check_db()
        with conn.cursor() as cur:
            cur.execute("SELECT closed_at, profit_abs, commission_paid FROM trades_v14 WHERE status='closed' ORDER BY closed_at ASC")
            rows = cur.fetchall()
            wins, loss, run_pnl = 0, 0, sett['base_capital']
            
            for r in rows:
                pnl = r['profit_abs']
                run_pnl += pnl
                if pnl > 0: wins += 1
                else: loss += abs(pnl)
                stats['comm'] += (r['commission_paid'] or 0)
                stats['hist'].append({'t': r['closed_at'].strftime('%d %H:%M'), 'v': run_pnl})
            
            stats['count'] = len(rows)
            stats['pnl'] = run_pnl - sett['base_capital'] # Net Profit Only
            stats['win'] = (wins/len(rows)*100) if rows else 0
            stats['pf'] = (stats['pnl'] + loss)/loss if loss > 0 else 0 # Simple PF appx
    except: pass
    
    return jsonify({"m": m, "s": s, "p": p, "st": stats, "l": l, "set": sett})

@app.route('/api/toggle', methods=['POST'])
def toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify("OK")

@app.route('/api/close/<sym>', methods=['POST'])
def manual(sym):
    p = live_prices.get(sym, 0)
    if p > 0:
        close_trade(sym, p, "إغلاق يدوي ⚡", BOT_SETTINGS['paper_trading_mode'])
        return jsonify("Closed")
    return jsonify("Error"), 400

@app.route('/api/save_settings', methods=['POST'])
def save_settings():
    d = request.json
    with locks['settings']:
        if 'base' in d: BOT_SETTINGS['base_capital'] = float(d['base'])
        if 'risk' in d: BOT_SETTINGS['risk_per_trade_pct'] = float(d['risk'])
        if 'comm' in d: BOT_SETTINGS['commission_rate'] = float(d['comm'])
    return jsonify("Saved")

# HTML أصلي (c4.py) + التعديلات
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot V14 - التحكم الكامل</title>
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
        .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text); margin-left: 10px; }
        .btn-sm-red { background: var(--red); color: #fff; padding: 4px 10px; font-size: 12px; border-radius: 4px; border:none; cursor:pointer; }
        
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: right; padding: 12px; border-bottom: 1px solid var(--border); }
        th { color: #848e9c; font-size: 12px; }
        .pnl-g { color: var(--green); } .pnl-r { color: var(--red); }
        
        /* Modal Styles */
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 100; }
        .modal-content { background: var(--panel); width: 400px; margin: 100px auto; padding: 25px; border-radius: 8px; border: 1px solid var(--border); }
        .inp-group { margin-bottom: 15px; }
        .inp-group label { display: block; margin-bottom: 8px; color: #848e9c; }
        .inp-group input { width: 100%; padding: 10px; background: var(--bg); border: 1px solid var(--border); color: #fff; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size:24px">SmartBot <span style="color:var(--accent)">V14</span></h1>
            <span style="font-size:12px; color:#848e9c">نظام التداول الذكي + إدارة المخاطر</span>
        </div>
        <div style="display:flex; align-items:center">
            <button class="btn btn-outline" onclick="openSettings()">⚙️ إعدادات</button>
            <div style="width:15px"></div>
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
            <div class="sub-text">عمولات مدفوعة: <span id="commPaid" style="color:#f6465d">0</span>$</div>
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
                        <th>تحكم</th>
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

    <!-- Settings Modal -->
    <div id="setModal" class="modal">
        <div class="modal-content">
            <h2 style="margin-top:0">⚙️ إعدادات النظام</h2>
            <div class="inp-group">
                <label>رأس المال (Base Capital)</label>
                <input type="number" id="inp_base">
            </div>
            <div class="inp-group">
                <label>نسبة المخاطرة (%)</label>
                <input type="number" id="inp_risk" step="0.1">
            </div>
            <div class="inp-group">
                <label>عمولة المنصة (%)</label>
                <input type="number" id="inp_comm" step="0.01">
            </div>
            <div style="margin-top:20px; display:flex; gap:10px">
                <button class="btn" onclick="saveSettings()">حفظ وتطبيق</button>
                <button class="btn btn-outline" onclick="closeSettings()">إلغاء</button>
            </div>
        </div>
    </div>

    <script>
        let equityChart, statsChart;
        Chart.defaults.color = '#848e9c';
        Chart.defaults.borderColor = '#2b3139';
        Chart.defaults.font.family = 'Tajawal';

        const regimeMap = {
            "Bull_Trend_Strong": "صاعد قوي 🐂",
            "Bull_Accumulation": "تجميع صاعد 📈",
            "Bear_Trend_Strong": "هابط قوي 🐻",
            "High_Volatility": "تذبذب عالي ⚡",
            "Ranging": "عرضي مستقر 🦀",
            "Neutral": "محايد ⚖️"
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
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { display: false } } }
            });
        }

        async function updateData() {
            try {
                const res = await fetch('/api/analytics');
                const d = await res.json();

                // 1. الأزرار والحالة
                const btn = document.getElementById('powerBtn');
                document.getElementById('connectionStatus').className = "status-dot dot-green";
                if(d.set.is_trading_enabled) {
                    btn.innerText = "إيقاف البوت 🛑"; btn.style.background = "var(--red)";
                } else {
                    btn.innerText = "تشغيل البوت 🚀"; btn.style.background = "var(--green)";
                }

                // 2. المؤشرات
                document.getElementById('regime').innerText = regimeMap[d.m.market_regime] || d.m.market_regime;
                document.getElementById('trendStr').innerText = d.m.trend_strength;
                document.getElementById('winRate').innerText = d.st.win.toFixed(1);
                document.getElementById('winRateCenter').innerText = d.st.win.toFixed(1) + "%";
                document.getElementById('tradeCount').innerText = d.st.count;
                
                const pnl = d.st.pnl;
                const pnlEl = document.getElementById('totalPnl');
                pnlEl.innerText = "$" + pnl.toFixed(2);
                pnlEl.style.color = pnl >= 0 ? "var(--green)" : "var(--red)";
                document.getElementById('commPaid').innerText = d.st.comm.toFixed(2);

                document.getElementById('activeCount').innerText = d.s.length;
                document.getElementById('openRisk').innerText = (d.s.length * d.set.risk_per_trade_pct).toFixed(1); 

                // 3. الشارتات
                if(d.st.hist.length > 0) {
                    equityChart.data.labels = d.st.hist.map(h => h.t);
                    equityChart.data.datasets[0].data = d.st.hist.map(h => h.v);
                    equityChart.update();
                    statsChart.data.datasets[0].data = [d.st.win, 100 - d.st.win];
                    statsChart.update();
                }

                // 4. الجدول (مع زر الإغلاق الجديد)
                document.getElementById('tradesBody').innerHTML = d.s.length ? d.s.map(s => {
                    const curr = d.p[s.symbol] || s.entry_price;
                    const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                    return `
                    <tr>
                        <td style="font-weight:bold; color:var(--text)">${s.symbol}</td>
                        <td><span style="background:#2b3139; padding:2px 6px; border-radius:4px; font-size:11px">${s.strategy}</span></td>
                        <td>${s.entry_price}</td>
                        <td>${curr}</td>
                        <td class="${pnl>=0?'pnl-g':'pnl-r'}">${pnl.toFixed(2)}%</td>
                        <td style="font-size:11px; color:#848e9c">${s.tp1}</td>
                        <td><button class="btn-sm-red" onclick="manualClose('${s.symbol}')">X</button></td>
                    </tr>`;
                }).join('') : "<tr><td colspan='7' style='text-align:center; padding:20px; color:#444'>لا توجد صفقات نشطة</td></tr>";

                // 5. السجل
                document.getElementById('logsBody').innerHTML = d.l.map(l => `
                    <tr><td style="color:#666">${l.t}</td><td style="font-weight:bold">${l.s}</td><td style="color:${l.st==='دخول'?'var(--green)':'#848e9c'}">${l.st}</td><td>${l.r}</td></tr>
                `).join('');

                // تعبئة المودال بالبيانات الحالية
                if(document.getElementById('setModal').style.display !== 'block') {
                    document.getElementById('inp_base').value = d.set.base_capital;
                    document.getElementById('inp_risk').value = d.set.risk_per_trade_pct;
                    document.getElementById('inp_comm').value = d.set.commission_rate;
                }

            } catch(e) { console.error(e); }
        }

        function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(updateData); }
        
        function manualClose(sym) {
            if(confirm('إغلاق الصفقة يدوياً؟')) fetch('/api/close/'+sym, {method:'POST'}).then(updateData);
        }

        function openSettings() { document.getElementById('setModal').style.display = 'block'; }
        function closeSettings() { document.getElementById('setModal').style.display = 'none'; }
        
        function saveSettings() {
            const load = {
                base: document.getElementById('inp_base').value,
                risk: document.getElementById('inp_risk').value,
                comm: document.getElementById('inp_comm').value
            };
            fetch('/api/save_settings', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify(load)
            }).then(()=>{ closeSettings(); updateData(); });
        }

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
    logger.info("🖥️ لوحة التحكم العربية تعمل على المنفذ 5000")
    app.run(host='0.0.0.0', port=5000)