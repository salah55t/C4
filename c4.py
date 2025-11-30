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
from binance.exceptions import BinanceAPIException, BinanceRequestException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- 1. إعدادات النظام ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('smart_bot_safe.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Safe')

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ في ملف .env: {e}")
    exit(1)

# --- 2. إعدادات التداول والمخاطر ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,       
    "risk_per_trade_pct": 2.0,    
    "max_open_trades": 5,         
    "min_usdt_volume": 10000000, 
    "timeframe_analysis": "1h",
    "atr_sl_mult": 1.5,
    "atr_tp_mult": 2.5
}

# --- إعدادات الأمان (الجوهرية لمنع الحظر) ---
SAFETY_CONFIG = {
    "MAX_WEIGHT_PER_MINUTE": 1000,  
    "SLEEP_BETWEEN_SYMBOLS": 2.5,   
    "SLEEP_ON_ERROR": 60,           
    "BAN_PROTECTION_SLEEP": 300     
}

system_state = {
    "market_regime": "Neutral",
    "market_score": 0,
    "trend_strength": 0,
    "last_update": None
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=200)

locks = {
    'signals': Lock(), 'prices': Lock(), 'market': Lock(), 
    'settings': Lock(), 'logs': Lock(), 'db': Lock(), 'api': Lock()
}

# --- 3. فئة حارس البوابة (Rate Limit Guard) ---
class RateLimitGuard:
    def __init__(self):
        self.used_weight_1m = 0
        self.last_reset = datetime.now()

    def check_limits(self, weight_cost=1):
        """فحص الحدود قبل تنفيذ الطلب"""
        now = datetime.now()
        if (now - self.last_reset).total_seconds() > 60:
            self.used_weight_1m = 0
            self.last_reset = now

        if self.used_weight_1m + weight_cost > SAFETY_CONFIG['MAX_WEIGHT_PER_MINUTE']:
            wait_time = 61 - (now - self.last_reset).total_seconds()
            wait_time = max(1, wait_time)
            logger.warning(f"⚠️ اقتربنا من حد API ({self.used_weight_1m}). إيقاف مؤقت {wait_time:.1f} ثانية...")
            time.sleep(wait_time)
            self.used_weight_1m = 0
            self.last_reset = datetime.now()
        
        self.used_weight_1m += weight_cost

rate_guard = RateLimitGuard()

# --- 4. قاعدة البيانات (V14) ---
conn = None
def init_db():
    global conn
    with locks['db']:
        try:
            if conn: conn.close()
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
                        exit_reason TEXT,
                        highest_price DOUBLE PRECISION
                    );
                """)
            logger.info("✅ قاعدة البيانات متصلة (V14 Structure).")
        except Exception as e: logger.error(f"خطأ قاعدة البيانات: {e}")

def check_db():
    global conn
    try:
        if conn is None or conn.closed != 0:
            init_db()
        else:
            with conn.cursor() as cur: cur.execute('SELECT 1')
    except:
        init_db()

# --- 5. التنبيهات ---
def clean_mk(text):
    return str(text).replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        mode_icon = "🧪 تجريبي" if payload.get('is_paper') else "💰 حقيقي"
        sym = clean_mk(payload.get('symbol', 'UNKNOWN'))
        msg = ""

        if event == "BUY":
            strat = clean_mk(payload.get('strategy', 'N/A'))
            msg = (
                f"🚀 *توصية جديدة | {sym}*\n"
                f"ــــــــــــــــــــــــــــــــــــــــ\n"
                f"📊 الاستراتيجية: `{strat}`\n"
                f"🌍 السوق: {payload.get('market_regime', 'N/A')}\n"
                f"💵 السعر: `{payload['entry_price']}`\n"
                f"🛑 الوقف: `{payload['stop_loss']}`\n"
                f"🎯 هدف 2: `{payload['tp2']}`\n"
                f"🕹️ الوضع: {mode_icon}"
            )
        elif event == "SELL":
            pnl = payload['profit']
            emoji = "✅ ربح" if pnl > 0 else "🔻 خسارة"
            msg = (
                f"{emoji} *إغلاق صفقة | {sym}*\n"
                f"ــــــــــــــــــــــــــــــــــــــــ\n"
                f"📉 الخروج: `{payload['price']}`\n"
                f"💰 النسبة: `{pnl:.2f}%`\n"
                f"📝 السبب: {clean_mk(payload['reason'])}\n"
                f"⏱️ المدة: {payload.get('duration', 0)} دقيقة"
            )
        elif event == "UPDATE":
            msg = (
                f"🛡️ *تحديث وقف | {sym}*\n"
                f"الجديد: `{payload['new_sl']}`\n"
                f"السبب: {clean_mk(payload['reason'])}"
            )

        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=5
        )
    except Exception as e:
        logger.error(f"فشل إرسال تليجرام: {e}")

# --- 6. التحليل الفني الآمن (Safe Fetch) ---
def safe_fetch_klines(client, symbol, interval, limit=100):
    try:
        with locks['api']:
            rate_guard.check_limits(weight_cost=2)
        
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        
        if not klines: return None
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # المؤشرات الأساسية
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        
        # Ichimoku
        h9 = df['high'].rolling(9).max(); l9 = df['low'].rolling(9).min()
        df['tenkan'] = (h9 + l9) / 2
        h26 = df['high'].rolling(26).max(); l26 = df['low'].rolling(26).min()
        df['kijun'] = (h26 + l26) / 2
        df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        h52 = df['high'].rolling(52).max(); l52 = df['low'].rolling(52).min()
        df['span_b'] = ((h52 + l52) / 2).shift(26)
        
        # RSI & ATR
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        df['atr'] = df['tr'].rolling(14).mean()
        
        return df

    except BinanceAPIException as e:
        if e.code == -1003:
            logger.critical(f"⛔ تحذير حظر (Way too much weight)! دخول وضع السبات لمدة 5 دقائق...")
            time.sleep(SAFETY_CONFIG['BAN_PROTECTION_SLEEP'])
            with locks['api']: rate_guard.used_weight_1m = 0
        elif e.code == -1021:
             logger.warning("Timestamp sync issue.. retrying next loop")
        else:
            logger.error(f"API Error ({symbol}): {e}")
        return None
    except Exception as e:
        logger.error(f"Fetch Error: {e}")
        return None

# --- 7. منطق السوق والرموز ---
def analyze_market_leaders(client):
    global system_state
    leaders = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    score = 0
    
    logger.info("🔎 تحديث حالة السوق...")
    for sym in leaders:
        df = safe_fetch_klines(client, sym, '4h', limit=50)
        time.sleep(1) 
        
        if df is None: continue
        last = df.iloc[-1]
        
        if last['close'] > last['span_a'] and last['close'] > last['span_b']: score += 1
        if last['close'] > last['ema200']: score += 1
        if last['close'] < last['span_a'] and last['close'] < last['span_b']: score -= 1

    norm_score = (score / (len(leaders) * 2)) * 100 
    
    regime = "Neutral"
    if norm_score > 30: regime = "Bull_Strong"
    elif norm_score > 10: regime = "Bull_Weak"
    elif norm_score < -30: regime = "Bear_Strong"
    elif norm_score < -10: regime = "Bear_Weak"
    
    with locks['market']:
        system_state['market_regime'] = regime
        system_state['market_score'] = norm_score
        system_state['last_update'] = datetime.now()
    
    logger.info(f"🧭 السوق: {regime} ({norm_score:.1f}%)")

def get_best_symbols(client):
    try:
        with locks['api']: rate_guard.check_limits(weight_cost=20)
        tickers = client.get_ticker()
        
        valid = []
        for t in tickers:
            if not t['symbol'].endswith('USDT'): continue
            vol = float(t['quoteVolume'])
            if vol < BOT_SETTINGS['min_usdt_volume']: continue
            if 'USD' in t['symbol'].replace('USDT', ''): continue
            
            score = vol * abs(float(t['priceChangePercent']))
            valid.append({'s': t['symbol'], 'sc': score})
            
        valid.sort(key=lambda x: x['sc'], reverse=True)
        return [x['s'] for x in valid[:30]]
    except Exception as e: 
        logger.error(f"Error fetching tickers: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

# --- 8. الاستراتيجيات ---
def get_signal(symbol, df, regime):
    if len(df) < 55: return None, None
    last = df.iloc[-1]
    
    if regime == "Bear_Strong": return None, None
    
    if "Bull" in regime:
        if last['close'] > last['span_a'] and last['close'] > last['span_b']:
            if last['close'] > last['ema50']:
                if last['tenkan'] > last['kijun']:
                     return "BUY", "Cloud_Breakout_Trend"
    
    if regime in ["Neutral", "Bull_Weak"]:
        dist = abs(last['close'] - last['ema50']) / last['close']
        if dist < 0.02 and last['close'] > last['ema50']:
            if last['rsi'] < 55: 
                return "BUY", "EMA50_Bounce"

    return None, None

def manage_trade(trade, curr_price, df):
    sl = trade['stop_loss']
    entry = trade['entry_price']
    highest = trade.get('highest_price', entry)
    
    if curr_price > highest: highest = curr_price
    
    atr = df['atr'].iloc[-1]
    profit_pct = (curr_price - entry) / entry * 100
    
    if curr_price >= trade['tp2']: return "CLOSE", "Hit Target 2 🎯", highest
    if curr_price <= sl: return "CLOSE", "Hit Stop Loss 🛑", highest
    
    new_sl = sl
    updated = False
    
    if profit_pct > 2.0 and sl < entry:
        new_sl = entry * 1.005
        updated = True
    elif profit_pct > 4.0:
        trailing = curr_price - (atr * 2.0)
        if trailing > sl:
            new_sl = trailing
            updated = True
            
    if updated: return "UPDATE", "Trailing", new_sl
    return "HOLD", "", highest

# --- 9. المحرك الرئيسي (مُحسّن) ---
def bot_engine():
    while True:
        try:
            client = Client(API_KEY, API_SECRET)
            client.get_system_status()
            break
        except Exception as e:
            logger.error("فشل الاتصال الأولي، إعادة المحاولة في 10 ثواني...")
            time.sleep(10)

    logger.info("🚀 SmartBot Ultimate (Safe Mode) Started")
    
    active_symbols = []
    last_scan_list = datetime.now() - timedelta(hours=2)
    last_market_check = datetime.now() - timedelta(minutes=30)
    
    while True:
        try:
            now = datetime.now()
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                max_trades = BOT_SETTINGS['max_open_trades']
                is_paper = BOT_SETTINGS['paper_trading_mode']
            
            if not enabled:
                time.sleep(10)
                continue

            if (now - last_market_check).total_seconds() > 900: 
                analyze_market_leaders(client)
                last_market_check = now
            
            if (now - last_scan_list).total_seconds() > 3600 or not active_symbols:
                active_symbols = get_best_symbols(client)
                logger.info(f"📋 القائمة النشطة: {len(active_symbols)} رمز")
                last_scan_list = now

            with locks['market']: regime = system_state['market_regime']
            
            with locks['signals']: trades = list(open_signals_cache.values())
            
            for t in trades:
                df = safe_fetch_klines(client, t['symbol'], '15m', 60)
                if df is None: continue
                
                curr = df['close'].iloc[-1]
                with locks['prices']: live_prices[t['symbol']] = curr
                
                action, reason, val = manage_trade(t, curr, df)
                
                if action == "CLOSE":
                    close_trade_db(t['id'], t['symbol'], curr, reason)
                elif action == "UPDATE":
                    # 🔥 FIX: ضمان التحويل إلى float قبل التحديث
                    val = float(val)
                    t['stop_loss'] = val
                    t['highest_price'] = float(max(t.get('highest_price', 0), curr))
                    
                    check_db()
                    with conn.cursor() as cur:
                        cur.execute("UPDATE trades_v14 SET stop_loss=%s, highest_price=%s WHERE id=%s", (val, t['highest_price'], t['id']))
                    send_telegram("UPDATE", {"symbol": t['symbol'], "new_sl": val, "reason": reason})

                time.sleep(1)

            if len(trades) < max_trades:
                for sym in active_symbols:
                    with locks['signals']:
                        if len(open_signals_cache) >= max_trades: break
                        if sym in open_signals_cache: continue
                        
                    df = safe_fetch_klines(client, sym, BOT_SETTINGS['timeframe_analysis'], 100)
                    
                    if df is not None:
                        sig, strat = get_signal(sym, df, regime)
                        
                        if sig == "BUY":
                            price = df['close'].iloc[-1]
                            atr = df['atr'].iloc[-1]
                            sl = price - (atr * BOT_SETTINGS['atr_sl_mult'])
                            tp1 = price + (atr * BOT_SETTINGS['atr_sl_mult'] * 1.5)
                            tp2 = price + (atr * BOT_SETTINGS['atr_tp_mult'])
                            
                            risk = price - sl
                            qty = (BOT_SETTINGS['base_capital'] * (BOT_SETTINGS['risk_per_trade_pct']/100)) / risk
                            
                            open_trade_db(sym, price, sl, tp1, tp2, qty, strat, regime, is_paper)
                    
                    time.sleep(SAFETY_CONFIG['SLEEP_BETWEEN_SYMBOLS'])
            
            time.sleep(10)

        except Exception as e:
            logger.error(f"Engine Loop Error: {e}")
            time.sleep(5)

# --- 10. عمليات قاعدة البيانات (تم الإصلاح هنا) ---
def open_trade_db(symbol, price, sl, tp1, tp2, qty, strat, regime, is_paper):
    check_db()
    try:
        # 🔥 FIX: تحويل كل متغيرات NumPy إلى Python native floats
        # هذا يمنع خطأ "schema np does not exist"
        price = float(price)
        sl = float(sl)
        tp1 = float(tp1)
        tp2 = float(tp2)
        qty = float(qty)
        
        mode = 'PAPER' if is_paper else 'REAL'
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v14 (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, status, mode, highest_price)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, %s) RETURNING id
            """, (symbol, price, sl, tp1, tp2, qty, strat, regime, mode, price))
            tid = cur.fetchone()['id']
            
        trade = {
            'id': tid, 'symbol': symbol, 'entry_price': price, 'stop_loss': sl,
            'tp1': tp1, 'tp2': tp2, 'quantity': qty, 'strategy': strat,
            'market_regime': regime, 'entry_time': datetime.now(), 'highest_price': price,
            'is_paper': is_paper
        }
        with locks['signals']: open_signals_cache[symbol] = trade
        send_telegram("BUY", trade)
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'دخول', 'r': strat})
        
    except Exception as e: logger.error(f"DB Open Error: {e}")

def close_trade_db(tid, symbol, price, reason):
    check_db()
    try:
        trade = None
        with locks['signals']:
            if symbol in open_signals_cache:
                trade = open_signals_cache[symbol]
                del open_signals_cache[symbol]
        
        if not trade: return
        
        # 🔥 FIX: تحويل القيم المحسوبة إلى float عادي
        price = float(price)
        entry_price = float(trade['entry_price'])
        quantity = float(trade['quantity'])
        
        profit_pct = float(((price - entry_price) / entry_price) * 100)
        profit_abs = float((price - entry_price) * quantity)
        dur = int((datetime.now() - trade['entry_time']).total_seconds() / 60)

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades_v14 SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, exit_reason=%s 
                WHERE id=%s
            """, (price, profit_pct, profit_abs, reason, tid))
            
        send_telegram("SELL", {'symbol': symbol, 'price': price, 'profit': profit_pct, 'reason': reason, 'duration': dur})
        logger.info(f"Closed {symbol} with {profit_pct:.2f}%")
        
    except Exception as e: logger.error(f"DB Close Error: {e}")

# --- 11. واجهة الويب الكاملة (النسخة الأصلية) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/toggle', methods=['POST'])
def toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify("OK")

@app.route('/api/close_manual/<symbol>', methods=['POST'])
def manual_close(symbol):
    with locks['signals']:
        if symbol in open_signals_cache:
            tid = open_signals_cache[symbol]['id']
            # إغلاق في الخلفية
            Thread(target=close_trade_db, args=(tid, symbol, live_prices.get(symbol, 0), "Manual Close 👤")).start()
            return jsonify("Closed")
    return jsonify("Not Found"), 404

@app.route('/api/analytics')
def analytics():
    # تجميع البيانات
    with locks['market']: m = system_state.copy()
    with locks['signals']: s = [{k: v for k, v in t.items() if k != 'entry_time'} for t in open_signals_cache.values()]
    with locks['prices']: p = live_prices.copy()
    with locks['logs']: l = list(scan_logs)
    
    stats = {'win_rate': 0, 'profit_factor': 0, 'total_pnl_usd': 0, 'trade_count': 0, 'history': []}
    try:
        check_db()
        with conn.cursor() as cur:
            cur.execute("SELECT closed_at, profit_pct, profit_abs FROM trades_v14 WHERE status='closed' ORDER BY closed_at ASC")
            rows = cur.fetchall()
            wins, gp, gl, cum = 0, 0, 0, 0
            for r in rows:
                if r['profit_pct'] > 0: wins += 1; gp += r['profit_abs']
                else: gl += abs(r['profit_abs'])
                cum += r['profit_pct']
                stats['history'].append({'t': r['closed_at'].strftime('%d %H:%M'), 'v': cum})
            
            if rows:
                stats['win_rate'] = (wins / len(rows)) * 100
                stats['profit_factor'] = (gp / gl) if gl > 0 else 99
                stats['total_pnl_usd'] = gp - gl
                stats['trade_count'] = len(rows)
    except: pass
    
    return jsonify({"market": m, "signals": s, "prices": p, "stats": stats, "logs": l, "settings": BOT_SETTINGS})

# HTML/JS/CSS الأصلي الكامل
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot Ultimate</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --panel: #151a1e; --border: #2b3139; --text: #eaecef; --green: #0ecb81; --red: #f6465d; --accent: #9932CC; }
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
        .btn { background: var(--accent); color: #fff; border: none; padding: 8px 20px; border-radius: 4px; font-weight: bold; cursor: pointer; transition: 0.2s; font-family: 'Tajawal'; }
        .btn-sm { padding: 4px 10px; font-size: 12px; background: var(--red); color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover, .btn-sm:hover { opacity: 0.9; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: right; padding: 12px; border-bottom: 1px solid var(--border); }
        th { color: #848e9c; font-size: 12px; }
        .pnl-g { color: var(--green); } .pnl-r { color: var(--red); }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
        @media(max-width: 768px) { .col-3, .col-4, .col-6, .col-8 { grid-column: span 12; } }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size:24px">SmartBot <span style="color:var(--accent)">Ultimate</span></h1>
            <span style="font-size:12px; color:#848e9c">نظام الحماية الذكي + V14 Backend</span>
        </div>
        <div style="display:flex; gap:15px; align-items:center">
            <div style="text-align:left; margin-left:15px">
                <span id="connectionStatus" class="status-dot"></span>
                <span style="font-size:12px">متصل</span>
            </div>
            <button id="powerBtn" class="btn" onclick="toggleBot()">جاري التحميل...</button>
        </div>
    </div>

    <div class="grid">
        <div class="card col-3">
            <h3>حالة السوق</h3>
            <div id="regime" class="big-num" style="color:var(--accent); font-size:20px">--</div>
            <div class="sub-text">Score: <span id="trendStr">0</span>%</div>
        </div>
        <div class="card col-3">
            <h3>نسبة النجاح</h3>
            <div class="big-num"><span id="winRate">0</span><small>%</small></div>
            <div class="sub-text">الصفقات: <span id="tradeCount">0</span></div>
        </div>
        <div class="card col-3">
            <h3>الأرباح (USDT)</h3>
            <div id="totalPnl" class="big-num">$0.00</div>
            <div class="sub-text">Profit Factor: <span id="profFact">0</span></div>
        </div>
        <div class="card col-3">
            <h3>المخاطرة</h3>
            <div class="big-num"><span id="openRisk">0</span><small>%</small></div>
            <div class="sub-text">نشطة: <span id="activeCount">0</span></div>
        </div>
    </div>

    <div class="grid">
        <div class="card col-8">
            <h3>نمو المحفظة</h3>
            <div style="height: 250px;"><canvas id="equityChart"></canvas></div>
        </div>
        <div class="card col-4">
            <h3>الأداء</h3>
            <div style="height: 250px; position:relative">
                <canvas id="statsChart"></canvas>
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); text-align:center">
                    <span style="font-size:20px; font-weight:bold" id="winRateCenter">0%</span><br>
                    <span style="font-size:10px; color:#888">فوز</span>
                </div>
            </div>
        </div>
    </div>

    <div class="grid">
        <div class="card col-8">
            <h3>المحفظة النشطة</h3>
            <table>
                <thead>
                    <tr>
                        <th>العملة</th>
                        <th>الاستراتيجية</th>
                        <th>الدخول</th>
                        <th>السعر</th>
                        <th>الربح %</th>
                        <th>هدف</th>
                        <th>إجراء</th>
                    </tr>
                </thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>
        <div class="card col-4">
            <h3>سجل النظام</h3>
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

        function initCharts() {
            const ctx1 = document.getElementById('equityChart').getContext('2d');
            const gradient = ctx1.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, 'rgba(153, 50, 204, 0.2)');
            gradient.addColorStop(1, 'rgba(153, 50, 204, 0)');

            equityChart = new Chart(ctx1, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'النمو %', data: [], borderColor: '#9932CC', backgroundColor: gradient, borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }] },
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
                
                const btn = document.getElementById('powerBtn');
                document.getElementById('connectionStatus').className = "status-dot dot-green";
                if(d.settings.is_trading_enabled) {
                    btn.innerText = "إيقاف البوت 🛑"; btn.style.background = "var(--red)";
                } else {
                    btn.innerText = "تشغيل البوت 🚀"; btn.style.background = "var(--green)";
                }

                document.getElementById('regime').innerText = d.market.market_regime;
                document.getElementById('trendStr').innerText = d.market.market_score.toFixed(1);
                
                document.getElementById('winRate').innerText = d.stats.win_rate.toFixed(1);
                document.getElementById('winRateCenter').innerText = d.stats.win_rate.toFixed(1) + "%";
                document.getElementById('tradeCount').innerText = d.stats.trade_count;
                
                const pnl = d.stats.total_pnl_usd;
                const pnlEl = document.getElementById('totalPnl');
                pnlEl.innerText = "$" + pnl.toFixed(2);
                pnlEl.style.color = pnl >= 0 ? "var(--green)" : "var(--red)";
                document.getElementById('profFact').innerText = d.stats.profit_factor.toFixed(2);

                document.getElementById('activeCount').innerText = d.signals.length;
                
                if(d.stats.history.length > 0) {
                    equityChart.data.labels = d.stats.history.map(h => h.t);
                    equityChart.data.datasets[0].data = d.stats.history.map(h => h.v);
                    equityChart.update();
                    statsChart.data.datasets[0].data = [d.stats.win_rate, 100 - d.stats.win_rate];
                    statsChart.update();
                }

                document.getElementById('tradesBody').innerHTML = d.signals.length ? d.signals.map(s => {
                    const curr = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                    return `
                    <tr>
                        <td style="font-weight:bold; color:var(--text)">${s.symbol}</td>
                        <td><span style="background:#2b3139; padding:2px 6px; border-radius:4px; font-size:11px">${s.strategy}</span></td>
                        <td>${s.entry_price}</td>
                        <td>${curr}</td>
                        <td class="${pnl>=0?'pnl-g':'pnl-r'}">${pnl.toFixed(2)}%</td>
                        <td style="font-size:11px">${s.tp2}</td>
                        <td><button class="btn-sm" onclick="closeTrade('${s.symbol}')">إغلاق</button></td>
                    </tr>`;
                }).join('') : "<tr><td colspan='7' style='text-align:center; padding:20px; color:#444'>لا توجد صفقات نشطة</td></tr>";

                document.getElementById('logsBody').innerHTML = d.logs.map(l => `
                    <tr>
                        <td style="color:#666">${l.t}</td>
                        <td style="font-weight:bold">${l.s}</td>
                        <td style="color:${l.st==='دخول'?'var(--green)':'#848e9c'}">${l.st}</td>
                        <td>${l.r}</td>
                    </tr>`).join('');
            } catch(e) { console.error(e); }
        }
        
        function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(updateData); }
        
        function closeTrade(sym) {
            if(confirm('هل أنت متأكد من إغلاق صفقة ' + sym + ' يدوياً؟')) {
                fetch('/api/close_manual/' + sym, {method: 'POST'})
                    .then(r => r.json())
                    .then(res => { alert(res); updateData(); });
            }
        }

        initCharts(); setInterval(updateData, 2000); updateData();
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