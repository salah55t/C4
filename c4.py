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

# --- 1. System Configuration & Logging ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('smart_bot_v12.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Pro')

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ System Start Failure: {e}")
    exit(1)

# --- 2. Professional Trading Parameters ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,       # المحفظة الافتراضية
    "risk_per_trade_pct": 2.0,    # المخاطرة 2% من المحفظة لكل صفقة
    "max_open_trades": 6,
    "max_drawdown_protect": 10.0, # إيقاف البوت لو خسرت المحفظة 10%
    "volume_lookback": 50,        # عدد الشموع لحساب متوسط الحجم
    "timeframe_analysis": "15m",  # فريم الدخول
    "timeframe_trend": "1h"       # فريم الاتجاه
}

LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']

# الحالة العامة للنظام
system_state = {
    "market_regime": "Neutral",   # Bull_Trend, Bear_Trend, Ranging, Volatile
    "trend_strength": 0,          # 0-100
    "volatility_index": "Low",    # Low, Med, High
    "active_strategies": [],
    "portfolio_value": BOT_SETTINGS['base_capital'],
    "last_update": None
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=200)

locks = {
    'signals': Lock(), 'prices': Lock(), 'market': Lock(), 
    'settings': Lock(), 'logs': Lock()
}

# --- 3. Advanced Database Schema ---
conn = None
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades_v12 (
                    id SERIAL PRIMARY KEY, 
                    symbol TEXT NOT NULL, 
                    entry_price DOUBLE PRECISION, 
                    stop_loss DOUBLE PRECISION, 
                    tp1 DOUBLE PRECISION,
                    tp2 DOUBLE PRECISION,
                    quantity DOUBLE PRECISION, 
                    strategy_name TEXT, 
                    market_regime TEXT,
                    status TEXT DEFAULT 'open', -- open, closed
                    mode TEXT,
                    entry_time TIMESTAMP DEFAULT NOW(),
                    closed_at TIMESTAMP, 
                    closing_price DOUBLE PRECISION, 
                    profit_abs DOUBLE PRECISION, -- الربح بالدولار
                    profit_pct DOUBLE PRECISION, -- الربح بالنسبة
                    exit_reason TEXT, 
                    max_favorable_excursion DOUBLE PRECISION DEFAULT 0 -- أقصى ربح وصلته الصفقة
                );
            """)
        logger.info("✅ SmartBot V12 Database Schema Ready.")
    except Exception as e: logger.error(f"DB Init Error: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 4. Notification System (Arabic Professional) ---
def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    mode_icon = "🧪 تجريبي" if payload.get('is_paper') else "💰 حقيقي"
    timestamp = datetime.now().strftime("%H:%M")
    msg = ""

    if event == "BUY":
        msg = (
            f"🚀 *تنفيذ دخول استراتيجي | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"📊 الاستراتيجية: `{payload['strategy']}`\n"
            f"🌍 بيئة السوق: {payload['regime']}\n"
            f"💵 السعر: `{payload['price']}`\n"
            f"🛑 الوقف: `{payload['sl']}`\n"
            f"🎯 الأهداف: `{payload['tp1']}` ➔ `{payload['tp2']}`\n"
            f"🕹️ الوضع: {mode_icon} | 🕒 {timestamp}"
        )
    elif event == "SELL":
        pnl = payload['profit']
        emoji = "✅ ربح" if pnl > 0 else "🔻 خسارة"
        msg = (
            f"{emoji} *إغلاق مركز | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"📉 الخروج: `{payload['price']}`\n"
            f"💰 الصافي: `{pnl:.2f}%`\n"
            f"📝 السبب: _{payload['reason']}_\n"
            f"⏱️ المدة: {payload['duration']} دقيقة"
        )
    elif event == "ALERT":
        msg = f"⚠️ *تنبيه نظام*: {payload['msg']}"

    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- 5. Technical Analysis Engine (The Core) ---
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
    # 1. Moving Averages
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # 2. RSI & Stochastic RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_k'] = ((df['rsi'] - min_rsi) / (max_rsi - min_rsi)) * 100
    
    # 3. MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. ADX (Trend Strength) & ATR (Volatility)
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

    # 5. Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # 6. Ichimoku (Simplified - Conversion Line only for triggers)
    high_9 = df['high'].rolling(9).max()
    low_9 = df['low'].rolling(9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    # 7. Volume Analysis
    df['vol_ma'] = df['volume'].rolling(20).mean()

    return df.fillna(0)

# --- 6. Market Regime Classifier (The Brain) ---
def analyze_market_regime(client):
    global system_state
    
    # نستخدم عملات قيادية لتحديد حالة السوق
    btc_df = fetch_data(client, 'BTCUSDT', '4h', 100)
    if btc_df is None: return

    btc_df = calculate_technical_indicators(btc_df)
    last = btc_df.iloc[-1]

    # 1. Trend Determination
    trend_score = 0
    if last['close'] > last['ema200']: trend_score += 1
    if last['ema50'] > last['ema200']: trend_score += 1
    if last['macd'] > last['macd_signal']: trend_score += 1
    
    # 2. Volatility & Strength
    adx = last['adx']
    atr_pct = (last['atr'] / last['close']) * 100
    
    regime = "Neutral"
    
    if trend_score == 3 and adx > 25:
        regime = "Bull_Trend_Strong"
    elif trend_score >= 2 and adx < 20:
        regime = "Bull_Accumulation" # تجميع
    elif trend_score == 0 and adx > 25:
        regime = "Bear_Trend_Strong"
    elif atr_pct > 2.0: # إذا كان الـ ATR عالي جداً بالنسبة للسعر
        regime = "High_Volatility_Choppy"
    else:
        regime = "Ranging" # عرضي

    with locks['market']:
        system_state['market_regime'] = regime
        system_state['trend_strength'] = int(adx)
        system_state['volatility_index'] = "High" if atr_pct > 1.5 else "Normal"
        system_state['last_update'] = datetime.now()
    
    logger.info(f"🧠 Market Regime Updated: {regime} | Strength: {int(adx)}")

# --- 7. Strategy Factory (Context Aware) ---
def get_smart_signal(symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # === فلتر السيولة الذكي (Adaptive Volume) ===
    # في حالة التقلب العالي، نتساهل في الحجم. في الهدوء، نطلب حجم عالي.
    vol_factor = 0.3 if "High_Volatility" in regime else 0.6
    if last['volume'] < last['vol_ma'] * vol_factor:
        # استثناء: اختراق سعري قوي
        if not (last['close'] > last['bb_upper']): 
            return None, "Low Liquidity"

    # === Strategy 1: Ichimoku + EMA Trend (Strong Bull) ===
    if "Bull_Trend" in regime:
        # السعر فوق السحابة (ممثلة بـ EMA50 هنا للتبسيط) + تقاطع Tenkan مع السعر
        if last['close'] > last['ema50'] and last['adx'] > 20:
            # Pullback Entry: السعر يلامس Tenkan ويرتد
            if last['low'] <= last['tenkan_sen'] and last['close'] > last['tenkan_sen']:
                 return "Trend_Pullback_Pro", "Bullish Re-Entry on Tenkan"
            # Breakout Entry
            if last['close'] > prev['high'] and last['macd_hist'] > 0:
                return "Momentum_Breakout", "Trend Continuation"

    # === Strategy 2: Mean Reversion Sniper (Ranging / Accumulation) ===
    elif "Ranging" in regime or "Accumulation" in regime:
        # BB Squeeze + RSI Oversold
        if last['bb_width'] < 0.10: # ضغط قوي
            if last['rsi'] < 40 and last['stoch_k'] < 20:
                if last['close'] > prev['close']: # شمعة تأكيد خضراء
                    return "Sniper_Reversion", "Oversold in Range + Green Candle"

    # === Strategy 3: Volatility Scalp (High Volatility) ===
    elif "High_Volatility" in regime:
        # خطف سريع: ابتعاد السعر عن المتوسطات كثيراً
        dist_ema = (last['close'] - last['ema9']) / last['ema9'] * 100
        if dist_ema < -3.0 and last['rsi'] < 25: # بعيد جداً للأسفل وتشبع بيعي
             return "Deep_Value_Scalp", "Extreme Deviation (Rebound Expected)"

    # === Strategy 4: Golden Cross (Universal) ===
    if last['ema50'] > last['ema200'] and prev['ema50'] <= prev['ema200']:
         return "Golden_Cross_Major", "Long Term Trend Change"

    return None, "No Strategic Fit"

# --- 8. Portfolio Manager (Risk & Position Sizing) ---
def manage_active_trade(symbol, signal, df):
    last = df.iloc[-1]
    curr = float(last['close'])
    entry = float(signal['entry_price'])
    tp1 = float(signal['tp1'])
    tp2 = float(signal['tp2'])
    sl = float(signal['stop_loss'])
    
    profit_pct = (curr - entry) / entry * 100
    duration = (datetime.now() - signal['entry_time']).total_seconds() / 3600

    health_msg = "Stable"
    action = "HOLD"
    new_val = 0

    # 1. Take Profit Levels Logic
    if curr >= tp2:
        # وصلنا للهدف الثاني، نرفع الوقف للهدف الأول لحجز ربح كبير
        if sl < tp1:
            return "UPDATE_SL", tp1, "Locked TP1 Profit", "Profitable 🟢"
    elif curr >= tp1:
        # وصلنا للهدف الأول، نرفع الوقف للدخول (Break Even)
        if sl < entry:
            return "UPDATE_SL", entry * 1.002, "Risk Free Trade", "Secure 🛡️"

    # 2. Smart Trailing Stop (ATR Based)
    # يلاحق السعر إذا تجاوز الربح 2%
    if profit_pct > 2.0:
        atr_trail = curr - (last['atr'] * 2.0)
        if atr_trail > sl:
             return "UPDATE_SL", atr_trail, "ATR Trailing", "Runner 🏃"

    # 3. Time Stop (Zombie Trade)
    # إذا مرت 6 ساعات والربح أقل من 0.5%، اخرج
    if duration > 6 and abs(profit_pct) < 0.5:
        return "CLOSE_NOW", curr, "Time Stop (Dead Money)", "Stagnant ⚠️"

    # 4. Technical Breakdown Exit
    # إذا كسرنا EMA50 بقوة في ترند صاعد
    if signal['market_regime'] == "Bull_Trend_Strong" and curr < last['ema50']:
         return "CLOSE_NOW", curr, "Trend Broken (EMA50)", "Reversal 🔻"

    return "HOLD", 0, "", health_msg

# --- 9. Execution Loop ---
def bot_engine():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 SmartBot V12 Professional Engine Started")
    
    # Load Symbols
    try:
        with open('crypto_list.txt') as f:
            symbols = [l.strip().upper() for l in f if l.strip()]
            symbols = [s if s.endswith('USDT') else s+'USDT' for s in symbols]
    except: 
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'LINKUSDT']

    while True:
        try:
            # Refresh Config
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                paper = BOT_SETTINGS['paper_trading_mode']
                max_t = BOT_SETTINGS['max_open_trades']
                
            if not enabled: 
                time.sleep(5)
                continue

            # 1. Analyze Market Context (Regime)
            analyze_market_regime(client)
            with locks['market']: regime = system_state['market_regime']

            # 2. Manage Active Trades
            with locks['signals']: active_trades = list(open_signals_cache.values())
            
            for trade in active_trades:
                sym = trade['symbol']
                df = fetch_data(client, sym, '5m', 60) # Fast tracking
                if df is None: continue
                df = calculate_technical_indicators(df)
                
                curr_price = df['close'].iloc[-1]
                with locks['prices']: live_prices[sym] = curr_price
                
                # Check Hard Stops
                exit_reason = None
                if curr_price <= trade['stop_loss']: exit_reason = "Stop Loss Hit 🛑"
                # We don't auto close at TP, we let it trail unless it hits max expectations
                
                if not exit_reason:
                    act, val, note, health = manage_active_trade(sym, trade, df)
                    
                    if act == "UPDATE_SL":
                        open_signals_cache[sym]['stop_loss'] = float(val)
                        check_db()
                        with conn.cursor() as cur:
                            cur.execute("UPDATE trades_v12 SET stop_loss=%s WHERE id=%s", (float(val), trade['id']))
                        send_telegram("UPDATE", {"symbol": sym, "new_sl": val, "reason": note})
                        
                    elif act == "CLOSE_NOW":
                        exit_reason = f"Smart Exit: {note}"
                
                if exit_reason:
                    close_trade_final(sym, curr_price, exit_reason, paper)

            # 3. Scan for New Opportunities
            if len(open_signals_cache) < max_t:
                tickers = client.get_ticker()
                valid = [t for t in tickers if t['symbol'] in symbols]
                # Sort by Volume * Change (Activity)
                valid.sort(key=lambda x: float(x['quoteVolume']) * abs(float(x['priceChangePercent'])), reverse=True)
                
                scanned_count = 0
                for t in valid:
                    if scanned_count > 20: break # Scan top 20 active coins only
                    scanned_count += 1
                    
                    sym = t['symbol']
                    if sym in open_signals_cache: continue
                    
                    df = fetch_data(client, sym, BOT_SETTINGS['timeframe_analysis'], 100)
                    if df is None: continue
                    df = calculate_technical_indicators(df)
                    
                    strat, reason = get_smart_signal(sym, df, regime)
                    
                    if strat:
                        curr = df['close'].iloc[-1]
                        atr = df['atr'].iloc[-1]
                        
                        # Advanced Position Sizing
                        sl = curr - (atr * 2.0)
                        tp1 = curr + (atr * 2.0) # 1:1
                        tp2 = curr + (atr * 4.0) # 1:2
                        
                        # Risk 2% of Capital
                        risk_amt = BOT_SETTINGS['base_capital'] * (BOT_SETTINGS['risk_per_trade_pct'] / 100)
                        price_diff = curr - sl
                        qty = risk_amt / price_diff if price_diff > 0 else 0
                        # Fallback safety
                        if qty * curr > BOT_SETTINGS['base_capital'] * 0.2: # Max 20% per trade
                            qty = (BOT_SETTINGS['base_capital'] * 0.2) / curr
                            
                        open_new_trade(sym, curr, sl, tp1, tp2, qty, strat, regime, paper)
                        time.sleep(1) # Cool down
                    else:
                        if random.random() < 0.05:
                             with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'Scan', 'r': reason})
                    
                    time.sleep(0.2)

            time.sleep(15)

        except Exception as e:
            logger.error(f"Engine Loop Error: {e}")
            time.sleep(10)

# --- 10. Database & System Utilities ---
def open_new_trade(symbol, price, sl, tp1, tp2, qty, strat, regime, is_paper):
    check_db()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v12 
                (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, status, mode, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, NOW())
                RETURNING id
            """, (symbol, float(price), float(sl), float(tp1), float(tp2), float(qty), strat, regime, mode))
            db_id = cur.fetchone()['id']
        
        trade = {
            'id': db_id, 'symbol': symbol, 'entry_price': float(price), 'stop_loss': float(sl),
            'tp1': float(tp1), 'tp2': float(tp2), 'quantity': float(qty), 'entry_time': datetime.now(),
            'strategy': strat, 'market_regime': regime, 'is_paper': is_paper
        }
        
        with locks['signals']: open_signals_cache[symbol] = trade
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'ENTRY', 'r': strat})
        send_telegram("BUY", {**trade, 'price': price, 'sl': sl})
        
    except Exception as e: logger.error(f"DB Insert Error: {e}")

def close_trade_final(symbol, price, reason, is_paper):
    check_db()
    try:
        trade = None
        with locks['signals']:
            if symbol in open_signals_cache:
                trade = open_signals_cache[symbol]
                del open_signals_cache[symbol]
        if not trade: return

        profit_pct = ((float(price) - trade['entry_price']) / trade['entry_price']) * 100
        profit_abs = (float(price) - trade['entry_price']) * trade['quantity']
        duration = int((datetime.now() - trade['entry_time']).total_seconds() / 60)

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades_v12 
                SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, exit_reason=%s
                WHERE id=%s
            """, (float(price), float(profit_pct), float(profit_abs), reason, trade['id']))
            
        send_telegram("SELL", {'symbol': symbol, 'price': price, 'profit': profit_pct, 'reason': reason, 'duration': duration})
        
    except Exception as e: logger.error(f"DB Close Error: {e}")

# --- 11. Professional Dashboard (Flask) ---
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
    
    # Advanced Statistics from DB
    stats = {'win_rate': 0, 'profit_factor': 0, 'total_pnl_usd': 0, 'trade_count': 0, 'history': []}
    try:
        check_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT closed_at, profit_pct, profit_abs 
                FROM trades_v12 WHERE status='closed' ORDER BY closed_at ASC
            """)
            rows = cur.fetchall()
            
            wins = 0
            gross_profit = 0
            gross_loss = 0
            cum_pnl = 0
            
            for r in rows:
                if r['profit_pct'] > 0: 
                    wins += 1
                    gross_profit += r['profit_abs']
                else: 
                    gross_loss += abs(r['profit_abs'])
                
                cum_pnl += r['profit_pct']
                stats['history'].append({'t': r['closed_at'].strftime('%d %H:%M'), 'v': cum_pnl})
            
            stats['trade_count'] = len(rows)
            stats['total_pnl_usd'] = gross_profit - gross_loss
            stats['win_rate'] = (wins / len(rows) * 100) if len(rows) > 0 else 0
            stats['profit_factor'] = (gross_profit / gross_loss) if gross_loss > 0 else 99.9
            
    except: pass
    
    return jsonify({"market": m, "signals": s, "prices": p, "stats": stats, "logs": l, "settings": BOT_SETTINGS})

@app.route('/api/toggle', methods=['POST'])
def toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify("OK")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot V12 Pro Terminal</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --panel: #151a1e; --border: #2b3139; --text: #eaecef; --green: #0ecb81; --red: #f6465d; --accent: #f0b90b; }
        * { box-sizing: border-box; }
        body { background: var(--bg); color: var(--text); font-family: 'Tajawal', sans-serif; margin: 0; padding: 20px; font-size: 14px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid var(--border); }
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; margin-bottom: 20px; }
        .col-3 { grid-column: span 3; } .col-4 { grid-column: span 4; } .col-6 { grid-column: span 6; } .col-8 { grid-column: span 8; } .col-12 { grid-column: span 12; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; position: relative; }
        .card h3 { margin: 0 0 15px 0; color: #848e9c; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
        .big-num { font-size: 28px; font-weight: 900; color: var(--text); }
        .sub-text { color: #848e9c; font-size: 12px; }
        .status-dot { height: 10px; width: 10px; background-color: #555; border-radius: 50%; display: inline-block; margin-left: 5px; }
        .dot-green { background-color: var(--green); box-shadow: 0 0 10px var(--green); }
        .btn { background: var(--accent); color: #000; border: none; padding: 8px 20px; border-radius: 4px; font-weight: bold; cursor: pointer; transition: 0.2s; }
        .btn:hover { opacity: 0.9; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: right; padding: 12px; border-bottom: 1px solid var(--border); }
        th { color: #848e9c; font-size: 12px; }
        .pnl-g { color: var(--green); } .pnl-r { color: var(--red); }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size:24px">SmartBot <span style="color:var(--accent)">V12 Pro</span></h1>
            <span style="font-size:12px; color:#848e9c">Institutional Grade Portfolio Manager</span>
        </div>
        <div style="display:flex; gap:15px; align-items:center">
            <div style="text-align:left">
                <span id="connectionStatus" class="status-dot"></span>
                <span style="font-size:12px">API Connected</span>
            </div>
            <button id="powerBtn" class="btn" onclick="toggleBot()">INITIALIZING...</button>
        </div>
    </div>

    <!-- Top KPIs -->
    <div class="grid">
        <div class="card col-3">
            <h3>بيئة السوق (Regime)</h3>
            <div id="regime" class="big-num" style="color:var(--accent)">--</div>
            <div class="sub-text">Trend Strength: <span id="trendStr">0</span></div>
        </div>
        <div class="card col-3">
            <h3>معدل النجاح (Win Rate)</h3>
            <div class="big-num"><span id="winRate">0</span><small>%</small></div>
            <div class="sub-text">Trades: <span id="tradeCount">0</span></div>
        </div>
        <div class="card col-3">
            <h3>العائد الإجمالي (PnL)</h3>
            <div id="totalPnl" class="big-num">$0.00</div>
            <div class="sub-text">Profit Factor: <span id="profFact">0</span></div>
        </div>
        <div class="card col-3">
            <h3>المخاطرة الحالية</h3>
            <div class="big-num"><span id="openRisk">0</span><small>%</small></div>
            <div class="sub-text">Active Positions: <span id="activeCount">0</span></div>
        </div>
    </div>

    <!-- Charts Section -->
    <div class="grid">
        <div class="card col-8">
            <h3>نمو المحفظة (Equity Curve)</h3>
            <div style="height: 250px;"><canvas id="equityChart"></canvas></div>
        </div>
        <div class="card col-4">
            <h3>توزيع الأداء</h3>
            <div style="height: 250px; position:relative">
                <canvas id="statsChart"></canvas>
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); text-align:center">
                    <span style="font-size:20px; font-weight:bold" id="winRateCenter">0%</span><br>
                    <span style="font-size:10px; color:#888">WIN RATE</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Active Trades & Logs -->
    <div class="grid">
        <div class="card col-8">
            <h3>المراكز المفتوحة (Active Portfolio)</h3>
            <table>
                <thead>
                    <tr>
                        <th>الرمز</th>
                        <th>الاستراتيجية</th>
                        <th>الدخول</th>
                        <th>الحالي</th>
                        <th>الربح %</th>
                        <th>الأهداف</th>
                    </tr>
                </thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>
        <div class="card col-4">
            <h3>سجل النظام (System Logs)</h3>
            <div style="height: 300px; overflow-y: auto;">
                <table style="font-size:12px">
                    <tbody id="logsBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Charts Config
        let equityChart, statsChart;
        Chart.defaults.color = '#848e9c';
        Chart.defaults.borderColor = '#2b3139';

        function initCharts() {
            // Equity Line Chart
            const ctx1 = document.getElementById('equityChart').getContext('2d');
            const gradient = ctx1.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, 'rgba(240, 185, 11, 0.2)');
            gradient.addColorStop(1, 'rgba(240, 185, 11, 0)');

            equityChart = new Chart(ctx1, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'PnL %', data: [], borderColor: '#f0b90b', backgroundColor: gradient, borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { borderDash: [5, 5] } } } }
            });

            // Doughnut Chart
            const ctx2 = document.getElementById('statsChart').getContext('2d');
            statsChart = new Chart(ctx2, {
                type: 'doughnut',
                data: { labels: ['Win', 'Loss'], datasets: [{ data: [50, 50], backgroundColor: ['#0ecb81', '#f6465d'], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { position: 'bottom' } } }
            });
        }

        async function updateData() {
            try {
                const res = await fetch('/api/analytics');
                const d = await res.json();

                // 1. Update State
                const btn = document.getElementById('powerBtn');
                document.getElementById('connectionStatus').className = "status-dot dot-green";
                if(d.settings.is_trading_enabled) {
                    btn.innerText = "RUNNING ⚡";
                    btn.style.background = "var(--green)";
                    btn.style.color = "#fff";
                } else {
                    btn.innerText = "STOPPED 🛑";
                    btn.style.background = "var(--red)";
                    btn.style.color = "#fff";
                }

                // 2. KPIs
                document.getElementById('regime').innerText = d.market.market_regime.replace(/_/g, ' ');
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
                // simple risk calc
                document.getElementById('openRisk').innerText = (d.signals.length * 2).toFixed(1); // 2% per trade assumption

                // 3. Charts
                if(d.stats.history.length > 0) {
                    equityChart.data.labels = d.stats.history.map(h => h.t);
                    equityChart.data.datasets[0].data = d.stats.history.map(h => h.v);
                    equityChart.update();
                    
                    statsChart.data.datasets[0].data = [d.stats.win_rate, 100 - d.stats.win_rate];
                    statsChart.update();
                }

                // 4. Trades Table
                const tb = document.getElementById('tradesBody');
                tb.innerHTML = d.signals.length ? d.signals.map(s => {
                    const curr = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                    return `
                    <tr>
                        <td style="font-weight:bold; color:var(--text)">${s.symbol}</td>
                        <td><span style="background:#2b3139; padding:2px 6px; border-radius:4px; font-size:11px">${s.strategy}</span></td>
                        <td>${s.entry_price}</td>
                        <td>${curr}</td>
                        <td class="${pnl>=0?'pnl-g':'pnl-r'}">${pnl.toFixed(2)}%</td>
                        <td style="font-size:11px; color:#848e9c">${s.tp1} ➔ ${s.tp2}</td>
                    </tr>`;
                }).join('') : "<tr><td colspan='6' style='text-align:center; padding:20px; color:#444'>No Active Positions</td></tr>";

                // 5. Logs
                document.getElementById('logsBody').innerHTML = d.logs.map(l => `
                    <tr>
                        <td style="color:#666">${l.t}</td>
                        <td style="font-weight:bold">${l.s}</td>
                        <td style="color:${l.st==='ENTRY'?'var(--green)':'#848e9c'}">${l.st}</td>
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
    logger.info("🖥️ SmartBot V12 Professional Interface on port 5000")
    app.run(host='0.0.0.0', port=5000)