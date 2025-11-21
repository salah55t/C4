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

# --- 1. System Configuration ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot_v10.log', encoding='utf-8'), logging.StreamHandler()]
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

# --- 2. Advanced Settings & State ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "trade_amount_usdt": 50.0,       # Increased for realistic simulation
    "max_open_trades": 4,            # Focus on quality over quantity
    "stop_loss_atr_multiplier": 1.8, # Tighter stops
    "trailing_atr_multiplier": 2.5,
    "volume_filter_limit": 50,       # Top 50 coins by volume
    "breakeven_trigger_pct": 1.5,    # Move SL to entry after 1.5% profit
    "max_hold_time_hours": 6         # Close if stagnant
}

LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
market_state = {
    "score": 50, "regime": "neutral", 
    "details": {"15m": 0, "1h": 0, "4h": 0},
    "trend_strength": "weak" # weak, strong, explosive
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=100) # Increased log size

locks = {
    'signals': Lock(), 'prices': Lock(), 'market': Lock(), 
    'settings': Lock(), 'logs': Lock()
}

# --- 3. Database Engine ---
conn = None
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals_v10 (
                    id SERIAL PRIMARY KEY, 
                    symbol TEXT NOT NULL, 
                    entry_price DOUBLE PRECISION, 
                    stop_loss DOUBLE PRECISION, 
                    target_price DOUBLE PRECISION,
                    quantity DOUBLE PRECISION, 
                    strategy_name TEXT, 
                    status TEXT DEFAULT 'open', 
                    mode TEXT, -- 'PAPER' or 'REAL'
                    entry_time TIMESTAMP DEFAULT NOW(),
                    closed_at TIMESTAMP, 
                    closing_price DOUBLE PRECISION, 
                    profit_pct DOUBLE PRECISION,
                    exit_reason TEXT, 
                    duration_minutes INTEGER
                );
            """)
        logger.info("✅ Elite Database Initialized.")
    except Exception as e: logger.error(f"DB Error: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 4. Pro Notification System ---
def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    mode_icon = "📝 PAPER" if payload.get('is_paper') else "💵 REAL"
    msg = ""

    if event == "BUY":
        msg = (
            f"🔵 *OPEN POSITION | {payload['symbol']}*\n"
            f"➖➖➖➖➖➖➖➖➖\n"
            f"🧩 Strategy: *{payload['strategy']}*\n"
            f"💰 Entry: `{payload['price']}`\n"
            f"🛡️ SL: `{payload['sl']}` | 🎯 TP: `{payload['tp']}`\n"
            f"📊 Market: *{payload['regime'].upper()}*\n"
            f"⚙️ Mode: {mode_icon}"
        )
    elif event == "SELL":
        pnl = payload['profit']
        icon = "🟢 PROFIT" if pnl > 0 else "🔴 LOSS"
        msg = (
            f"{icon} *CLOSE POSITION | {payload['symbol']}*\n"
            f"➖➖➖➖➖➖➖➖➖\n"
            f"📉 Exit Price: `{payload['price']}`\n"
            f"💸 PNL: `{pnl:.2f}%`\n"
            f"⏱️ Duration: {payload['duration']} min\n"
            f"📝 Reason: _{payload['reason']}_\n"
            f"⚙️ Mode: {mode_icon}"
        )
    elif event == "UPDATE":
        msg = (
            f"🛡️ *SL UPDATED | {payload['symbol']}*\n"
            f"New Stop Loss: `{payload['new_sl']}`\n"
            f"Reason: {payload['reason']}"
        )

    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except Exception as e: logger.error(f"Telegram Error: {e}")

# --- 5. Professional Technical Analysis (Pandas) ---
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
    # EMAs
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
    
    # Stochastic RSI (More sensitive)
    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi)
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ADX (Simplified for Pandas without TA-Lib)
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
    
    # Awesome Oscillator
    median_price = (df['high'] + df['low']) / 2
    df['ao'] = median_price.rolling(5).mean() - median_price.rolling(34).mean()

    return df.fillna(0)

# --- 6. THE ELITE STRATEGY ENGINE ---
def get_strategy_signal(symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 0. Pre-Filter: Volume & Volatility
    vol_ma = last['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 0.4 # Relaxed volume
    if not vol_ma: return None, "Low Relative Volume"

    # --- REGIME: BULLISH (Trend Trading) ---
    if "bull" in regime:
        # 1. The "Trend Sniper" (EMA + ADX + MACD)
        # Strong Trend confirmation
        if last['adx'] > 20: # Trend is present
            if last['close'] > last['ema200'] and last['ema9'] > last['ema21']:
                # Entry Trigger: MACD Cross OR AO Cross
                if (last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']) or \
                   (last['ao'] > 0 and prev['ao'] <= 0):
                    return "Trend_Sniper_Pro", "EMA Align + ADX > 20 + Momentum Trigger"

        # 2. Volatility Breakout (Squeeze)
        if prev['bb_width'] < 0.12: # Tighter squeeze
            if last['close'] > last['bb_upper'] and last['volume'] > prev['volume']:
                 return "Vol_Explosion", "BB Squeeze Breakout with Volume"

    # --- REGIME: SIDEWAYS (Mean Reversion) ---
    elif "sideways" in regime:
        # 3. Precision Reversion
        # Confluence: Price at Lower BB + RSI Oversold + Stoch Oversold
        if last['close'] < last['bb_lower']:
            if last['rsi'] < 32 and last['stoch_rsi'] < 0.2:
                return "Sniper_Reversion", "BB Lower + RSI < 32 + Stoch < 0.2"

    # --- REGIME: BEARISH (Scalping Only) ---
    elif "bearish" in regime:
        # 4. Dead Cat Bounce (Very Strict)
        if last['rsi'] < 22 and last['close'] < last['bb_lower'] * 0.98:
            return "Deep_Value_Scalp", "Extreme Oversold (RSI < 22)"

    # --- UNIVERSAL: Golden Cross ---
    if last['close'] > last['ema200']:
        if last['ema50'] > last['ema200'] and prev['ema50'] <= prev['ema200']:
             return "Golden_Cross_Major", "EMA 50/200 Cross"

    return None, "No Valid Setup"

# --- 7. Trade Management Logic (The "Fund Manager") ---
def manage_trade(symbol, signal, df, regime):
    last = df.iloc[-1]
    curr = float(last['close'])
    entry = float(signal['entry_price'])
    sl = float(signal['stop_loss'])
    
    profit_pct = (curr - entry) / entry * 100
    duration = (datetime.now() - signal['entry_time']).total_seconds() / 3600 # hours

    # 1. Breakeven Logic (Risk Free)
    if profit_pct >= BOT_SETTINGS['breakeven_trigger_pct'] and sl < entry:
        new_sl = entry * 1.001 # Slightly above entry to cover fees
        return "UPDATE_SL", new_sl, "Moved to Breakeven (Risk Free)"

    # 2. Trailing Stop (Protect Profits)
    if profit_pct >= 2.0:
        atr = last['atr']
        new_sl = curr - (atr * BOT_SETTINGS['trailing_atr_multiplier'])
        if new_sl > sl:
            return "UPDATE_SL", new_sl, "ATR Trailing Stop"

    # 3. Time-Based Exit (Opportunity Cost)
    if duration > BOT_SETTINGS['max_hold_time_hours'] and profit_pct < 0.5:
        return "CLOSE_NOW", curr, "Time Limit Exceeded (Stagnant)"

    # 4. Trend Reversal Panic Exit
    if regime == "bearish" and curr < last['ema50'] and profit_pct < -1:
        return "CLOSE_NOW", curr, "Regime Change Panic Exit"
    
    # 5. RSI Overbought Exit (For Reversion Trades)
    if "Reversion" in signal['strategy_name'] and last['rsi'] > 65:
         return "CLOSE_NOW", curr, "RSI Target Hit (Reversion)"

    return "HOLD", 0, ""

# --- 8. Market Analysis (Score Based) ---
def analyze_market_structure(client):
    global market_state
    tf_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.5}
    total_score = 0
    
    details = {}
    
    for sym in LEADING_SYMBOLS:
        sym_score = 0
        for tf, weight in tf_weights.items():
            df = fetch_data(client, sym, tf, 60)
            if df is None: continue
            
            # Indicator checks
            close = df['close'].iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            rsi = 100 - (100 / (1 + (df['close'].diff().where(lambda x: x>0, 0).rolling(14).mean() / -df['close'].diff().where(lambda x: x<0, 0).rolling(14).mean()).iloc[-1]))
            
            tf_points = 0
            if close > ema200: tf_points += 50
            if rsi > 50: tf_points += 50
            
            sym_score += tf_points * weight
            
        total_score += sym_score
    
    final_avg_score = total_score / len(LEADING_SYMBOLS)
    
    if final_avg_score >= 75: regime = "strong_bull"
    elif final_avg_score >= 55: regime = "bullish"
    elif final_avg_score <= 25: regime = "bearish"
    elif final_avg_score <= 45: regime = "weak_bear"
    else: regime = "sideways"
    
    with locks['market']:
        market_state = {
            "score": final_avg_score, 
            "regime": regime,
            "last_update": datetime.now()
        }
    
    logger.info(f"📊 Market Analysis: Score {final_avg_score:.1f} | Regime: {regime}")

# --- 9. Execution & Main Loops ---
def execute_trade(client, symbol, side, qty):
    # Paper trading always returns True immediately
    if BOT_SETTINGS['paper_trading_mode']: return True
    try:
        # Real execution logic (would need precision handling here)
        # client.create_order(...) 
        return True
    except Exception as e:
        logger.error(f"Binance Execution Error: {e}")
        return False

def bot_loop():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 SmartBot Elite Engine Started")
    
    # Load symbols
    try:
        with open('crypto_list.txt') as f:
            symbols = [l.strip().upper() for l in f if l.strip()]
            symbols = [s if s.endswith('USDT') else s+'USDT' for s in symbols]
    except: symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT']

    while True:
        try:
            # 1. Refresh Settings
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                limit = BOT_SETTINGS['volume_filter_limit']
                amt = BOT_SETTINGS['trade_amount_usdt']
                max_t = BOT_SETTINGS['max_open_trades']
                paper = BOT_SETTINGS['paper_trading_mode']
            
            if not enabled:
                time.sleep(10)
                continue

            # 2. Update Market Structure
            analyze_market_structure(client)
            with locks['market']: regime = market_state['regime']

            # 3. Manage Open Trades
            with locks['signals']: open_trades = list(open_signals_cache.values())
            
            for trade in open_trades:
                sym = trade['symbol']
                df = fetch_data(client, sym, '5m', 60)
                if df is None: continue
                df = add_indicators(df)
                
                curr_price = df['close'].iloc[-1]
                with locks['prices']: live_prices[sym] = curr_price
                
                # Check Stop Loss / Take Profit
                exit_reason = None
                if curr_price <= trade['stop_loss']: exit_reason = "Stop Loss Hit 🛑"
                elif curr_price >= trade['target_price']: exit_reason = "Take Profit Hit 🎯"
                
                if not exit_reason:
                    action, val, note = manage_trade(sym, trade, df, regime)
                    if action == "UPDATE_SL":
                        open_signals_cache[sym]['stop_loss'] = val
                        check_db()
                        with conn.cursor() as cur: 
                            cur.execute("UPDATE signals_v10 SET stop_loss=%s WHERE id=%s", (val, trade['id']))
                        send_telegram("UPDATE", {"symbol": sym, "new_sl": val, "reason": note, "is_paper": paper})
                    elif action == "CLOSE_NOW":
                        exit_reason = f"Smart Exit: {note}"

                if exit_reason:
                    execute_trade(client, sym, 'SELL', trade['quantity'])
                    close_trade_system(sym, curr_price, exit_reason, paper)

            # 4. Scan for New Trades (if slots available)
            if len(open_signals_cache) < max_t:
                # Get top volume coins
                tickers = client.get_ticker()
                valid_tickers = [t for t in tickers if t['symbol'] in symbols]
                valid_tickers.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                top_coins = valid_tickers[:limit]
                
                for t in top_coins:
                    if len(open_signals_cache) >= max_t: break
                    sym = t['symbol']
                    if sym in open_signals_cache: continue
                    
                    df = fetch_data(client, sym, '15m', 100) # Using 15m for Entry confirmation
                    if df is None: continue
                    df = add_indicators(df)
                    
                    strat, reason = get_strategy_signal(sym, df, regime)
                    
                    if strat:
                        curr = df['close'].iloc[-1]
                        atr = df['atr'].iloc[-1]
                        
                        # Calculate SL/TP
                        sl = curr - (atr * BOT_SETTINGS['stop_loss_atr_multiplier'])
                        tp = curr + (atr * 3.0) # 1:3 Risk Reward default
                        qty = amt / curr
                        
                        if execute_trade(client, sym, 'BUY', qty):
                            record_new_trade(sym, curr, sl, tp, qty, strat, paper)
                            logger.info(f"✅ ENTRY: {sym} | {strat}")
                    else:
                         if random.random() < 0.02: # Minimal logging
                            with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'SKIP', 'r': reason})
                    
                    time.sleep(0.5) # Avoid API limits

            time.sleep(30) # Main loop delay

        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            time.sleep(10)

# --- 10. System Helpers (DB & Cache) ---
def record_new_trade(symbol, price, sl, tp, qty, strat, is_paper):
    check_db()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals_v10 
                (symbol, entry_price, stop_loss, target_price, quantity, strategy_name, status, mode, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, 'open', %s, NOW())
                RETURNING id
            """, (symbol, price, sl, tp, qty, strat, mode))
            db_id = cur.fetchone()['id']
        
        # Add to cache
        trade_obj = {
            'id': db_id, 'symbol': symbol, 'entry_price': price, 
            'stop_loss': sl, 'target_price': tp, 'quantity': qty, 
            'strategy_name': strat, 'entry_time': datetime.now(),
            'is_paper': is_paper
        }
        with locks['signals']: open_signals_cache[symbol] = trade_obj
        
        # Add to logs
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'ENTRY', 'r': strat})
        
        # Notify
        send_telegram("BUY", {'symbol': symbol, 'strategy': strat, 'price': price, 'sl': sl, 'tp': tp, 'regime': market_state['regime'], 'is_paper': is_paper})
        
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
                UPDATE signals_v10 
                SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, exit_reason=%s, duration_minutes=%s
                WHERE id=%s
            """, (price, profit_pct, reason, duration, trade['id']))
            
        # Notify
        send_telegram("SELL", {'symbol': symbol, 'price': price, 'profit': profit_pct, 'reason': reason, 'duration': duration, 'is_paper': is_paper})
        
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'CLOSED', 'r': f"{profit_pct:.2f}%"})

    except Exception as e: logger.error(f"Close Trade Error: {e}")

# --- 11. Web Dashboard (Flask) ---
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
            cur.execute("SELECT closed_at, profit_pct FROM signals_v10 WHERE status='closed' ORDER BY closed_at ASC")
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
    <title>SmartBot Elite V10</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --card: #151a1e; --text: #eaecef; --green: #0ecb81; --red: #f6465d; --accent: #f0b90b; }
        body { background: var(--bg); color: var(--text); font-family: 'Cairo', sans-serif; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; margin-bottom: 20px; border-bottom: 1px solid #2b3139; padding-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: var(--card); padding: 20px; border-radius: 8px; border: 1px solid #2b3139; }
        .btn { background: var(--accent); color: #000; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; }
        .status-badge { padding: 5px 10px; border-radius: 4px; font-size: 0.8em; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        th, td { padding: 10px; text-align: right; border-bottom: 1px solid #2b3139; }
        .val-green { color: var(--green); } .val-red { color: var(--red); }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h2>💎 SmartBot <span style="color:var(--accent)">Elite V10</span></h2>
            <small>Portfolio Manager Logic • Sniper Strategies</small>
        </div>
        <div>
            <button id="toggleBtn" class="btn" onclick="toggleBot()">--</button>
        </div>
    </div>

    <div class="grid" style="margin-bottom: 20px;">
        <div class="card">
            <h3>📊 حالة السوق</h3>
            <h1 id="regime" style="color:var(--accent)">--</h1>
            <p>Score: <b id="score">0</b>/100</p>
        </div>
        <div class="card">
            <h3>💰 إحصائيات الأداء</h3>
            <div style="display:flex; justify-content:space-between;">
                <span>الصفقات الرابحة: <b id="wins" style="color:var(--green)">0</b></span>
                <span>الخاسرة: <b id="losses" style="color:var(--red)">0</b></span>
            </div>
            <h2 id="totalPnl">0.00%</h2>
        </div>
        <div class="card">
            <h3>📈 الرسم البياني (Equity Curve)</h3>
            <div style="height:100px"><canvas id="pnlChart"></canvas></div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>⚡ الصفقات المفتوحة</h3>
            <table>
                <thead><tr><th>العملة</th><th>الاستراتيجية</th><th>الدخول</th><th>السعر الحالي</th><th>الربح %</th></tr></thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>
        <div class="card">
            <h3>📝 سجل العمليات (Logs)</h3>
            <div style="height:300px; overflow-y:auto;">
                <table>
                    <thead><tr><th>الوقت</th><th>العملة</th><th>الحدث</th><th>تفاصيل</th></tr></thead>
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
                data: { labels: [], datasets: [{ label: 'PNL %', data: [], borderColor: '#f0b90b', tension: 0.4, borderWidth: 2 }] },
                options: { plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } }, maintainAspectRatio: false }
            });
        }

        async function update() {
            const res = await fetch('/api/data');
            const d = await res.json();

            // Header & Settings
            const btn = document.getElementById('toggleBtn');
            btn.innerText = d.settings.enabled ? "إيقاف البوت 🛑" : "تشغيل البوت 🚀";
            btn.style.background = d.settings.enabled ? "var(--red)" : "var(--green)";
            btn.style.color = "#fff";

            // Market
            document.getElementById('regime').innerText = d.market.regime.toUpperCase().replace('_', ' ');
            document.getElementById('score').innerText = d.market.score.toFixed(1);

            // Stats
            document.getElementById('wins').innerText = d.stats.wins;
            document.getElementById('losses').innerText = d.stats.losses;
            const tpnl = document.getElementById('totalPnl');
            tpnl.innerText = d.stats.total_pnl.toFixed(2) + "%";
            tpnl.className = d.stats.total_pnl >= 0 ? "val-green" : "val-red";

            // Chart
            chart.data.labels = d.history.map(h => h.date);
            chart.data.datasets[0].data = d.history.map(h => h.pnl);
            chart.update();

            // Trades
            const tb = document.getElementById('tradesBody');
            tb.innerHTML = d.signals.map(s => {
                const curr = d.prices[s.symbol] || s.entry_price;
                const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                return `<tr>
                    <td><b>${s.symbol}</b></td>
                    <td><small>${s.strategy_name}</small></td>
                    <td>${s.entry_price}</td>
                    <td>${curr}</td>
                    <td class="${pnl>=0?'val-green':'val-red'}">${pnl.toFixed(2)}%</td>
                </tr>`;
            }).join('');

            // Logs
            document.getElementById('logsBody').innerHTML = d.logs.map(l => 
                `<tr><td>${l.t}</td><td>${l.s}</td><td>${l.st}</td><td><small>${l.r}</small></td></tr>`
            ).join('');
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