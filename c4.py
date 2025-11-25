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
    handlers=[logging.FileHandler('smart_bot_v15.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Arab')

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ في الإعدادات: {e}")
    exit(1)

# --- 2. إعدادات التداول (المحفظة الذكية) ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,       
    "risk_per_trade_pct": 2.0,    
    "max_open_trades": 6,         
    "min_volume_24h": 10000000,
    "max_spread_pct": 0.2,
    "min_signal_score": 70,
    "trading_fee_pct": 0.1,       # (جديد) نسبة عمولة المنصة (0.1% لباينانس)
    "timeframe_analysis": "15m",
    "timeframe_trend": "1h"
}

LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']

system_state = {
    "market_regime": "Neutral",
    "trend_strength": 0,
    "volatility_index": "Low",
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

# --- 3. قاعدة البيانات ---
conn = None
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        with conn.cursor() as cur:
            # تم تحديث الجدول إلى V15 ليشمل العمولات وصافي الربح
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades_v15 (
                    id SERIAL PRIMARY KEY, 
                    symbol TEXT NOT NULL, 
                    entry_price DOUBLE PRECISION, 
                    stop_loss DOUBLE PRECISION, 
                    tp1 DOUBLE PRECISION,
                    tp2 DOUBLE PRECISION,
                    quantity DOUBLE PRECISION, 
                    strategy_name TEXT, 
                    market_regime TEXT,
                    signal_score INTEGER,
                    status TEXT DEFAULT 'open', 
                    mode TEXT,
                    entry_time TIMESTAMP DEFAULT NOW(),
                    closed_at TIMESTAMP, 
                    closing_price DOUBLE PRECISION, 
                    profit_abs DOUBLE PRECISION, 
                    profit_pct DOUBLE PRECISION, 
                    total_fees DOUBLE PRECISION,   -- (جديد) إجمالي العمولات
                    net_profit_abs DOUBLE PRECISION, -- (جديد) الربح الصافي كرقم
                    net_profit_pct DOUBLE PRECISION, -- (جديد) الربح الصافي كنسبة
                    exit_reason TEXT
                );
            """)
        logger.info("✅ قاعدة البيانات جاهزة (V15 مع حساب العمولات).")
    except Exception as e: logger.error(f"خطأ قاعدة البيانات: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 4. نظام التنبيهات العربي ---
def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    mode_icon = "🧪 تجريبي" if payload.get('is_paper') else "💰 حقيقي"
    msg = ""

    if event == "BUY":
        score_bar = "▓" * (payload.get('score', 50) // 10) + "░" * (10 - (payload.get('score', 50) // 10))
        msg = (
            f"🚀 *إشارة دخول جديدة | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"📊 الاستراتيجية: `{payload['strategy']}`\n"
            f"⭐️ الجودة: {payload.get('score', 0)}/100\n"
            f"{score_bar}\n"
            f"💵 السعر: `{payload['price']}`\n"
            f"🛑 الوقف: `{payload['sl']}`\n"
            f"🎯 الأهداف: `{payload['tp1']}` ➔ `{payload['tp2']}`\n"
            f"🕹️ الوضع: {mode_icon}"
        )
    elif event == "SELL":
        net_pnl = payload.get('net_profit_pct', 0)
        fees = payload.get('fees', 0)
        emoji = "✅ ربح صافي" if net_pnl > 0 else "🔻 خسارة صافية"
        msg = (
            f"{emoji} *إغلاق صفقة | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"📉 سعر الخروج: `{payload['price']}`\n"
            f"💰 العائد الصافي: `{net_pnl:.2f}%`\n"
            f"💸 العمولات: `{fees:.4f}$`\n"
            f"📝 السبب: _{payload['reason']}_\n"
            f"⏱️ المدة: {payload['duration']} دقيقة"
        )
    elif event == "UPDATE":
        msg = (
            f"🛡️ *تحديث وقف الخسارة | {payload['symbol']}*\n"
            f"الوقف الجديد: `{payload['new_sl']}`\n"
            f"السبب: {payload['reason']}"
        )

    try:
        Thread(target=requests.post, args=(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",), 
               kwargs={"data": {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, "timeout": 5}).start()
    except Exception as e: logger.error(f"Telegram Error: {e}")

# --- 5. محرك التحليل الفني ---
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
    # المتوسطات
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # RSI & Stochastic
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_k'] = ((df['rsi'] - min_rsi) / (max_rsi - min_rsi)) * 100
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ADX & ATR
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

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # Volume Indicators
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['tp'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    # Ichimoku Tenkan
    high_9 = df['high'].rolling(9).max()
    low_9 = df['low'].rolling(9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    return df.fillna(0)

def get_trend_bias(client, symbol):
    try:
        df_trend = fetch_data(client, symbol, BOT_SETTINGS['timeframe_trend'], 50)
        if df_trend is None: return "Neutral"
        last = df_trend.iloc[-1]
        ema50 = last['close'].ewm(span=50).mean().iloc[-1] 
        ema200 = last['close'].ewm(span=200).mean().iloc[-1]
        
        if last['close'] > ema50 and ema50 > ema200: return "Bullish"
        elif last['close'] < ema50 and ema50 < ema200: return "Bearish"
        return "Neutral"
    except: return "Neutral"

# --- 6. نظام تقييم الإشارة ---
def calculate_signal_score(row, trend_bias, strategy_type):
    score = 50
    if strategy_type in ["Trend_Pullback", "Momentum_Breakout", "Golden_Cross"]:
        if trend_bias == "Bullish": score += 20
        elif trend_bias == "Bearish": score -= 20
    
    if row['volume'] > row['vol_ma'] * 1.5: score += 10
    if row['close'] > row['vwap']: score += 5
    if 40 < row['rsi'] < 70: score += 5
    if row['adx'] > 25: score += 5
    if row['macd_hist'] > 0 and row['macd'] > 0: score += 5
    
    return min(100, max(0, score))

# --- 7. مصنع الاستراتيجيات ---
def get_smart_signal(client, symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    trend_bias = get_trend_bias(client, symbol)
    
    strategy = None
    reason = ""

    # 1. استراتيجية سحابة الاتجاه
    if "Bull" in regime or trend_bias == "Bullish":
        if last['close'] > last['ema50'] and last['adx'] > 20:
            if last['obv'] > df['obv'].iloc[-5]: 
                if last['low'] <= last['tenkan_sen'] and last['close'] > last['tenkan_sen']:
                    strategy = "Trend_Pullback"
                    reason = "إعادة دخول مع الاتجاه"
                elif last['close'] > prev['high'] and last['macd_hist'] > 0 and last['volume'] > last['vol_ma']:
                    strategy = "Momentum_Breakout"
                    reason = "اختراق زخم مع فوليوم"

    # 2. استراتيجية القناص
    elif "Ranging" in regime:
        if last['bb_width'] < 0.10: 
            if last['rsi'] < 40 and last['stoch_k'] < 20:
                if last['close'] > prev['close'] and last['close'] > last['vwap']:
                    strategy = "Sniper_Reversion"
                    reason = "ارتداد من القاع مع دعم VWAP"

    # 3. التقاطع الذهبي
    if last['ema50'] > last['ema200'] and prev['ema50'] <= prev['ema200']:
         strategy = "Golden_Cross"
         reason = "تقاطع ذهبي كلاسيكي"

    if strategy:
        score = calculate_signal_score(last, trend_bias, strategy)
        if score >= BOT_SETTINGS['min_signal_score']:
            return strategy, reason, score
        else:
            return None, f"تقييم ضعيف ({score})", 0

    return None, "لا توجد فرصة", 0

# --- 8. مدير المحفظة ---
def manage_active_trade(symbol, signal, df):
    last = df.iloc[-1]
    curr = float(last['close'])
    entry = float(signal['entry_price'])
    tp1 = float(signal['tp1'])
    tp2 = float(signal['tp2'])
    sl = float(signal['stop_loss'])
    
    # حساب الربح الخام (بدون عمولة) للعرض السريع
    profit_pct = (curr - entry) / entry * 100
    duration = (datetime.now() - signal['entry_time']).total_seconds() / 3600
    health_msg = "مستقر"

    if curr < last['vwap'] and profit_pct < -1.0:
        health_msg = "تحت VWAP ⚠️"

    if curr >= tp2:
        if sl < tp1: return "UPDATE_SL", tp1, "تأمين ربح الهدف الأول", "ربح ممتاز 🟢"
    elif curr >= tp1:
        if sl < entry: return "UPDATE_SL", entry * 1.002, "صفقة خالية من المخاطر", "مؤمنة 🛡️"

    if profit_pct > 2.0:
        atr_trail = curr - (last['atr'] * 2.0)
        if atr_trail > sl: return "UPDATE_SL", atr_trail, "ملاحقة الأرباح (ATR)", "منطلق 🏃"

    if duration > 6 and abs(profit_pct) < 0.5:
        return "CLOSE_NOW", curr, "تجميد رأس المال (خروج زمني)", "راكد ⚠️"

    return "HOLD", 0, "", health_msg

def analyze_market_regime(client):
    global system_state
    btc_df = fetch_data(client, 'BTCUSDT', '4h', 100)
    if btc_df is None: return
    btc_df = calculate_technical_indicators(btc_df)
    last = btc_df.iloc[-1]
    
    trend_score = 0
    if last['close'] > last['ema200']: trend_score += 1
    if last['macd'] > last['macd_signal']: trend_score += 1
    if last['obv'] > btc_df['obv'].iloc[-10]: trend_score += 1 
    
    adx = last['adx']
    regime = "Neutral"
    
    if trend_score == 3 and adx > 25: regime = "Bull_Trend_Strong"
    elif trend_score >= 2: regime = "Bull_Accumulation"
    elif trend_score == 0 and adx > 25: regime = "Bear_Trend_Strong"
    else: regime = "Ranging"

    with locks['market']:
        system_state['market_regime'] = regime
        system_state['trend_strength'] = int(adx)
        system_state['last_update'] = datetime.now()

# --- 9. المحرك الرئيسي ---
def bot_engine():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 SmartBot V15 Engine Started")
    
    while True:
        try:
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                paper = BOT_SETTINGS['paper_trading_mode']
                max_t = BOT_SETTINGS['max_open_trades']
                min_vol = BOT_SETTINGS['min_volume_24h']
                max_spread = BOT_SETTINGS['max_spread_pct']
                
            if not enabled: 
                time.sleep(5)
                continue

            analyze_market_regime(client)
            with locks['market']: regime = system_state['market_regime']

            # إدارة الصفقات
            with locks['signals']: active_trades = list(open_signals_cache.values())
            for trade in active_trades:
                sym = trade['symbol']
                df = fetch_data(client, sym, '5m', 60)
                if df is None: continue
                df = calculate_technical_indicators(df)
                
                curr_price = df['close'].iloc[-1]
                with locks['prices']: live_prices[sym] = curr_price
                
                exit_reason = None
                if curr_price <= trade['stop_loss']: exit_reason = "ضرب وقف الخسارة 🛑"
                
                if not exit_reason:
                    act, val, note, health = manage_active_trade(sym, trade, df)
                    if act == "UPDATE_SL":
                        open_signals_cache[sym]['stop_loss'] = float(val)
                        check_db()
                        with conn.cursor() as cur:
                            cur.execute("UPDATE trades_v15 SET stop_loss=%s WHERE id=%s", (float(val), trade['id']))
                        send_telegram("UPDATE", {"symbol": sym, "new_sl": val, "reason": note})
                    elif act == "CLOSE_NOW":
                        exit_reason = f"خروج ذكي: {note}"
                
                if exit_reason:
                    close_trade_final(sym, curr_price, exit_reason, paper)

            # البحث عن فرص
            if len(open_signals_cache) < max_t:
                tickers = client.get_ticker()
                valid_candidates = []
                for t in tickers:
                    if not t['symbol'].endswith('USDT'): continue
                    
                    quote_vol = float(t['quoteVolume'])
                    if quote_vol < min_vol: continue 
                    
                    bid = float(t['bidPrice'])
                    ask = float(t['askPrice'])
                    if ask == 0: continue
                    spread = ((ask - bid) / ask) * 100
                    if spread > max_spread: continue
                    
                    valid_candidates.append(t)
                
                valid_candidates.sort(key=lambda x: float(x['quoteVolume']) * abs(float(x['priceChangePercent'])), reverse=True)
                
                count = 0
                for t in valid_candidates:
                    if count > 15: break
                    
                    sym = t['symbol']
                    if sym in open_signals_cache: continue
                    
                    df = fetch_data(client, sym, BOT_SETTINGS['timeframe_analysis'], 100)
                    if df is None: continue
                    df = calculate_technical_indicators(df)
                    
                    strat, reason, score = get_smart_signal(client, sym, df, regime)
                    
                    if strat:
                        count += 1
                        curr = df['close'].iloc[-1]
                        atr = df['atr'].iloc[-1]
                        
                        sl = curr - (atr * 2.0)
                        tp1 = curr + (atr * 2.0)
                        tp2 = curr + (atr * 5.0)
                        
                        risk_amt = BOT_SETTINGS['base_capital'] * (BOT_SETTINGS['risk_per_trade_pct'] / 100)
                        qty = risk_amt / (curr - sl) if (curr - sl) > 0 else 0
                        
                        if qty * curr > BOT_SETTINGS['base_capital'] * 0.2:
                            qty = (BOT_SETTINGS['base_capital'] * 0.2) / curr
                            
                        open_new_trade(sym, curr, sl, tp1, tp2, qty, strat, regime, score, paper)
                        time.sleep(1)
                    else:
                         if random.random() < 0.05:
                             with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'فحص', 'r': reason})
                    
                    time.sleep(0.5)

            time.sleep(10)

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            time.sleep(10)

# --- 10. أدوات قاعدة البيانات (مع حساب العمولات) ---
def open_new_trade(symbol, price, sl, tp1, tp2, qty, strat, regime, score, is_paper):
    check_db()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        price, sl, tp1, tp2, qty = float(price), float(sl), float(tp1), float(tp2), float(qty)
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v15 
                (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, signal_score, status, mode, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, NOW())
                RETURNING id
            """, (symbol, price, sl, tp1, tp2, qty, strat, regime, score, mode))
            db_id = cur.fetchone()['id']
        
        trade = {
            'id': db_id, 'symbol': symbol, 'entry_price': price, 'stop_loss': sl,
            'tp1': tp1, 'tp2': tp2, 'quantity': qty, 'entry_time': datetime.now(),
            'strategy': strat, 'market_regime': regime, 'score': score, 'is_paper': is_paper
        }
        
        with locks['signals']: open_signals_cache[symbol] = trade
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'دخول', 'r': f"{strat} ({score})"})
        send_telegram("BUY", {**trade, 'price': price, 'sl': sl})
        
    except Exception as e: logger.error(f"DB Insert Error: {e}")

def close_trade_final(symbol, price, reason, is_paper):
    """
    إغلاق الصفقة وحساب العمولات وصافي الربح
    """
    check_db()
    try:
        trade = None
        with locks['signals']:
            if symbol in open_signals_cache:
                trade = open_signals_cache[symbol]
                del open_signals_cache[symbol]
        if not trade: return

        exit_price = float(price)
        entry_price = trade['entry_price']
        qty = trade['quantity']
        fee_rate = BOT_SETTINGS['trading_fee_pct'] / 100.0  # 0.1% = 0.001

        # 1. حساب القيم الأساسية
        entry_cost = entry_price * qty
        exit_value = exit_price * qty
        
        # 2. حساب العمولات (دخول + خروج)
        # ملاحظة: في التداول الفوري الحقيقي، يتم خصم العمولة من العملة، ولكن هنا نخصمها من القيمة الدولارية للتبسيط
        entry_fee = entry_cost * fee_rate
        exit_fee = exit_value * fee_rate
        total_fees = entry_fee + exit_fee

        # 3. حساب الأرباح
        gross_profit_abs = exit_value - entry_cost # الربح قبل العمولة
        net_profit_abs = gross_profit_abs - total_fees # الربح الصافي (بعد العمولة)
        
        # نسبة الربح الصافي مقارنة برأس المال المستثمر
        net_profit_pct = (net_profit_abs / entry_cost) * 100
        
        # نسبة الربح الخام (للأرشفة فقط)
        gross_profit_pct = (gross_profit_abs / entry_cost) * 100

        duration = int((datetime.now() - trade['entry_time']).total_seconds() / 60)

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades_v15 
                SET status='closed', closed_at=NOW(), closing_price=%s, 
                    profit_pct=%s, profit_abs=%s, 
                    total_fees=%s, net_profit_abs=%s, net_profit_pct=%s,
                    exit_reason=%s
                WHERE id=%s
            """, (exit_price, gross_profit_pct, gross_profit_abs, total_fees, net_profit_abs, net_profit_pct, reason, trade['id']))
            
        send_telegram("SELL", {
            'symbol': symbol, 'price': exit_price, 
            'profit': gross_profit_pct, 'net_profit_pct': net_profit_pct,
            'fees': total_fees,
            'reason': reason, 'duration': duration
        })
        
    except Exception as e: logger.error(f"DB Close Error: {e}")

# --- 11. واجهة التحكم (تحديث العرض ليشمل الصافي) ---
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
    
    stats = {'win_rate': 0, 'total_pnl_usd': 0, 'net_pnl_usd': 0, 'trade_count': 0, 'fees_paid': 0, 'history': []}
    try:
        check_db()
        with conn.cursor() as cur:
            # جلب البيانات من الجدول الجديد
            cur.execute("""
                SELECT closed_at, net_profit_pct, net_profit_abs, total_fees 
                FROM trades_v15 WHERE status='closed' ORDER BY closed_at ASC
            """)
            rows = cur.fetchall()
            
            wins = sum(1 for r in rows if r['net_profit_pct'] > 0)
            stats['trade_count'] = len(rows)
            stats['total_pnl_usd'] = sum(r['net_profit_abs'] for r in rows) # الآن يعرض الصافي
            stats['fees_paid'] = sum(r['total_fees'] for r in rows)
            stats['win_rate'] = (wins / len(rows) * 100) if len(rows) > 0 else 0
            
    except: pass
    
    return jsonify({"market": m, "signals": s, "prices": p, "stats": stats, "logs": l, "settings": BOT_SETTINGS})

@app.route('/api/toggle', methods=['POST'])
def toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify("OK")

@app.route('/api/close/<symbol>', methods=['POST'])
def manual_close(symbol):
    with locks['signals']:
        if symbol not in open_signals_cache: return jsonify("Not found"), 404
    curr = live_prices.get(symbol, 0)
    if curr == 0: return jsonify("Price error"), 400
    close_trade_final(symbol, curr, "إغلاق يدوي من اللوحة 👤", BOT_SETTINGS['paper_trading_mode'])
    return jsonify("Closed")

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    with locks['settings']:
        for k, v in data.items():
            if k in BOT_SETTINGS:
                if isinstance(BOT_SETTINGS[k], float): BOT_SETTINGS[k] = float(v)
                elif isinstance(BOT_SETTINGS[k], int): BOT_SETTINGS[k] = int(v)
                else: BOT_SETTINGS[k] = v
    return jsonify("Updated")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot V15 - لوحة التحكم</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --panel: #151a1e; --border: #2b3139; --text: #eaecef; --green: #0ecb81; --red: #f6465d; --accent: #f0b90b; }
        * { box-sizing: border-box; }
        body { background: var(--bg); color: var(--text); font-family: 'Tajawal', sans-serif; margin: 0; padding: 20px; font-size: 14px; text-align: right; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 15px; }
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; margin-bottom: 20px; }
        .col-3 { grid-column: span 3; } .col-4 { grid-column: span 4; } .col-8 { grid-column: span 8; } .col-12 { grid-column: span 12; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
        .btn { background: var(--accent); color: #000; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: inherit; margin-left: 5px; }
        .btn-red { background: var(--red); color: white; }
        .btn-sm { padding: 4px 10px; font-size: 11px; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; }
        .modal-content { background: var(--panel); margin: 10% auto; padding: 20px; width: 50%; border-radius: 8px; border: 1px solid var(--border); }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #848e9c; }
        .form-group input { width: 100%; padding: 8px; background: var(--bg); border: 1px solid var(--border); color: white; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: right; padding: 12px; border-bottom: 1px solid var(--border); }
        .score-badge { background: #333; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
        .fee-info { font-size: 10px; color: #666; display: block; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SmartBot <span style="color:var(--accent)">V15</span> <small style="font-size:12px; color:#666">مع حساب العمولات</small></h1>
        <div>
            <button class="btn" onclick="openSettings()">⚙️ الإعدادات</button>
            <button id="powerBtn" class="btn" onclick="toggleBot()">تحميل...</button>
        </div>
    </div>

    <div class="grid">
        <div class="card col-3">
            <h3>حالة السوق</h3>
            <div id="regime" style="color:var(--accent); font-size:18px; font-weight:bold">--</div>
        </div>
        <div class="card col-3">
            <h3>صافي الأرباح (Net PnL)</h3>
            <div id="totalPnl" style="font-size:24px; font-weight:bold">$0.00</div>
            <span class="fee-info">بعد خصم العمولات</span>
        </div>
        <div class="card col-3">
            <h3>العمولات المدفوعة</h3>
            <div id="feesPaid" style="font-size:18px; color:#848e9c">$0.00</div>
        </div>
    </div>

    <div class="grid">
        <div class="card col-12">
            <h3>الصفقات النشطة</h3>
            <table>
                <thead>
                    <tr>
                        <th>العملة</th>
                        <th>الاستراتيجية</th>
                        <th>الجودة</th>
                        <th>السعر</th>
                        <th>الربح العائم %</th>
                        <th>إجراء</th>
                    </tr>
                </thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>
    </div>
    
    <div class="grid">
        <div class="card col-12">
            <h3>سجل النظام</h3>
            <div id="logsBody" style="font-family:monospace; color:#848e9c; max-height:200px; overflow-y:auto"></div>
        </div>
    </div>

    <div id="settingsModal" class="modal">
        <div class="modal-content">
            <h2>⚙️ تعديل الإعدادات</h2>
            <div class="form-group">
                <label>أقصى عدد صفقات (Max Trades)</label>
                <input type="number" id="set_max_trades">
            </div>
            <div class="form-group">
                <label>تقييم الدخول (Min Score)</label>
                <input type="number" id="set_min_score">
            </div>
            <div class="form-group">
                <label>المخاطرة لكل صفقة %</label>
                <input type="number" id="set_risk">
            </div>
             <div class="form-group">
                <label>عمولة المنصة % (Trading Fee)</label>
                <input type="number" step="0.01" id="set_fee">
            </div>
            <button class="btn" onclick="saveSettings()">حفظ وتطبيق</button>
            <button class="btn btn-red" onclick="document.getElementById('settingsModal').style.display='none'">إلغاء</button>
        </div>
    </div>

    <script>
        async function updateData() {
            const res = await fetch('/api/analytics');
            const d = await res.json();
            
            const btn = document.getElementById('powerBtn');
            if(d.settings.is_trading_enabled) {
                btn.innerText = "إيقاف البوت 🛑"; btn.style.background = "var(--red)";
            } else {
                btn.innerText = "تشغيل البوت 🚀"; btn.style.background = "var(--green)";
            }

            document.getElementById('regime').innerText = d.market.market_regime;
            
            // عرض صافي الربح والعمولات
            const pnl = d.stats.total_pnl_usd;
            const fees = d.stats.fees_paid;
            
            document.getElementById('totalPnl').innerText = "$" + pnl.toFixed(2);
            document.getElementById('totalPnl').style.color = pnl >= 0 ? "var(--green)" : "var(--red)";
            document.getElementById('feesPaid').innerText = "$" + fees.toFixed(2);

            document.getElementById('tradesBody').innerHTML = d.signals.map(s => {
                const curr = d.prices[s.symbol] || s.entry_price;
                // حساب تقريبي للربح العائم (شامل العمولة المتوقعة للدخول والخروج)
                // Fee approx = 2 * fee_pct (دخول وخروج)
                const fee_pct_total = (d.settings.trading_fee_pct || 0.1) * 2;
                const gross_pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                const net_pnl_est = gross_pnl - fee_pct_total;
                
                let scoreColor = s.score >= 80 ? '#0ecb81' : (s.score >= 60 ? '#f0b90b' : '#f6465d');
                
                return `
                <tr>
                    <td><b>${s.symbol}</b></td>
                    <td>${s.strategy}</td>
                    <td><span class="score-badge" style="color:${scoreColor}">${s.score || 0}</span></td>
                    <td>${curr}</td>
                    <td>
                        <span style="color:${gross_pnl>=0?'var(--green)':'var(--red)'}">${gross_pnl.toFixed(2)}%</span>
                        <br><span style="font-size:10px; color:#666">صافي: ${net_pnl_est.toFixed(2)}%</span>
                    </td>
                    <td><button class="btn btn-red btn-sm" onclick="closeTrade('${s.symbol}')">خروج 🔴</button></td>
                </tr>`;
            }).join('');
            
            document.getElementById('logsBody').innerHTML = d.logs.map(l => 
                `<div><span style="color:var(--accent)">[${l.t}]</span> ${l.s}: ${l.st} - ${l.r}</div>`
            ).join('');
            
            window.currentSettings = d.settings;
        }

        function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(updateData); }
        
        function closeTrade(symbol) {
            if(confirm("هل أنت متأكد من الخروج اليدوي؟ سيتم احتساب عمولة الخروج.")) {
                fetch('/api/close/'+symbol, {method:'POST'}).then(updateData);
            }
        }

        function openSettings() {
            const s = window.currentSettings;
            if(!s) return;
            document.getElementById('set_max_trades').value = s.max_open_trades;
            document.getElementById('set_min_score').value = s.min_signal_score;
            document.getElementById('set_risk').value = s.risk_per_trade_pct;
            document.getElementById('set_fee').value = s.trading_fee_pct || 0.1;
            document.getElementById('settingsModal').style.display = 'block';
        }

        function saveSettings() {
            const data = {
                max_open_trades: document.getElementById('set_max_trades').value,
                min_signal_score: document.getElementById('set_min_score').value,
                risk_per_trade_pct: document.getElementById('set_risk').value,
                trading_fee_pct: document.getElementById('set_fee').value
            };
            fetch('/api/update_settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            }).then(() => {
                document.getElementById('settingsModal').style.display = 'none';
                updateData();
            });
        }

        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    Thread(target=bot_engine, daemon=True).start()
    logger.info("🖥️ لوحة التحكم العربية (V15) تعمل على المنفذ 5000")
    app.run(host='0.0.0.0', port=5000)