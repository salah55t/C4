import time
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import random
import re
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
logger = logging.getLogger('SmartBot_Pro')

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ في الإعدادات: {e}")
    exit(1)

# --- 2. إعدادات التداول المتقدمة ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,
    "risk_per_trade_pct": 2.0,    # المخاطرة من رأس المال لكل صفقة
    "max_open_trades": 5,
    "min_usdt_volume": 10000000,  # الحد الأدنى لحجم التداول اليومي (10 مليون)
    "atr_multiplier_sl": 1.5,     # معامل ATR لوقف الخسارة
    "atr_multiplier_tp": 2.5      # معامل ATR للهدف
}

# المتغيرات العامة
system_state = {
    "market_regime": "Neutral",      # Bull_Strong, Bull_Weak, Neutral, Bear_Weak, Bear_Strong
    "market_score": 0,               # من -100 إلى +100
    "btc_dominance_trend": "Flat",
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
                    highest_price DOUBLE PRECISION  -- لتتبع الوقف المتحرك
                );
            """)
        logger.info("✅ قاعدة البيانات جاهزة (V14 - Advanced Logic).")
    except Exception as e: logger.error(f"خطأ قاعدة البيانات: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: init_db()

# --- 4. نظام التنبيهات (تم إصلاح مشاكل الإرسال) ---
def clean_markdown(text):
    """تنظيف النص لتجنب أخطاء مارك داون في تلغرام"""
    return str(text).replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    try:
        mode_icon = "🧪 تجريبي" if payload.get('is_paper') else "💰 حقيقي"
        msg = ""
        symbol_safe = clean_markdown(payload.get('symbol', 'UNKNOWN'))

        if event == "BUY":
            strategy_safe = clean_markdown(payload['strategy'])
            msg = (
                f"🚀 *إشارة شراء جديدة | {symbol_safe}*\n"
                f"ــــــــــــــــــــــــــــــــــ\n"
                f"📊 الاستراتيجية: `{strategy_safe}`\n"
                f"🌍 حالة السوق: {payload['regime']}\n"
                f"💵 الدخول: `{payload['price']}`\n"
                f"🛑 الوقف: `{payload['sl']}`\n"
                f"🎯 الهدف: `{payload['tp2']}`\n"
                f"⚖️ المخاطرة: `{payload.get('risk_usd', 0):.2f}$`\n"
                f"🕹️ الوضع: {mode_icon}"
            )
        elif event == "SELL":
            pnl = payload['profit']
            emoji = "✅ ربح" if pnl > 0 else "🔻 خسارة"
            msg = (
                f"{emoji} *إغلاق صفقة | {symbol_safe}*\n"
                f"ــــــــــــــــــــــــــــــــــ\n"
                f"📉 سعر الخروج: `{payload['price']}`\n"
                f"💰 النسبة: `{pnl:.2f}%`\n"
                f"📝 السبب: {clean_markdown(payload['reason'])}\n"
                f"⏱️ المدة: {payload['duration']} دقيقة"
            )
        elif event == "UPDATE":
            msg = (
                f"🛡️ *تحديث وقف متحرك | {symbol_safe}*\n"
                f"الوقف الجديد: `{payload['new_sl']}`\n"
                f"الربح الحالي: `{payload.get('current_profit', 0):.2f}%`"
            )

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        
        resp = requests.post(url, data=data, timeout=5)
        if resp.status_code != 200:
            logger.error(f"Telegram Send Error: {resp.text}")
            
    except Exception as e:
        logger.error(f"خطأ في إرسال تلغرام: {e}")

# --- 5. التحليل الفني المتقدم (Deep Analysis) ---
def fetch_klines(client, symbol, interval, limit=100): 
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
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Ichimoku
    high_9 = df['high'].rolling(9).max(); low_9 = df['low'].rolling(9).min()
    df['tenkan'] = (high_9 + low_9) / 2
    high_26 = df['high'].rolling(26).max(); low_26 = df['low'].rolling(26).min()
    df['kijun'] = (high_26 + low_26) / 2
    df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
    high_52 = df['high'].rolling(52).max(); low_52 = df['low'].rolling(52).min()
    df['span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    return df

# --- 6. تحليل هيكل السوق (الرموز القيادية) ---
def analyze_market_structure_advanced(client):
    """
    يحلل العملات الأربعة الكبرى على 3 فريمات زمنية لتحديد الاتجاه العام بدقة
    """
    global system_state
    leaders = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    timeframes = ['15m', '4h', '1d']
    weights = {'15m': 1, '4h': 2, '1d': 3} # الفريم الأكبر له وزن أكبر
    
    total_score = 0
    max_score = len(leaders) * sum(weights.values()) # تطبيع النتيجة
    
    logger.info("📡 جاري تحليل عمق السوق (Market Depth Analysis)...")
    
    for sym in leaders:
        for tf in timeframes:
            df = fetch_klines(client, sym, tf, limit=60)
            if df is None: continue
            df = add_indicators(df)
            last = df.iloc[-1]
            
            # منطق النقاط لكل فريم
            score = 0
            # 1. السعر فوق EMA200 (اتجاه عام)
            if last['close'] > last['ema_200']: score += 0.5
            else: score -= 0.5
            
            # 2. السعر فوق السحابة (إيشيموكو)
            if last['close'] > last['span_a'] and last['close'] > last['span_b']: score += 0.5
            elif last['close'] < last['span_a'] and last['close'] < last['span_b']: score -= 0.5
            
            total_score += (score * weights[tf])

    # تحويل النتيجة إلى نسبة مئوية (-100% إلى +100%)
    normalized_score = (total_score / max_score) * 100
    
    regime = "Neutral"
    if normalized_score >= 60: regime = "Bull_Strong"
    elif normalized_score >= 20: regime = "Bull_Weak"
    elif normalized_score <= -60: regime = "Bear_Strong"
    elif normalized_score <= -20: regime = "Bear_Weak"
    
    with locks['market']:
        system_state['market_regime'] = regime
        system_state['market_score'] = normalized_score
        system_state['last_update'] = datetime.now()
        
    logger.info(f"🧠 نتيجة تحليل السوق: {regime} ({normalized_score:.1f}%)")

# --- 7. اختيار الرموز الذكي (Dynamic Selection) ---
def filter_best_symbols(client):
    """
    يرشح أفضل العملات بناءً على الحجم والزخم بالنسبة للسعر
    """
    try:
        tickers = client.get_ticker()
        valid_symbols = []
        
        min_vol = BOT_SETTINGS['min_usdt_volume']
        
        for t in tickers:
            symbol = t['symbol']
            if not symbol.endswith('USDT'): continue
            
            quote_vol = float(t['quoteVolume'])
            price_change = float(t['priceChangePercent'])
            last_price = float(t['lastPrice'])
            
            # 1. تصفية العملات ذات السيولة الضعيفة
            if quote_vol < min_vol: continue
            
            # 2. استبعاد العملات المستقرة (Stablecoins)
            if 'USD' in symbol.replace('USDT', ''): continue
            
            # 3. حساب "نقاط الجودة" (Quality Score)
            # نفضل: حجم عالي + حركة سعرية إيجابية (للشراء)
            # أو حجم عالي + تقلب (Volatility)
            score = quote_vol * (1 + (abs(price_change) / 100))
            
            valid_symbols.append({
                'symbol': symbol,
                'score': score,
                'change': price_change,
                'price': last_price
            })
            
        # ترتيب حسب النقاط واختيار أفضل 30
        valid_symbols.sort(key=lambda x: x['score'], reverse=True)
        return [x['symbol'] for x in valid_symbols[:30]]
        
    except Exception as e:
        logger.error(f"Symbol Filter Error: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT']

# --- 8. استراتيجيات ديناميكية حسب حالة السوق ---
def get_strategy_signal(symbol, df, regime):
    if len(df) < 55: return None, None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # تحييد القرارات بناءً على السوق
    allow_long = "Bull" in regime or regime == "Neutral"
    # allow_short = "Bear" in regime # (غير مفعل حاليا لأن البوت سبوت)
    
    signal = None
    strategy_name = None
    
    # --- استراتيجية 1: اختراق الزخم القوي (Trend Follow) ---
    # مناسبة للسوق الصاعد القوي
    if regime == "Bull_Strong" and allow_long:
        if last['close'] > last['ema_50'] and last['close'] > last['ema_200']:
            if last['rsi'] > 50 and last['rsi'] < 75: # ليس متشبعاً جداً
                if last['close'] > last['span_a'] and last['close'] > last['span_b']: # فوق السحابة
                    if last['volume'] > df['volume'].mean() * 1.5: # تأكيد فوليوم
                         signal = "BUY"
                         strategy_name = "Strong_Trend_Breakout"

    # --- استراتيجية 2: الارتداد من الدعم (Dip Buy) ---
    # مناسبة للسوق الصاعد الضعيف أو المحايد
    elif (regime == "Bull_Weak" or regime == "Neutral") and allow_long:
        # السعر قريب من EMA50 أو Kijun
        dist_ema = abs(last['close'] - last['ema_50']) / last['close']
        if dist_ema < 0.015 and last['close'] > last['ema_50']:
            if last['rsi'] < 45: # تشبع بيعي خفيف في ترند صاعد
                 signal = "BUY"
                 strategy_name = "EMA50_Bounce"

    # --- استراتيجية 3: اختراق الكثافة (Volatility Pop) ---
    # تعمل في أي سوق بشرط وجود فوليوم استثنائي
    vol_spike = last['volume'] > df['volume'].rolling(50).mean().iloc[-1] * 3
    if vol_spike and last['close'] > last['open'] and allow_long:
        signal = "BUY"
        strategy_name = "Volume_Spike_Scalp"

    return signal, strategy_name

# --- 9. إدارة الصفقات (Logic Update + Trailing) ---
def manage_trade_logic(trade, current_price, df):
    """
    منطق محدث للوقف المتحرك الديناميكي
    """
    entry = trade['entry_price']
    sl = trade['stop_loss']
    tp1 = trade['tp1']
    highest = trade.get('highest_price', entry)
    
    # تحديث أعلى سعر وصل له الرمز
    if current_price > highest:
        highest = current_price
        # تحديث في قاعدة البيانات (في الذاكرة فقط هنا للأداء، وقاعدة البيانات لاحقاً)
        trade['highest_price'] = highest

    profit_pct = (current_price - entry) / entry * 100
    drawdown_from_peak = (highest - current_price) / highest * 100
    
    atr = df['atr'].iloc[-1]
    
    action = "HOLD"
    new_sl = sl
    reason = ""

    # 1. تحديث الوقف المتحرك (Trailing Stop)
    # إذا تجاوز الربح 1.5%، حرك الوقف لنقطة الدخول (Breakeven)
    if profit_pct > 1.5 and sl < entry:
        new_sl = entry * 1.002 # زيادة بسيطة لتغطية العمولات
        action = "UPDATE_SL"
        reason = "Breakeven Activation"

    # إذا تجاوز الربح 3%، ابدأ بملاحقة السعر بمسافة 2 ATR
    elif profit_pct > 3.0:
        proposed_sl = current_price - (atr * 2.0)
        if proposed_sl > sl:
            new_sl = proposed_sl
            action = "UPDATE_SL"
            reason = "ATR Trailing"

    # 2. جني الأرباح (Take Profit)
    if current_price >= trade['tp2']:
        return "CLOSE", "Target 2 Hit 🎯", 0
    
    # 3. وقف الخسارة (Stop Loss)
    if current_price <= sl:
        return "CLOSE", "Stop Loss Hit 🛑", 0

    # 4. الخروج الذكي (Smart Exit)
    # إذا كسر السعر الدعم المهم بقوة (إغلاق تحت EMA50) والصفقة رابحة قليلاً
    if current_price < df['ema_50'].iloc[-1] and profit_pct > 0.5:
         return "CLOSE", "Trend Broken (EMA50)", 0

    return action, reason, new_sl

# --- 10. المحرك الرئيسي ---
def bot_engine():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 SmartBot V14 Engine Started...")
    
    # دورة تحديث حالة السوق (كل 15 دقيقة تقريباً)
    last_market_analysis = datetime.now() - timedelta(minutes=20)
    # دورة تحديث قائمة الرموز (كل ساعة)
    last_symbol_update = datetime.now() - timedelta(hours=2)
    active_symbols = []

    while True:
        try:
            # 1. تحديث الإعدادات وحالة السوق
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                max_trades = BOT_SETTINGS['max_open_trades']
                is_paper = BOT_SETTINGS['paper_trading_mode']

            if not enabled:
                time.sleep(5)
                continue

            now = datetime.now()
            
            # تحديث حالة السوق (الرموز القيادية)
            if (now - last_market_analysis).seconds > 900: # 15 min
                analyze_market_structure_advanced(client)
                last_market_analysis = now
            
            # تحديث قائمة الرموز المرشحة
            if (now - last_symbol_update).seconds > 3600 or not active_symbols:
                active_symbols = filter_best_symbols(client)
                logger.info(f"📋 تم تحديث قائمة المراقبة: {len(active_symbols)} رمز")
                last_symbol_update = now

            with locks['market']: regime = system_state['market_regime']

            # 2. إدارة الصفقات المفتوحة
            with locks['signals']: trades_copy = list(open_signals_cache.values())
            
            for trade in trades_copy:
                sym = trade['symbol']
                df = fetch_klines(client, sym, '15m', limit=60) # فريم سريع للإدارة
                if df is None: continue
                df = add_indicators(df)
                
                curr_price = float(df['close'].iloc[-1])
                with locks['prices']: live_prices[sym] = curr_price
                
                # استدعاء منطق الإدارة المحدث
                action, reason, val = manage_trade_logic(trade, curr_price, df)
                
                if action == "CLOSE":
                    close_trade(trade['id'], sym, curr_price, reason, is_paper)
                
                elif action == "UPDATE_SL":
                    # تحديث في الذاكرة وقاعدة البيانات وتلغرام
                    open_signals_cache[sym]['stop_loss'] = val
                    open_signals_cache[sym]['highest_price'] = max(trade.get('highest_price', 0), curr_price)
                    
                    check_db()
                    with conn.cursor() as cur:
                        cur.execute("UPDATE trades_v14 SET stop_loss=%s, highest_price=%s WHERE id=%s", 
                                   (val, open_signals_cache[sym]['highest_price'], trade['id']))
                    
                    p_pct = (curr_price - trade['entry_price']) / trade['entry_price'] * 100
                    send_telegram("UPDATE", {"symbol": sym, "new_sl": val, "reason": reason, "current_profit": p_pct})

                time.sleep(0.5)

            # 3. البحث عن فرص جديدة (Scanning)
            if len(open_signals_cache) < max_trades and regime != "Bear_Strong":
                for sym in active_symbols:
                    with locks['signals']:
                        if len(open_signals_cache) >= max_trades: break
                        if sym in open_signals_cache: continue
                    
                    # تحليل العملة
                    df = fetch_klines(client, sym, '1h', limit=100)
                    if df is None: continue
                    df = add_indicators(df)
                    
                    sig, strat_name = get_strategy_signal(sym, df, regime)
                    
                    if sig == "BUY":
                        price = df['close'].iloc[-1]
                        atr = df['atr'].iloc[-1]
                        
                        # حساب الأهداف والوقف بناءً على ATR
                        sl_dist = atr * BOT_SETTINGS['atr_multiplier_sl']
                        tp_dist = atr * BOT_SETTINGS['atr_multiplier_tp']
                        
                        sl = price - sl_dist
                        tp1 = price + (tp_dist * 0.6) # هدف أول قريب
                        tp2 = price + tp_dist
                        
                        # حساب الكمية بناءً على المخاطرة
                        risk_per_share = price - sl
                        if risk_per_share <= 0: continue
                        
                        capital = BOT_SETTINGS['base_capital']
                        risk_amt = capital * (BOT_SETTINGS['risk_per_trade_pct'] / 100)
                        qty = risk_amt / risk_per_share
                        
                        # سقف للكمية (لا يزيد عن 20% من المحفظة)
                        max_pos_size = capital * 0.20
                        if (qty * price) > max_pos_size:
                            qty = max_pos_size / price

                        execute_trade(sym, price, sl, tp1, tp2, qty, strat_name, regime, is_paper, risk_amt)
                        time.sleep(1) # تفادي ضغط API
            
            time.sleep(10) # انتظار قليل بين الدورات

        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            time.sleep(5)

# --- 11. تنفيذ العمليات ---
def execute_trade(symbol, price, sl, tp1, tp2, qty, strat, regime, is_paper, risk_usd):
    check_db()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v14 
                (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, status, mode, highest_price)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, %s)
                RETURNING id
            """, (symbol, price, sl, tp1, tp2, qty, strat, regime, mode, price))
            db_id = cur.fetchone()['id']
        
        trade_obj = {
            'id': db_id, 'symbol': symbol, 'entry_price': price, 'stop_loss': sl,
            'tp1': tp1, 'tp2': tp2, 'quantity': qty, 'strategy': strat,
            'market_regime': regime, 'entry_time': datetime.now(), 'highest_price': price,
            'is_paper': is_paper
        }
        
        with locks['signals']: open_signals_cache[symbol] = trade_obj
        
        # إرسال التنبيه مع تفاصيل المخاطرة
        send_telegram("BUY", {**trade_obj, "regime": regime, "risk_usd": risk_usd})
        
        with locks['logs']: 
            scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'دخول', 'r': strat})

    except Exception as e: logger.error(f"Execution Error: {e}")

def close_trade(db_id, symbol, price, reason, is_paper):
    check_db()
    try:
        trade = None
        with locks['signals']:
            if symbol in open_signals_cache:
                trade = open_signals_cache[symbol]
                del open_signals_cache[symbol]
        
        if not trade: return

        profit_pct = ((price - trade['entry_price']) / trade['entry_price']) * 100
        profit_abs = (price - trade['entry_price']) * trade['quantity']
        duration = int((datetime.now() - trade['entry_time']).total_seconds() / 60)

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades_v14
                SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, exit_reason=%s
                WHERE id=%s
            """, (price, profit_pct, profit_abs, reason, db_id))
            
        send_telegram("SELL", {'symbol': symbol, 'price': price, 'profit': profit_pct, 'reason': reason, 'duration': duration})
        logger.info(f"Closed {symbol} PnL: {profit_pct:.2f}% Reason: {reason}")

    except Exception as e: logger.error(f"Closing Error: {e}")

# --- 12. واجهة الويب (Flask) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/close_trade/<symbol>', methods=['POST'])
def manual_close(symbol):
    with locks['settings']: is_paper = BOT_SETTINGS['paper_trading_mode']
    with locks['signals']:
        if symbol in open_signals_cache:
            trade_id = open_signals_cache[symbol]['id']
            # نغلق الصفقة في خيط منفصل لتجنب تجميد الواجهة
            Thread(target=close_trade, args=(trade_id, symbol, live_prices.get(symbol, 0), "Manual Close 👤", is_paper)).start()
            return jsonify({"status": "success", "message": f"Closing {symbol}..."})
    return jsonify({"status": "error", "message": "Symbol not found"}), 404

@app.route('/api/data')
def get_data():
    with locks['market']: m = system_state.copy()
    with locks['signals']: s = [{k: v for k, v in t.items() if k != 'entry_time'} for t in open_signals_cache.values()]
    with locks['prices']: p = live_prices.copy()
    with locks['logs']: l = list(scan_logs)
    return jsonify({"market": m, "trades": s, "prices": p, "logs": l, "settings": BOT_SETTINGS})

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>SmartBot Pro V14</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { background: #0f172a; color: #e2e8f0; font-family: 'Tajawal'; padding: 20px; }
        .card { background: #1e293b; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #334155; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: right; border-bottom: 1px solid #334155; }
        .btn-close { background: #ef4444; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; }
        .status-bull { color: #10b981; } .status-bear { color: #ef4444; }
    </style>
</head>
<body>
    <div class="header">
        <h1>SmartBot <span style="color:#8b5cf6">Pro V14</span></h1>
    </div>

    <div class="grid">
        <div class="card">
            <h3>حالة السوق (Market Regime)</h3>
            <h2 id="regimeText">--</h2>
            <p>Score: <span id="regimeScore">0</span>%</p>
        </div>
        <div class="card">
            <h3>الصفقات المفتوحة</h3>
            <h2 id="openCount">0</h2>
        </div>
    </div>

    <div class="card">
        <h3>الصفقات النشطة</h3>
        <table>
            <thead>
                <tr>
                    <th>العملة</th>
                    <th>الاستراتيجية</th>
                    <th>دخول</th>
                    <th>حالياً</th>
                    <th>P&L</th>
                    <th>إجراء</th>
                </tr>
            </thead>
            <tbody id="tradesTable"></tbody>
        </table>
    </div>

    <script>
        function update() {
            fetch('/api/data').then(r => r.json()).then(d => {
                const r = d.market.market_regime;
                const el = document.getElementById('regimeText');
                el.innerText = r;
                el.className = r.includes('Bull') ? 'status-bull' : (r.includes('Bear') ? 'status-bear' : '');
                document.getElementById('regimeScore').innerText = d.market.market_score.toFixed(1);
                document.getElementById('openCount').innerText = d.trades.length;

                const tbody = document.getElementById('tradesTable');
                tbody.innerHTML = d.trades.map(t => {
                    const curr = d.prices[t.symbol] || t.entry_price;
                    const pnl = ((curr - t.entry_price) / t.entry_price) * 100;
                    return `<tr>
                        <td><b>${t.symbol}</b></td>
                        <td>${t.strategy}</td>
                        <td>${t.entry_price}</td>
                        <td>${curr}</td>
                        <td style="color:${pnl>=0?'#10b981':'#ef4444'}">${pnl.toFixed(2)}%</td>
                        <td><button class="btn-close" onclick="closeTrade('${t.symbol}')">إغلاق</button></td>
                    </tr>`;
                }).join('');
            });
        }
        
        function closeTrade(sym) {
            if(confirm('هل أنت متأكد من إغلاق ' + sym + ' يدوياً؟')) {
                fetch('/api/close_trade/'+sym, {method:'POST'});
            }
        }
        
        setInterval(update, 2000);
        update();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    # تشغيل البوت في خيط منفصل
    t = Thread(target=bot_engine)
    t.daemon = True
    t.start()
    
    logger.info("🖥️ Starting Web Server...")
    app.run(host='0.0.0.0', port=5000)