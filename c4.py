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
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('smart_bot_v13.log', encoding='utf-8'), logging.StreamHandler()]
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
    "base_capital": 1000.0,       # رأس المال الافتراضي
    "risk_per_trade_pct": 2.0,    # المخاطرة لكل صفقة
    "max_open_trades": 5,         
    "max_drawdown_protect": 10.0, 
    "volume_lookback": 50,
    "timeframe_analysis": "15m",  
    "timeframe_trend": "4h"       
}

# سيتم تحديث القائمة ديناميكياً
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']

# حالة النظام
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades_v13 (
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
        logger.info("✅ قاعدة البيانات جاهزة (V13 - Candlestick Enhanced).")
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
        msg = (
            f"🚀 *إشارة دخول مؤكدة | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"🌊 الاستراتيجية: `{payload['strategy']}`\n"
            f"🕯️ الشمعة: `{payload.get('candle_pattern', 'Generic')}`\n"
            f"☁️ السحابة: {payload['regime']}\n"
            f"💵 السعر: `{payload['price']}`\n"
            f"🛑 الوقف: `{payload['sl']}`\n"
            f"🎯 هدف أول: `{payload['tp1']}`\n"
            f"🕹️ الوضع: {mode_icon}"
        )
    elif event == "SELL":
        pnl = payload['profit']
        emoji = "✅ ربح" if pnl > 0 else "🔻 خسارة"
        msg = (
            f"{emoji} *إغلاق صفقة | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"📉 الخروج: `{payload['price']}`\n"
            f"💰 الصافي: `{pnl:.2f}%`\n"
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
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- 5. محرك التحليل الفني (Ichimoku & Elliott + Candles) ---
def fetch_data(client, symbol, interval, limit=130): 
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except: return None

def calculate_technical_indicators(df):
    df = df.copy()
    
    # 1. Ichimoku Cloud (Full)
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    df['chikou_span'] = df['close'].shift(-26)

    # 2. Elliott Wave Helper (Awesome Oscillator - AO)
    median_price = (df['high'] + df['low']) / 2
    df['ao'] = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    df['ao_prev'] = df['ao'].shift(1)

    # 3. Standard Indicators
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # RSI & ATR
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()

    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    return df

# --- 5.1 تحليل الشموع اليابانية (جديد) ---
def detect_bullish_pattern(df):
    """
    دالة للكشف عن أنماط الشموع الصعودية في آخر شمعة مكتملة
    Returns: (is_bullish, pattern_name)
    """
    try:
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # حساب جسم الشمعة والظلال
        body = abs(curr['close'] - curr['open'])
        upper_wick = curr['high'] - max(curr['close'], curr['open'])
        lower_wick = min(curr['close'], curr['open']) - curr['low']
        is_green = curr['close'] > curr['open']
        
        # 1. الابتلاع الشرائي (Bullish Engulfing)
        # شمعة سابقة حمراء، شمعة حالية خضراء تبتلع جسم السابقة بالكامل
        if (prev['close'] < prev['open']) and is_green:
            if (curr['open'] < prev['close']) and (curr['close'] > prev['open']):
                return True, "Bullish Engulfing (ابتلاع شرائي)"

        # 2. المطرقة (Hammer)
        # ظل سفلي طويل (ضعف الجسم على الأقل) وظل علوي صغير جداً
        if lower_wick >= (body * 2) and upper_wick <= (body * 0.5):
            # يفضل أن تكون خضراء أو حمراء في قاع
            return True, "Hammer (مطرقة)"

        # 3. خط الثقب (Piercing Line)
        # شمعة حمراء قوية ثم خضراء تفتح بفجوة هابطة وتغلق فوق منتصف الحمراء
        if (prev['close'] < prev['open']) and is_green:
            mid_point = prev['open'] - (abs(prev['open'] - prev['close']) / 2)
            if (curr['open'] < prev['low']) and (curr['close'] > mid_point):
                return True, "Piercing Line (خط الثقب)"

        # 4. ثلاثة جنود بيض (Three White Soldiers)
        # ثلاث شموع خضراء متتالية، كل واحدة تغلق أعلى من السابقة
        if (curr['close'] > curr['open']) and (prev['close'] > prev['open']) and (prev2['close'] > prev2['open']):
            if (curr['close'] > prev['close']) and (prev['close'] > prev2['close']):
                # تأكد أن الذيول العلوية ليست طويلة جداً
                if upper_wick < body: 
                    return True, "3 White Soldiers (جنود بيض)"

        # 5. شمعة زخم قوية (Strong Momentum Candle)
        # شمعة خضراء كبيرة تغلق قريباً جداً من الهاي (Marubozu-like)
        avg_body = abs(df['close'] - df['open']).rolling(10).mean().iloc[-1]
        if is_green and (body > avg_body * 1.5):
            if (curr['close'] - curr['low']) > (curr['high'] - curr['low']) * 0.85: # إغلاق في الربع العلوي
                return True, "Strong Momentum (زخم قوي)"

        return False, None
    except:
        return False, None

# --- 6. محلل بيئة السوق (Market Regime) ---
def analyze_market_regime(client):
    global system_state
    btc_df = fetch_data(client, 'BTCUSDT', '4h', 150)
    if btc_df is None: return

    btc_df = calculate_technical_indicators(btc_df)
    last = btc_df.iloc[-1]

    cloud_status = "Neutral"
    if last['close'] > last['senkou_span_a'] and last['close'] > last['senkou_span_b']:
        cloud_status = "Bull_Cloud" 
    elif last['close'] < last['senkou_span_a'] and last['close'] < last['senkou_span_b']:
        cloud_status = "Bear_Cloud" 
    else:
        cloud_status = "In_Cloud_Turbulence" 

    trend_strength = 0
    if last['ao'] > 0 and last['ao'] > last['ao_prev']: trend_strength = 1 
    elif last['ao'] < 0 and last['ao'] < last['ao_prev']: trend_strength = -1 

    regime = cloud_status
    
    with locks['market']:
        system_state['market_regime'] = regime
        system_state['trend_strength'] = trend_strength
        system_state['volatility_index'] = "Normal" 
        system_state['last_update'] = datetime.now()
    
    logger.info(f"🧠 حالة السوق: {regime} | AO: {trend_strength}")

# --- 7. مصنع الاستراتيجيات (Ichimoku & Elliott + Candle Confirmation) ---
def get_smart_signal(symbol, df, regime):
    if len(df) < 52: return None, "بيانات غير كافية", None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. فلتر السيولة
    if last['volume'] < last['vol_ma'] * 0.5:
        return None, "سيولة ضعيفة", None

    # 2. فحص أنماط الشموع (الفلتر الجديد)
    is_candle_valid, candle_pattern = detect_bullish_pattern(df)
    
    # إذا لم توجد شمعة تأكيد، لا تكمل التحليل (إلا في حالات خاصة جداً)
    if not is_candle_valid:
        return None, "انتظار تأكيد شمعة صاعدة", None

    # شروط الإيشيموكو الأساسية
    above_cloud = (last['close'] > last['senkou_span_a']) and (last['close'] > last['senkou_span_b'])
    tk_cross = (last['tenkan_sen'] >= last['kijun_sen']) # السماح بالتلامس أو التقاطع
    
    # --- الاستراتيجيات ---

    # أ) استراتيجية موجة 3 (Elliott Wave 3 Breakout)
    if above_cloud and tk_cross and last['ao'] > 0:
        if last['ao'] > last['ao_prev']: # تسارع الزخم
            if last['close'] > prev['high']: # تأكيد حركة السعر
                return "Elliott_Wave_3", "اختراق موجة 3 + " + candle_pattern, candle_pattern

    # ب) استراتيجية ارتداد موجة 4 (Elliott Wave 4 Pullback)
    # الارتداد من Kijun Sen أو سقف السحابة
    if above_cloud:
        # السعر لامس Kijun وارتد
        dist_to_kijun = abs(last['low'] - last['kijun_sen']) / last['close']
        if dist_to_kijun < 0.02 and last['close'] > last['kijun_sen']:
             # هنا الشمعة (مثل المطرقة) ضرورية جداً
             return "Elliott_Wave_4_Bounce", "ارتداد Kijun + " + candle_pattern, candle_pattern

    # ج) تقاطع TK قوي فوق السحابة (Strong TK Cross)
    if above_cloud and (prev['tenkan_sen'] <= prev['kijun_sen']) and (last['tenkan_sen'] > last['kijun_sen']):
        return "Ichimoku_TK_Cross", "تقاطع TK ذهبي + " + candle_pattern, candle_pattern

    return None, "لا توجد فرصة مؤكدة", None

# --- 8. مدير المحفظة والمخاطر ---
def manage_active_trade(symbol, signal, df):
    last = df.iloc[-1]
    curr = float(last['close'])
    entry = float(signal['entry_price'])
    tp1 = float(signal['tp1'])
    tp2 = float(signal['tp2'])
    sl = float(signal['stop_loss'])
    kijun = last['kijun_sen']
    
    profit_pct = (curr - entry) / entry * 100
    duration = (datetime.now() - signal['entry_time']).total_seconds() / 3600

    # جني الأرباح
    if curr >= tp2:
        return "UPDATE_SL", tp1, "تأمين ربح الهدف الأول", "ربح ممتاز 🟢"
    elif curr >= tp1:
        if sl < entry: return "UPDATE_SL", entry * 1.002, "Breakeven", "مؤمنة 🛡️"

    # وقف الخسارة المتحرك (Ichimoku Trailing)
    if profit_pct > 3.0:
        new_sl = kijun * 0.99
        if new_sl > sl:
             return "UPDATE_SL", new_sl, "Trailing Stop (Kijun)", "ملاحقة Kijun"

    # الخروج الفني (كسر السحابة)
    if curr < last['senkou_span_b'] and curr < last['senkou_span_a']:
         return "CLOSE_NOW", curr, "كسر السحابة للأسفل", "خطر ⚠️"

    # وقف الوقت
    if duration > 12 and profit_pct < 1.0:
         return "CLOSE_NOW", curr, "Time Stop", "راكد"

    return "HOLD", 0, "", "مستمر"

# --- 9. المحرك الرئيسي ---
def bot_engine():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 SmartBot V13 (Elliott + Ichimoku + Candles) Started")
    
    try:
        symbols = LEADING_SYMBOLS
        try:
            tickers = client.get_ticker()
            valid = [t for t in tickers if t['symbol'].endswith('USDT')]
            valid.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            symbols = [x['symbol'] for x in valid[:25]] 
            logger.info(f"✅ تم تحميل {len(symbols)} عملة للتحليل")
        except Exception as e:
            logger.warning(f"⚠️ استخدام القائمة الاحتياطية: {e}")

    except Exception as e: 
        logger.error(f"Initialization Error: {e}")
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

    while True:
        try:
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                paper = BOT_SETTINGS['paper_trading_mode']
                max_t = BOT_SETTINGS['max_open_trades']
                
            if not enabled: 
                time.sleep(10)
                continue

            analyze_market_regime(client)
            with locks['market']: regime = system_state['market_regime']
            
            # السماح بالتداول فقط إذا لم يكن السوق في حالة انهيار شديد
            trading_allowed = True
            
            # إدارة الصفقات
            with locks['signals']: active_trades = list(open_signals_cache.values())
            
            for trade in active_trades:
                sym = trade['symbol']
                df = fetch_data(client, sym, '1h', 60)
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
                            cur.execute("UPDATE trades_v13 SET stop_loss=%s WHERE id=%s", (float(val), trade['id']))
                        send_telegram("UPDATE", {"symbol": sym, "new_sl": val, "reason": note})
                        
                    elif act == "CLOSE_NOW":
                        exit_reason = f"خروج فني: {note}"
                
                if exit_reason:
                    close_trade_final(sym, curr_price, exit_reason, paper)
                time.sleep(0.5)

            # البحث (Scanning)
            if len(open_signals_cache) < max_t and trading_allowed:
                for sym in symbols:
                    with locks['signals']: 
                        if len(open_signals_cache) >= max_t: break
                        if sym in open_signals_cache: continue
                    
                    df = fetch_data(client, sym, BOT_SETTINGS['timeframe_analysis'], 100)
                    if df is None: 
                        time.sleep(0.2)
                        continue
                        
                    df = calculate_technical_indicators(df)
                    
                    # استدعاء الدالة المحدثة التي ترجع 3 قيم
                    strat, reason, candle_pat = get_smart_signal(sym, df, regime)
                    
                    if strat:
                        curr = df['close'].iloc[-1]
                        atr = df['atr'].iloc[-1]
                        
                        support_level = min(df['senkou_span_a'].iloc[-1], df['senkou_span_b'].iloc[-1], df['kijun_sen'].iloc[-1])
                        sl = min(support_level, curr - (atr * 1.5))
                        
                        if (curr - sl) / curr > 0.05: sl = curr * 0.95
                        
                        risk = curr - sl
                        tp1 = curr + (risk * 1.5)
                        tp2 = curr + (risk * 3.0)
                        
                        risk_amt = BOT_SETTINGS['base_capital'] * (BOT_SETTINGS['risk_per_trade_pct'] / 100)
                        qty = risk_amt / risk if risk > 0 else 0
                        
                        if qty * curr > BOT_SETTINGS['base_capital'] * 0.25:
                            qty = (BOT_SETTINGS['base_capital'] * 0.25) / curr
                            
                        open_new_trade(sym, curr, sl, tp1, tp2, qty, strat, regime, paper, candle_pat)
                        time.sleep(1) 
                    else:
                        if random.random() < 0.02:
                             with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'فحص', 'r': reason})
                    
                    time.sleep(0.8) 

            logger.info("💤 انتهاء دورة البحث، انتظار 20 ثانية...")
            time.sleep(20)

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            time.sleep(10)

# --- 10. أدوات قاعدة البيانات ---
def open_new_trade(symbol, price, sl, tp1, tp2, qty, strat, regime, is_paper, candle_pat="None"):
    check_db()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        price, sl, tp1, tp2, qty = float(price), float(sl), float(tp1), float(tp2), float(qty)
        
        # إضافة اسم الشمعة للاستراتيجية للتخزين
        full_strat_name = f"{strat} | {candle_pat}"

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v13 
                (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, status, mode, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, NOW())
                RETURNING id
            """, (symbol, price, sl, tp1, tp2, qty, full_strat_name, regime, mode))
            db_id = cur.fetchone()['id']
        
        trade = {
            'id': db_id, 'symbol': symbol, 'entry_price': price, 'stop_loss': sl,
            'tp1': tp1, 'tp2': tp2, 'quantity': qty, 'entry_time': datetime.now(),
            'strategy': strat, 'market_regime': regime, 'is_paper': is_paper,
            'candle_pattern': candle_pat
        }
        
        with locks['signals']: open_signals_cache[symbol] = trade
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'دخول', 'r': strat})
        send_telegram("BUY", trade) # تم تمرير trade كاملة بما فيها candle_pattern
        
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

        price = float(price)
        profit_pct = ((price - trade['entry_price']) / trade['entry_price']) * 100
        profit_abs = (price - trade['entry_price']) * trade['quantity']
        duration = int((datetime.now() - trade['entry_time']).total_seconds() / 60)

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades_v13 
                SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, exit_reason=%s
                WHERE id=%s
            """, (price, profit_pct, profit_abs, reason, trade['id']))
            
        send_telegram("SELL", {'symbol': symbol, 'price': price, 'profit': profit_pct, 'reason': reason, 'duration': duration})
        
    except Exception as e: logger.error(f"DB Close Error: {e}")

# --- 11. واجهة التحكم (Flask) ---
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
    
    stats = {'win_rate': 0, 'profit_factor': 0, 'total_pnl_usd': 0, 'trade_count': 0, 'history': []}
    try:
        check_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT closed_at, profit_pct, profit_abs 
                FROM trades_v13 WHERE status='closed' ORDER BY closed_at ASC
            """)
            rows = cur.fetchall()
            wins, gross_profit, gross_loss, cum_pnl = 0, 0, 0, 0
            for r in rows:
                if r['profit_pct'] > 0: wins += 1; gross_profit += r['profit_abs']
                else: gross_loss += abs(r['profit_abs'])
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

# HTML Dashboard (نفس الواجهة)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot V13 - Candle Confirmation</title>
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
        .btn:hover { opacity: 0.9; }
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
            <h1 style="margin:0; font-size:24px">SmartBot <span style="color:var(--accent)">V13 Candles</span></h1>
            <span style="font-size:12px; color:#848e9c">نظام إدارة المحفظة (إليوت + إيشيموكو + شموع)</span>
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
            <h3>حالة سحابة الإيشيموكو</h3>
            <div id="regime" class="big-num" style="color:var(--accent); font-size:20px">--</div>
            <div class="sub-text">زخم AO: <span id="trendStr">0</span></div>
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
            <h3>المحفظة النشطة</h3>
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

        const regimeMap = {
            "Bull_Cloud": "سحابة صاعدة (شراء) 🟢",
            "Bear_Cloud": "سحابة هابطة (بيع) 🔴",
            "In_Cloud_Turbulence": "داخل السحابة (تذبذب) ☁️",
            "Neutral": "محايد ⚖️"
        };
        const stratMap = {
            "Elliott_Wave_3": "اختراق موجة 3 🚀",
            "Elliott_Wave_4_Bounce": "ارتداد موجة 4 🛡️",
            "Ichimoku_TK_Cross": "تقاطع TK ذهبي ✨"
        };

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
                        <td><span style="background:#2b3139; padding:2px 6px; border-radius:4px; font-size:11px">${stratMap[s.strategy] || s.strategy}</span></td>
                        <td>${s.entry_price}</td>
                        <td>${curr}</td>
                        <td class="${pnl>=0?'pnl-g':'pnl-r'}">${pnl.toFixed(2)}%</td>
                        <td style="font-size:11px; color:#848e9c">${s.tp1} ➔ ${s.tp2}</td>
                    </tr>`;
                }).join('') : "<tr><td colspan='6' style='text-align:center; padding:20px; color:#444'>لا توجد صفقات نشطة حالياً</td></tr>";

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
        initCharts(); setInterval(updateData, 2000); updateData();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    Thread(target=bot_engine, daemon=True).start()
    logger.info("🖥️ لوحة التحكم العربية تعمل على المنفذ 5000")
    app.run(host='0.0.0.0', port=5000)