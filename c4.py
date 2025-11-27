import time
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
from threading import Thread, Lock
from datetime import datetime, timedelta
from collections import deque
from decouple import config
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- 1. إعدادات النظام المتقدمة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('smart_bot_v14.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Pro')

# تكوين الاتصال
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ فادح في الإعدادات: {e}")
    exit(1)

# --- 2. إعدادات التداول المتقدمة وإدارة المخاطر ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,     # تفعيل الوضع الوهمي للتجربة
    "base_capital": 1000.0,         # رأس المال المخصص
    "risk_per_trade_pct": 1.5,      # المخاطرة بالنسبة المئوية من رأس المال
    "max_open_trades": 5,           # عدد الصفقات المتزامنة
    "leverage": 1,                  # الرافعة (1 تعني سبوت)
    "exchange_fee_rate": 0.001,     # عمولة المنصة (0.1% للسبوت)
    "timeframe_analysis": "15m",    # إطار التحليل للدخول
    "timeframe_trend": "4h",        # إطار تحديد الاتجاه العام
    "use_candlestick_confirm": True # تفعيل تأكيد الشموع
}

# قوائم المراقبة
IGNORED_SYMBOLS = ['USDCUSDT', 'TUSDUSDT', 'FDUSDUSDT', 'EURUSDT']

# حالة النظام المتغيرة
system_state = {
    "market_regime": "Neutral",
    "btc_trend": "Unknown",
    "volatility_status": "Normal",
    "last_api_call": 0
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=200)

locks = {
    'signals': Lock(), 'prices': Lock(), 'market': Lock(), 
    'settings': Lock(), 'logs': Lock(), 'db': Lock()
}

# --- 3. إدارة قاعدة البيانات ---
conn = None
def get_db_connection():
    global conn
    try:
        if conn is None or conn.closed != 0:
            conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
            conn.autocommit = True
    except Exception as e:
        logger.error(f"خطأ اتصال قاعدة البيانات: {e}")
    return conn

def init_db():
    try:
        c = get_db_connection()
        with c.cursor() as cur:
            # تم تحديث الجدول ليشمل العمولات وتفاصيل أكثر دقة
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades_v14 (
                    id SERIAL PRIMARY KEY, 
                    symbol TEXT NOT NULL, 
                    entry_price DOUBLE PRECISION, 
                    stop_loss DOUBLE PRECISION, 
                    tp1 DOUBLE PRECISION,
                    tp2 DOUBLE PRECISION,
                    tp3_moonbag DOUBLE PRECISION,
                    quantity DOUBLE PRECISION, 
                    strategy_name TEXT, 
                    market_structure TEXT,
                    status TEXT DEFAULT 'open', 
                    mode TEXT,
                    entry_time TIMESTAMP DEFAULT NOW(),
                    closed_at TIMESTAMP, 
                    closing_price DOUBLE PRECISION, 
                    gross_profit DOUBLE PRECISION,
                    net_profit DOUBLE PRECISION, -- بعد خصم العمولة
                    fees_paid DOUBLE PRECISION,
                    exit_reason TEXT,
                    max_price_reached DOUBLE PRECISION -- لتتبع أقصى ربح وصلته الصفقة
                );
            """)
        logger.info("✅ تم تهيئة قاعدة البيانات الاحترافية (V14).")
    except Exception as e: logger.error(f"خطأ تهيئة قاعدة البيانات: {e}")

# --- 4. محرك التحليل الفني المتقدم (Core Logic) ---

# إدارة طلبات API لتجنب الحظر (Rate Limiting)
def safe_api_request(func, *args, **kwargs):
    now = time.time()
    # تأخير بسيط ديناميكي
    if now - system_state['last_api_call'] < 0.1:
        time.sleep(0.1)
    
    try:
        res = func(*args, **kwargs)
        system_state['last_api_call'] = time.time()
        return res
    except BinanceAPIException as e:
        logger.warning(f"Binance API Warning: {e}")
        if "Too many requests" in str(e):
            time.sleep(60) # عقوبة ذاتية لتجنب الحظر الطويل
        return None
    except Exception as e:
        logger.error(f"API Error: {e}")
        return None

def fetch_data(client, symbol, interval, limit=100):
    klines = safe_api_request(client.get_historical_klines, symbol, interval, limit=limit)
    if not klines: return None
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# مكتبة التحليل الفني اليدوية (بدون الاعتماد على مكتبات خارجية ثقيلة)
def calculate_advanced_indicators(df):
    df = df.copy()
    
    # 1. المتوسطات
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # 2. ATR (للمخاطر)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # 3. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 4. ADX (لقوة الاتجاه)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    df['plus_di'] = 100 * (pd.Series(plus_dm).rolling(14).mean() / df['atr'])
    df['minus_di'] = 100 * (pd.Series(minus_dm).rolling(14).mean() / df['atr'])
    dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.rolling(14).mean()
    
    # 5. تحديد هيكل السوق (Market Structure) - محلي
    # تحديد القمم والقيعان المحلية (Local Extrema) لآخر 5 شمعات
    df['is_pivot_high'] = df['high'] == df['high'].rolling(5, center=True).max()
    df['is_pivot_low'] = df['low'] == df['low'].rolling(5, center=True).min()
    
    return df.fillna(0)

# التعرف على نماذج الشموع اليابانية (Price Action)
def detect_candlestick_pattern(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    body = abs(last['close'] - last['open'])
    wick_upper = last['high'] - max(last['close'], last['open'])
    wick_lower = min(last['close'], last['open']) - last['low']
    avg_body = abs(df['close'] - df['open']).rolling(10).mean().iloc[-1]
    
    # 1. Bullish Engulfing (الابتلاع الشرائي)
    if (prev['close'] < prev['open']) and (last['close'] > last['open']) and \
       (last['close'] > prev['open']) and (last['open'] < prev['close']):
        return "Bullish_Engulfing"
    
    # 2. Hammer / Pinbar (المطرقة)
    if (wick_lower > 2 * body) and (wick_upper < body * 0.5) and (body > 0):
        # تأكيد المطرقة في اتجاه هابط قصير المدى
        if df['close'].iloc[-3] > df['close'].iloc[-2]: 
            return "Hammer_Reversal"
            
    # 3. Marubozu (شمعة زخم قوية)
    if (body > avg_body * 1.5) and (wick_upper < body * 0.1) and (wick_lower < body * 0.1):
        if last['close'] > last['open']: return "Bullish_Marubozu"
        
    return None

# --- 5. منطق استراتيجيات التداول المحترف ---

def analyze_market_structure_trend(df):
    """
    تحديد هيكل السوق بدقة:
    - Higher Highs + Higher Lows = Bullish Structure
    - Lower Lows + Lower Highs = Bearish Structure
    """
    pivots_high = df[df['is_pivot_high']]['high'].tail(3).values
    pivots_low = df[df['is_pivot_low']]['low'].tail(3).values
    
    if len(pivots_high) < 2 or len(pivots_low) < 2: return "Unclear"
    
    if pivots_high[-1] > pivots_high[-2] and pivots_low[-1] > pivots_low[-2]:
        return "Bullish_Structure"
    elif pivots_low[-1] < pivots_low[-2] and pivots_high[-1] < pivots_high[-2]:
        return "Bearish_Structure"
    
    return "Ranging"

def get_professional_signal(symbol, df_trend, df_entry):
    """
    مولد الإشارات المتقدم
    df_trend: فريم 4 ساعات (للاتجاه العام)
    df_entry: فريم 15 دقيقة (للدخول الدقيق)
    """
    # 1. تحليل الإطار الزمني الكبير (4H)
    last_trend = df_trend.iloc[-1]
    structure_4h = analyze_market_structure_trend(df_trend)
    
    trend_bias = "Neutral"
    if structure_4h == "Bullish_Structure" and last_trend['close'] > last_trend['ema50']:
        trend_bias = "Bullish"
    elif structure_4h == "Bearish_Structure":
        trend_bias = "Bearish"
        
    # نحن نبحث عن الشراء فقط في البوت الحالي (Spot)
    if trend_bias != "Bullish": 
        # استثناء: إذا كان هناك ذروة بيع عنيفة (Mean Reversion)
        if last_trend['rsi'] < 25 and analyze_market_structure_trend(df_entry) == "Bullish_Structure":
             pass # السماح بالمرور لاختبار الارتداد
        else:
             return None, "الاتجاه العام ليس صاعداً"

    # 2. تحليل إطار الدخول (15m)
    last_entry = df_entry.iloc[-1]
    prev_entry = df_entry.iloc[-2]
    structure_15m = analyze_market_structure_trend(df_entry)
    
    # فلتر: هل الشمعة الحالية خضراء قوية؟
    candle_pattern = detect_candlestick_pattern(df_entry)
    
    # الاستراتيجية 1: (Trend Pullback) إعادة اختبار في اتجاه صاعد
    if trend_bias == "Bullish":
        # السعر فوق EMA200 لكنه تراجع قليلاً (RSI < 60)
        if last_entry['close'] > last_entry['ema200']:
            # شرط: كسر هيكل صغير للأعلى أو ارتداد من EMA50
            near_ema50 = abs(last_entry['close'] - last_entry['ema50']) / last_entry['ema50'] < 0.005
            if near_ema50 and candle_pattern in ["Bullish_Engulfing", "Hammer_Reversal"]:
                return "Trend_Pullback_EMA50", "ارتداد من المتوسط 50 مع تأكيد شموع"

    # الاستراتيجية 2: (Volatility Breakout) انفجار سعري
    if last_entry['adx'] > 25: # يوجد زخم
        # اختراق مقاومة (البولينجر أو قمة سابقة)
        # هنا سنستخدم اختراق آخر قمة PIVOT
        last_pivot_high = df_entry[df_entry['is_pivot_high']]['high'].max()
        if last_entry['close'] > last_pivot_high and last_entry['volume'] > last_entry['volume'].mean() * 1.5:
            return "Volatility_Breakout", "اختراق قمة مع فوليوم عالي"

    # الاستراتيجية 3: (Sniper Entry) التقاط القاع
    if last_entry['rsi'] < 30: # تشبع بيعي
        if candle_pattern == "Bullish_Engulfing": # تأكيد قوي
             return "Sniper_Bottom", "ارتداد من تشبع بيعي مع ابتلاع شرائي"

    return None, "لا توجد شروط مكتملة"

# --- 6. إعادة تحليل الصفقة (Logic Update) ---
def reverify_trade_thesis(trade, df_entry):
    """
    هذه الدالة تعمل كمدير مخاطر ذكي.
    تقرر ما إذا كانت "فرضية" الصفقة لا تزال قائمة أم لا.
    """
    last = df_entry.iloc[-1]
    entry_price = float(trade['entry_price'])
    current_price = float(last['close'])
    profit_pct = (current_price - entry_price) / entry_price * 100
    
    # 1. قاعدة الزمن (Time Stop)
    # إذا مرت 4 ساعات والربح أقل من 0.2%، السوق ميت، اخرج ووفر السيولة
    duration_mins = (datetime.now() - trade['entry_time']).total_seconds() / 60
    if duration_mins > 240 and profit_pct < 0.2 and profit_pct > -1.0:
        return "CLOSE", "ركود سعري (Time Limit)"
        
    # 2. انهيار الزخم (Momentum Decay)
    # إذا كنا في صفقة اختراق (Breakout) ولكن السعر عاد تحت نقطة الاختراق بسرعة
    if "Breakout" in trade['strategy_name']:
        if current_price < entry_price and last['rsi'] < 45:
             return "CLOSE", "فشل الاختراق (False Breakout)"

    # 3. التأكيد السلبي القوي
    # ظهور نموذج سلبي قوي (مثل ابتلاع بيعي ضخم)
    # can_pattern = detect_candlestick_pattern(df_entry)
    # if can_pattern == "Bearish_Engulfing" and profit_pct > 1.0:
    #     return "CLOSE", "ظهور إشارة خروج عكسية"

    return "HOLD", "الفرضية قائمة"

# --- 7. محرك البوت الرئيسي ---
def bot_engine():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 بدء تشغيل المحرك الذكي (V14) - وضع احترافي")
    
    # تحميل العملات القابلة للتداول
    trading_pairs = []
    try:
        # نحضر أعلى 30 عملة بالسيولة لتجنب العملات الميتة
        tickers = client.get_ticker()
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and float(t['quoteVolume']) > 10000000]
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        trading_pairs = [t['symbol'] for t in usdt_pairs[:30]]
        # استثناء العملات المستقرة
        trading_pairs = [s for s in trading_pairs if s not in IGNORED_SYMBOLS]
        logger.info(f"📋 تم اختيار {len(trading_pairs)} أصل للتداول بناءً على السيولة.")
    except Exception as e:
        logger.error(f"فشل تحميل العملات: {e}")
        trading_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']

    while True:
        try:
            # تحديث الإعدادات
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                is_paper = BOT_SETTINGS['paper_trading_mode']
            
            if not enabled:
                time.sleep(5)
                continue

            # 1. إدارة الصفقات المفتوحة (الأولوية القصوى)
            with locks['signals']:
                active_trades = list(open_signals_cache.values())
            
            for trade in active_trades:
                symbol = trade['symbol']
                df_entry = fetch_data(client, symbol, BOT_SETTINGS['timeframe_analysis'], 60)
                
                if df_entry is None: continue
                df_entry = calculate_advanced_indicators(df_entry)
                
                curr_price = df_entry['close'].iloc[-1]
                max_p = trade.get('max_price_reached', trade['entry_price'])
                if curr_price > max_p:
                    trade['max_price_reached'] = curr_price # تحديث أعلى سعر وصل له

                # فحص الخروج الطارئ أو الذكي
                decision, reason = reverify_trade_thesis(trade, df_entry)
                
                if decision == "CLOSE":
                    close_trade(symbol, curr_price, reason, is_paper)
                    continue
                
                # منطق إدارة الأهداف والوقف (محسن)
                manage_trade_execution(symbol, trade, curr_price, df_entry, is_paper)

            # 2. البحث عن فرص جديدة (إذا كان هناك متسع)
            if len(active_trades) < BOT_SETTINGS['max_open_trades']:
                for symbol in trading_pairs:
                    # تخطي العملات المفتوحة حالياً
                    if symbol in open_signals_cache: continue
                    
                    # جلب البيانات (4 ساعات للاتجاه، 15 دقيقة للدخول)
                    df_4h = fetch_data(client, symbol, BOT_SETTINGS['timeframe_trend'], 100)
                    df_15m = fetch_data(client, symbol, BOT_SETTINGS['timeframe_analysis'], 60)
                    
                    if df_4h is None or df_15m is None: continue
                    
                    df_4h = calculate_advanced_indicators(df_4h)
                    df_15m = calculate_advanced_indicators(df_15m)
                    
                    signal, reason = get_professional_signal(symbol, df_4h, df_15m)
                    
                    if signal:
                        # حساب الكمية بناءً على المخاطرة
                        curr = df_15m['close'].iloc[-1]
                        atr = df_15m['atr'].iloc[-1]
                        
                        # الوقف يكون أسفل آخر قاع محلي أو 2 ATR
                        sl_level = curr - (atr * 2.0)
                        risk_per_share = curr - sl_level
                        
                        if risk_per_share <= 0: continue
                        
                        capital = BOT_SETTINGS['base_capital']
                        risk_amount = capital * (BOT_SETTINGS['risk_per_trade_pct'] / 100)
                        qty = risk_amount / risk_per_share
                        
                        # تحديد الأهداف (R:R Ratio)
                        tp1 = curr + (risk_per_share * 1.5) # R:1.5
                        tp2 = curr + (risk_per_share * 3.0) # R:3
                        tp3 = curr * 1.5 # Moonbag (هدف مفتوح بعيد)
                        
                        execute_entry(symbol, curr, sl_level, tp1, tp2, tp3, qty, signal, reason, is_paper)
                        
                    time.sleep(0.5) # احترام Rate Limit

            time.sleep(10) # دورة تحديث سريعة

        except Exception as e:
            logger.error(f"Global Engine Error: {e}")
            time.sleep(5)

# --- 8. تنفيذ الأوامر وإدارة الخروج ---
def execute_entry(symbol, price, sl, tp1, tp2, tp3, qty, strategy, structure, is_paper):
    c = get_db_connection()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        with c.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v14 
                (symbol, entry_price, stop_loss, tp1, tp2, tp3_moonbag, quantity, strategy_name, market_structure, status, mode, max_price_reached)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, %s)
                RETURNING id
            """, (symbol, price, sl, tp1, tp2, tp3, qty, strategy, structure, mode, price))
            t_id = cur.fetchone()['id']
            
        trade_obj = {
            'id': t_id, 'symbol': symbol, 'entry_price': price, 'stop_loss': sl,
            'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'quantity': qty, 'strategy_name': strategy,
            'entry_time': datetime.now(), 'max_price_reached': price, 'tp1_hit': False, 'tp2_hit': False
        }
        
        with locks['signals']: open_signals_cache[symbol] = trade_obj
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'm': f"دخول: {strategy}"})
        
        # إرسال تنبيه (محاكاة)
        logger.info(f"✅ دخول صفقة: {symbol} بسعر {price}")

    except Exception as e: logger.error(f"Entry Error: {e}")

def manage_trade_execution(symbol, trade, current_price, df, is_paper):
    # منطق الوقف المتحرك الذكي (بناءً على الهيكل)
    # إذا تحرك السعر لصالحنا وشكل قاعاً جديداً أعلى من الدخول، ننقل الوقف تحته
    last_pivot_low = df[df['is_pivot_low']]['low'].iloc[-1] if not df[df['is_pivot_low']].empty else 0
    
    # 1. تحديث الوقف المتحرك
    if current_price > trade['entry_price'] * 1.015: # ربح 1.5%
        new_sl = max(trade['stop_loss'], last_pivot_low * 0.995) # 0.5% تحت القاع
        if new_sl > trade['stop_loss']:
            trade['stop_loss'] = new_sl
            # تحديث في قاعدة البيانات
            c = get_db_connection()
            with c.cursor() as cur:
                cur.execute("UPDATE trades_v14 SET stop_loss=%s WHERE id=%s", (new_sl, trade['id']))

    # 2. جني الأرباح الجزئي
    if not trade.get('tp1_hit') and current_price >= trade['tp1']:
        trade['tp1_hit'] = True
        trade['stop_loss'] = trade['entry_price'] * 1.002 # Breakeven + Fees
        logger.info(f"💰 {symbol} تحقيق الهدف الأول. تم نقل الوقف للدخول.")
        
    # 3. الخروج (وقف الخسارة أو الهدف النهائي)
    if current_price <= trade['stop_loss']:
        close_trade(symbol, trade['stop_loss'], "ضرب وقف الخسارة (SL)", is_paper)
    elif current_price >= trade['tp2']:
        close_trade(symbol, current_price, "تحقيق الهدف الثاني (TP2)", is_paper)

def close_trade(symbol, exit_price, reason, is_paper):
    trade = None
    with locks['signals']:
        if symbol in open_signals_cache:
            trade = open_signals_cache.pop(symbol)
            
    if not trade: return

    # حسابات الأرباح الدقيقة مع العمولة
    entry_val = trade['entry_price'] * trade['quantity']
    exit_val = float(exit_price) * trade['quantity']
    gross_pnl = exit_val - entry_val
    
    fee_rate = BOT_SETTINGS['exchange_fee_rate']
    fees = (entry_val * fee_rate) + (exit_val * fee_rate)
    net_pnl = gross_pnl - fees
    
    c = get_db_connection()
    try:
        with c.cursor() as cur:
            cur.execute("""
                UPDATE trades_v14 
                SET status='closed', closed_at=NOW(), closing_price=%s, 
                    gross_profit=%s, net_profit=%s, fees_paid=%s, exit_reason=%s
                WHERE id=%s
            """, (exit_price, gross_pnl, net_pnl, fees, reason, trade['id']))
        logger.info(f"🚫 إغلاق {symbol}: {reason} | صافي الربح: {net_pnl:.2f}$")
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'm': f"خروج: {reason}"})
    except Exception as e: logger.error(f"Close Error: {e}")

# --- 9. واجهة الويب والتحكم (API & UI) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    return render_template_string(HTML_DASHBOARD)

@app.route('/api/data')
def get_data():
    with locks['signals']: trades = list(open_signals_cache.values())
    with locks['logs']: logs = list(scan_logs)
    
    # حساب الإحصائيات من قاعدة البيانات
    c = get_db_connection()
    stats = {}
    try:
        with c.cursor() as cur:
            cur.execute("SELECT count(*) as cnt, sum(net_profit) as pnl FROM trades_v14 WHERE status='closed'")
            res = cur.fetchone()
            stats['total_trades'] = res['cnt']
            stats['total_pnl'] = res['pnl'] if res['pnl'] else 0.0
    except: stats = {'total_trades': 0, 'total_pnl': 0}

    return jsonify({
        'trades': trades,
        'logs': logs,
        'stats': stats,
        'settings': BOT_SETTINGS
    })

@app.route('/api/close_trade', methods=['POST'])
def manual_close():
    data = request.json
    symbol = data.get('symbol')
    price = data.get('price') # السعر الحالي من الواجهة لتسريع العملية
    close_trade(symbol, float(price), "إغلاق يدوي من لوحة التحكم", BOT_SETTINGS['paper_trading_mode'])
    return jsonify({"status": "success"})

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    with locks['settings']:
        BOT_SETTINGS['is_trading_enabled'] = data.get('enabled', BOT_SETTINGS['is_trading_enabled'])
        BOT_SETTINGS['risk_per_trade_pct'] = float(data.get('risk', BOT_SETTINGS['risk_per_trade_pct']))
    return jsonify({"status": "saved"})

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>SmartBot Pro V14</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body { background: #0b0e11; color: #eaecef; font-family: 'Tajawal', sans-serif; padding: 20px; }
        .card { background: #1e2329; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #2b3139; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: right; border-bottom: 1px solid #2b3139; }
        th { color: #848e9c; }
        .btn { padding: 8px 16px; border-radius: 4px; border: none; cursor: pointer; font-weight: bold; }
        .btn-red { background: #f6465d; color: white; }
        .btn-green { background: #0ecb81; color: white; }
        .profit { color: #0ecb81; } .loss { color: #f6465d; }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center">
        <h1>SmartBot Pro <span style="color:#f0b90b">V14</span></h1>
        <div id="statusIndicator">...</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>الإحصائيات العامة (الصافي بعد العمولة)</h3>
            <h2 id="totalPnl" dir="ltr">0.00 $</h2>
            <p>إجمالي الصفقات المغلقة: <span id="totalTrades">0</span></p>
        </div>
        <div class="card">
            <h3>التحكم السريع</h3>
            <label>حالة البوت: <input type="checkbox" id="botToggle" onchange="toggleBot()"></label><br><br>
            <label>المخاطرة %: <input type="number" id="riskInput" step="0.1" style="width:60px"></label>
            <button class="btn btn-green" onclick="saveSettings()">حفظ الإعدادات</button>
        </div>
    </div>

    <div class="card">
        <h3>الصفقات المفتوحة</h3>
        <table>
            <thead>
                <tr>
                    <th>العملة</th>
                    <th>الاستراتيجية</th>
                    <th>الدخول</th>
                    <th>الربح الحالي</th>
                    <th>الأهداف</th>
                    <th>إجراء</th>
                </tr>
            </thead>
            <tbody id="tradesTable"></tbody>
        </table>
    </div>

    <div class="card">
        <h3>سجل الأحداث</h3>
        <div id="logs" style="max-height: 200px; overflow-y: auto; font-size: 0.9em; color: #ccc;"></div>
    </div>

    <script>
        let currentTrades = {};

        async function fetchData() {
            const res = await axios.get('/api/data');
            const data = res.data;
            
            // Stats
            document.getElementById('totalPnl').innerText = data.stats.total_pnl.toFixed(2) + ' $';
            document.getElementById('totalPnl').className = data.stats.total_pnl >= 0 ? 'profit' : 'loss';
            document.getElementById('totalTrades').innerText = data.stats.total_trades;
            
            // Settings
            document.getElementById('botToggle').checked = data.settings.is_trading_enabled;
            if(document.activeElement.id !== 'riskInput') {
                document.getElementById('riskInput').value = data.settings.risk_per_trade_pct;
            }

            // Trades
            const tbody = document.getElementById('tradesTable');
            tbody.innerHTML = data.trades.map(t => {
                // محاكاة السعر الحالي للواجهة فقط
                const curr = t.max_price_reached; // في الواقع يجب جلبه من live_prices
                const pnl = ((curr - t.entry_price) / t.entry_price) * 100;
                currentTrades[t.symbol] = curr;
                
                return `<tr>
                    <td><b>${t.symbol}</b></td>
                    <td><small>${t.strategy_name}</small></td>
                    <td>${t.entry_price}</td>
                    <td class="${pnl>=0?'profit':'loss'}" dir="ltr">${pnl.toFixed(2)}%</td>
                    <td><small>${t.tp1} -> ${t.tp2}</small></td>
                    <td><button class="btn btn-red" onclick="closeTrade('${t.symbol}')">إغلاق</button></td>
                </tr>`;
            }).join('');

            // Logs
            document.getElementById('logs').innerHTML = data.logs.map(l => 
                `<div><span style="color:#f0b90b">[${l.t}]</span> <b>${l.s}</b>: ${l.m}</div>`
            ).join('');
        }

        async function closeTrade(symbol) {
            if(!confirm('هل أنت متأكد من إغلاق الصفقة يدوياً؟')) return;
            await axios.post('/api/close_trade', {symbol: symbol, price: currentTrades[symbol]});
            fetchData();
        }

        async function toggleBot() {
            const enabled = document.getElementById('botToggle').checked;
            await axios.post('/api/update_settings', {enabled: enabled});
        }
        
        async function saveSettings() {
            const risk = document.getElementById('riskInput').value;
            await axios.post('/api/update_settings', {risk: risk});
            alert('تم حفظ الإعدادات');
        }

        setInterval(fetchData, 2000);
        fetchData();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    # تشغيل خيوط المعالجة
    t = Thread(target=bot_engine)
    t.daemon = True
    t.start()
    
    logger.info("🖥️ تشغيل الخادم على المنفذ 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)