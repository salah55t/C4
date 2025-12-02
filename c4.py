import time
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import random
from threading import Thread, Lock
from datetime import datetime
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
    handlers=[logging.FileHandler('smart_bot_fib.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Pro')

# تحميل المتغيرات البيئية (يفضل وضعها في ملف .env)
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    # قيم افتراضية للتجربة في حالة عدم وجود ملف .env
    API_KEY = ""
    API_SECRET = ""
    DB_URL = "postgresql://user:password@localhost/dbname" 
    logger.warning(f"⚠️ تنبيه: لم يتم تحميل الإعدادات كاملة ({e}). سيعمل النظام بوضع المحاكاة.")

# --- 2. إعدادات التداول المتقدمة ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,       # رأس المال
    "risk_per_trade_pct": 2.0,    # المخاطرة لكل صفقة
    "max_open_trades": 5,         # عدد الصفقات المتزامنة (تم تقليله لزيادة التركيز)
    "fib_lookback": 144,          # فترة البحث عن قمم وقيعان فيبوناتشي
    "timeframe_analysis": "15m",
    "request_delay": 0.8          # التأخير لتجنب الحظر (ثانية)
}

# تخزين الحالة العامة
system_state = {
    "market_regime": "Neutral",
    "trend_strength": 0,
    "active_symbols_pool": [],    # القائمة المفلترة للعملات
    "last_scan_time": None
}

open_signals_cache = {}
live_prices = {}
scan_logs = deque(maxlen=200)

locks = {
    'signals': Lock(), 'prices': Lock(), 'market': Lock(), 
    'settings': Lock(), 'logs': Lock()
}

# --- 3. إدارة قاعدة البيانات ---
conn = None
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades_fib_v1 (
                    id SERIAL PRIMARY KEY, 
                    symbol TEXT NOT NULL, 
                    entry_price DOUBLE PRECISION, 
                    stop_loss DOUBLE PRECISION, 
                    tp1 DOUBLE PRECISION,
                    tp2 DOUBLE PRECISION,
                    fib_level_entry TEXT,
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
        logger.info("✅ قاعدة البيانات جاهزة (نسخة فيبوناتشي).")
    except Exception as e: 
        logger.error(f"خطأ قاعدة البيانات: {e}")

def check_db():
    global conn
    if conn is None or conn.closed != 0: 
        try:
            init_db()
        except:
            pass

# --- 4. التحليل الفني ومستويات فيبوناتشي ---
def calculate_fibonacci_levels(df, period=144):
    """حساب مستويات فيبوناتشي بناءً على أعلى قمة وأدنى قاع"""
    # نأخذ نافذة زمنية محددة
    window = df.iloc[-period:]
    high_price = window['high'].max()
    low_price = window['low'].min()
    diff = high_price - low_price
    
    levels = {
        '0.0': high_price, # القمة
        '0.236': high_price - 0.236 * diff,
        '0.382': high_price - 0.382 * diff,
        '0.5': high_price - 0.5 * diff,
        '0.618': high_price - 0.618 * diff, # النسبة الذهبية
        '0.786': high_price - 0.786 * diff,
        '1.0': low_price, # القاع
        # امتدادات للأهداف
        'ext_1.272': high_price + 0.272 * diff,
        'ext_1.618': high_price + 0.618 * diff
    }
    return levels, high_price, low_price

def calculate_indicators(df):
    df = df.copy()
    # المتوسطات المتحركة
    df['ema9'] = df['close'].ewm(span=9).mean()
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
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    return df.fillna(0)

# --- 5. فلترة السوق المتقدمة ---
def filter_top_symbols(client):
    """اختيار أفضل 20 عملة بناءً على السيولة والنشاط"""
    try:
        tickers = client.get_ticker()
        # 1. فلتر أولي: عملات USDT فقط وتجاهل العملات المستقرة المعروفة
        stablecoins = ['USDCUSDT', 'TUSDUSDT', 'FDUSDUSDT', 'DAIUSDT', 'USDPUSDT']
        valid = []
        
        for t in tickers:
            s = t['symbol']
            if not s.endswith('USDT') or s in stablecoins: continue
            
            # تجاهل العملات ذات الحجم الضئيل (أقل من 10 مليون دولار) لتجنب الانزلاق
            q_vol = float(t['quoteVolume'])
            if q_vol < 10_000_000: continue 
            
            valid.append({
                'symbol': s,
                'volume': q_vol,
                'change': float(t['priceChangePercent']),
                'count': int(t['count']) # عدد الصفقات
            })
            
        # 2. الترتيب حسب معيار مركب (الحجم * القيمة المطلقة للتغير)
        # نركز على العملات التي تتحرك ولها سيولة
        valid.sort(key=lambda x: x['volume'] * abs(x['change']), reverse=True)
        
        top_20 = [x['symbol'] for x in valid[:20]]
        logger.info(f"🔎 تم تحديث القائمة المختارة: {top_20}")
        return top_20
        
    except Exception as e:
        logger.error(f"خطأ في فلترة الرموز: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'] # قائمة احتياطية

# --- 6. استراتيجيات التداول الديناميكية ---
def analyze_signal(symbol, df, regime):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # حساب فيبوناتشي
    fibs, high_p, low_p = calculate_fibonacci_levels(df, BOT_SETTINGS['fib_lookback'])
    current_price = last['close']
    
    signal = None
    reason = ""
    fib_note = ""
    
    # -- الفلاتر الديناميكية حسب حالة السوق --
    
    # 1. استراتيجية الارتداد من فيبوناتشي 0.618 (Golden Pocket)
    # فعالة في الاتجاه الصاعد (Bull Trend)
    if "Bull" in regime:
        # السعر قريب من مستوى 0.618 أو 0.5
        dist_to_618 = abs(current_price - fibs['0.618']) / current_price
        dist_to_050 = abs(current_price - fibs['0.5']) / current_price
        
        if (dist_to_618 < 0.005 or dist_to_050 < 0.005): # قريب جداً (0.5%)
            # شرط التأكيد: RSI ليس مشبعاً بالشراء + شمعة خضراء
            if last['rsi'] < 70 and last['close'] > last['open']:
                signal = "Fib_Retracement_Entry"
                reason = "ارتداد من المنطقة الذهبية (0.5-0.618)"
                fib_note = "Bounce 0.618"

    # 2. استراتيجية اختراق فيبوناتشي (للأسواق القوية)
    elif "Volatile" in regime or "Bull" in regime:
        # اختراق القمة السابقة (مستوى 0) مع زخم
        if prev['close'] < fibs['0.0'] and last['close'] > fibs['0.0']:
            if last['volume'] > last['vol_ma'] * 1.5: # شرط حجم تداول عالي
                signal = "Fib_Breakout"
                reason = "اختراق القمة السابقة بزخم عالي"
                fib_note = "Break Level 0"

    # 3. استراتيجية السكالبينج (للسوق العرضي)
    elif "Ranging" in regime:
        # الشراء عند الدعم (BB Lower) والبيع عند المقاومة
        if last['close'] < last['bb_lower'] and last['rsi'] < 30:
             signal = "BB_Reversal"
             reason = "ارتداد من قاع بولنجر (تشبع بيعي)"
             fib_note = "Support Bounce"

    if signal:
        return signal, reason, fibs, fib_note
    return None, None, None, None

# --- 7. المحرك الرئيسي للبوت ---
def bot_engine():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 تم تشغيل محرك SmartBot Fib Pro")
    
    # تأخير أولي
    time.sleep(2)
    
    while True:
        try:
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                paper = BOT_SETTINGS['paper_trading_mode']
                max_trades = BOT_SETTINGS['max_open_trades']
                delay = BOT_SETTINGS['request_delay']

            if not enabled:
                time.sleep(5)
                continue

            # 1. تحديث القائمة المختارة (كل 30 دقيقة تقريباً أو إذا كانت فارغة)
            if not system_state['active_symbols_pool'] or datetime.now().minute % 30 == 0:
                with locks['market']:
                    system_state['active_symbols_pool'] = filter_top_symbols(client)
                time.sleep(delay)

            # 2. تحليل حالة السوق (على البيتكوين)
            btc_df = fetch_data(client, 'BTCUSDT', '4h', 100)
            if btc_df is not None:
                btc_df = calculate_indicators(btc_df)
                update_market_regime(btc_df)
            time.sleep(delay)

            regime = system_state['market_regime']
            
            # 3. إدارة الصفقات المفتوحة
            manage_open_trades(client, paper, delay)
            
            # 4. البحث عن فرص جديدة
            with locks['signals']: current_opens = len(open_signals_cache)
            
            if current_opens < max_trades:
                for sym in system_state['active_symbols_pool']:
                    if current_opens >= max_trades: break
                    if sym in open_signals_cache: continue # تخطي العملات المفتوحة
                    
                    # جلب البيانات
                    df = fetch_data(client, sym, BOT_SETTINGS['timeframe_analysis'], 200) # نحتاج 200 للفيبوناتشي
                    time.sleep(delay) # 🛑 الانتظار لتجنب الحظر
                    
                    if df is None: continue
                    
                    df = calculate_indicators(df)
                    signal_name, reason, fibs, fib_note = analyze_signal(sym, df, regime)
                    
                    if signal_name:
                        execute_trade_logic(sym, df, fibs, signal_name, reason, fib_note, paper)
                        current_opens += 1
                    else:
                        # تسجيل محاولة فحص عشوائية (ليس كل مرة لتخفيف الضغط)
                        if random.random() < 0.1:
                            log_scan(sym, 'فحص', 'لا توجد إشارة')

            time.sleep(10) # راحة قصيرة قبل الدورة التالية

        except Exception as e:
            logger.error(f"خطأ في المحرك الرئيسي: {e}")
            time.sleep(10)

def fetch_data(client, symbol, interval, limit):
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def update_market_regime(df):
    last = df.iloc[-1]
    # منطق مبسط لتحديد الحالة
    trend = "Neutral"
    if last['ema50'] > last['ema200']: trend = "Bull"
    elif last['ema50'] < last['ema200']: trend = "Bear"
    
    if last['adx'] < 20: trend += "_Ranging" # عرضي
    elif last['adx'] > 30: trend += "_Strong" # قوي
    
    # قياس التذبذب
    atr = last['high'] - last['low'] # تقريبي
    atr_pct = (atr / last['close']) * 100
    if atr_pct > 2.0: trend = "High_Volatility"
    
    with locks['market']:
        system_state['market_regime'] = trend
        system_state['trend_strength'] = round(last['adx'], 2)

# --- 8. تنفيذ وإدارة الصفقات ---
def execute_trade_logic(symbol, df, fibs, strategy, reason, fib_note, is_paper):
    last_price = df['close'].iloc[-1]
    
    # تحديد الأهداف والوقف بناءً على مستويات فيبوناتشي
    # نجد أقرب مستوى فيبوناتشي تحت السعر ليكون وقف الخسارة
    sorted_levels = sorted([v for k,v in fibs.items() if not k.startswith('ext')])
    
    # الافتراضي
    sl = last_price * 0.98
    tp1 = last_price * 1.02
    tp2 = last_price * 1.04
    
    # محاولة استخدام الفيبوناتشي بدقة
    try:
        below_levels = [l for l in sorted_levels if l < last_price]
        above_levels = [l for l in sorted_levels if l > last_price]
        
        if below_levels:
            # الوقف تحت أقرب مستوى دعم
            sl = below_levels[-1] * 0.995 # هامش بسيط تحت الدعم
        
        if above_levels:
            tp1 = above_levels[0] # أول مقاومة
            if len(above_levels) > 1:
                tp2 = above_levels[1] # ثاني مقاومة
            else:
                tp2 = fibs.get('ext_1.272', last_price * 1.05) # استخدام الامتداد
        else:
            # نحن في قمة جديدة، نستخدم الامتدادات
            tp1 = fibs.get('ext_1.272', last_price * 1.03)
            tp2 = fibs.get('ext_1.618', last_price * 1.06)

    except: pass # الرجوع للافتراضي عند الخطأ

    # حساب الكمية (إدارة المخاطر)
    risk_amt = BOT_SETTINGS['base_capital'] * (BOT_SETTINGS['risk_per_trade_pct'] / 100)
    loss_per_share = last_price - sl
    if loss_per_share <= 0: loss_per_share = last_price * 0.01 # حماية من القسمة على صفر
    
    qty = risk_amt / loss_per_share
    
    # فتح الصفقة
    trade_data = {
        'symbol': symbol, 'entry_price': last_price, 'sl': sl, 
        'tp1': tp1, 'tp2': tp2, 'qty': qty, 'strat': strategy, 
        'note': f"{reason} | {fib_note}", 'regime': system_state['market_regime']
    }
    
    # حفظ في الذاكرة وقاعدة البيانات
    save_new_trade(trade_data, is_paper)

def manage_open_trades(client, is_paper, delay):
    with locks['signals']: trades = list(open_signals_cache.values())
    
    for trade in trades:
        sym = trade['symbol']
        
        # جلب سعر لحظي
        try:
            ticker = client.get_symbol_ticker(symbol=sym)
            curr_price = float(ticker['price'])
            with locks['prices']: live_prices[sym] = curr_price
            time.sleep(delay) # 🛑 تأخير
        except: continue
        
        # فحص الشروط
        sl = trade['stop_loss']
        tp1 = trade['tp1']
        tp2 = trade['tp2']
        
        exit_reason = None
        
        # 1. وقف الخسارة
        if curr_price <= sl:
            exit_reason = "ضرب وقف الخسارة (Fib Support Broken)"
        
        # 2. الهدف الثاني (خروج كامل)
        elif curr_price >= tp2:
            exit_reason = "تحقق الهدف النهائي (TP2)"
            
        # 3. إدارة الهدف الأول (حجز أرباح)
        elif curr_price >= tp1:
            # هنا يمكننا رفع الوقف لنقطة الدخول (Breakeven) بدلاً من الإغلاق
            if sl < trade['entry_price']:
                new_sl = trade['entry_price'] * 1.002 # فوق الدخول بقليل لتغطية العمولات
                update_sl_in_db(trade['id'], new_sl, sym)
                # نرسل تنبيه فقط ولا نغلق
        
        if exit_reason:
            close_trade_final(sym, curr_price, exit_reason, is_paper)

def save_new_trade(data, is_paper):
    mode = 'PAPER' if is_paper else 'REAL'
    try:
        check_db()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_fib_v1 
                (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, status, mode, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, NOW())
                RETURNING id
            """, (data['symbol'], data['entry_price'], data['sl'], data['tp1'], data['tp2'], data['qty'], data['strat'], data['regime'], mode))
            new_id = cur.fetchone()['id']
            
        cache_item = {
            'id': new_id, 'symbol': data['symbol'], 'entry_price': data['entry_price'],
            'stop_loss': data['sl'], 'tp1': data['tp1'], 'tp2': data['tp2'],
            'quantity': data['qty'], 'strategy': data['strat'], 'entry_time': datetime.now()
        }
        with locks['signals']: open_signals_cache[data['symbol']] = cache_item
        log_scan(data['symbol'], 'دخول', data['note'])
        send_telegram(f"🟢 شراء جديد: {data['symbol']}\nالسعر: {data['entry_price']}\nالهدف: {data['tp2']}\nالوقف: {data['sl']}")
        
    except Exception as e: logger.error(f"DB Insert: {e}")

def close_trade_final(symbol, price, reason, is_paper):
    trade = None
    with locks['signals']:
        if symbol in open_signals_cache:
            trade = open_signals_cache.pop(symbol)
    
    if trade:
        profit_pct = ((price - trade['entry_price']) / trade['entry_price']) * 100
        profit_abs = (price - trade['entry_price']) * trade['quantity']
        
        try:
            check_db()
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trades_fib_v1 
                    SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, exit_reason=%s
                    WHERE id=%s
                """, (price, profit_pct, profit_abs, reason, trade['id']))
            
            emoji = "✅" if profit_pct > 0 else "🔻"
            send_telegram(f"{emoji} إغلاق صفقة: {symbol}\nالربح: {profit_pct:.2f}%\nالسبب: {reason}")
            log_scan(symbol, 'إغلاق', reason)
        except Exception as e: logger.error(f"DB Close: {e}")

def update_sl_in_db(trade_id, new_sl, symbol):
    try:
        with locks['signals']:
            if symbol in open_signals_cache:
                open_signals_cache[symbol]['stop_loss'] = new_sl
        
        check_db()
        with conn.cursor() as cur:
            cur.execute("UPDATE trades_fib_v1 SET stop_loss=%s WHERE id=%s", (new_sl, trade_id))
        logger.info(f"🛡️ تم تحديث الوقف لـ {symbol} إلى {new_sl}")
    except: pass

def log_scan(sym, status, reason):
    with locks['logs']:
        scan_logs.appendleft({
            't': datetime.now().strftime('%H:%M:%S'),
            's': sym, 'st': status, 'r': reason
        })

def send_telegram(msg):
    if not TELEGRAM_TOKEN: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except: pass

# --- 9. واجهة الويب (Flask) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    with locks['market']: m = system_state.copy()
    with locks['signals']: s = list(open_signals_cache.values())
    with locks['prices']: p = live_prices.copy()
    with locks['logs']: l = list(scan_logs)
    
    # إزالة التواريخ من الرد لضمان توافق JSON
    safe_signals = []
    for t in s:
        temp = t.copy()
        if 'entry_time' in temp: del temp['entry_time']
        safe_signals.append(temp)

    return jsonify({
        "market": str(m['market_regime']),
        "signals": safe_signals,
        "prices": p,
        "logs": l,
        "settings": BOT_SETTINGS
    })

@app.route('/api/close_manual', methods=['POST'])
def manual_close():
    """واجهة برمجة التطبيقات للإغلاق اليدوي"""
    data = request.json
    symbol = data.get('symbol')
    price = live_prices.get(symbol, 0)
    
    if price == 0: # محاولة جلب السعر إذا لم يكن متاحاً
        try:
             client = Client(API_KEY, API_SECRET)
             price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        except: return jsonify({"status": "error", "msg": "Could not fetch price"}), 400

    close_trade_final(symbol, price, "إغلاق يدوي من المشرف", BOT_SETTINGS['paper_trading_mode'])
    return jsonify({"status": "success", "symbol": symbol})

@app.route('/api/toggle', methods=['POST'])
def toggle():
    with locks['settings']: 
        BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify("OK")

# --- HTML Dashboard (Arabic) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot Fibonacci - لوحة التحكم</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #131722; --card: #1e222d; --text: #d1d4dc; --green: #00b59b; --red: #fa3c58; --accent: #2962ff; }
        body { background: var(--bg); color: var(--text); font-family: 'Tajawal', sans-serif; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid #2a2e39; padding-bottom: 20px; }
        .status-badge { padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 14px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: var(--card); padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .card h3 { color: #787b86; font-size: 14px; margin-top: 0; }
        .value { font-size: 24px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 12px; text-align: right; border-bottom: 1px solid #2a2e39; }
        th { color: #787b86; font-size: 12px; }
        .btn { border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-family: 'Tajawal'; font-weight: bold; transition: 0.3s; }
        .btn-red { background: rgba(250, 60, 88, 0.1); color: var(--red); }
        .btn-red:hover { background: var(--red); color: white; }
        .btn-main { background: var(--accent); color: white; width: 100%; padding: 15px; font-size: 16px; }
        .scroll-box { max-height: 400px; overflow-y: auto; }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: #363a45; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>💎 SmartBot <span style="color:var(--accent)">Fibonacci</span></h1>
            <div style="font-size:14px; color:#787b86">نظام تداول آلي يعتمد على مستويات الدعم والمقاومة الذكية</div>
        </div>
        <div id="statusIndicator">
            جاري التحميل...
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>حالة البوت</h3>
            <button id="toggleBtn" class="btn btn-main" onclick="toggleBot()">...</button>
        </div>
        <div class="card">
            <h3>حالة السوق (Regime)</h3>
            <div class="value" id="marketRegime">--</div>
        </div>
        <div class="card">
            <h3>الصفقات المفتوحة</h3>
            <div class="value" id="openCount">0</div>
        </div>
        <div class="card">
            <h3>الأرباح المحققة (الجلسة)</h3>
            <div class="value" style="color:var(--green)">$0.00</div> <!-- Placeholder -->
        </div>
    </div>

    <div class="grid" style="grid-template-columns: 2fr 1fr;">
        <div class="card">
            <h3>📊 الصفقات النشطة</h3>
            <table>
                <thead>
                    <tr>
                        <th>العملة</th>
                        <th>سعر الدخول</th>
                        <th>السعر الحالي</th>
                        <th>الربح/الخسارة</th>
                        <th>المستهدفات (Fibs)</th>
                        <th>إجراء</th>
                    </tr>
                </thead>
                <tbody id="tradesTable"></tbody>
            </table>
        </div>
        <div class="card scroll-box">
            <h3>📜 سجل العمليات (Logs)</h3>
            <div id="logsArea"></div>
        </div>
    </div>

    <script>
        async function fetchData() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();
                
                // 1. تحديث زر التشغيل
                const btn = document.getElementById('toggleBtn');
                if (data.settings.is_trading_enabled) {
                    btn.innerText = "🛑 إيقاف النظام";
                    btn.style.background = "var(--red)";
                } else {
                    btn.innerText = "🚀 تشغيل النظام";
                    btn.style.background = "var(--green)";
                }

                // 2. تحديث المعلومات
                document.getElementById('marketRegime').innerText = data.market;
                document.getElementById('openCount').innerText = data.signals.length;

                // 3. جدول الصفقات
                const tbody = document.getElementById('tradesTable');
                tbody.innerHTML = '';
                
                if (data.signals.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:20px; color:#555">لا توجد صفقات مفتوحة حالياً</td></tr>';
                } else {
                    data.signals.forEach(t => {
                        const curr = data.prices[t.symbol] || t.entry_price;
                        const pnl = ((curr - t.entry_price) / t.entry_price) * 100;
                        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
                        
                        const row = `
                            <tr>
                                <td style="font-weight:bold; color:#fff">${t.symbol}</td>
                                <td>${t.entry_price.toFixed(4)}</td>
                                <td style="color:#fff">${curr.toFixed(4)}</td>
                                <td style="color:${pnlColor}; direction:ltr; font-weight:bold">${pnl.toFixed(2)}%</td>
                                <td style="font-size:12px">
                                    <span style="color:var(--red)">SL: ${t.stop_loss.toFixed(4)}</span><br>
                                    <span style="color:var(--green)">TP: ${t.tp2.toFixed(4)}</span>
                                </td>
                                <td>
                                    <button class="btn btn-red" onclick="closeManual('${t.symbol}')">إغلاق ✖</button>
                                </td>
                            </tr>
                        `;
                        tbody.innerHTML += row;
                    });
                }

                // 4. السجلات
                const logsDiv = document.getElementById('logsArea');
                logsDiv.innerHTML = data.logs.map(l => `
                    <div style="padding:8px; border-bottom:1px solid #2a2e39; font-size:13px;">
                        <span style="color:#787b86">${l.t}</span>
                        <span style="color:#fff; font-weight:bold; margin:0 5px">${l.s}</span>
                        <span style="color:${l.st === 'دخول' ? 'var(--green)' : l.st === 'إغلاق' ? 'var(--red)' : '#787b86'}">${l.st}</span>
                        <span style="display:block; color:#555; font-size:11px">${l.r}</span>
                    </div>
                `).join('');

            } catch (err) { console.error(err); }
        }

        async function toggleBot() {
            await fetch('/api/toggle', { method: 'POST' });
            fetchData();
        }

        async function closeManual(symbol) {
            if(!confirm('هل أنت متأكد من إغلاق صفقة ' + symbol + ' يدوياً؟')) return;
            await fetch('/api/close_manual', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ symbol: symbol })
            });
            fetchData();
        }

        setInterval(fetchData, 2000);
        fetchData();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    # تشغيل البوت في مسار منفصل
    t = Thread(target=bot_engine)
    t.daemon = True
    t.start()
    
    # تشغيل السيرفر
    print("🚀 Server running on port 5000...")
    app.run(host='0.0.0.0', port=5000)