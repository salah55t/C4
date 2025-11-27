import time
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import json
from threading import Thread, Lock
from datetime import datetime, timedelta
from collections import deque
from decouple import config
from binance.client import Client
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor
import warnings

# --- 1. إعدادات النظام الأساسية ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('smart_bot_v2.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Pro')

# تحميل المتغيرات البيئية
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ في الإعدادات (ملف .env): {e}")
    # قيم افتراضية لمنع توقف الكود عند الاختبار بدون ملف بيئة
    API_KEY = "test"
    API_SECRET = "test"
    DB_URL = "postgres://user:pass@localhost:5432/db" 

# --- 2. الإعدادات الديناميكية (Dynamic Settings) ---
# هذه الإعدادات قابلة للتعديل الآن من واجهة الويب
DEFAULT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,
    "risk_per_trade_pct": 2.0,  # 2% مخاطرة من رأس المال
    "max_open_trades": 5,
    "leverage": 1,              # للتداول الفوري (Spot) اتركها 1
    "take_profit_ratio": 2.0,   # العائد مقابل المخاطرة (Risk:Reward)
    "stop_loss_atr_mult": 2.0,  # مضاعف ATR لوقف الخسارة
    "timeframe": "15m"
}

# محاولة تحميل الإعدادات من ملف محلي للحفاظ عليها بعد إعادة التشغيل
try:
    with open('bot_settings.json', 'r') as f:
        BOT_SETTINGS = json.load(f)
except:
    BOT_SETTINGS = DEFAULT_SETTINGS.copy()

def save_settings():
    with locks['settings']:
        with open('bot_settings.json', 'w') as f:
            json.dump(BOT_SETTINGS, f)

# --- 3. المتغيرات العامة والأقفال ---
system_state = {
    "market_regime": "Analyzing...",
    "btc_price": 0,
    "last_update": datetime.now()
}

open_signals_cache = {} # الصفقات المفتوحة في الذاكرة
live_prices = {}
scan_logs = deque(maxlen=100)

locks = {
    'signals': Lock(), 'prices': Lock(), 'market': Lock(), 
    'settings': Lock(), 'logs': Lock(), 'db': Lock()
}

# --- 4. قاعدة البيانات ---
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
        logger.info("✅ قاعدة البيانات جاهزة (V14).")
    except Exception as e: 
        logger.error(f"⚠️ تنبيه قاعدة البيانات: {e} - سيعمل البوت في وضع الذاكرة فقط.")

def check_db():
    global conn
    if conn is None or conn.closed != 0: 
        try: init_db()
        except: pass

# --- 5. التحليل الفني المتقدم (Advanced Technical Engine) ---
def fetch_data(client, symbol, interval, limit=100):
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        logger.debug(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_indicators(df):
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
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR (Volatility)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2*std)
    df['bb_lower'] = df['bb_mid'] - (2*std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # ADX (Trend Strength)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    df['plus_di'] = 100 * (pd.Series(plus_dm).rolling(14).mean() / df['atr'])
    df['minus_di'] = 100 * (pd.Series(minus_dm).rolling(14).mean() / df['atr'])
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(14).mean()

    return df.fillna(0)

# --- 6. منطق الاستراتيجيات (The Brain) ---
def analyze_strategy(symbol, df, regime):
    """
    تحليل السوق باستخدام استراتيجيات مطورة
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. فلتر السيولة (Volume Filter)
    # لا ندخل إذا كان حجم التداول الحالي أقل بكثير من المتوسط
    vol_ma = df['volume'].rolling(20).mean().iloc[-1]
    if last['volume'] < (vol_ma * 0.4):
        return None, "Low Volume"

    # --- Strategy A: BB Squeeze Breakout (انفجار البولنجر) ---
    # فعالة في بداية تحرك قوي بعد هدوء
    if last['bb_width'] < 0.15: # السوق كان هادئاً (Squeeze)
        # شرط الاختراق للأعلى
        if last['close'] > last['bb_upper'] and last['volume'] > vol_ma:
            if last['rsi'] > 50 and last['rsi'] < 75: # ليس متشبعاً جداً بعد
                return "BB_Squeeze_Breakout", "اختراق سعري مع سيولة"

    # --- Strategy B: RSI Bullish Divergence (الانحراف الإيجابي) ---
    # استراتيجية انعكاس قوية (Price Low, RSI Higher Low)
    # تتطلب منطقاً معقداً قليلاً لمقارنة القيعان، سنستخدم نسخة مبسطة
    if last['rsi'] < 40:
        # السعر الحالي أقل من أدنى سعر في 10 شمعات سابقة، لكن RSI الحالي أعلى من RSI السابق
        lowest_price_10 = df['close'].rolling(10).min().iloc[-2]
        if last['close'] <= lowest_price_10 and last['rsi'] > df['rsi'].iloc[-2]:
             if last['macd_hist'] > prev['macd_hist']: # بداية تحسن الزخم
                 return "RSI_Div_Reversal", "انحراف إيجابي (ارتداد)"

    # --- Strategy C: Trend Pullback (إعادة اختبار الاتجاه) ---
    # الدخول مع الاتجاه العام
    if last['adx'] > 25 and last['close'] > last['ema200']: # اتجاه قوي
        # السعر عاد للمتوسط 21 أو 50 وارتد منه
        if last['low'] <= last['ema21'] and last['close'] > last['ema21']:
            if last['stoch_k'] if 'stoch_k' in last else last['rsi'] < 50:
                 return "Trend_Smart_Pullback", "تصحيح في اتجاه صاعد"

    return None, ""

# --- 7. محرك البوت (Engine) ---
def bot_engine():
    # محاكاة العميل في حالة عدم وجود مفاتيح
    try:
        client = Client(API_KEY, API_SECRET)
        # اختبار الاتصال
        if API_KEY != 'test': client.get_account()
    except:
        logger.warning("⚠️ يعمل البوت في وضع المحاكاة الكاملة (بدون اتصال بـ Binance)")
        client = None

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT']

    logger.info("🚀 SmartBot Pro Engine Started...")

    while True:
        try:
            # قراءة الإعدادات الحالية
            with locks['settings']:
                cfg = BOT_SETTINGS.copy()
            
            # 1. تحديث حالة السوق العامة (BTC)
            if client:
                try:
                    btc_data = fetch_data(client, 'BTCUSDT', '1h', 50)
                    if btc_data is not None:
                        df_btc = calculate_indicators(btc_data)
                        last_btc = df_btc.iloc[-1]
                        
                        regime = "Neutral"
                        if last_btc['close'] > last_btc['ema200']: regime = "Bullish"
                        elif last_btc['close'] < last_btc['ema200']: regime = "Bearish"
                        
                        with locks['market']:
                            system_state['market_regime'] = regime
                            system_state['btc_price'] = last_btc['close']
                            system_state['last_update'] = datetime.now()
                except Exception as e: logger.error(f"Market Check Error: {e}")

            if not cfg['is_trading_enabled']:
                time.sleep(5)
                continue

            # 2. إدارة الصفقات المفتوحة (Trailing Stop & TP)
            with locks['signals']:
                active_trades = list(open_signals_cache.values())

            for trade in active_trades:
                sym = trade['symbol']
                current_price = trade['entry_price'] # افتراضي
                
                # جلب السعر الحالي
                if client:
                    try:
                        ticker = client.get_symbol_ticker(symbol=sym)
                        current_price = float(ticker['price'])
                    except: pass
                else:
                    # محاكاة حركة السعر للاختبار
                    import random
                    move = random.uniform(-0.002, 0.003)
                    current_price = trade['entry_price'] * (1 + move)

                with locks['prices']: live_prices[sym] = current_price

                # فحص الخروج
                exit_reason = None
                pnl_pct = (current_price - trade['entry_price']) / trade['entry_price'] * 100

                # أ) وقف الخسارة
                if current_price <= trade['stop_loss']:
                    exit_reason = "Stop Loss Hit 🛑"
                
                # ب) الهدف الثاني (Full TP)
                elif current_price >= trade['tp2']:
                    exit_reason = "Take Profit 2 Hit 🎯"

                # ج) التحديث الذكي (Trailing Stop)
                elif current_price >= trade['tp1']:
                    # رفع الوقف إلى نقطة الدخول لحماية الأرباح
                    new_sl = trade['entry_price'] * 1.002 # دخول + رسوم بسيطة
                    if new_sl > trade['stop_loss']:
                        trade['stop_loss'] = new_sl
                        update_trade_sl_db(trade['id'], new_sl)
                        logger.info(f"🛡️ Trailing Stop Activated for {sym}")

                if exit_reason:
                    close_trade(sym, current_price, exit_reason)

            # 3. البحث عن فرص جديدة
            if len(open_signals_cache) < cfg['max_open_trades']:
                for sym in symbols:
                    if sym in open_signals_cache: continue # العملة موجودة بالفعل
                    
                    if client:
                        df = fetch_data(client, sym, cfg['timeframe'], 100)
                    else:
                        # بيانات وهمية للاختبار
                        continue 

                    if df is not None:
                        df = calculate_indicators(df)
                        strategy_name, reason = analyze_strategy(sym, df, system_state['market_regime'])
                        
                        if strategy_name:
                            # حساب الكمية وإدارة المخاطر
                            curr = df['close'].iloc[-1]
                            atr = df['atr'].iloc[-1]
                            
                            # المسافة لوقف الخسارة بناء على ATR
                            sl_dist = atr * cfg['stop_loss_atr_mult']
                            sl = curr - sl_dist
                            
                            # المسافة للهدف بناء على نسبة العائد للمخاطرة
                            risk = curr - sl
                            reward = risk * cfg['take_profit_ratio']
                            tp1 = curr + (reward * 0.5)
                            tp2 = curr + reward
                            
                            # حجم الصفقة بناء على المخاطرة بالدولار
                            # Risk Amount = Capital * Risk%
                            risk_amt = cfg['base_capital'] * (cfg['risk_per_trade_pct'] / 100)
                            # Qty = Risk Amount / (Entry - SL)
                            qty = risk_amt / (curr - sl)
                            
                            # حماية: لا تتجاوز 20% من رأس المال في صفقة واحدة
                            max_pos_size = cfg['base_capital'] * 0.25
                            if (qty * curr) > max_pos_size:
                                qty = max_pos_size / curr

                            enter_trade(sym, curr, sl, tp1, tp2, qty, strategy_name, cfg['paper_trading_mode'])
                            time.sleep(1) # تفادي حظر API
                        else:
                            # تسجيل للمراجعة
                            pass

            time.sleep(10) # انتظار الدورة التالية

        except Exception as e:
            logger.error(f"Engine Loop Error: {e}")
            time.sleep(5)

# --- 8. وظائف إدارة الصفقات (DB & State) ---
def enter_trade(symbol, price, sl, tp1, tp2, qty, strat, is_paper):
    check_db()
    trade_id = 0
    mode = 'PAPER' if is_paper else 'REAL'
    try:
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades_v14 
                    (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, status, mode)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s) RETURNING id
                """, (symbol, price, sl, tp1, tp2, qty, strat, mode))
                trade_id = cur.fetchone()['id']
        else:
            trade_id = int(time.time()) # ID مؤقت

        trade_obj = {
            'id': trade_id, 'symbol': symbol, 'entry_price': price, 
            'stop_loss': sl, 'tp1': tp1, 'tp2': tp2, 'quantity': qty,
            'strategy': strat, 'entry_time': datetime.now()
        }
        
        with locks['signals']: open_signals_cache[symbol] = trade_obj
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'a': 'BUY', 'm': strat})
        
        send_telegram(f"🚀 *New Trade: {symbol}*\nStrat: {strat}\nEntry: {price}\nSL: {sl}\nTP: {tp2}")

    except Exception as e: logger.error(f"Trade Entry Error: {e}")

def update_trade_sl_db(trade_id, new_sl):
    check_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE trades_v14 SET stop_loss=%s WHERE id=%s", (new_sl, trade_id))
        except: pass

def close_trade(symbol, price, reason):
    check_db()
    trade = None
    with locks['signals']:
        if symbol in open_signals_cache:
            trade = open_signals_cache.pop(symbol)
    
    if trade:
        profit_pct = ((price - trade['entry_price']) / trade['entry_price']) * 100
        profit_abs = (price - trade['entry_price']) * trade['quantity']
        
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE trades_v14 
                        SET status='closed', closed_at=NOW(), closing_price=%s, profit_pct=%s, profit_abs=%s, exit_reason=%s
                        WHERE id=%s
                    """, (price, profit_pct, profit_abs, reason, trade['id']))
            except Exception as e: logger.error(f"DB Close Error: {e}")
        
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'a': 'SELL', 'm': f"{profit_pct:.2f}% ({reason})"})
        
        emoji = "✅" if profit_pct > 0 else "🔻"
        send_telegram(f"{emoji} *Closed: {symbol}*\nPrice: {price}\nPnL: {profit_pct:.2f}%\nReason: {reason}")
        return True
    return False

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- 9. واجهة الويب (Flask + HTML/JS SPA) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    with locks['market']: m = system_state.copy()
    with locks['signals']: s = list(open_signals_cache.values())
    with locks['prices']: p = live_prices.copy()
    with locks['logs']: l = list(scan_logs)
    with locks['settings']: cfg = BOT_SETTINGS.copy()
    
    # تحويل التواريخ لنصوص لJSON
    for trade in s:
        if isinstance(trade.get('entry_time'), datetime):
            trade['entry_time'] = trade['entry_time'].strftime('%H:%M:%S')

    return jsonify({"market": m, "trades": s, "prices": p, "logs": l, "config": cfg})

@app.route('/api/close_trade/<symbol>', methods=['POST'])
def manual_close(symbol):
    """API لإغلاق صفقة يدوياً"""
    with locks['prices']: price = live_prices.get(symbol, 0)
    if price == 0: 
        # محاولة جلب السعر من الصفقات إذا لم يتوفر السعر المباشر
        with locks['signals']:
            if symbol in open_signals_cache:
                price = open_signals_cache[symbol]['entry_price']

    result = close_trade(symbol, price, "Manual Close via Dashboard 👤")
    if result: return jsonify({"status": "success", "msg": f"تم إغلاق {symbol} بنجاح"})
    return jsonify({"status": "error", "msg": "الصفقة غير موجودة أو مغلقة بالفعل"}), 400

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    """API لتحديث الإعدادات"""
    data = request.json
    global BOT_SETTINGS
    with locks['settings']:
        # تحديث القيم الموجودة فقط
        for key in BOT_SETTINGS:
            if key in data:
                # تحويل الأنواع
                if isinstance(BOT_SETTINGS[key], bool):
                    BOT_SETTINGS[key] = bool(data[key])
                elif isinstance(BOT_SETTINGS[key], float):
                    BOT_SETTINGS[key] = float(data[key])
                elif isinstance(BOT_SETTINGS[key], int):
                    BOT_SETTINGS[key] = int(data[key])
                else:
                    BOT_SETTINGS[key] = data[key]
        save_settings()
    
    logger.info("⚙️ Settings updated via Dashboard")
    return jsonify({"status": "success", "settings": BOT_SETTINGS})

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot Pro V2</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <style>
        :root { --bg: #12161c; --card: #1e2329; --border: #2b3139; --text: #eaecef; --green: #0ecb81; --red: #f6465d; --gold: #f0b90b; --blue: #3b82f6; }
        body { background: var(--bg); color: var(--text); font-family: 'Tajawal', sans-serif; margin: 0; padding: 0; }
        
        /* Layout */
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }
        .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; }
        
        /* Cards */
        .card { background: var(--card); border-radius: 12px; padding: 20px; border: 1px solid var(--border); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .card-header { display: flex; justify-content: space-between; margin-bottom: 15px; color: #848e9c; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
        
        /* Metrics */
        .metric-value { font-size: 24px; font-weight: 700; margin-top: 5px; }
        .metric-label { font-size: 12px; color: #848e9c; }
        
        /* Table */
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th { text-align: right; color: #848e9c; padding: 10px; font-weight: 500; font-size: 12px; border-bottom: 1px solid var(--border); }
        td { padding: 12px 10px; border-bottom: 1px solid #2b3139; font-size: 14px; }
        
        /* Buttons & Badges */
        .btn { padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer; font-family: inherit; font-weight: 600; transition: 0.2s; }
        .btn-primary { background: var(--gold); color: #000; }
        .btn-danger { background: rgba(246, 70, 93, 0.1); color: var(--red); }
        .btn-danger:hover { background: var(--red); color: #fff; }
        .btn-settings { background: var(--card); border: 1px solid var(--border); color: var(--text); }
        
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
        .badge-strategy { background: rgba(59, 130, 246, 0.1); color: var(--blue); }
        
        /* Settings Modal */
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; display: none; align-items: center; justify-content: center; }
        .modal { background: var(--card); width: 500px; max-width: 90%; border-radius: 12px; padding: 25px; border: 1px solid var(--border); position: relative; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 8px; font-size: 13px; color: #848e9c; }
        .form-control { width: 100%; background: #12161c; border: 1px solid var(--border); color: #fff; padding: 10px; border-radius: 6px; box-sizing: border-box; }
        .switch { position: relative; display: inline-block; width: 50px; height: 24px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }
        .slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: var(--green); }
        input:checked + .slider:before { transform: translateX(26px); }

        /* Helper Classes */
        .text-green { color: var(--green); }
        .text-red { color: var(--red); }
        .col-3 { grid-column: span 3; } .col-4 { grid-column: span 4; } .col-8 { grid-column: span 8; } .col-12 { grid-column: span 12; }
        @media(max-width: 768px) { .col-3, .col-4, .col-8 { grid-column: span 12; } }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <h2 style="margin:0">SmartBot <span style="color:var(--gold)">Pro</span></h2>
                <span style="font-size:12px; color:#848e9c">نظام تداول آلي متقدم</span>
            </div>
            <div style="display:flex; gap:10px">
                <button class="btn btn-settings" onclick="openSettings()"><i class="fas fa-cog"></i> الإعدادات</button>
                <div id="statusIndicator" style="padding: 8px 15px; background: rgba(14, 203, 129, 0.1); color: var(--green); border-radius: 6px; font-size: 12px; font-weight: bold;">
                    <i class="fas fa-circle" style="font-size:8px"></i> النظام يعمل
                </div>
            </div>
        </div>

        <!-- Metrics Grid -->
        <div class="grid" style="margin-bottom: 20px;">
            <div class="card col-3">
                <div class="card-header"><i class="fas fa-chart-line"></i> حالة السوق</div>
                <div class="metric-value" id="marketRegime">--</div>
                <div class="metric-label">سعر BTC: <span id="btcPrice">--</span></div>
            </div>
            <div class="card col-3">
                <div class="card-header"><i class="fas fa-wallet"></i> الرصيد المتوقع</div>
                <div class="metric-value" id="equity">$0.00</div>
                <div class="metric-label">بناءً على الصفقات المغلقة</div>
            </div>
            <div class="card col-3">
                <div class="card-header"><i class="fas fa-bolt"></i> الصفقات النشطة</div>
                <div class="metric-value" id="activeCount">0</div>
                <div class="metric-label">من أصل <span id="maxTrades">--</span> مسموح بها</div>
            </div>
            <div class="card col-3">
                <div class="card-header"><i class="fas fa-shield-alt"></i> وضع التداول</div>
                <div class="metric-value" id="tradeMode">--</div>
                <div class="metric-label">نوع التنفيذ</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="grid">
            <!-- Trades Table -->
            <div class="card col-8">
                <div class="card-header">
                    <span>الصفقات الجارية</span>
                    <i class="fas fa-list"></i>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>الزوج</th>
                            <th>الاستراتيجية</th>
                            <th>الدخول</th>
                            <th>الحالي</th>
                            <th>الربح %</th>
                            <th>TP/SL</th>
                            <th>إجراء</th>
                        </tr>
                    </thead>
                    <tbody id="tradesBody">
                        <tr><td colspan="7" style="text-align:center; padding:20px">جاري تحميل البيانات...</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Logs -->
            <div class="card col-4">
                <div class="card-header">
                    <span>سجل الأحداث (Live Logs)</span>
                    <i class="fas fa-history"></i>
                </div>
                <div id="logsBody" style="height: 300px; overflow-y: auto; font-size: 12px;">
                </div>
            </div>
        </div>
    </div>

    <!-- Settings Modal -->
    <div class="modal-overlay" id="settingsModal">
        <div class="modal">
            <div style="display:flex; justify-content:space-between; margin-bottom:20px">
                <h3>⚙️ إعدادات البوت</h3>
                <button onclick="closeSettings()" style="background:none; border:none; color:#fff; cursor:pointer; font-size:18px">&times;</button>
            </div>
            
            <div class="form-group" style="display:flex; justify-content:space-between; align-items:center; background:#12161c; padding:10px; border-radius:6px">
                <label style="margin:0; color:#fff">تفعيل التداول (Master Switch)</label>
                <label class="switch">
                    <input type="checkbox" id="set_enabled">
                    <span class="slider"></span>
                </label>
            </div>

            <div class="form-group" style="display:flex; justify-content:space-between; align-items:center; background:#12161c; padding:10px; border-radius:6px">
                <label style="margin:0; color:#fff">وضع التجربة (Paper Trading)</label>
                <label class="switch">
                    <input type="checkbox" id="set_paper">
                    <span class="slider"></span>
                </label>
            </div>

            <div class="grid" style="grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="form-group">
                    <label>رأس المال الأساسي ($)</label>
                    <input type="number" id="set_capital" class="form-control">
                </div>
                <div class="form-group">
                    <label>المخاطرة لكل صفقة (%)</label>
                    <input type="number" id="set_risk" class="form-control" step="0.1">
                </div>
                <div class="form-group">
                    <label>أقصى صفقات متزامنة</label>
                    <input type="number" id="set_max_trades" class="form-control">
                </div>
                <div class="form-group">
                    <label>نسبة الهدف للمخاطرة (R:R)</label>
                    <input type="number" id="set_rr" class="form-control" step="0.1">
                </div>
            </div>

            <button class="btn btn-primary" style="width:100%; margin-top:15px; padding:12px" onclick="saveSettings()">حفظ وتطبيق التغييرات</button>
        </div>
    </div>

    <script>
        // دالة لجلب البيانات وتحديث الواجهة
        async function updateDashboard() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();
                
                // تحديث المؤشرات العلوية
                document.getElementById('marketRegime').innerText = data.market.market_regime;
                document.getElementById('btcPrice').innerText = "$" + data.market.btc_price.toLocaleString();
                document.getElementById('activeCount').innerText = data.trades.length;
                document.getElementById('maxTrades').innerText = data.config.max_open_trades;
                document.getElementById('tradeMode').innerText = data.config.paper_trading_mode ? "Paper (تجريبي)" : "Real (حقيقي)";
                document.getElementById('tradeMode').style.color = data.config.paper_trading_mode ? "var(--gold)" : "var(--green)";

                const statusEl = document.getElementById('statusIndicator');
                if(!data.config.is_trading_enabled) {
                    statusEl.style.background = "rgba(246, 70, 93, 0.1)";
                    statusEl.style.color = "var(--red)";
                    statusEl.innerHTML = '<i class="fas fa-pause-circle"></i> متوقف مؤقتاً';
                } else {
                    statusEl.style.background = "rgba(14, 203, 129, 0.1)";
                    statusEl.style.color = "var(--green)";
                    statusEl.innerHTML = '<i class="fas fa-circle" style="font-size:8px"></i> النظام يعمل';
                }

                // تحديث الجدول
                const tbody = document.getElementById('tradesBody');
                if (data.trades.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:30px; color:#444">لا توجد صفقات نشطة، جاري المسح... <i class="fas fa-radar"></i></td></tr>';
                } else {
                    tbody.innerHTML = data.trades.map(t => {
                        const curr = data.prices[t.symbol] || t.entry_price;
                        const pnl = ((curr - t.entry_price) / t.entry_price) * 100;
                        const pnlClass = pnl >= 0 ? 'text-green' : 'text-red';
                        const sign = pnl >= 0 ? '+' : '';
                        
                        return `
                        <tr>
                            <td style="font-weight:bold; color:#fff">${t.symbol}</td>
                            <td><span class="badge badge-strategy">${t.strategy}</span></td>
                            <td>${t.entry_price.toFixed(4)}</td>
                            <td style="color:#fff">${curr.toFixed(4)}</td>
                            <td class="${pnlClass}" style="font-weight:bold">${sign}${pnl.toFixed(2)}%</td>
                            <td style="font-size:11px; color:#848e9c">TP: ${t.tp2.toFixed(4)}<br>SL: ${t.stop_loss.toFixed(4)}</td>
                            <td>
                                <button class="btn btn-danger" style="padding:4px 10px; font-size:12px" onclick="manualClose('${t.symbol}')">
                                    <i class="fas fa-times"></i> إغلاق
                                </button>
                            </td>
                        </tr>
                        `;
                    }).join('');
                }

                // تحديث السجل
                const logsContainer = document.getElementById('logsBody');
                logsContainer.innerHTML = data.logs.map(l => `
                    <div style="padding:8px; border-bottom:1px solid #2b3139; display:flex; justify-content:space-between">
                        <span style="color:#666">${l.t}</span>
                        <span style="font-weight:bold; color:${l.a==='BUY'?'var(--green)':'var(--red)'}">${l.a} ${l.s}</span>
                        <span style="color:#848e9c">${l.m}</span>
                    </div>
                `).join('');

                // تحديث قيم الـ Modal مرة واحدة (عند فتحها فقط لتجنب الكتابة فوق المدخلات)
                if(!window.settingsOpened) {
                    document.getElementById('set_enabled').checked = data.config.is_trading_enabled;
                    document.getElementById('set_paper').checked = data.config.paper_trading_mode;
                    document.getElementById('set_capital').value = data.config.base_capital;
                    document.getElementById('set_risk').value = data.config.risk_per_trade_pct;
                    document.getElementById('set_max_trades').value = data.config.max_open_trades;
                    document.getElementById('set_rr').value = data.config.take_profit_ratio;
                }

            } catch (e) { console.error("Update Error:", e); }
        }

        // إغلاق صفقة يدوياً
        function manualClose(symbol) {
            Swal.fire({
                title: 'هل أنت متأكد؟',
                text: `سيتم إغلاق صفقة ${symbol} بسعر السوق فوراً`,
                icon: 'warning',
                showCancelButton: true,
                confirmButtonColor: '#f6465d',
                cancelButtonColor: '#2b3139',
                confirmButtonText: 'نعم، إغلاق الآن',
                cancelButtonText: 'إلغاء',
                background: '#1e2329',
                color: '#fff'
            }).then((result) => {
                if (result.isConfirmed) {
                    fetch(`/api/close_trade/${symbol}`, { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        if(data.status === 'success') {
                            Swal.fire({title:'تم!', text: data.msg, icon:'success', background:'#1e2329', color:'#fff', timer: 1500, showConfirmButton:false});
                            updateDashboard();
                        } else {
                            Swal.fire('خطأ', data.msg, 'error');
                        }
                    });
                }
            })
        }

        // إدارة المودال
        function openSettings() {
            document.getElementById('settingsModal').style.display = 'flex';
            window.settingsOpened = true;
        }
        function closeSettings() {
            document.getElementById('settingsModal').style.display = 'none';
            window.settingsOpened = false;
        }

        // حفظ الإعدادات
        function saveSettings() {
            const payload = {
                is_trading_enabled: document.getElementById('set_enabled').checked,
                paper_trading_mode: document.getElementById('set_paper').checked,
                base_capital: parseFloat(document.getElementById('set_capital').value),
                risk_per_trade_pct: parseFloat(document.getElementById('set_risk').value),
                max_open_trades: parseInt(document.getElementById('set_max_trades').value),
                take_profit_ratio: parseFloat(document.getElementById('set_rr').value)
            };

            fetch('/api/update_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).then(r => r.json()).then(data => {
                closeSettings();
                Swal.fire({
                    title: 'تم الحفظ',
                    text: 'تم تحديث إعدادات البوت بنجاح',
                    icon: 'success',
                    background: '#1e2329',
                    color: '#fff',
                    timer: 1500,
                    showConfirmButton: false
                });
                updateDashboard();
            });
        }

        // التشغيل التلقائي
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    # تشغيل البوت في خيط منفصل
    bot_thread = Thread(target=bot_engine, daemon=True)
    bot_thread.start()
    
    logger.info("🖥️ SmartBot Pro Dashboard running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)