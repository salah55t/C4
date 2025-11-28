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
    "max_open_trades": 5,         # تقليل العدد للتركيز على الجودة
    "max_drawdown_protect": 10.0, # حماية من الانهيار
    "volume_lookback": 50,
    "timeframe_analysis": "15m",  # فريم الدخول
    "timeframe_trend": "1h"
}

# الرموز القيادية التي تحدد اتجاه السوق
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']

# حالة النظام
system_state = {
    "market_regime": "Neutral",   # Bull_Trend_Strong, Bull_Accumulation, Ranging, High_Volatility_Choppy, Bear_Trend
    "trend_strength": 0,          # 0 to 100
    "volatility_index": "Low",
    "global_score": 0,            # مجموع نقاط قوة السوق
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
        logger.info("✅ قاعدة البيانات جاهزة (V13).")
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
            f"🚀 *تنفيذ دخول استراتيجي | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"📊 الاستراتيجية: `{payload['strategy']}`\n"
            f"🌍 حالة السوق: {payload['regime']}\n"
            f"💵 السعر: `{payload['price']}`\n"
            f"🛑 الوقف: `{payload['sl']}`\n"
            f"🎯 الأهداف: `{payload['tp1']}` ➔ `{payload['tp2']}`\n"
            f"🕹️ الوضع: {mode_icon}"
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
    df['ema20'] = df['close'].ewm(span=20).mean() # Bollinger Mid
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
    df['bb_upper'] = df['ema20'] + (2 * df['close'].rolling(20).std())
    df['bb_lower'] = df['ema20'] - (2 * df['close'].rolling(20).std())
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ema20']

    # Ichimoku Tenkan (Simplified)
    high_9 = df['high'].rolling(9).max()
    low_9 = df['low'].rolling(9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2

    df['vol_ma'] = df['volume'].rolling(20).mean()
    return df.fillna(0)

# --- 6. محلل بيئة السوق المتطور (Advanced Market Regime) ---
def analyze_market_regime(client):
    """
    يقوم بتحليل السوق بناءً على الرموز القيادية في 3 فريمات زمنية:
    - 4 ساعات: لتحديد الهيكل العام (Structure).
    - 1 ساعة: لتحديد الاتجاه الحالي (Trend).
    - 15 دقيقة: لتحديد الزخم (Momentum).
    """
    global system_state
    
    total_score = 0
    analyzed_count = 0
    total_adx = 0
    total_atr_pct = 0
    
    # الفريمات المستخدمة
    timeframes = ['4h', '1h', '15m']
    tf_weights = {'4h': 0.5, '1h': 0.3, '15m': 0.2} # وزن أكبر للفريمات الكبيرة

    for symbol in LEADING_SYMBOLS:
        symbol_score = 0
        try:
            # جلب البيانات لكل الفريمات
            klines_4h = fetch_data(client, symbol, '4h', 60)
            klines_1h = fetch_data(client, symbol, '1h', 60)
            klines_15m = fetch_data(client, symbol, '15m', 60)
            
            if klines_4h is None or klines_1h is None or klines_15m is None:
                continue
                
            # حساب المؤشرات
            df_4h = calculate_technical_indicators(klines_4h).iloc[-1]
            df_1h = calculate_technical_indicators(klines_1h).iloc[-1]
            df_15m = calculate_technical_indicators(klines_15m).iloc[-1]
            
            # --- 1. تحليل هيكل 4 ساعات (الوزن 50%) ---
            score_4h = 0
            if df_4h['close'] > df_4h['ema200']: score_4h += 1    # فوق المتوسط طويل الأمد
            if df_4h['ema50'] > df_4h['ema200']: score_4h += 1    # ترتيب المتوسطات إيجابي
            if df_4h['rsi'] > 50: score_4h += 0.5                 # قوة نسبية
            # إذا كان السعر تحت المتوسطات، النقاط تصبح سلبية
            if df_4h['close'] < df_4h['ema200']: score_4h -= 1
            if df_4h['ema50'] < df_4h['ema200']: score_4h -= 1
            
            # --- 2. تحليل اتجاه 1 ساعة (الوزن 30%) ---
            score_1h = 0
            if df_1h['close'] > df_1h['ema50']: score_1h += 1
            if df_1h['macd_hist'] > 0: score_1h += 1
            if df_1h['close'] < df_1h['ema50']: score_1h -= 1
            
            # --- 3. تحليل زخم 15 دقيقة (الوزن 20%) ---
            score_15m = 0
            if df_15m['close'] > df_15m['ema20']: score_15m += 1 # فوق خط البولنجر الأوسط
            if df_15m['adx'] > 25: score_15m += 0.5              # وجود زخم
            if df_15m['close'] < df_15m['ema20']: score_15m -= 1

            # حساب الدرجة النهائية للعملة
            final_sym_score = (score_4h * tf_weights['4h']) + \
                              (score_1h * tf_weights['1h']) + \
                              (score_15m * tf_weights['15m'])
            
            total_score += final_sym_score
            analyzed_count += 1
            
            # تجميع بيانات التذبذب وقوة الاتجاه
            total_adx += df_1h['adx']
            total_atr_pct += (df_1h['atr'] / df_1h['close']) * 100
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    if analyzed_count == 0: return

    # حساب المتوسطات العامة
    avg_score = total_score / analyzed_count # Range approx -2.5 to +2.5
    avg_adx = total_adx / analyzed_count
    avg_atr_pct = total_atr_pct / analyzed_count
    
    # تحديد حالة السوق بناءً على النقاط المجمعة
    regime = "Neutral"
    
    if avg_score >= 1.5 and avg_adx > 25:
        regime = "Bull_Trend_Strong"      # صعود قوي جداً ومتفق عليه
    elif avg_score >= 0.5:
        if avg_adx < 20:
            regime = "Bull_Accumulation"  # صعود ضعيف أو تجميع
        else:
            regime = "Bull_Trend_Strong"  # صعود جيد
    elif avg_score <= -1.0:
        regime = "Bear_Trend_Strong"      # هبوط صريح
    elif avg_atr_pct > 2.5:
        regime = "High_Volatility_Choppy" # تذبذب عالي وخطير
    else:
        regime = "Ranging"                # حركة عرضية (بين -1 و 0.5)

    with locks['market']:
        system_state['market_regime'] = regime
        system_state['trend_strength'] = int(avg_adx)
        system_state['global_score'] = round(avg_score, 2)
        system_state['volatility_index'] = "High" if avg_atr_pct > 2.0 else "Normal"
        system_state['last_update'] = datetime.now()
    
    logger.info(f"🧠 Market Analysis: {regime} | Score: {avg_score:.2f} | ADX: {int(avg_adx)}")

# --- 7. مصنع الاستراتيجيات (Strategy Factory) ---
def get_smart_signal(symbol, df, regime):
    """
    يختار الاستراتيجية المناسبة بناءً على حالة السوق التي تم تحديدها
    من تحليل الرموز القيادية، مع تطبيق شروط الفريم الحالي (15 دقيقة).
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- فلتر السيولة الأساسي ---
    vol_factor = 0.5
    if last['volume'] < last['vol_ma'] * vol_factor:
        return None, "سيولة ضعيفة"

    # -----------------------------------------------------
    # الحالة 1: سوق صاعد قوي (Bull_Trend_Strong)
    # الاستراتيجية: Momentum Breakout (اختراق الزخم)
    # -----------------------------------------------------
    if regime == "Bull_Trend_Strong":
        # شروط الاختراق:
        # 1. السعر فوق جميع المتوسطات
        # 2. MACD هستوجرام إيجابي ومتزايد
        # 3. اختراق قمة الشمعة السابقة
        if last['close'] > last['ema20'] and last['close'] > last['ema50']:
            if last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']:
                if last['close'] > prev['high']:
                    return "Momentum_Breakout", "اختراق زخم مع الاتجاه العام"

    # -----------------------------------------------------
    # الحالة 2: تجميع صاعد (Bull_Accumulation)
    # الاستراتيجية: Trend Pullback (الشراء من الانخفاضات)
    # -----------------------------------------------------
    elif regime == "Bull_Accumulation":
        # شروط التصحيح:
        # 1. الاتجاه العام صاعد (فوق EMA200)
        # 2. السعر يقوم بتصحيح نحو Tenkan Sen أو EMA50
        # 3. شمعة خضراء تفلتر الدخول
        if last['close'] > last['ema200']:
            # السعر قريب من الدعم الديناميكي
            dist_to_ema50 = abs(last['close'] - last['ema50']) / last['close'] * 100
            if dist_to_ema50 < 1.5: # قريب جداً من المتوسط
                if last['rsi'] < 55 and last['rsi'] > 40: # ليس في تشبع شرائي
                    if last['close'] > last['open']: # شمعة تأكيد
                        return "Trend_Pullback", "ارتداد من دعم (تجميع)"

    # -----------------------------------------------------
    # الحالة 3: سوق عرضي (Ranging)
    # الاستراتيجية: Sniper Reversion (ارتداد من البولنجر)
    # -----------------------------------------------------
    elif regime == "Ranging":
        # شروط الارتداد:
        # 1. البولنجر باند ضيق (انضغاط)
        # 2. RSI في مناطق تشبع بيعي
        # 3. السعر يلمس الحد السفلي ثم يغلق فوقه
        if last['bb_width'] < 0.15: # السوق هادئ
            if last['rsi'] < 35: # تشبع بيعي
                if last['low'] <= last['bb_lower'] and last['close'] > last['bb_lower']:
                    return "Sniper_Reversion", "اقتناص قاع النطاق العرضي"

    # -----------------------------------------------------
    # الحالة 4: تذبذب عالي (High_Volatility)
    # الاستراتيجية: Deep Value Scalp (خطف سريع للانحرافات)
    # -----------------------------------------------------
    elif "High_Volatility" in regime:
        # انحراف سعري حاد عن المتوسط القصير (9)
        dist_ema9 = (last['close'] - last['ema9']) / last['ema9'] * 100
        if dist_ema9 < -4.0: # هبوط عنيف جداً بعيد عن المتوسط
            if last['volume'] > last['vol_ma'] * 2: # ذروة بيع (Climax)
                return "Deep_Value_Scalp", "ارتداد فني من ذروة البيع"

    # -----------------------------------------------------
    # الحالة 5: استراتيجية إضافية (التقاطع الذهبي)
    # تعمل في الأسواق الصاعدة أو المحايدة
    # -----------------------------------------------------
    if regime in ["Bull_Trend_Strong", "Bull_Accumulation", "Neutral"]:
        if last['ema50'] > last['ema200'] and prev['ema50'] <= prev['ema200']:
             return "Golden_Cross", "تقاطع إيجابي للمتوسطات"

    return None, "لا توجد إشارة"

# --- 8. مدير المحفظة والمخاطر ---
def manage_active_trade(symbol, signal, df):
    last = df.iloc[-1]
    curr = float(last['close'])
    entry = float(signal['entry_price'])
    tp1 = float(signal['tp1'])
    tp2 = float(signal['tp2'])
    sl = float(signal['stop_loss'])
    
    profit_pct = (curr - entry) / entry * 100
    duration = (datetime.now() - signal['entry_time']).total_seconds() / 3600

    health_msg = "مستقر"

    # 1. جني الأرباح المرحلي
    if curr >= tp2:
        # عند الوصول للهدف الثاني، نرفع الوقف للهدف الأول
        if sl < tp1: return "UPDATE_SL", tp1, "تأمين ربح الهدف الأول", "ربح ممتاز 🟢"
    elif curr >= tp1:
        # عند الوصول للهدف الأول، نرفع الوقف لنقطة الدخول + قليل من الربح
        if sl < entry: return "UPDATE_SL", entry * 1.002, "صفقة خالية من المخاطر", "مؤمنة 🛡️"

    # 2. وقف الخسارة المتحرك (Trailing Stop)
    # يتم تفعيله فقط إذا تجاوز الربح 1.5% والسوق صاعد
    if profit_pct > 1.5 and "Bull" in signal['market_regime']:
        atr_trail = curr - (last['atr'] * 2.0)
        if atr_trail > sl: return "UPDATE_SL", atr_trail, "ملاحقة الأرباح (ATR)", "منطلق 🏃"

    # 3. وقف الوقت (Time Stop)
    # إذا مرت 4 ساعات (16 شمعة ربع ساعة) والسعر لم يتحرك
    if duration > 4 and abs(profit_pct) < 0.6:
        return "CLOSE_NOW", curr, "تجميد رأس المال (خروج زمني)", "راكد ⚠️"

    # 4. الخروج الفني المبكر
    # إذا كسر السعر EMA50 بقوة في صفقة "Trend"
    if "Trend" in signal['strategy'] and curr < last['ema50']:
        if profit_pct < -0.5: # تأكيد السلبية
             return "CLOSE_NOW", curr, "فشل الاتجاه (كسر EMA50)", "انعكاس 🔻"

    return "HOLD", 0, "", health_msg

# --- 9. المحرك الرئيسي ---
def bot_engine():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 SmartBot V13 Enhanced Engine Started")
    
    try:
        with open('crypto_list.txt') as f:
            symbols = [l.strip().upper() for l in f if l.strip()]
            symbols = [s if s.endswith('USDT') else s+'USDT' for s in symbols]
    except: 
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT']

    while True:
        try:
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                paper = BOT_SETTINGS['paper_trading_mode']
                max_t = BOT_SETTINGS['max_open_trades']
                
            if not enabled: 
                time.sleep(5)
                continue

            # 1. تحليل السوق الشامل (تحديث كل دورة)
            analyze_market_regime(client)
            with locks['market']: regime = system_state['market_regime']

            # 2. إدارة الصفقات المفتوحة
            with locks['signals']: active_trades = list(open_signals_cache.values())
            
            for trade in active_trades:
                sym = trade['symbol']
                # نراقب الصفقات المفتوحة على فريم 5 دقائق لسرعة التفاعل
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
                            cur.execute("UPDATE trades_v13 SET stop_loss=%s WHERE id=%s", (float(val), trade['id']))
                        send_telegram("UPDATE", {"symbol": sym, "new_sl": val, "reason": note})
                        
                    elif act == "CLOSE_NOW":
                        exit_reason = f"خروج ذكي: {note}"
                
                if exit_reason:
                    close_trade_final(sym, curr_price, exit_reason, paper)

            # 3. البحث عن فرص جديدة
            if len(open_signals_cache) < max_t:
                tickers = client.get_ticker()
                valid = [t for t in tickers if t['symbol'] in symbols]
                # ترتيب العملات حسب الحجم والتغيير للعثور على العملات النشطة
                valid.sort(key=lambda x: float(x['quoteVolume']) * abs(float(x['priceChangePercent'])), reverse=True)
                
                count = 0
                for t in valid:
                    if count > 15: break # فحص أفضل 15 عملة فقط لتسريع الدورة
                    count += 1
                    
                    sym = t['symbol']
                    if sym in open_signals_cache: continue
                    
                    # تحليل العملة على الفريم المحدد في الإعدادات (عادة 15m)
                    df = fetch_data(client, sym, BOT_SETTINGS['timeframe_analysis'], 80)
                    if df is None: continue
                    df = calculate_technical_indicators(df)
                    
                    strat, reason = get_smart_signal(sym, df, regime)
                    
                    if strat:
                        curr = df['close'].iloc[-1]
                        atr = df['atr'].iloc[-1]
                        
                        # إدارة المخاطر: الأهداف والوقف بناءً على ATR
                        sl = curr - (atr * 2.0)
                        tp1 = curr + (atr * 2.5) # R:R > 1.2
                        tp2 = curr + (atr * 4.5) # R:R > 2.2
                        
                        # حساب حجم الصفقة
                        risk_amt = BOT_SETTINGS['base_capital'] * (BOT_SETTINGS['risk_per_trade_pct'] / 100)
                        price_diff = curr - sl
                        qty = risk_amt / price_diff if price_diff > 0 else 0
                        
                        if qty * curr > BOT_SETTINGS['base_capital'] * 0.25:
                            qty = (BOT_SETTINGS['base_capital'] * 0.25) / curr
                            
                        open_new_trade(sym, curr, sl, tp1, tp2, qty, strat, regime, paper)
                        time.sleep(1)
                    else:
                        # تسجيل عمليات الفحص بشكل عشوائي لعدم ملء السجل
                        if random.random() < 0.05:
                             with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'فحص', 'r': reason})
                    
                    time.sleep(0.1)

            time.sleep(10) # انتظار قليل قبل الدورة التالية

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            time.sleep(10)

# --- 10. أدوات قاعدة البيانات ---
def open_new_trade(symbol, price, sl, tp1, tp2, qty, strat, regime, is_paper):
    check_db()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        price, sl, tp1, tp2, qty = float(price), float(sl), float(tp1), float(tp2), float(qty)
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v13 
                (symbol, entry_price, stop_loss, tp1, tp2, quantity, strategy_name, market_regime, status, mode, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, NOW())
                RETURNING id
            """, (symbol, price, sl, tp1, tp2, qty, strat, regime, mode))
            db_id = cur.fetchone()['id']
        
        trade = {
            'id': db_id, 'symbol': symbol, 'entry_price': price, 'stop_loss': sl,
            'tp1': tp1, 'tp2': tp2, 'quantity': qty, 'entry_time': datetime.now(),
            'strategy': strat, 'market_regime': regime, 'is_paper': is_paper
        }
        
        with locks['signals']: open_signals_cache[symbol] = trade
        with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': symbol, 'st': 'دخول', 'r': strat})
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

# --- 11. واجهة التحكم العربية (Flask) ---
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
    <title>SmartBot V13 - لوحة التحكم</title>
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
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: right; padding: 12px; border-bottom: 1px solid var(--border); }
        th { color: #848e9c; font-size: 12px; }
        .pnl-g { color: var(--green); } .pnl-r { color: var(--red); }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
        
        @media(max-width: 768px) { .col-3, .col-4, .col-6, .col-8 { grid-column: span 12; } }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size:24px">SmartBot <span style="color:var(--accent)">V13 AR</span></h1>
            <span style="font-size:12px; color:#848e9c">نظام إدارة المحفظة الذكي - النسخة العربية</span>
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

    <script>
        let equityChart, statsChart;
        Chart.defaults.color = '#848e9c';
        Chart.defaults.borderColor = '#2b3139';
        Chart.defaults.font.family = 'Tajawal';

        // قاموس الترجمة لحالات السوق
        const regimeMap = {
            "Bull_Trend_Strong": "اتجاه صاعد قوي 🐂",
            "Bull_Accumulation": "تجميع صاعد 📈",
            "Bear_Trend_Strong": "اتجاه هابط قوي 🐻",
            "High_Volatility_Choppy": "تذبذب عالي ⚡",
            "Ranging": "عرضي مستقر 🦀",
            "Neutral": "محايد ⚖️"
        };

        // قاموس الترجمة للاستراتيجيات
        const stratMap = {
            "Trend_Pullback": "إعادة دخول (ترند)",
            "Momentum_Breakout": "اختراق زخم",
            "Sniper_Reversion": "قناص مرتد",
            "Deep_Value_Scalp": "خطف سيولة",
            "Golden_Cross": "تقاطع ذهبي"
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
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { position: 'bottom' } } }
            });
        }

        async function updateData() {
            try {
                const res = await fetch('/api/analytics');
                const d = await res.json();

                // 1. تحديث الأزرار
                const btn = document.getElementById('powerBtn');
                document.getElementById('connectionStatus').className = "status-dot dot-green";
                if(d.settings.is_trading_enabled) {
                    btn.innerText = "إيقاف البوت 🛑";
                    btn.style.background = "var(--red)";
                    btn.style.color = "#fff";
                } else {
                    btn.innerText = "تشغيل البوت 🚀";
                    btn.style.background = "var(--green)";
                    btn.style.color = "#fff";
                }

                // 2. المؤشرات
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

                // 3. تحديث الشارت
                if(d.stats.history.length > 0) {
                    equityChart.data.labels = d.stats.history.map(h => h.t);
                    equityChart.data.datasets[0].data = d.stats.history.map(h => h.v);
                    equityChart.update();
                    
                    statsChart.data.datasets[0].data = [d.stats.win_rate, 100 - d.stats.win_rate];
                    statsChart.update();
                }

                // 4. جدول الصفقات
                const tb = document.getElementById('tradesBody');
                tb.innerHTML = d.signals.length ? d.signals.map(s => {
                    const curr = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                    const stratName = stratMap[s.strategy] || s.strategy;
                    return `
                    <tr>
                        <td style="font-weight:bold; color:var(--text)">${s.symbol}</td>
                        <td><span style="background:#2b3139; padding:2px 6px; border-radius:4px; font-size:11px">${stratName}</span></td>
                        <td>${s.entry_price}</td>
                        <td>${curr}</td>
                        <td class="${pnl>=0?'pnl-g':'pnl-r'}">${pnl.toFixed(2)}%</td>
                        <td style="font-size:11px; color:#848e9c">${s.tp1} ➔ ${s.tp2}</td>
                    </tr>`;
                }).join('') : "<tr><td colspan='6' style='text-align:center; padding:20px; color:#444'>لا توجد صفقات نشطة حالياً</td></tr>";

                // 5. السجل
                document.getElementById('logsBody').innerHTML = d.logs.map(l => `
                    <tr>
                        <td style="color:#666">${l.t}</td>
                        <td style="font-weight:bold">${l.s}</td>
                        <td style="color:${l.st==='دخول'?'var(--green)':'#848e9c'}">${l.st}</td>
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
    logger.info("🖥️ لوحة التحكم العربية تعمل على المنفذ 5000")
    app.run(host='0.0.0.0', port=5000)