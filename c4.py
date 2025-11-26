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
    handlers=[logging.FileHandler('smart_bot_v14.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_Arab_V14')

try:
    API_KEY = config('BINANCE_API_KEY', default='')
    API_SECRET = config('BINANCE_API_SECRET', default='')
    DB_URL = config('DATABASE_URL', default='')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ خطأ في الإعدادات: {e}")
    # لا نغلق البرنامج للسماح بالتشغيل التجريبي بدون API أحياناً
    
# --- 2. إعدادات التداول (المحفظة الذكية) ---
BOT_SETTINGS = {
    "is_trading_enabled": False,
    "paper_trading_mode": True,
    "base_capital": 1000.0,       # رأس المال
    "risk_per_trade_pct": 2.0,    # نسبة المخاطرة لكل صفقة
    "max_open_trades": 5,         # أقصى عدد صفقات
    "timeframe_analysis": "15m",  # فريم التحليل
    "commission_rate": 0.1,       # عمولة المنصة % (لكل جانب)
    "min_score_entry": 60         # أقل نقاط للدخول
}

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
    if not DB_URL: return
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
                    commission_paid DOUBLE PRECISION,
                    exit_reason TEXT
                );
            """)
        logger.info("✅ قاعدة البيانات جاهزة (V14).")
    except Exception as e: logger.error(f"خطأ قاعدة البيانات: {e}")

def check_db():
    global conn
    if conn is None or (conn and conn.closed != 0): init_db()

# --- 4. نظام التنبيهات ---
def send_telegram(event, payload):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    mode_icon = "🧪 تجريبي" if payload.get('is_paper') else "💰 حقيقي"
    msg = ""

    if event == "BUY":
        msg = (
            f"🚀 *دخول ذكي جديد | {payload['symbol']}*\n"
            f"ــــــــــــــــــــــــــــــــــــــــ\n"
            f"📊 الاستراتيجية: `{payload['strategy']}`\n"
            f"💵 السعر: `{payload['price']}`\n"
            f"🛑 الوقف: `{payload['sl']}`\n"
            f"🎯 هدف 1: `{payload['tp1']}`\n"
            f"🎯 هدف 2: `{payload['tp2']}`\n"
            f"⚖️ المخاطرة: `{payload['risk_r']:.2f}`\n"
            f"🕹️ الوضع: {mode_icon}"
        )
    elif event == "SELL":
        pnl = payload['profit']
        net_pnl = payload.get('net_profit', 0)
        emoji = "✅ ربح" if net_pnl > 0 else "🔻 خسارة"
        msg = (
            f"{emoji} *إغلاق صفقة | {payload['symbol']}*\n"
            f"ــــــــــــــــــــــــــــــــــــــــ\n"
            f"📉 الخروج: `{payload['price']}`\n"
            f"💰 الصافي (بعد العمولة): `{net_pnl:.2f}$`\n"
            f"📊 النسبة: `{pnl:.2f}%`\n"
            f"📝 السبب: _{payload['reason']}_\n"
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

# --- 5. محرك التحليل الفني المتقدم (New Logic) ---
def fetch_data(client, symbol, interval, limit=100):
    try:
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except: return None

# 🟢 دوال التحليل المتقدمة والمؤشرات
def advanced_volume_analysis(df, period=20):
    """تحليل متقدم للسيولة"""
    df = df.copy()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(period).mean()
    df['volume_spike'] = df['volume_ratio'] > 2.0
    # ADL
    chl = (df['close'] - df['low']) - (df['high'] - df['close'])
    h_l = df['high'] - df['low']
    # تجنب القسمة على صفر
    h_l = h_l.replace(0, 0.000001)
    df['adl'] = (chl / h_l) * df['volume']
    df['adl_ma'] = df['adl'].rolling(period).mean()
    return df

def volume_confirmation(df, min_ratio=1.2):
    last = df.iloc[-1]
    return last['volume_ratio'] >= min_ratio or last['volume_spike']

def add_leading_indicators(df):
    """إضافة مؤشرات متقدمة"""
    # TR calculation first for Vortex
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Vortex
    df['vm+'] = abs(df['high'] - df['low'].shift(1))
    df['vm-'] = abs(df['low'] - df['high'].shift(1))
    df['vm_14+'] = df['vm+'].rolling(14).sum()
    df['vm_14-'] = df['vm-'].rolling(14).sum()
    tr14 = df['tr'].rolling(14).sum()
    df['vortex+'] = df['vm_14+'] / tr14
    df['vortex-'] = df['vm_14-'] / tr14
    
    # Supertrend
    atr_multiplier = 3
    hl2 = (df['high'] + df['low']) / 2
    df['supertrend_upper'] = hl2 + (atr_multiplier * df['atr'])
    df['supertrend_lower'] = hl2 - (atr_multiplier * df['atr'])
    
    return df

def price_structure_analysis(df, lookback=50):
    """تحليل هيكل السعر والمقاومة والدعم"""
    highs = df['high'].tail(lookback)
    lows = df['low'].tail(lookback)
    
    resistance = highs.rolling(5).max()
    support = lows.rolling(5).min()
    
    df['higher_high'] = df['high'] > df['high'].shift(1)
    df['higher_low'] = df['low'] > df['low'].shift(1)
    df['lower_high'] = df['high'] < df['high'].shift(1)
    df['lower_low'] = df['low'] < df['low'].shift(1)
    
    uptrend = df['higher_high'].tail(3).sum() >= 2 and df['higher_low'].tail(3).sum() >= 2
    downtrend = df['lower_high'].tail(3).sum() >= 2 and df['lower_low'].tail(3).sum() >= 2
    
    last_close = df['close'].iloc[-1]
    last_res = resistance.iloc[-1]
    last_sup = support.iloc[-1]

    return {
        'resistance': last_res,
        'support': last_sup,
        'trend': 'uptrend' if uptrend else 'downtrend' if downtrend else 'ranging',
        'distance_to_resistance': (last_res - last_close) / last_close * 100,
        'distance_to_support': (last_close - last_sup) / last_close * 100
    }

def calculate_full_technical_indicators(df):
    # دمج كل المؤشرات
    df = df.copy()
    # أساسيات
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic
    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_k'] = ((df['rsi'] - min_rsi) / (max_rsi - min_rsi)) * 100
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    # ATR is calculated in add_leading_indicators but needed here roughly
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    atr_tmp = tr.rolling(14).mean()
    df['plus_di'] = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr_tmp)
    df['minus_di'] = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr_tmp)
    dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.rolling(14).mean()

    # الإضافات الجديدة
    df = advanced_volume_analysis(df)
    df = add_leading_indicators(df)
    
    return df.fillna(0)

def weighted_signal_scoring(symbol, df, regime, price_structure):
    """نظام تصنيف مرجح للإشارات"""
    score = 0
    last = df.iloc[-1]
    
    # 1. المؤشرات التقنية (40 نقطة)
    tech_score = 0
    if last['ema50'] > last['ema200']: tech_score += 8
    if last['close'] > last['ema50']: tech_score += 7
    if 30 < last['rsi'] < 70: tech_score += 5
    if last['macd'] > last['macd_signal']: tech_score += 5
    if last['stoch_k'] > 20 and last['stoch_k'] < 80: tech_score += 5
    if last['adx'] > 25: tech_score += 10
    score += tech_score
    
    # 2. السيولة والحجم (20 نقطة)
    volume_score = 0
    if volume_confirmation(df): volume_score += 20
    score += volume_score
    
    # 3. الهيكل السعري (25 نقطة)
    price_score = 0
    if price_structure['trend'] == 'uptrend': price_score += 10
    if price_structure['distance_to_resistance'] > 2: price_score += 10
    if price_structure['distance_to_support'] < 3: price_score += 5
    score += price_score
    
    # 4. سياق السوق (15 نقطة)
    market_score = 0
    if "Bull" in regime: market_score += 15
    elif "Ranging" in regime: market_score += 8
    elif "Bear" in regime: market_score += 3
    score += market_score
    
    grade = 'A' if score >= 80 else 'B' if score >= 65 else 'C' if score >= 50 else 'D'
    
    return {
        'total_score': score,
        'tech_score': tech_score,
        'volume_score': volume_score,
        'grade': grade,
        'market_score': market_score # Added for checks
    }

def adaptive_risk_management(df, regime, signal_score):
    """إدارة مخاطر تكيفية"""
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    # تحديد المضاعفات حسب السوق
    if "High_Volatility" in regime:
        atr_multiplier_sl = 1.5; atr_multiplier_tp = 2.5; risk_multiplier = 0.7
    elif "Bull_Trend" in regime:
        atr_multiplier_sl = 1.8; atr_multiplier_tp = 3.0; risk_multiplier = 1.2
    else:
        atr_multiplier_sl = 2.0; atr_multiplier_tp = 3.5; risk_multiplier = 1.0
    
    # تعديل حسب الجودة
    grade = signal_score['grade']
    if grade == 'A': risk_multiplier *= 1.2
    elif grade == 'C': risk_multiplier *= 0.8
    elif grade == 'D': risk_multiplier *= 0.5
    
    sl = close - (atr * atr_multiplier_sl)
    tp1 = close + (atr * atr_multiplier_tp)
    tp2 = close + (atr * (atr_multiplier_tp + 1.0))
    
    return sl, tp1, tp2, risk_multiplier

def get_enhanced_smart_signal(symbol, df, regime):
    """الاستراتيجية المحسنة"""
    # التحليلات
    price_structure = price_structure_analysis(df)
    signal_score = weighted_signal_scoring(symbol, df, regime, price_structure)
    
    min_score = BOT_SETTINGS.get('min_score_entry', 60)
    if signal_score['total_score'] < min_score:
        # تسجيل محاولات مثيرة للاهتمام فقط
        return None, None, f"سكور ضعيف: {signal_score['total_score']}"
    
    last = df.iloc[-1]
    strategy_name = None
    reason = ""

    # 1. استراتيجية الاتجاه القوي
    if signal_score['tech_score'] >= 30 and signal_score['market_score'] >= 10:
        if (last['vortex+'] > last['vortex-'] and 
            last['close'] > last['supertrend_upper'] and 
            volume_confirmation(df)):
            strategy_name = "Enhanced_Trend"
            reason = f"اتجاه قوي A+ (Score: {signal_score['total_score']})"
    
    # 2. استراتيجية الارتداد المدعوم
    elif (price_structure['distance_to_support'] < 1.5 and 
          signal_score['volume_score'] >= 15 and 
          last['rsi'] < 35):
        strategy_name = "Enhanced_Reversion"
        reason = f"ارتداد دعم (Score: {signal_score['total_score']})"
    
    # 3. استراتيجية الاختراق
    elif (price_structure['distance_to_resistance'] < 1.0 and 
          volume_confirmation(df) and 
          last['adx'] > 25):
        strategy_name = "Enhanced_Breakout"
        reason = f"اختراق مؤكد (Score: {signal_score['total_score']})"

    if strategy_name:
        return strategy_name, signal_score, reason
    
    return None, None, f"لا توجد استراتيجية مطابقة ({signal_score['total_score']})"

# --- 6. مدير السوق والمحفظة ---
def analyze_market_regime(client):
    global system_state
    try:
        btc_df = fetch_data(client, 'BTCUSDT', '4h', 100)
        if btc_df is None: return

        btc_df = calculate_full_technical_indicators(btc_df)
        last = btc_df.iloc[-1]

        trend_score = 0
        if last['close'] > last['ema200']: trend_score += 1
        if last['ema50'] > last['ema200']: trend_score += 1
        
        adx = last['adx']
        atr_pct = (last['atr'] / last['close']) * 100
        
        regime = "Neutral"
        if trend_score >= 2 and adx > 25: regime = "Bull_Trend_Strong"
        elif trend_score >= 1 and adx < 20: regime = "Bull_Accumulation"
        elif trend_score == 0 and adx > 25: regime = "Bear_Trend"
        elif atr_pct > 2.0: regime = "High_Volatility"
        else: regime = "Ranging"

        with locks['market']:
            system_state['market_regime'] = regime
            system_state['trend_strength'] = int(adx)
            system_state['last_update'] = datetime.now()
    except Exception as e:
        logger.error(f"Market Regime Error: {e}")

def manage_active_trade(symbol, signal, df):
    last = df.iloc[-1]
    curr = float(last['close'])
    entry = float(signal['entry_price'])
    sl = float(signal['stop_loss'])
    tp1 = float(signal['tp1'])
    tp2 = float(signal['tp2'])
    
    # إذا كان هناك طلب إغلاق يدوي محفوظ (يمكن إضافته هنا إذا لزم الأمر، لكننا سنعالج الإغلاق اليدوي مباشرة عبر الـ API)
    
    profit_pct = (curr - entry) / entry * 100
    
    # الخروج الطارئ
    if curr <= sl: return "CLOSE_STOP", sl, "ضرب وقف الخسارة", "خروج 🔴"
    
    # تحديثات الأهداف
    if curr >= tp2:
        if sl < tp1: return "UPDATE_SL", tp1, "تأمين الربح عند الهدف 1", "ممتاز 🟢"
    elif curr >= tp1:
        if sl < entry: return "UPDATE_SL", entry * 1.002, "تأمين الدخول (Break-Even)", "مؤمن 🛡️"
        
    # وقف الخسارة المتحرك (Trailing Stop)
    if profit_pct > 1.5:
        # استخدام ATR للتحريك
        atr_trail = curr - (last['atr'] * 2.5)
        if atr_trail > sl: return "UPDATE_SL", atr_trail, "ملاحقة (ATR Trailing)", "متابع 🏃"

    return "HOLD", 0, "", "مستقر"

# --- 7. المحرك الرئيسي ---
def bot_engine():
    client = Client(API_KEY, API_SECRET)
    logger.info("🚀 SmartBot V14 Enhanced Engine Started")
    
    # قائمة العملات (يمكن تحسينها)
    default_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'MATICUSDT', 'DOGEUSDT']
    
    while True:
        try:
            with locks['settings']:
                enabled = BOT_SETTINGS['is_trading_enabled']
                paper = BOT_SETTINGS['paper_trading_mode']
                max_t = BOT_SETTINGS['max_open_trades']
                timeframe = BOT_SETTINGS['timeframe_analysis']
                
            if not enabled: 
                time.sleep(5)
                continue

            # تحديث حالة السوق
            analyze_market_regime(client)
            with locks['market']: regime = system_state['market_regime']

            # إدارة الصفقات المفتوحة
            with locks['signals']: active_trades = list(open_signals_cache.values())
            
            for trade in active_trades:
                sym = trade['symbol']
                df = fetch_data(client, sym, '5m', 60) # فريم سريع للإدارة
                if df is None: continue
                df = calculate_full_technical_indicators(df)
                
                curr_price = df['close'].iloc[-1]
                with locks['prices']: live_prices[sym] = curr_price
                
                act, val, note, health = manage_active_trade(sym, trade, df)
                
                if act == "CLOSE_STOP":
                    close_trade_final(sym, curr_price, note, paper)
                elif act == "UPDATE_SL":
                    open_signals_cache[sym]['stop_loss'] = float(val)
                    check_db()
                    with conn.cursor() as cur:
                        cur.execute("UPDATE trades_v14 SET stop_loss=%s WHERE id=%s", (float(val), trade['id']))
                    send_telegram("UPDATE", {"symbol": sym, "new_sl": val, "reason": note})

            # البحث عن فرص جديدة
            if len(open_signals_cache) < max_t:
                tickers = client.get_ticker()
                # فلترة وترتيب سريع حسب الحجم
                valid = [t for t in tickers if t['symbol'].endswith('USDT') and float(t['quoteVolume']) > 10000000]
                valid.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                
                # فحص أعلى 15 عملة
                for t in valid[:15]:
                    sym = t['symbol']
                    if sym in open_signals_cache: continue
                    
                    df = fetch_data(client, sym, timeframe, 100)
                    if df is None: continue
                    df = calculate_full_technical_indicators(df)
                    
                    strat, score_obj, reason = get_enhanced_smart_signal(sym, df, regime)
                    
                    if strat:
                        # حساب المخاطر التكيفي
                        sl, tp1, tp2, risk_mult = adaptive_risk_management(df, regime, score_obj)
                        
                        # حساب الكمية
                        curr = df['close'].iloc[-1]
                        base_risk_pct = BOT_SETTINGS['risk_per_trade_pct']
                        account_risk = BOT_SETTINGS['base_capital'] * (base_risk_pct / 100) * risk_mult
                        
                        dist_to_sl = curr - sl
                        if dist_to_sl <= 0: continue # حماية
                        
                        qty = account_risk / dist_to_sl
                        # حماية الحد الأقصى للصفقة الواحدة (25% من المحفظة)
                        if qty * curr > BOT_SETTINGS['base_capital'] * 0.25:
                            qty = (BOT_SETTINGS['base_capital'] * 0.25) / curr
                        
                        open_new_trade(sym, curr, sl, tp1, tp2, qty, strat, regime, paper, risk_mult)
                        time.sleep(1) # تفادي حدود الـ API
                    else:
                        if "سكور" in reason and random.random() < 0.1:
                             with locks['logs']: scan_logs.appendleft({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'فحص', 'r': reason})
                    
                    time.sleep(0.1)

            time.sleep(10)

        except Exception as e:
            logger.error(f"Engine Loop Error: {e}")
            time.sleep(5)

# --- 8. عمليات قاعدة البيانات والتنفيذ ---
def open_new_trade(symbol, price, sl, tp1, tp2, qty, strat, regime, is_paper, risk_r):
    check_db()
    try:
        mode = 'PAPER' if is_paper else 'REAL'
        price, sl, tp1, tp2, qty = map(float, [price, sl, tp1, tp2, qty])
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v14 
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
        
        send_telegram("BUY", {**trade, 'price': price, 'sl': sl, 'risk_r': risk_r})
        
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
        qty = float(trade['quantity'])
        entry = float(trade['entry_price'])
        
        # حساب الربح والعمولة
        gross_profit = (price - entry) * qty
        comm_rate = BOT_SETTINGS.get('commission_rate', 0.1) / 100
        commission = (entry * qty * comm_rate) + (price * qty * comm_rate)
        net_profit_abs = gross_profit - commission
        profit_pct = ((price - entry) / entry) * 100
        
        # تحديث المحفظة الافتراضية
        with locks['settings']:
            if is_paper:
                BOT_SETTINGS['base_capital'] += net_profit_abs

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trades_v14 
                SET status='closed', closed_at=NOW(), closing_price=%s, 
                    profit_pct=%s, profit_abs=%s, commission_paid=%s, exit_reason=%s
                WHERE id=%s
            """, (price, profit_pct, net_profit_abs, commission, reason, trade['id']))
            
        send_telegram("SELL", {'symbol': symbol, 'price': price, 'profit': profit_pct, 'net_profit': net_profit_abs, 'reason': reason})
        
    except Exception as e: logger.error(f"DB Close Error: {e}")

# --- 9. واجهة الويب API ---
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
    with locks['settings']: settings = BOT_SETTINGS.copy()
    
    stats = {'win_rate': 0, 'profit_factor': 0, 'total_pnl_usd': 0, 'trade_count': 0, 'history': [], 'total_commissions': 0}
    try:
        check_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT closed_at, profit_pct, profit_abs, commission_paid
                FROM trades_v14 WHERE status='closed' ORDER BY closed_at ASC
            """)
            rows = cur.fetchall()
            
            wins = 0
            gross_p = 0
            gross_l = 0
            cum_pnl = settings['base_capital'] # البدء برأس المال
            
            for r in rows:
                net_pnl = r['profit_abs'] # مخزنة بالفعل صافية بعد العمولة
                if net_pnl > 0: 
                    wins += 1
                    gross_p += net_pnl
                else: 
                    gross_l += abs(net_pnl)
                
                cum_pnl += net_pnl
                # إذا كانت commission_paid غير موجودة (قديمة) نعتبرها 0
                comm = r.get('commission_paid') or 0
                stats['total_commissions'] += comm
                
                stats['history'].append({'t': r['closed_at'].strftime('%d %H:%M'), 'v': cum_pnl})
            
            stats['trade_count'] = len(rows)
            stats['total_pnl_usd'] = gross_p - gross_l
            stats['win_rate'] = (wins / len(rows) * 100) if len(rows) > 0 else 0
            stats['profit_factor'] = (gross_p / gross_l) if gross_l > 0 else 99.9
            
    except: pass
    
    return jsonify({"market": m, "signals": s, "prices": p, "stats": stats, "logs": l, "settings": settings})

@app.route('/api/toggle', methods=['POST'])
def toggle():
    with locks['settings']: BOT_SETTINGS['is_trading_enabled'] = not BOT_SETTINGS['is_trading_enabled']
    return jsonify("OK")

@app.route('/api/close/<symbol>', methods=['POST'])
def manual_close(symbol):
    current_price = live_prices.get(symbol, 0)
    if current_price == 0: return jsonify({"error": "السعر غير متوفر"}), 400
    
    # إغلاق فوري
    close_trade_final(symbol, current_price, "إغلاق يدوي من اللوحة", BOT_SETTINGS['paper_trading_mode'])
    return jsonify({"status": "closed", "symbol": symbol})

@app.route('/api/save_settings', methods=['POST'])
def save_settings():
    data = request.json
    with locks['settings']:
        # تحديث القيم القابلة للتعديل فقط
        if 'base_capital' in data: BOT_SETTINGS['base_capital'] = float(data['base_capital'])
        if 'risk_per_trade_pct' in data: BOT_SETTINGS['risk_per_trade_pct'] = float(data['risk_per_trade_pct'])
        if 'max_open_trades' in data: BOT_SETTINGS['max_open_trades'] = int(data['max_open_trades'])
        if 'min_score_entry' in data: BOT_SETTINGS['min_score_entry'] = int(data['min_score_entry'])
    return jsonify("Settings Updated")

# --- HTML Dashboard (Updated) ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot V14 - Enhanced</title>
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
        .btn { background: var(--accent); color: #000; border: none; padding: 8px 20px; border-radius: 4px; font-weight: bold; cursor: pointer; transition: 0.2s; font-family: 'Tajawal'; }
        .btn:hover { opacity: 0.9; }
        .btn-red { background: var(--red); color: white; padding: 4px 10px; font-size: 12px; }
        .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text); }
        
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: right; padding: 12px; border-bottom: 1px solid var(--border); }
        th { color: #848e9c; font-size: 12px; }
        .pnl-g { color: var(--green); } .pnl-r { color: var(--red); }
        
        /* Modal */
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 100; }
        .modal-content { background: var(--panel); width: 400px; margin: 100px auto; padding: 20px; border-radius: 8px; border: 1px solid var(--border); }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #848e9c; }
        .form-group input { width: 100%; padding: 8px; background: var(--bg); border: 1px solid var(--border); color: white; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="margin:0; font-size:24px">SmartBot <span style="color:var(--accent)">V14</span></h1>
            <span style="font-size:12px; color:#848e9c">المحرك المتقدم + إدارة المخاطر التكيفية</span>
        </div>
        <div style="display:flex; gap:10px;">
            <button class="btn btn-outline" onclick="openSettings()">⚙️ الإعدادات</button>
            <button id="powerBtn" class="btn" onclick="toggleBot()">جاري التحميل...</button>
        </div>
    </div>

    <!-- مؤشرات الأداء -->
    <div class="grid">
        <div class="card col-3">
            <h3>حالة السوق</h3>
            <div id="regime" class="big-num" style="color:var(--accent); font-size:18px">--</div>
            <div class="sub-text">قوة: <span id="trendStr">0</span> | V: <span id="volIdx">--</span></div>
        </div>
        <div class="card col-3">
            <h3>صافي الربح (بعد العمولة)</h3>
            <div id="totalPnl" class="big-num">$0.00</div>
            <div class="sub-text">العمولات المدفوعة: <span id="commPaid" style="color:#f6465d">0</span>$</div>
        </div>
        <div class="card col-3">
            <h3>أداء الاستراتيجية</h3>
            <div class="big-num"><span id="winRate">0</span><small>%</small></div>
            <div class="sub-text">PF: <span id="profFact">0</span> | صفقات: <span id="tradeCount">0</span></div>
        </div>
        <div class="card col-3">
            <h3>رأس المال (Equity)</h3>
            <div class="big-num" id="equityVal">0</div>
            <div class="sub-text">متوفر للتداول</div>
        </div>
    </div>

    <!-- الرسم البياني -->
    <div class="grid">
        <div class="card col-12">
            <h3>منحنى نمو المحفظة</h3>
            <div style="height: 300px;"><canvas id="equityChart"></canvas></div>
        </div>
    </div>

    <!-- الصفقات -->
    <div class="grid">
        <div class="card col-8">
            <h3>الصفقات النشطة</h3>
            <table>
                <thead>
                    <tr>
                        <th>الرمز</th>
                        <th>الاستراتيجية</th>
                        <th>دخول</th>
                        <th>حالي</th>
                        <th>P&L%</th>
                        <th>الهدف</th>
                        <th>إجراء</th>
                    </tr>
                </thead>
                <tbody id="tradesBody"></tbody>
            </table>
        </div>
        <div class="card col-4">
            <h3>سجل العمليات</h3>
            <div style="height: 300px; overflow-y: auto;">
                <table style="font-size:12px">
                    <tbody id="logsBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Modal Settings -->
    <div id="settingsModal" class="modal">
        <div class="modal-content">
            <h2 style="margin-top:0">⚙️ إعدادات البوت</h2>
            <div class="form-group">
                <label>رأس المال الأساسي (Base Capital)</label>
                <input type="number" id="set_capital">
            </div>
            <div class="form-group">
                <label>المخاطرة لكل صفقة (%)</label>
                <input type="number" id="set_risk" step="0.1">
            </div>
            <div class="form-group">
                <label>أقصى عدد صفقات</label>
                <input type="number" id="set_max_trades">
            </div>
            <div class="form-group">
                <label>أقل سكور للدخول (Entry Score)</label>
                <input type="number" id="set_min_score">
            </div>
            <div style="display:flex; gap:10px; margin-top:20px">
                <button class="btn" onclick="saveSettings()">حفظ وتطبيق</button>
                <button class="btn btn-outline" onclick="closeSettings()">إلغاء</button>
            </div>
        </div>
    </div>

    <script>
        let equityChart;
        Chart.defaults.color = '#848e9c';
        Chart.defaults.borderColor = '#2b3139';
        
        // Mappings
        const regimeMap = { "Bull_Trend_Strong": "صاعد قوي 🚀", "Bull_Accumulation": "تجميع 📈", "Bear_Trend": "هابط 🐻", "High_Volatility": "تذبذب عالي ⚡", "Ranging": "عرضي 🦀", "Neutral": "محايد" };

        function initCharts() {
            const ctx = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(ctx, {
                type: 'line',
                data: { labels: [], datasets: [{ label: 'رصيد المحفظة', data: [], borderColor: '#f0b90b', backgroundColor: 'rgba(240, 185, 11, 0.1)', fill: true, tension: 0.3 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false } } }
            });
        }

        async function updateData() {
            try {
                const res = await fetch('/api/analytics');
                const d = await res.json();

                // تحديث القيم العلوية
                const btn = document.getElementById('powerBtn');
                if(d.settings.is_trading_enabled) { btn.innerText = "🛑 إيقاف"; btn.style.background = "var(--red)"; }
                else { btn.innerText = "🚀 تشغيل"; btn.style.background = "var(--green)"; }

                document.getElementById('regime').innerText = regimeMap[d.market.market_regime] || d.market.market_regime;
                document.getElementById('trendStr').innerText = d.market.trend_strength;
                document.getElementById('totalPnl').innerText = "$" + d.stats.total_pnl_usd.toFixed(2);
                document.getElementById('totalPnl').className = "big-num " + (d.stats.total_pnl_usd >= 0 ? "pnl-g" : "pnl-r");
                document.getElementById('commPaid').innerText = d.stats.total_commissions.toFixed(2);
                document.getElementById('winRate').innerText = d.stats.win_rate.toFixed(1);
                document.getElementById('profFact').innerText = d.stats.profit_factor.toFixed(2);
                document.getElementById('tradeCount').innerText = d.stats.trade_count;
                document.getElementById('equityVal').innerText = "$" + d.settings.base_capital.toFixed(0);

                // الرسم البياني
                if(d.stats.history.length > 0) {
                    equityChart.data.labels = d.stats.history.map(h => h.t);
                    equityChart.data.datasets[0].data = d.stats.history.map(h => h.v);
                    equityChart.update();
                }

                // الصفقات
                document.getElementById('tradesBody').innerHTML = d.signals.length ? d.signals.map(s => {
                    const curr = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                    return `
                    <tr>
                        <td style="font-weight:bold">${s.symbol}</td>
                        <td><span style="background:#2b3139; padding:2px 6px; border-radius:4px; font-size:11px">${s.strategy}</span></td>
                        <td>${s.entry_price}</td>
                        <td>${curr}</td>
                        <td class="${pnl>=0?'pnl-g':'pnl-r'}">${pnl.toFixed(2)}%</td>
                        <td>${s.tp1}</td>
                        <td><button class="btn btn-red" onclick="closeTrade('${s.symbol}')">X</button></td>
                    </tr>`;
                }).join('') : "<tr><td colspan='7' style='text-align:center; padding:20px; color:#444'>لا توجد صفقات</td></tr>";

                // السجل
                document.getElementById('logsBody').innerHTML = d.logs.map(l => `
                    <tr><td style="color:#666">${l.t}</td><td style="font-weight:bold">${l.s}</td><td style="color:${l.st==='دخول'?'var(--green)':'#848e9c'}">${l.st}</td><td>${l.r}</td></tr>
                `).join('');

                // تعبئة بيانات الإعدادات (مرة واحدة أو عند الفتح يمكن تحسينها)
                if(!document.getElementById('settingsModal').style.display || document.getElementById('settingsModal').style.display === 'none') {
                    document.getElementById('set_capital').value = d.settings.base_capital;
                    document.getElementById('set_risk').value = d.settings.risk_per_trade_pct;
                    document.getElementById('set_max_trades').value = d.settings.max_open_trades;
                    document.getElementById('set_min_score').value = d.settings.min_score_entry;
                }

            } catch(e) { console.error(e); }
        }

        function toggleBot() { fetch('/api/toggle', {method:'POST'}).then(updateData); }
        
        function closeTrade(symbol) {
            if(!confirm('هل أنت متأكد من إغلاق صفقة ' + symbol + ' يدوياً؟')) return;
            fetch('/api/close/'+symbol, {method:'POST'})
            .then(r => r.json())
            .then(d => { alert('تم الإغلاق: ' + d.symbol); updateData(); });
        }

        // Settings Functions
        function openSettings() { document.getElementById('settingsModal').style.display = 'block'; }
        function closeSettings() { document.getElementById('settingsModal').style.display = 'none'; }
        function saveSettings() {
            const data = {
                base_capital: document.getElementById('set_capital').value,
                risk_per_trade_pct: document.getElementById('set_risk').value,
                max_open_trades: document.getElementById('set_max_trades').value,
                min_score_entry: document.getElementById('set_min_score').value
            };
            fetch('/api/save_settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            }).then(() => { closeSettings(); updateData(); });
        }

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