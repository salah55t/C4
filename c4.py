# smart_bot_part1.py
# --- الجزء الأول: الإعدادات، قاعدة البيانات، وتحليل هيكل السوق المتقدم ---

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import statistics
import random
from decimal import Decimal, ROUND_DOWN, getcontext
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from flask_sock import Sock
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any
from collections import deque
import warnings
from scipy.signal import argrelextrema

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
getcontext().prec = 18

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('smart_bot.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('SmartBot_V2')

# --- المشفر المخصص JSON ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Decimal): return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        return super(NpEncoder, self).default(obj)

# --- متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ فشل تحميل المتغيرات: {e}"); exit(1)

# --- المتغيرات العامة وحالة السوق ---
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'] # الرموز القيادية لتحديد اتجاه السوق
current_market_regime: str = "sideways" # bullish, bearish, sideways, volatile
market_score: int = 50 # 0-100
is_trading_enabled: bool = False
paper_trading_mode: bool = True
usdt_balance: float = 0.0
open_signals_cache: Dict[str, Dict] = {}
live_prices: Dict[str, float] = {}

# أقفال (Locks)
locks = {
    'trade': Lock(), 'balance': Lock(), 'signals': Lock(), 
    'prices': Lock(), 'market': Lock(), 'log': Lock()
}

# --- إعدادات قاعدة البيانات (Postgres) ---
conn: Optional[psycopg2.extensions.connection] = None

def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        with conn.cursor() as cur:
            # جدول الصفقات
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION,
                    stop_loss DOUBLE PRECISION, target_price_1 DOUBLE PRECISION, target_price_2 DOUBLE PRECISION,
                    status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                    profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                    is_real_trade BOOLEAN DEFAULT FALSE, quantity DOUBLE PRECISION, closing_reason TEXT,
                    last_analysis_time TIMESTAMP, analysis_notes TEXT
                );
            """)
            # جدول الإشعارات
            cur.execute("CREATE TABLE IF NOT EXISTS notifications (id SERIAL PRIMARY KEY, timestamp TIMESTAMP DEFAULT NOW(), type TEXT, message TEXT);")
        logger.info("✅ تم تهيئة قاعدة البيانات بنجاح.")
    except Exception as e:
        logger.error(f"❌ خطأ في قاعدة البيانات: {e}")

def check_db_connection():
    global conn
    try:
        if conn is None or conn.closed != 0: init_db()
        with conn.cursor() as cur: cur.execute("SELECT 1")
        return True
    except: return False

# --- إعداد Redis ---
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# --- 🚀 الجديد: تحليل هيكل السوق المتقدم (Market Structure Analysis) ---
def fetch_historical_data(client, symbol, interval, days) -> Optional[pd.DataFrame]:
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in df.columns: df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except: return None

def analyze_market_structure(client):
    """
    تحليل السوق بناءً على الرموز القيادية وليس فقط البيتكوين.
    يحدد: الاتجاه العام (Regime) وقوة الاتجاه (Score).
    """
    global current_market_regime, market_score
    
    scores = []
    details = {}
    
    logger.info("🔍 جاري تحليل هيكل السوق عبر الرموز القيادية...")
    
    for symbol in LEADING_SYMBOLS:
        df = fetch_historical_data(client, symbol, '1h', 3) # فريم الساعة لآخر 3 أيام
        if df is None or len(df) < 50: continue
        
        # حساب المؤشرات البسيطة للاتجاه
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        df['rsi'] = 100 - (100 / (1 + df['close'].diff().apply(lambda x: x if x>0 else 0).rolling(14).mean() / df['close'].diff().apply(lambda x: -x if x<0 else 0).rolling(14).mean()))
        
        last = df.iloc[-1]
        symbol_score = 50
        
        # تقييم الاتجاه
        if last['close'] > last['ema50']: symbol_score += 10
        if last['close'] > last['ema200']: symbol_score += 10
        if last['ema50'] > last['ema200']: symbol_score += 10 # Golden Cross Alignment
        if 50 < last['rsi'] < 70: symbol_score += 5
        if last['rsi'] > 70: symbol_score -= 5 # Overbought warning
        
        scores.append(symbol_score)
        details[symbol] = symbol_score

    if not scores: return

    avg_score = sum(scores) / len(scores)
    
    # تحديد نظام السوق بناءً على متوسط درجات العملات القيادية
    with locks['market']:
        market_score = avg_score
        if avg_score >= 75:
            current_market_regime = "bullish_strong" # صاعد بقوة
        elif avg_score >= 60:
            current_market_regime = "bullish" # صاعد
        elif avg_score <= 30:
            current_market_regime = "bearish" # هابط (خطر للدخول)
        elif avg_score <= 45:
            current_market_regime = "bearish_weak" # هابط بضعف
        else:
            current_market_regime = "sideways" # جانبي

    logger.info(f"🌐 حالة السوق المحدثة: {current_market_regime.upper()} (Score: {avg_score:.1f})")
    # حفظ الحالة في Redis
    redis_client.set('market_regime', json.dumps({'regime': current_market_regime, 'score': avg_score, 'details': details}))

# --- نهاية الجزء الأول ---
# smart_bot_part2.py
# --- الجزء الثاني: حساب المؤشرات وتعريف الاستراتيجيات ---

# --- دالة حساب المؤشرات الفنية الشاملة ---
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # المتوسطات المتحركة
    for span in [7, 21, 50, 100, 200]:
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # ADX (مبسط)
    df['adx'] = df['atr_pct'].rolling(14).mean() * 10 # تقريبي للأغراض السريعة

    return df.fillna(0)

# --- 🚀 الجديد: استراتيجيات مخصصة لكل حالة سوق ---

# 1. استراتيجية الزخم (للسوق الصاعد)
def strategy_momentum_bullish(df):
    last = df.iloc[-1]
    # شروط قوية: السعر فوق كل المتوسطات، RSI قوي لكن ليس متشبع جداً، الماكد إيجابي
    if (last['close'] > last['ema21'] > last['ema50']) and \
       (55 < last['rsi'] < 75) and \
       (last['macd_hist'] > 0) and \
       (last['adx'] > 20): # وجود اتجاه
        return True
    return False

# 2. استراتيجية الارتداد (Pullback) (للسوق الصاعد القوي)
def strategy_pullback_bullish(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # اتجاه عام صاعد (فوق EMA200)
    if last['close'] > last['ema200']:
        # السعر يلمس EMA50 أو يقترب منها ثم يرتد
        if (last['low'] <= last['ema50'] * 1.005) and (last['close'] > last['ema50']):
            # تأكيد من الستوكاستك (تشبع بيعي ثم تقاطع)
            if last['stoch_k'] < 40 and last['stoch_k'] > last['stoch_d']:
                return True
    return False

# 3. استراتيجية التذبذب (Range Trading) (للسوق الجانبي)
def strategy_sideways_scalp(df):
    last = df.iloc[-1]
    # البولنجر باند مستوي (عرض قليل نسبياً)
    if last['bb_width'] < 0.05: # يعتمد على العملة، هذه قيمة تقريبية
        # الشراء من الحد السفلي
        if last['close'] <= last['bb_lower'] * 1.01 and last['rsi'] < 40:
            return True
    return False

# 4. استراتيجية القنص (Oversold Bounce) (للسوق الهابط - خطرة)
def strategy_bearish_bounce(df):
    last = df.iloc[-1]
    # ابحث عن انحراف شديد عن EMA50
    dist_from_ema = (last['ema50'] - last['close']) / last['ema50']
    # انخفاض حاد + RSI منخفض جداً
    if dist_from_ema > 0.08 and last['rsi'] < 25:
        return True
    return False

# --- دالة اختيار الاستراتيجية المناسبة ---
def get_signal_from_strategies(symbol, df, regime):
    """
    تختار الاستراتيجية بناءً على نظام السوق الحالي.
    """
    signal_found = None
    strategy_name = None

    # تطبيق الاستراتيجيات بناءً على الوضع
    if "bullish" in regime:
        # في السوق الصاعد نبحث عن زخم أو ارتدادات
        if strategy_momentum_bullish(df):
            return "Momentum_Bullish"
        if strategy_pullback_bullish(df):
            return "Pullback_Bullish"
    
    elif regime == "sideways":
        # في السوق الجانبي نبحث عن تداول النطاق
        if strategy_sideways_scalp(df):
            return "Range_Scalp"

    elif "bearish" in regime:
        # في السوق الهابط نكون حذرين جداً (قنص فقط)
        if strategy_bearish_bounce(df):
            return "Oversold_Bounce"
    
    return None

# --- حساب الأهداف والوقف (ديناميكي) ---
def calculate_entry_params(df, strategy_name):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    # إعدادات افتراضية
    sl_dist = atr * 2
    tp1_dist = atr * 3
    tp2_dist = atr * 5
    
    # تخصيص حسب الاستراتيجية
    if strategy_name == "Momentum_Bullish":
        sl_dist = atr * 1.5 # وقف قريب للحفاظ على الزخم
        tp1_dist = atr * 2.5
        tp2_dist = atr * 6 # طموح
    elif strategy_name == "Range_Scalp":
        sl_dist = atr * 1.2
        tp1_dist = (last['bb_mid'] - close) * 0.9 # الهدف هو خط المنتصف
        tp2_dist = (last['bb_upper'] - close) * 0.9 # الحد العلوي
    elif strategy_name == "Oversold_Bounce":
        sl_dist = atr * 2.5 # وقف واسع للتقلبات
        tp1_dist = atr * 2 # خروج سريع
        tp2_dist = atr * 3

    stop_loss = close - sl_dist
    target1 = close + tp1_dist
    target2 = close + tp2_dist
    
    return stop_loss, target1, target2

# --- نهاية الجزء الثاني ---
# smart_bot_part3.py
# --- الجزء الثالث: إعادة التحليل، حلقة التداول، والواجهة ---

# --- 🚀 الجديد: آلية إعادة تحليل الإشارات المفتوحة (The Brain) ---
def reanalyze_open_position(symbol, signal_data, df, market_regime):
    """
    تقوم هذه الدالة بفحص الصفقة المفتوحة واتخاذ قرار:
    - HOLD: استمرار
    - EXIT_NOW: خروج فوري (تغير السوق، أو فشل النمط)
    - TIGHTEN_SL: رفع وقف الخسارة لحجز الربح
    - EXTEND_TP: رفع الهدف (إذا كان الزخم قوي جداً)
    """
    last = df.iloc[-1]
    entry_price = float(signal_data['entry_price'])
    current_price = float(last['close'])
    profit_pct = (current_price - entry_price) / entry_price * 100
    duration = (datetime.now() - datetime.fromisoformat(signal_data['closed_at'] if 'closed_at' in signal_data else signal_data['created_at'] if 'created_at' in signal_data else datetime.now().isoformat())).total_seconds() / 60
    
    action = "HOLD"
    reason = ""

    # 1. الحماية ضد انقلاب السوق
    if market_regime == "bearish" and signal_data['strategy_name'] in ["Momentum_Bullish", "Pullback_Bullish"]:
        # السوق انقلب للهبوط ونحن في صفقة شراء
        if profit_pct < -0.5:
            return "EXIT_NOW", "Market regime flipped to bearish"
        elif profit_pct > 0.5:
            return "TIGHTEN_SL", "Protect profit as market turned bearish"

    # 2. الخروج الزمني (Time-based Exit)
    # إذا مرت 12 شمعة (ساعة) والسعر لم يتحرك كثيراً
    if duration > 60 and -0.5 < profit_pct < 0.5:
         return "EXIT_NOW", "Stagnation (Price not moving)"

    # 3. التحليل الفني العكسي (Technical Breakdown)
    if signal_data['strategy_name'] == "Momentum_Bullish":
        # إذا فقدنا الزخم (RSI كسر 50 لأسفل أو MACD سلبي)
        if last['rsi'] < 48 or last['macd_hist'] < 0:
            if profit_pct > 0: return "TIGHTEN_SL", "Momentum lost"
            else: return "EXIT_NOW", "Momentum breakdown"
    
    # 4. تعزيز الربح (Trend Following)
    if profit_pct > 2.0 and last['rsi'] > 70 and last['macd_hist'] > 0:
        # الزخم قوي جداً، يمكننا رفع الهدف
        return "EXTEND_TP", "Strong momentum detected"

    return action, reason

# --- حلقة إدارة الصفقات (محدثة) ---
def trade_management_loop(client):
    logger.info("🛡️ بدء حلقة إدارة الصفقات وإعادة التحليل...")
    while True:
        try:
            # تحديث قائمة الصفقات المفتوحة
            with locks['signals']:
                signals = list(open_signals_cache.values())
            
            if not signals:
                time.sleep(5)
                continue

            for signal in signals:
                symbol = signal['symbol']
                # جلب بيانات حديثة للتحليل
                df = fetch_historical_data(client, symbol, '5m', 2) # فريم 5 دقائق
                if df is None: continue
                df = calculate_features(df)
                
                current_price = df.iloc[-1]['close']
                
                # 1. فحص الأهداف والوقف الكلاسيكي
                sl = float(signal['stop_loss'])
                tp1 = float(signal['target_price_1'])
                tp2 = float(signal['target_price_2'])
                
                if current_price <= sl:
                    close_trade(client, signal, current_price, "Stop Loss Hit")
                    continue
                elif current_price >= tp2:
                    close_trade(client, signal, current_price, "TP2 Hit")
                    continue
                
                # 2. 🚀 إعادة التحليل الذكي
                with locks['market']: regime = current_market_regime
                
                action, reason = reanalyze_open_position(symbol, signal, df, regime)
                
                if action == "EXIT_NOW":
                    logger.info(f"⚠️ خروج مبكر ذكي لـ {symbol}: {reason}")
                    close_trade(client, signal, current_price, f"Smart Exit: {reason}")
                
                elif action == "TIGHTEN_SL":
                    new_sl = current_price * 0.995 # وضع الوقف تحت السعر الحالي بقليل
                    if new_sl > sl:
                        update_signal_sl(signal['id'], new_sl)
                        logger.info(f"🔧 تم رفع وقف الخسارة لـ {symbol} إلى {new_sl} ({reason})")

                elif action == "EXTEND_TP":
                    new_tp2 = tp2 * 1.02 # زيادة الهدف 2%
                    update_signal_tp(signal['id'], tp2=new_tp2)
                    logger.info(f"🚀 تم رفع الهدف لـ {symbol} بسبب قوة الزخم")

            time.sleep(10) # فحص كل 10 ثواني
        except Exception as e:
            logger.error(f"خطأ في إدارة الصفقات: {e}")
            time.sleep(10)

# --- دوال مساعدة للتنفيذ ---
def close_trade(client, signal, price, reason):
    # تنفيذ البيع في باينانس وتحديث قاعدة البيانات (تبسيط للكود)
    logger.info(f"💰 إغلاق الصفقة {signal['symbol']} عند {price}. السبب: {reason}")
    # ... (كود البيع الفعلي هنا - مشابه للسابق)
    # حذف من الكاش
    with locks['signals']:
        if signal['symbol'] in open_signals_cache:
            del open_signals_cache[signal['symbol']]

def update_signal_sl(id, new_sl):
    # تحديث ال SQL والكاش
    with locks['signals']:
        for sym, sig in open_signals_cache.items():
            if sig['id'] == id:
                sig['stop_loss'] = new_sl
                break

def update_signal_tp(id, tp2):
    with locks['signals']:
        for sym, sig in open_signals_cache.items():
            if sig['id'] == id:
                sig['target_price_2'] = tp2
                break

# --- الحلقة الرئيسية (Main Loop) ---
def main_bot_loop():
    logger.info("🚀 بدء البوت الذكي...")
    
    # تهيئة الاتصال
    client = Client(API_KEY, API_SECRET)
    
    # تشغيل خيوط الخلفية
    Thread(target=trade_management_loop, args=(client,), daemon=True).start()
    
    while True:
        try:
            if not is_trading_enabled:
                time.sleep(10)
                continue
            
            # 1. تحديث هيكل السوق (كل 15 دقيقة مثلاً أو كل دورة كبيرة)
            analyze_market_structure(client)
            
            # 2. جلب العملات للمسح
            tickers = client.get_ticker()
            # فلترة بسيطة لأعلى حجم تداول
            symbols = [t['symbol'] for t in tickers if t['symbol'].endswith('USDT') and float(t['quoteVolume']) > 10000000]
            
            for symbol in symbols[:30]: # فحص أفضل 30 عملة فقط للسرعة
                with locks['signals']:
                    if symbol in open_signals_cache: continue # لدينا صفقة بالفعل
                
                # جلب البيانات
                df = fetch_historical_data(client, symbol, '5m', 2)
                if df is None: continue
                df = calculate_features(df)
                
                # تحديد الاستراتيجية المناسبة للوضع الحالي
                with locks['market']: regime = current_market_regime
                
                strategy = get_signal_from_strategies(symbol, df, regime)
                
                if strategy:
                    logger.info(f"✨ إشارة جديدة ({strategy}) للعملة {symbol} في وضع {regime}")
                    sl, tp1, tp2 = calculate_entry_params(df, strategy)
                    
                    # تنفيذ الدخول (محاكاة أو حقيقي)
                    # save_signal_to_db(...)
                    # add_to_cache(...)
            
            time.sleep(300) # دورة كل 5 دقائق
            
        except Exception as e:
            logger.error(f"خطأ في الحلقة الرئيسية: {e}")
            time.sleep(60)

# --- تطبيق Flask (واجهة بسيطة) ---
# smart_bot_part4.py
# --- الجزء الرابع: واجهة المستخدم المتقدمة والخادم ---
# ملاحظة: انسخ هذا الكود وألصقه بدلاً من أسطر تشغيل Flask في نهاية الجزء الثالث.

# --- قالب HTML المتقدم (يدعم حالة السوق الجديدة) ---
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة تحكم البوت الذكي V2</title>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--text:#e8f1ff;--muted:#8aa0c8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif}
.container{max-width:1400px;margin:0 auto;padding:20px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;border-bottom:1px solid #233056;padding-bottom:15px}
h1{margin:0;font-size:20px;color:#d7e4ff}
.status-badge{padding:5px 10px;border-radius:20px;font-size:12px;background:#0d1730;border:1px solid #233056}

/* شبكة التخطيط */
.grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));gap:20px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid #233056;border-radius:12px;padding:20px;box-shadow:0 4px 20px rgba(0,0,0,0.2)}
.card h2{margin-top:0;font-size:16px;color:var(--muted);border-bottom:1px solid #233056;padding-bottom:10px}

/* مقياس حالة السوق */
.market-meter{text-align:center}
.score-circle{width:100px;height:100px;border-radius:50%;background:conic-gradient(var(--accent) 0%, #1e2c52 0%);margin:0 auto;display:flex;align-items:center;justify-content:center;position:relative;transition:background 1s}
.score-circle::before{content:'';position:absolute;width:80px;height:80px;background:var(--panel);border-radius:50%}
.score-value{position:relative;font-size:24px;font-weight:bold}
.regime-text{font-size:18px;font-weight:bold;margin-top:10px;display:block}
.regime-bullish{color:var(--ok)}
.regime-bearish{color:var(--bad)}
.regime-sideways{color:var(--warn)}

/* جدول الصفقات */
.trades-grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));gap:15px}
.trade-card{background:#0d1730;border:1px solid #233056;border-radius:8px;padding:15px;position:relative;overflow:hidden}
.trade-header{display:flex;justify-content:space-between;margin-bottom:10px}
.symbol{font-weight:bold;font-size:16px}
.strategy-tag{font-size:10px;padding:3px 6px;border-radius:4px;background:#1e2c52;color:#aac}
.price-row{display:flex;justify-content:space-between;font-size:14px;margin-bottom:5px}
.pnl{font-weight:bold}
.pnl.pos{color:var(--ok)}
.pnl.neg{color:var(--bad)}
.progress-bar{height:4px;background:#1e2c52;border-radius:2px;margin-top:10px;overflow:hidden}
.progress-fill{height:100%;background:var(--accent);width:0%}

/* التحكم */
.btn{background:var(--accent);color:white;border:none;padding:10px 20px;border-radius:6px;cursor:pointer;font-weight:bold}
.btn:hover{opacity:0.9}
.btn.stop{background:var(--bad)}

table {width: 100%; border-collapse: collapse; font-size: 13px;}
th {text-align: right; color: var(--muted); padding: 8px;}
td {padding: 8px; border-top: 1px solid #233056;}
</style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>🤖 البوت الذكي (Smart Bot V2)</h1>
            <small style="color:var(--muted)">التحليل الهيكلي + الإدارة النشطة</small>
        </div>
        <div id="connectionStatus" class="status-badge">متصل 🟢</div>
    </header>

    <div class="grid">
        <!-- كارد حالة السوق -->
        <div class="card">
            <h2>📊 هيكل السوق (Market Structure)</h2>
            <div class="market-meter">
                <div class="score-circle" id="marketScoreCircle">
                    <span class="score-value" id="marketScoreVal">--</span>
                </div>
                <span class="regime-text" id="marketRegimeText">جاري التحليل...</span>
                <p style="font-size:12px;color:var(--muted);margin-top:5px">يعتمد على تحليل الرموز القيادية (BTC, ETH, SOL...)</p>
            </div>
        </div>

        <!-- كارد الإحصائيات -->
        <div class="card">
            <h2>📈 الأداء المباشر</h2>
            <table>
                <tr><td>الرصيد (USDT):</td><td id="balanceVal">--</td></tr>
                <tr><td>الصفقات المفتوحة:</td><td id="openCount">0</td></tr>
                <tr><td>وضع التداول:</td><td id="tradingMode">--</td></tr>
            </table>
            <div style="margin-top:20px;text-align:center">
                <button class="btn" id="toggleBtn" onclick="toggleTrading()">تشغيل / إيقاف</button>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>⚡ الصفقات النشطة (Active Positions)</h2>
        <div id="tradesContainer" class="trades-grid">
            <!-- سيتم ملء الصفقات هنا عبر JS -->
            <p style="color:var(--muted)">جاري انتظار الإشارات...</p>
        </div>
    </div>
    
    <div class="card" style="margin-top: 20px;">
        <h2>📝 سجل نشاط "العقل" (Re-analysis Logs)</h2>
        <div id="logsContainer" style="max-height: 150px; overflow-y: auto; font-family: monospace; font-size: 12px; color: #aac;">
            <!-- سجلات النظام -->
        </div>
    </div>

</div>

<script>
    function updateDashboard() {
        fetch('/api/data')
            .then(response => response.json())
            .then(data => {
                // تحديث بيانات السوق
                const score = data.market_score;
                const regime = data.market_regime;
                
                document.getElementById('marketScoreVal').innerText = score.toFixed(0);
                const circle = document.getElementById('marketScoreCircle');
                // تلوين الدائرة حسب النتيجة
                let color = '#ff9f1a'; // sideways
                if(score > 60) color = '#15c46a'; // bullish
                if(score < 40) color = '#ff4757'; // bearish
                circle.style.background = `conic-gradient(${color} ${score}%, #1e2c52 ${score}%)`;
                
                const regimeText = document.getElementById('marketRegimeText');
                regimeText.innerText = regime.toUpperCase().replace('_', ' ');
                regimeText.className = 'regime-text ' + (score > 60 ? 'regime-bullish' : (score < 40 ? 'regime-bearish' : 'regime-sideways'));

                // تحديث الرصيد والوضع
                document.getElementById('balanceVal').innerText = data.balance.toFixed(2);
                document.getElementById('openCount').innerText = data.open_signals.length;
                document.getElementById('tradingMode').innerText = data.is_enabled ? "يعمل ✅" : "متوقف 🛑";
                
                const btn = document.getElementById('toggleBtn');
                btn.className = data.is_enabled ? "btn stop" : "btn";
                btn.innerText = data.is_enabled ? "إيقاف التداول" : "بدء التداول";

                // تحديث الصفقات
                const tradesContainer = document.getElementById('tradesContainer');
                if(data.open_signals.length === 0) {
                    tradesContainer.innerHTML = '<p style="color:var(--muted)">لا توجد صفقات نشطة حالياً.</p>';
                } else {
                    let html = '';
                    data.open_signals.forEach(sig => {
                        const pnl = ((data.live_prices[sig.symbol] - sig.entry_price) / sig.entry_price * 100) || 0;
                        const pnlClass = pnl >= 0 ? 'pos' : 'neg';
                        
                        // حساب التقدم نحو الهدف
                        const progress = Math.min(100, Math.max(0, ((data.live_prices[sig.symbol] - sig.entry_price) / (sig.target_price_1 - sig.entry_price)) * 100));
                        
                        html += `
                        <div class="trade-card">
                            <div class="trade-header">
                                <span class="symbol">${sig.symbol}</span>
                                <span class="pnl ${pnlClass}">${pnl.toFixed(2)}%</span>
                            </div>
                            <div class="strategy-tag">${sig.strategy_name}</div>
                            <div style="margin-top:10px">
                                <div class="price-row"><span>الدخول:</span><span>${sig.entry_price}</span></div>
                                <div class="price-row"><span>الحالي:</span><span>${data.live_prices[sig.symbol] || '...'}</span></div>
                                <div class="price-row"><span>وقف:</span><span style="color:var(--bad)">${sig.stop_loss.toFixed(4)}</span></div>
                            </div>
                            <div class="progress-bar"><div class="progress-fill" style="width:${progress}%"></div></div>
                        </div>
                        `;
                    });
                    tradesContainer.innerHTML = html;
                }
            });
    }

    function toggleTrading() {
        fetch('/api/toggle', {method: 'POST'})
        .then(() => updateDashboard());
    }

    // تحديث كل 2 ثانية
    setInterval(updateDashboard, 2000);
    updateDashboard();
</script>
</body>
</html>
"""

# --- تطبيق Flask (النسخة الكاملة) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/data')
def get_data():
    # تجميع البيانات من المتغيرات العامة في الأجزاء السابقة
    with locks['market']: 
        regime = current_market_regime
        score = market_score
    
    with locks['signals']: 
        # تحويل القيم للكاش للعرض
        signals_list = list(open_signals_cache.values())
    
    with locks['prices']:
        current_prices = live_prices.copy()
        
    with locks['balance']:
        bal = usdt_balance

    return jsonify({
        "market_regime": regime,
        "market_score": score,
        "open_signals": signals_list,
        "live_prices": current_prices,
        "balance": bal,
        "is_enabled": is_trading_enabled
    })

@app.route('/api/toggle', methods=['POST'])
def toggle_bot():
    global is_trading_enabled
    is_trading_enabled = not is_trading_enabled
    logger.info(f"تم تغيير حالة البوت إلى: {is_trading_enabled}")
    return jsonify({"status": is_trading_enabled})

# --- تشغيل البرنامج ---
if __name__ == "__main__":
    # تهيئة قاعدة البيانات
    init_db()
    
    # تشغيل خادم الويب في خيط منفصل لكي لا يوقف البوت
    # أو تشغيل البوت في خيط منفصل (الخيار الأفضل هنا هو تشغيل الحلقة الرئيسية في خيط)
    
    bot_thread = Thread(target=main_bot_loop, daemon=True)
    bot_thread.start()
    
    logger.info("🌐 بدء خادم الويب ولوحة التحكم...")
    app.run(host='0.0.0.0', port=5000, debug=False)