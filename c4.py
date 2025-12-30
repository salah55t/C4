# smart_bot_v2.py
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
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ فشل تحميل المتغيرات: {e}"); exit(1)

# --- المتغيرات العامة وحالة السوق ---
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'] # الرموز القيادية لتحديد اتجاه السوق
current_market_regime: str = "sideways" # bullish, bearish, sideways, volatile
market_score: int = 50 # 0-100
is_trading_enabled: bool = False
paper_trading_mode: bool = True
usdt_balance: float = 10000.0  # رصيد افتراضي 10000 USDT
open_signals_cache: Dict[str, Dict] = {}
live_prices: Dict[str, float] = {}

# إعدادات إدارة المخاطر
risk_per_trade: float = 0.02  # 2% من الرصيد لكل صفقة
max_open_trades: int = 5  # الحد الأقصى للصفقات المفتوحة
sl_atr_multiplier: float = 2.0  # مضاعف ATR لحساب وقف الخسارة
strategy_performance: Dict[str, Dict] = {}  # أداء كل استراتيجية

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
                    last_analysis_time TIMESTAMP, analysis_notes TEXT, entry_time TIMESTAMP
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
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
            if not klines: 
                return None
                
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            for col in df.columns: 
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # التحقق من جودة البيانات
            if df.isnull().values.any():
                logger.warning(f"Data contains null values for {symbol}, filling with forward fill")
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching data for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # زيادة وقت الانتظار
                continue
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            return None
    
    return None

def analyze_market_structure(client):
    """
    تحليل السوق بناءً على الرموز القيادية مع مؤشرات إضافية
    """
    global current_market_regime, market_score
    
    scores = []
    details = {}
    volume_analysis = {}
    volatility_analysis = {}
    
    logger.info("🔍 جاري تحليل هيكل السوق عبر الرموز القيادية...")
    
    for symbol in LEADING_SYMBOLS:
        df = fetch_historical_data(client, symbol, '1h', 7) # زيادة فترة التحليل إلى 7 أيام
        if df is None or len(df) < 100: continue
        
        # حساب المؤشرات
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        
        # تحسين حساب RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # إضافة مؤشر ADX الصحيح
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        df['atr'] = df['tr'].rolling(14).mean()
        df['plus_di'] = 100 * (df['high'].diff().where(df['high'].diff() > 0, 0).rolling(14).mean() / df['atr'])
        df['minus_di'] = 100 * (df['low'].diff().where(df['low'].diff() < 0, 0).abs().rolling(14).mean() / df['atr'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()
        
        # إضافة مؤشر MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # تحليل حجم التداول
        df['volume_sma'] = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1]
        volume_analysis[symbol] = volume_ratio
        
        # تحليل التقلبات
        df['volatility'] = df['close'].pct_change().rolling(14).std() * 100
        volatility_analysis[symbol] = df['volatility'].iloc[-1]
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        symbol_score = 50
        
        # تقييم الاتجاه المحسن
        if last['close'] > last['ema20']: symbol_score += 5
        if last['close'] > last['ema50']: symbol_score += 10
        if last['close'] > last['ema200']: symbol_score += 15
        if last['ema20'] > last['ema50']: symbol_score += 5
        if last['ema50'] > last['ema200']: symbol_score += 10
        if last['macd_hist'] > 0: symbol_score += 5
        if last['macd_hist'] > prev['macd_hist']: symbol_score += 5  # MACD الصاعد
        
        # تقييم القوة
        if last['adx'] > 25: symbol_score += 10
        if last['adx'] > 40: symbol_score += 5  # قوة زائدة
        
        # تقييم الحجم
        if volume_ratio > 1.5: symbol_score += 5
        
        # تقييم RSI مع الانحرافات
        if 40 < last['rsi'] < 60: symbol_score += 5
        if 60 < last['rsi'] < 70: symbol_score += 5
        if last['rsi'] > 70: symbol_score -= 10  # تشبع شرائي
        if last['rsi'] < 30: symbol_score -= 10  # تشبع بيعي
        
        scores.append(symbol_score)
        details[symbol] = {
            'score': symbol_score,
            'volume_ratio': volume_ratio,
            'volatility': df['volatility'].iloc[-1],
            'adx': last['adx'],
            'rsi': last['rsi']
        }

    if not scores: return

    avg_score = sum(scores) / len(scores)
    avg_volume = sum(volume_analysis.values()) / len(volume_analysis)
    avg_volatility = sum(volatility_analysis.values()) / len(volatility_analysis)
    
    # تحسين تحديد نظام السوق
    with locks['market']:
        market_score = avg_score
        
        # إضافة عامل التقلبات في تحديد النظام
        volatility_factor = min(1.5, avg_volatility / 2)  # عامل التقلبات
        
        if avg_score >= 75:
            current_market_regime = "bullish_strong" if volatility_factor < 1.2 else "bullish_volatile"
        elif avg_score >= 60:
            current_market_regime = "bullish"
        elif avg_score <= 30:
            current_market_regime = "bearish" if volatility_factor < 1.2 else "bearish_volatile"
        elif avg_score <= 45:
            current_market_regime = "bearish_weak"
        else:
            current_market_regime = "sideways" if volatility_factor < 1.2 else "sideways_volatile"

    logger.info(f"🌐 حالة السوق المحدثة: {current_market_regime.upper()} (Score: {avg_score:.1f}, Volatility: {avg_volatility:.2f}%)")
    # حفظ الحالة في Redis
    redis_client.set('market_regime', json.dumps({
        'regime': current_market_regime, 
        'score': avg_score, 
        'details': details,
        'avg_volume': avg_volume,
        'avg_volatility': avg_volatility
    }))

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

    # حجم التداول
    df['volume_sma'] = df['volume'].rolling(20).mean()
    
    return df.fillna(0)

# --- 🚀 الجديد: استراتيجيات مخصصة لكل حالة سوق ---

# 1. استراتيجية الزخم (للسوق الصاعد)
def strategy_momentum_bullish(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # شروط أكثر صرامة: سعر فوق المتوسطات، حجم متزايد، MACD قوي
    if (last['close'] > last['ema21'] > last['ema50'] > last['ema200']) and \
       (55 < last['rsi'] < 75) and \
       (last['macd_hist'] > 0) and \
       (last['macd_hist'] > prev['macd_hist']) and  # MACD صاعد
       (last['adx'] > 25) and  # وجود اتجاه واضح
       (last['volume'] > last['volume_sma'] * 1.2):  # حجم متزايد
        return True
    return False

# 2. استراتيجية الارتداد (Pullback) (للسوق الصاعد القوي)
def strategy_pullback_bullish(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # اتجاه عام صاعد (فوق EMA200)
    if last['close'] > last['ema200']:
        # السعر يلمس EMA50 أو يقترب منها ثم يرتد
        if (last['low'] <= last['ema50'] * 1.01) and (last['close'] > last['ema50']):
            # تأكيد من الستوكاستك (تشبع بيعي ثم تقاطع)
            if last['stoch_k'] < 40 and last['stoch_k'] > last['stoch_d']:
                # تأكيد من حجم التداول
                if last['volume'] > last['volume_sma'] * 1.1:
                    return True
    return False

# 3. استراتيجية التذبذب (Range Trading) (للسوق الجانبي)
def strategy_sideways_scalp(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # البولنجر باند مستوي (عرض قليل نسبياً)
    if last['bb_width'] < 0.05:
        # الشراء من الحد السفلي مع تأكيد الحجم
        if last['close'] <= last['bb_lower'] * 1.01 and last['rsi'] < 40:
            if last['volume'] > last['volume_sma'] * 1.2:
                return True
    return False

# 4. استراتيجية القنص (Oversold Bounce) (للسوق الهابط - خطرة)
def strategy_bearish_bounce(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # انحراف شديد عن EMA50
    dist_from_ema = (last['ema50'] - last['close']) / last['ema50']
    # انخفاض حاد + RSI منخفض جداً + حجم مرتفع
    if dist_from_ema > 0.08 and last['rsi'] < 25:
        if last['volume'] > last['volume_sma'] * 1.5:  # حجم مرتفع عند الارتداد
            return True
    return False

# استراتيجية جديدة: استراتيجية الاختراق
def strategy_breakout(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # حساب نطاق التداول الأخير
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    
    # شرط الاختراق فوق مستوى مقاومة مع حجم مرتفع
    if (last['close'] > df['high_20'].iloc[-2]) and \
       (last['volume'] > last['volume_sma'] * 1.5) and \
       (last['rsi'] > 50):
        return True
    return False

# استراتيجية جديدة: استراتيجية التراجع القوي
def strategy_deep_pullback(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # اتجاه عام صاعد
    if last['ema50'] > last['ema200']:
        # تراجع عميق إلى EMA200 مع مؤشرات قوة
        if (last['close'] < last['ema50']) and (last['close'] > last['ema200'] * 0.98):
            if (last['rsi'] < 35) and (last['volume'] > last['volume_sma'] * 1.3):
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
        if strategy_breakout(df):
            return "Breakout"
    
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
    sl_dist = atr * sl_atr_multiplier
    tp1_dist = atr * 3
    tp2_dist = atr * 5
    
    # تخصيص حسب الاستراتيجية
    if strategy_name == "Momentum_Bullish":
        sl_dist = atr * 1.5
        tp1_dist = atr * 2.5
        tp2_dist = atr * 6
    elif strategy_name == "Range_Scalp":
        sl_dist = atr * 1.2
        tp1_dist = (last['bb_mid'] - close) * 0.9
        tp2_dist = (last['bb_upper'] - close) * 0.9
    elif strategy_name == "Oversold_Bounce":
        sl_dist = atr * 2.5
        tp1_dist = atr * 2
        tp2_dist = atr * 3
    elif strategy_name == "Breakout":
        sl_dist = atr * 1.8
        tp1_dist = atr * 3
        tp2_dist = atr * 7
    elif strategy_name == "Deep_Pullback":
        sl_dist = atr * 2
        tp1_dist = atr * 3
        tp2_dist = atr * 5

    stop_loss = close - sl_dist
    target1 = close + tp1_dist
    target2 = close + tp2_dist
    
    return stop_loss, target1, target2

# --- الجزء الثالث: التنفيذ، تلغرام، وإعادة التحليل ---

# --- إعدادات تلغرام ---
def send_telegram_alert(message):
    """إرسال تنبيهات إلى تلغرام مع معالجة الأخطاء"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        # محاولة الإرسال في خيط منفصل لتجنب تعطيل البوت
        Thread(target=requests.post, args=(url,), kwargs={'data': payload}).start()
    except Exception as e:
        logger.error(f"فشل إرسال تلغرام: {e}")

# --- تحسين إدارة المخاطر ---
def calculate_position_size(symbol, price, atr, risk_per_trade=0.02):
    """
    حساب حجم المركز بناءً على نسبة المخاطرة المحددة وقيمة ATR
    """
    try:
        # الحصول على رصيد الحساب
        with locks['balance']:
            balance = usdt_balance
        
        # حساب حجم المركز بناءً على نسبة المخاطرة
        risk_amount = balance * risk_per_trade
        
        # حساب حجم الموقف بناءً على مسافة وقف الخسارة (2x ATR)
        stop_distance = atr * sl_atr_multiplier
        position_size = risk_amount / stop_distance
        
        # الحصول على معلومات الرمز لتحديد الحد الأدنى للكمية
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            position_size = round(position_size / step_size) * step_size
        
        # التأكد من أن حجم الموقف لا يتجاوز 20% من الرصيد
        max_position_value = balance * 0.2
        if position_size * price > max_position_value:
            position_size = max_position_value / price
            position_size = round(position_size / step_size) * step_size
        
        return position_size
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0.0

# --- دالة التنفيذ الذكية (ورقي / حقيقي) ---
def execute_order(client, symbol, side, quantity, price=None, order_type='MARKET'):
    """
    تنفذ الطلب بناءً على وضع التداول (ورقي أو حقيقي) مع تحسينات
    """
    global usdt_balance, paper_trading_mode
    
    # 1. التنفيذ الورقي (Paper)
    if paper_trading_mode:
        logger.info(f"📝 تنفيذ ورقي: {side} {symbol} الكمية: {quantity}")
        # محاكاة انزلاق السعر
        slippage = 0.0005 if order_type == 'MARKET' else 0
        executed_price = (price if price else live_prices.get(symbol, 0)) * (1 + slippage if side == Client.SIDE_BUY else 1 - slippage)
        return {"status": "FILLED", "executedQty": quantity, "price": executed_price, "orderId": "PAPER_123"}
    
    # 2. التنفيذ الحقيقي (Real)
    else:
        try:
            logger.info(f"🚨 تنفيذ حقيقي: {side} {symbol}...")
            
            # تحويل الكمية للصيغة المقبولة (Precision)
            symbol_info = client.get_symbol_info(symbol)
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                quantity = round(quantity / step_size) * step_size
            
            # محاولة تنفيذ الأمر
            if order_type == 'MARKET':
                order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
            else:
                # تحويل السعر للصيغة المقبولة
                price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if price_filter:
                    tick_size = float(price_filter['tickSize'])
                    price = round(price / tick_size) * tick_size
                
                order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity, price=str(price))
            
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ خطأ في باينانس: {e}")
            send_telegram_alert(f"❌ فشل تنفيذ صفقة حقيقية لـ {symbol}: {e}")
            return None

# --- تحسين آلية إعادة تحليل الإشارات المفتوحة ---
def reanalyze_open_position(symbol, signal_data, df, market_regime):
    last = df.iloc[-1]
    entry_price = float(signal_data['entry_price'])
    current_price = float(last['close'])
    profit_pct = (current_price - entry_price) / entry_price * 100
    
    action = "HOLD"
    reason = ""
    
    # حساب التقلبات الحالية
    current_atr = last['atr']
    atr_pct = (current_atr / current_price) * 100
    
    # 1. الحماية: السوق انقلب ضدنا
    if "bearish" in market_regime and signal_data['strategy_name'] in ["Momentum_Bullish", "Pullback_Bullish", "Breakout"]:
        if profit_pct < -0.5: return "EXIT_NOW", "انقلاب السوق للهبوط"
        elif profit_pct > 0.5: return "TIGHTEN_SL", "حماية الربح (سوق هابط)"
    
    # 2. ضعف الزخم الفني
    if signal_data['strategy_name'] == "Momentum_Bullish":
        if last['rsi'] < 48:
            return ("TIGHTEN_SL", "ضعف الزخم (RSI)") if profit_pct > 0 else ("EXIT_NOW", "فشل الزخم")
        # انحراف MACD السلبي
        if last['macd_hist'] < 0 and profit_pct < 1.0:
            return "EXIT_NOW", "انحراف MACD سلبي"
    
    # 3. تعزيز الربح
    if profit_pct > 2.5 and last['macd_hist'] > 0:
        return "EXTEND_TP", "زخم قوي جداً 🚀"
    
    # 4. إدارة التقلبات
    if atr_pct > 3.0:  # تقلبات عالية
        if profit_pct > 1.0:
            return "TIGHTEN_SL", "حماية الربح في سوق متقلب"
        elif profit_pct < -1.0:
            return "EXIT_NOW", "خروج بسبب تقلبات عالية"
    
    # 5. وقف متحرك
    if profit_pct > 3.0:
        new_sl = current_price - (current_atr * 1.5)
        if new_sl > float(signal_data['stop_loss']):
            return "TRAILING_SL", f"رفع الوقف المتحرك إلى {new_sl:.4f}"
    
    return action, reason

# --- حلقة إدارة الصفقات ---
def trade_management_loop(client):
    logger.info("🛡️ بدء حلقة إدارة الصفقات...")
    while True:
        try:
            with locks['signals']: signals = list(open_signals_cache.values())
            if not signals: time.sleep(5); continue

            for signal in signals:
                symbol = signal['symbol']
                df = fetch_historical_data(client, symbol, '5m', 1)
                if df is None: continue
                df = calculate_features(df)
                current_price = df.iloc[-1]['close']
                
                # تحديث السعر المباشر للكاش
                with locks['prices']: live_prices[symbol] = current_price

                sl = float(signal['stop_loss'])
                tp1 = float(signal['target_price_1'])
                tp2 = float(signal['target_price_2'])
                
                # 1. التحقق من الأهداف والوقف
                exit_reason = None
                if current_price <= sl: exit_reason = "ضرب وقف الخسارة 🛑"
                elif current_price >= tp2: exit_reason = "تحقيق الهدف النهائي 🎯"
                
                if exit_reason:
                    close_trade(client, signal, current_price, exit_reason)
                    continue

                # 2. إعادة التحليل الذكي
                with locks['market']: regime = current_market_regime
                action, reason = reanalyze_open_position(symbol, signal, df, regime)
                
                if action == "EXIT_NOW":
                    close_trade(client, signal, current_price, f"خروج ذكي: {reason}")
                
                elif action == "TIGHTEN_SL":
                    new_sl = current_price * 0.998 # حجز ربح
                    if new_sl > sl:
                        update_signal_sl(signal['id'], new_sl)
                        msg = f"🛡️ *تحديث {symbol}*\nتم رفع الوقف إلى: `{new_sl:.4f}`\nالسبب: {reason}"
                        logger.info(msg)
                        send_telegram_alert(msg)
                
                elif action == "TRAILING_SL":
                    new_sl = current_price - (df.iloc[-1]['atr'] * 1.5)
                    if new_sl > sl:
                        update_signal_sl(signal['id'], new_sl)
                        msg = f"🛡️ *تحديث {symbol}*\nتم رفع الوقف المتحرك إلى: `{new_sl:.4f}`\nالسبب: {reason}"
                        logger.info(msg)
                        send_telegram_alert(msg)

            time.sleep(5)
        except Exception as e:
            logger.error(f"Error in trade loop: {e}"); time.sleep(10)

def close_trade(client, signal, price, reason):
    qty = float(signal['quantity'])
    
    # تنفيذ أمر البيع
    order = execute_order(client, signal['symbol'], Client.SIDE_SELL, qty)
    
    if order:
        # حساب السعر الفعلي من الطلب
        executed_price = float(order.get('price', price))
        entry_price = float(signal['entry_price'])
        profit_pct = (executed_price - entry_price) / entry_price * 100
        
        # تحديث قاعدة البيانات وحذف من الكاش
        with locks['signals']:
            if signal['symbol'] in open_signals_cache: del open_signals_cache[signal['symbol']]
        
        # تحديث أداء الاستراتيجية
        strategy_name = signal['strategy_name']
        with locks['market']:
            if strategy_name not in strategy_performance:
                strategy_performance[strategy_name] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': 0,
                    'total_loss': 0
                }
            
            strategy_performance[strategy_name]['total_trades'] += 1
            if profit_pct > 0:
                strategy_performance[strategy_name]['winning_trades'] += 1
                strategy_performance[strategy_name]['total_profit'] += profit_pct
            else:
                strategy_performance[strategy_name]['total_loss'] += abs(profit_pct)
            
            # حفظ أداء الاستراتيجية في Redis
            redis_client.set('strategy_performance', json.dumps(strategy_performance))
        
        # إرسال تنبيه تلغرام
        emoji = "✅" if profit_pct > 0 else "🔻"
        msg = (f"{emoji} *إغلاق صفقة {signal['symbol']}*\n"
               f"الاستراتيجية: {strategy_name}\n"
               f"الربح: `{profit_pct:.2f}%`\n"
               f"سعر الدخول: `{entry_price}`\n"
               f"سعر الخروج: `{executed_price}`\n"
               f"السبب: {reason}")
        send_telegram_alert(msg)
        logger.info(f"Closed {signal['symbol']}: {profit_pct:.2f}%")
        
        # تحديث الرصيد في الوضع الورقي
        if paper_trading_mode:
            with locks['balance']:
                usdt_balance += (executed_price * qty) * (1 - 0.001)  # خصم رسوم تقديرية

def update_signal_sl(id, new_sl):
    with locks['signals']:
        for s in open_signals_cache.values():
            if s['id'] == id: s['stop_loss'] = new_sl

# --- الحلقة الرئيسية (Main Loop) ---
def main_bot_loop():
    logger.info("🚀 بدء البوت الرئيسي...")
    global client
    client = Client(API_KEY, API_SECRET)
    Thread(target=trade_management_loop, args=(client,), daemon=True).start()
    
    # تحميل الإعدادات من Redis
    settings = redis_client.get('bot_settings')
    if settings:
        settings = json.loads(settings)
        global risk_per_trade, max_open_trades, sl_atr_multiplier
        risk_per_trade = settings.get('risk_per_trade', 0.02)
        max_open_trades = settings.get('max_open_trades', 5)
        sl_atr_multiplier = settings.get('sl_atr_multiplier', 2.0)
    
    while True:
        try:
            if not is_trading_enabled: 
                time.sleep(10)
                continue
            
            # تحليل هيكل السوق
            analyze_market_structure(client)
            
            # جلب رموز نشطة
            tickers = client.get_ticker()
            symbols = [t['symbol'] for t in tickers if t['symbol'].endswith('USDT') and float(t['quoteVolume']) > 20000000]
            random.shuffle(symbols)  # تنويع البحث

            # تحديد عدد الرموز التي سيتم تحليلها بناءً على حالة السوق
            scan_limit = 30 if "bullish" in current_market_regime else 20
            
            for symbol in symbols[:scan_limit]:
                with locks['signals']:
                    if symbol in open_signals_cache: 
                        continue
                    
                    # التحقق من الحد الأقصى للصفقات المفتوحة
                    if len(open_signals_cache) >= max_open_trades:
                        break
                
                df = fetch_historical_data(client, symbol, '5m', 2)
                if df is None: 
                    continue
                
                df = calculate_features(df)
                
                with locks['market']: 
                    regime = current_market_regime
                
                strategy = get_signal_from_strategies(symbol, df, regime)
                
                if strategy:
                    current_price = df.iloc[-1]['close']
                    sl, tp1, tp2 = calculate_entry_params(df, strategy)
                    
                    # حساب حجم الموقف بناءً على نسبة المخاطرة
                    quantity = calculate_position_size(symbol, current_price, df.iloc[-1]['atr'], risk_per_trade)
                    
                    if quantity <= 0:
                        logger.warning(f"كمية غير صالحة لـ {symbol}: {quantity}")
                        continue
                    
                    # تنفيذ الشراء
                    order = execute_order(client, symbol, Client.SIDE_BUY, quantity)
                    
                    if order:
                        # حفظ الصفقة
                        signal_data = {
                            'id': int(time.time()), 
                            'symbol': symbol, 
                            'entry_price': current_price,
                            'stop_loss': sl, 
                            'target_price_1': tp1, 
                            'target_price_2': tp2,
                            'quantity': quantity, 
                            'strategy_name': strategy, 
                            'status': 'open',
                            'entry_time': datetime.now(timezone.utc).isoformat()
                        }
                        
                        with locks['signals']: 
                            open_signals_cache[symbol] = signal_data
                        
                        # إرسال تنبيه تلغرام
                        msg = (f"🚀 *توصية جديدة ({strategy})*\n"
                               f"العملة: `{symbol}`\n"
                               f"الدخول: `{current_price}`\n"
                               f"الهدف: `{tp2}` | الوقف: `{sl}`\n"
                               f"الكمية: `{quantity}`\n"
                               f"حالة السوق: {regime}")
                        send_telegram_alert(msg)
                        logger.info(f"New signal: {symbol} - {strategy}")

            # تحديث الرصيد في الوضع الورقي
            if paper_trading_mode:
                with locks['balance']:
                    # حساب القيمة الإجمالية للمحفظة
                    total_value = usdt_balance
                    for signal in open_signals_cache.values():
                        symbol_price = live_prices.get(signal['symbol'], signal['entry_price'])
                        total_value += symbol_price * signal['quantity']
                    
                    # تحديث الرصيد في Redis
                    redis_client.set('paper_balance', json.dumps({
                        'usdt': usdt_balance,
                        'total_value': total_value,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }))

            # فترة انتظار متغيرة بناءً على حالة السوق
            wait_time = 60 if "volatile" in current_market_regime else 120
            time.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            time.sleep(60)

# --- الجزء الرابع: واجهة المستخدم المتقدمة والخادم ---

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

/* مؤشرات الأداء */
.performance-metrics{display:grid;grid-template-columns:repeat(3, 1fr);gap:10px;margin-top:15px}
.metric{text-align:center}
.metric-value{font-size:18px;font-weight:bold}
.metric-label{font-size:12px;color:var(--muted)}

/* إعدادات البوت */
.settings-grid{display:grid;grid-template-columns:repeat(2, 1fr);gap:15px}
.setting-item{display:flex;flex-direction:column;margin-bottom:10px}
.setting-label{font-size:12px;color:var(--muted);margin-bottom:5px}
.setting-input{background:#0d1730;border:1px solid #233056;color:var(--text);padding:8px;border-radius:4px}
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
                <div style="margin-top:10px">
                    <div class="metric">
                        <div class="metric-value" id="avgVolumeVal">--</div>
                        <div class="metric-label">متوسط حجم التداول</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="avgVolatilityVal">--</div>
                        <div class="metric-label">متوسط التقلب (%)</div>
                    </div>
                </div>
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
            <div class="performance-metrics">
                <div class="metric">
                    <div class="metric-value" id="winRateVal">--</div>
                    <div class="metric-label">نسبة الفوز (%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="profitFactorVal">--</div>
                    <div class="metric-label">عامل الربح</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="maxDDVal">--</div>
                    <div class="metric-label">أقصى انخفاض (%)</div>
                </div>
            </div>
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
        <h2>⚙️ إعدادات البوت</h2>
        <div class="settings-grid">
            <div class="setting-item">
                <label class="setting-label">نسبة المخاطرة للصفقة الواحدة (%)</label>
                <input type="number" class="setting-input" id="riskPerTrade" value="2" min="0.5" max="5" step="0.5">
            </div>
            <div class="setting-item">
                <label class="setting-label">الحد الأقصى للصفقات المفتوحة</label>
                <input type="number" class="setting-input" id="maxOpenTrades" value="5" min="1" max="10" step="1">
            </div>
            <div class="setting-item">
                <label class="setting-label">نسبة وقف الخسارة من ATR</label>
                <input type="number" class="setting-input" id="slAtrMultiplier" value="2" min="1" max="5" step="0.5">
            </div>
            <div class="setting-item">
                <label class="setting-label">تفعيل وضع التداول الورقي</label>
                <select class="setting-input" id="paperTradingMode">
                    <option value="true" selected>نعم</option>
                    <option value="false">لا</option>
                </select>
            </div>
        </div>
        <div style="margin-top:15px;text-align:center">
            <button class="btn" onclick="saveSettings()">حفظ الإعدادات</button>
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

                // تحديث بيانات السوق الإضافية
                document.getElementById('avgVolumeVal').innerText = data.avg_volume ? data.avg_volume.toFixed(2) : '--';
                document.getElementById('avgVolatilityVal').innerText = data.avg_volatility ? data.avg_volatility.toFixed(2) + '%' : '--';

                // تحديث الرصيد والوضع
                document.getElementById('balanceVal').innerText = data.balance.toFixed(2);
                document.getElementById('openCount').innerText = data.open_signals.length;
                document.getElementById('tradingMode').innerText = data.is_enabled ? "يعمل ✅" : "متوقف 🛑";
                
                const btn = document.getElementById('toggleBtn');
                btn.className = data.is_enabled ? "btn stop" : "btn";
                btn.innerText = data.is_enabled ? "إيقاف التداول" : "بدء التداول";

                // تحديث مؤشرات الأداء
                document.getElementById('winRateVal').innerText = data.win_rate ? data.win_rate.toFixed(1) + '%' : '--';
                document.getElementById('profitFactorVal').innerText = data.profit_factor ? data.profit_factor.toFixed(2) : '--';
                document.getElementById('maxDDVal').innerText = data.max_drawdown ? data.max_drawdown.toFixed(1) + '%' : '--';

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
    
    function saveSettings() {
        const settings = {
            risk_per_trade: parseFloat(document.getElementById('riskPerTrade').value),
            max_open_trades: parseInt(document.getElementById('maxOpenTrades').value),
            sl_atr_multiplier: parseFloat(document.getElementById('slAtrMultiplier').value),
            paper_trading_mode: document.getElementById('paperTradingMode').value === 'true'
        };
        
        fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .then(data => {
            if(data.success) {
                alert('تم حفظ الإعدادات بنجاح');
                updateDashboard();
            } else {
                alert('فشل حفظ الإعدادات: ' + data.error);
            }
        });
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
    
    # الحصول على بيانات السوق الإضافية من Redis
    market_data = redis_client.get('market_regime')
    avg_volume = 0
    avg_volatility = 0
    if market_data:
        market_data = json.loads(market_data)
        avg_volume = market_data.get('avg_volume', 0)
        avg_volatility = market_data.get('avg_volatility', 0)
    
    # الحصول على بيانات الأداء
    performance = get_performance().get_json()
    
    return jsonify({
        "market_regime": regime,
        "market_score": score,
        "avg_volume": avg_volume,
        "avg_volatility": avg_volatility,
        "open_signals": signals_list,
        "live_prices": current_prices,
        "balance": bal,
        "is_enabled": is_trading_enabled,
        "win_rate": performance.get('win_rate', 0),
        "profit_factor": performance.get('profit_factor', 0),
        "max_drawdown": performance.get('max_drawdown', 0)
    })

@app.route('/api/toggle', methods=['POST'])
def toggle_bot():
    global is_trading_enabled
    is_trading_enabled = not is_trading_enabled
    logger.info(f"تم تغيير حالة البوت إلى: {is_trading_enabled}")
    return jsonify({"status": is_trading_enabled})

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global risk_per_trade, max_open_trades, sl_atr_multiplier, paper_trading_mode
    
    if request.method == 'GET':
        return jsonify({
            'risk_per_trade': risk_per_trade,
            'max_open_trades': max_open_trades,
            'sl_atr_multiplier': sl_atr_multiplier,
            'paper_trading_mode': paper_trading_mode
        })
    else:  # POST
        try:
            settings = request.json
            risk_per_trade = settings.get('risk_per_trade', 0.02)
            max_open_trades = settings.get('max_open_trades', 5)
            sl_atr_multiplier = settings.get('sl_atr_multiplier', 2.0)
            paper_trading_mode = settings.get('paper_trading_mode', True)
            
            # حفظ الإعدادات في Redis
            redis_client.set('bot_settings', json.dumps(settings))
            
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return jsonify({'success': False, 'error': str(e)})

@app.route('/api/performance')
def get_performance():
    try:
        # حساب مؤشرات الأداء من قاعدة البيانات
        with conn.cursor() as cur:
            # حساب نسبة الفوز
            cur.execute("""
                SELECT 
                    COUNT(CASE WHEN profit_percentage > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate,
                    SUM(CASE WHEN profit_percentage > 0 THEN profit_percentage ELSE 0 END) / 
                    NULLIF(SUM(CASE WHEN profit_percentage < 0 THEN ABS(profit_percentage) ELSE 0 END), 0) as profit_factor
                FROM signals 
                WHERE status = 'closed' AND closing_price IS NOT NULL
            """)
            result = cur.fetchone()
            win_rate = result['win_rate'] if result['win_rate'] else 0
            profit_factor = result['profit_factor'] if result['profit_factor'] else 0
            
            # حساب أقصى انخفاض
            cur.execute("""
                WITH equity_curve AS (
                    SELECT 
                        closing_at,
                        SUM(profit_percentage) OVER (ORDER BY closing_at) as cumulative_profit
                    FROM signals 
                    WHERE status = 'closed' AND closing_price IS NOT NULL
                    ORDER BY closing_at
                ),
                peaks AS (
                    SELECT 
                        closing_at,
                        cumulative_profit,
                        MAX(cumulative_profit) OVER (ORDER BY closing_at) as peak
                    FROM equity_curve
                )
                SELECT 
                    MIN((cumulative_profit - peak) / peak * 100) as max_drawdown
                FROM peaks
            """)
            result = cur.fetchone()
            max_drawdown = result['max_drawdown'] if result['max_drawdown'] else 0
            
            return jsonify({
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': abs(max_drawdown)
            })
    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        return jsonify({
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0
        })

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