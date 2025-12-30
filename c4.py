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
import random
from decimal import Decimal, ROUND_DOWN, getcontext
from psycopg2 import pool, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any
import warnings

# --- إعدادات اللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
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
    API_KEY: str = config('BINANCE_API_KEY', default='')
    API_SECRET: str = config('BINANCE_API_SECRET', default='')
    DB_URL: str = config('DATABASE_URL', default='')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
    PORT = int(config('PORT', default=5000))
except Exception as e:
    logger.critical(f"Environment Variable Error: {e}")
    # استخدام قيم افتراضية للتجربة إذا لم توجد متغيرات
    API_KEY = "TEST"
    API_SECRET = "TEST"

# --- المتغيرات العامة ---
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
current_market_regime: str = "sideways"
market_score: int = 50
is_trading_enabled: bool = True
paper_trading_mode: bool = True
usdt_balance: float = 10000.0
open_signals_cache: Dict[str, Dict] = {}
live_prices: Dict[str, float] = {}

# إعدادات إدارة المخاطر
risk_per_trade: float = 0.02
max_open_trades: int = 5
sl_atr_multiplier: float = 2.0
strategy_performance: Dict[str, Dict] = {}

# أقفال (Locks) لمنع تضارب البيانات بين الخيوط
locks = {
    'trade': Lock(), 'balance': Lock(), 'signals': Lock(), 
    'prices': Lock(), 'market': Lock(), 'log': Lock()
}

# --- إعدادات قاعدة البيانات (Postgres Connection Pool) ---
db_pool: Optional[pool.ThreadedConnectionPool] = None

def init_db():
    global db_pool
    if not DB_URL:
        logger.warning("DATABASE_URL not found, using in-memory mode only.")
        return
        
    try:
        db_pool = pool.ThreadedConnectionPool(minconn=1, maxconn=5, dsn=DB_URL)
        conn = db_pool.getconn()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION,
                    stop_loss DOUBLE PRECISION, target_price_1 DOUBLE PRECISION, target_price_2 DOUBLE PRECISION,
                    status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                    profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                    is_real_trade BOOLEAN DEFAULT FALSE, quantity DOUBLE PRECISION, closing_reason TEXT,
                    entry_time TIMESTAMP
                );
            """)
        db_pool.putconn(conn)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database Error: {e}")

def get_db_connection():
    global db_pool
    if db_pool:
        try:
            return db_pool.getconn()
        except OperationalError:
            logger.error("DB Connection failed, attempting re-init...")
            init_db()
            return db_pool.getconn()
    return None

# --- إعداد Redis ---
redis_client = None
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.warning(f"Redis Error: {e}")

# --- تحليل هيكل السوق ---
def fetch_historical_data(client, symbol, interval, days) -> Optional[pd.DataFrame]:
    max_retries = 2
    for attempt in range(max_retries):
        try:
            klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
            if not klines: return None
                
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'x', 'y', 'z', 'a', 'b', 'c'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            for col in df.columns: df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.fillna(method='ffill')
            return df
        except BinanceAPIException as e:
            logger.error(f"API Error {symbol}: {e}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Unexpected Error {symbol}: {e}")
            time.sleep(1)
    return None

def analyze_market_structure(client):
    global current_market_regime, market_score
    
    scores = []
    details = {}
    
    logger.info("Analyzing Market Structure...")
    
    for symbol in LEADING_SYMBOLS:
        df = fetch_historical_data(client, symbol, '1h', 7)
        if df is None or len(df) < 100: continue
        
        # حساب المؤشرات
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        df['atr'] = df['tr'].rolling(14).mean()
        
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        symbol_score = 50
        
        # نقاط التقييم
        if last['close'] > last['ema20']: symbol_score += 5
        if last['close'] > last['ema50']: symbol_score += 10
        if last['close'] > last['ema200']: symbol_score += 15
        if last['ema20'] > last['ema50']: symbol_score += 5
        if last['ema50'] > last['ema200']: symbol_score += 10
        
        # إصلاح محتمل للأخطاء البرمجية في الشروط
        is_macd_rising = last['macd_hist'] > prev['macd_hist']
        if last['macd_hist'] > 0: symbol_score += 5
        if is_macd_rising: symbol_score += 5
        
        # تقييم ADX
        df['plus_di'] = 100 * (df['high'].diff().where(df['high'].diff() > 0, 0).rolling(14).mean() / df['atr'])
        df['minus_di'] = 100 * (df['low'].diff().where(df['low'].diff() < 0, 0).abs().rolling(14).mean() / df['atr'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()
        
        if last['adx'] > 25: symbol_score += 10
        
        if last['rsi'] > 70: symbol_score -= 10
        if last['rsi'] < 30: symbol_score -= 10
        
        scores.append(symbol_score)
        details[symbol] = {'score': symbol_score, 'rsi': last['rsi'], 'adx': last['adx']}

    if not scores: return

    avg_score = sum(scores) / len(scores)
    
    with locks['market']:
        market_score = avg_score
        
        if avg_score >= 75: current_market_regime = "bullish_strong"
        elif avg_score >= 60: current_market_regime = "bullish"
        elif avg_score <= 30: current_market_regime = "bearish"
        elif avg_score <= 45: current_market_regime = "bearish_weak"
        else: current_market_regime = "sideways"

    logger.info(f"Market Regime: {current_market_regime} (Score: {avg_score:.1f})")
    
    if redis_client:
        try:
            redis_client.set('market_regime', json.dumps({'regime': current_market_regime, 'score': avg_score}))
        except:
            pass

# --- الجزء الثاني: حساب المؤشرات وتحسين الاستراتيجيات ---

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMAs
    for span in [7, 21, 50, 200]:
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
    
    df['volume_sma'] = df['volume'].rolling(20).mean()
    
    return df.fillna(0)

# --- الاستراتيجيات المحسنة ---

def strategy_momentum_bullish(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. الاتجاه العام صاعد
    is_trend_up = last['close'] > last['ema21'] > last['ema50'] > last['ema200']
    
    # 2. الزخم (MACD) صاعد
    is_momentum_up = last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']
    
    # 3. تحسين: فلتر القرب من EMA لتجنب الشراء عند القمة (Late Entry)
    # نقبل الشراء فقط إذا كان السعر قريباً من EMA21 (مسافة 1.5%) أو يلمسه
    dist_to_ema = (last['close'] - last['ema21']) / last['ema21']
    is_near_entry = 0 < dist_to_ema < 0.015
    
    # 4. RSI في منطقة صحية (ليس تشبع شرائي)
    is_rsi_ok = 55 < last['rsi'] < 75
    
    if is_trend_up and is_momentum_up and is_near_entry and is_rsi_ok:
        return True
    return False

def strategy_pullback_bullish(df):
    last = df.iloc[-1]
    prev = df.iloc[-1]
    
    # الاتجاه العام صاعد (فوق EMA200)
    if last['close'] < last['ema200']: return False
    
    # تحسين: تأكيد الانعكاس (الشمعة الحالية تغلق فوق EMA50)
    # بدلاً من مجرد اللمس
    prev_close = df.iloc[-2]['close']
    was_touching = prev_close <= last['ema50']
    is_bouncing = last['close'] > last['ema50']
    
    if was_touching and is_bouncing:
        # تأكيد Stochastic (تقاطع صاعد)
        if last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < 40:
            if last['volume'] > last['volume_sma'] * 1.1:
                return True
    return False

def strategy_sideways_scalp(df):
    last = df.iloc[-1]
    
    # تحسين: تعطيل الاستراتيجية إذا كان السوق متقلباً للغاية
    # (هذا يتم التحقق منه خارج الدالة عبر regime، لكن نضيف شرطاً إضافياً هنا للعرض)
    if last['bb_width'] < 0.04: # نطاق ضيق
        # شراء من القاع
        if last['close'] <= last['bb_lower'] * 1.005 and last['rsi'] < 35:
            if last['volume'] > last['volume_sma'] * 1.2:
                return True
    return False

def strategy_bearish_bounce(df):
    last = df.iloc[-1]
    
    # مسافة الانحراف الكبير
    dist_from_ema = (last['ema50'] - last['close']) / last['ema50']
    
    if dist_from_ema > 0.08 and last['rsi'] < 25:
        if last['volume'] > last['volume_sma'] * 1.5:
            return True
    return False

def strategy_breakout(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    df['high_20'] = df['high'].rolling(20).max()
    prev_high_20 = df['high_20'].iloc[-2]
    
    # تحسين: تأكيد الاختراق (الإغلاق فوق المستوى)
    # مع حجم مرتفع
    if prev['close'] < prev_high_20 and last['close'] > prev_high_20:
        if last['volume'] > last['volume_sma'] * 1.5 and last['rsi'] > 50:
            return True
    return False

def get_signal_from_strategies(symbol, df, regime):
    # تعطيل التداول الجانبي إذا كان السوق متقلباً أو هابطاً
    if "bearish" in regime or "volatile" in regime:
        # لا نستخدم استراتيجية التذبذب في السوق الخطير
        pass
    else:
        if strategy_sideways_scalp(df): return "Range_Scalp"

    if "bullish" in regime:
        if strategy_momentum_bullish(df): return "Momentum_Bullish"
        if strategy_pullback_bullish(df): return "Pullback_Bullish"
        if strategy_breakout(df): return "Breakout"
    
    # استراتيجية القنص خطيرة جداً، نستخدمها فقط في السوق الجانبي العادي بحذر
    if regime == "sideways":
        if strategy_bearish_bounce(df): return "Oversold_Bounce"
    
    return None

# --- حساب الأهداف والوقف ---
def calculate_entry_params(df, strategy_name):
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    
    # إعدادات افتراضية
    sl_dist = atr * sl_atr_multiplier
    tp1_dist = atr * 3
    tp2_dist = atr * 5
    
    if strategy_name == "Momentum_Bullish":
        sl_dist = atr * 1.5
        tp1_dist = atr * 2.5
        tp2_dist = atr * 6
    elif strategy_name == "Range_Scalp":
        sl_dist = atr * 1.0 # وقف أضيق للسكالبينج
        tp1_dist = (last['bb_mid'] - close) * 0.8
        tp2_dist = (last['bb_upper'] - close) * 0.8
    elif strategy_name == "Oversold_Bounce":
        sl_dist = atr * 2.5
        tp1_dist = atr * 2
        tp2_dist = atr * 3
    elif strategy_name == "Breakout":
        sl_dist = atr * 1.8
        tp1_dist = atr * 3
        tp2_dist = atr * 7

    stop_loss = close - sl_dist
    target1 = close + tp1_dist
    target2 = close + tp2_dist
    
    return stop_loss, target1, target2

# --- الجزء الثالث: التنفيذ وإدارة المخاطر المحسنة ---

def send_telegram_alert(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    def _send():
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except: pass
    Thread(target=_send, daemon=True).start()

client_binance = None

def calculate_position_size(symbol, price, atr, strategy_name):
    global client_binance
    try:
        with locks['balance']: balance = usdt_balance
        
        # تحسين: تقليل المخاطرة لصفقات القنص (Oversold Bounce)
        current_risk = risk_per_trade
        if strategy_name == "Oversold_Bounce":
            current_risk = risk_per_trade * 0.5 # نصف المخاطرة
        
        risk_amount = balance * current_risk
        stop_distance = atr * sl_atr_multiplier
        position_size = risk_amount / stop_distance
        
        # ضبط الدقة (Step Size)
        try:
            symbol_info = client_binance.get_symbol_info(symbol)
            lot_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_filter:
                step = float(lot_filter['stepSize'])
                position_size = round(position_size / step) * step
        except:
            pass
        
        # الحد الأقصى لحجم الموقف (20% من الرصيد)
        max_pos_val = balance * 0.2
        if position_size * price > max_pos_val:
            position_size = max_pos_val / price
        
        return position_size
    except Exception as e:
        logger.error(f"Position Size Error: {e}")
        return 0.0

def execute_order(client, symbol, side, quantity, price=None, order_type='MARKET'):
    global usdt_balance, paper_trading_mode
    
    if paper_trading_mode:
        slippage = 0.0005 if order_type == 'MARKET' else 0
        exec_price = (price if price else live_prices.get(symbol, 0)) * (1 + slippage if side == Client.SIDE_BUY else 1 - slippage)
        return {"status": "FILLED", "executedQty": quantity, "price": exec_price, "orderId": "PAPER"}
    
    try:
        # ضبط الدقة للكمية والسعر
        info = client.get_symbol_info(symbol)
        lot_step = float(next(f['stepSize'] for f in info['filters'] if f['filterType'] == 'LOT_SIZE'))
        quantity = round(quantity / lot_step) * lot_step
        
        if order_type == 'MARKET':
            return client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        else:
            tick = float(next(f['tickSize'] for f in info['filters'] if f['filterType'] == 'PRICE_FILTER'))
            price = round(price / tick) * tick
            return client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity, price=str(price))
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        send_telegram_alert(f"Order Failed {symbol}: {e}")
        return None

# --- تحسين إدارة الصفقات (Emergency Exit Logic) ---
def reanalyze_open_position(symbol, signal_data, df, market_regime):
    last = df.iloc[-1]
    entry_price = float(signal_data['entry_price'])
    current_price = float(last['close'])
    profit_pct = (current_price - entry_price) / entry_price * 100
    
    action = "HOLD"
    reason = ""
    current_atr = last['atr']
    
    # --- منطق الطوارئ الجديد: خروج فوري في السوق الخطير ---
    # إذا كان السوق هابطاً ومتقلباً ونحن في خسارة، اخرج فوراً
    if ("bearish" in market_regime and "volatile" in market_regime) and profit_pct < -1.0:
        return "EXIT_NOW", "EMERGENCY: Market Crash Detected"
    
    # إدارة وقف الخسارة
    if signal_data['strategy_name'] == "Momentum_Bullish":
        if last['rsi'] < 48 and profit_pct < 0:
            return "EXIT_NOW", "Momentum Failed"
        if last['macd_hist'] < 0 and profit_pct < 1.0:
            return "EXIT_NOW", "MACD Divergence"
            
    # وقف متحرك
    if profit_pct > 3.0:
        new_sl = current_price - (current_atr * 1.5)
        if new_sl > float(signal_data['stop_loss']):
            return "TRAILING_SL", f"Trailing SL to {new_sl:.4f}"
    
    # حماية الربح في السوق المتذبذب
    if profit_pct > 1.5 and profit_pct < 2.0 and "volatile" in market_regime:
        return "TIGHTEN_SL", "Secure Profit (Volatile)"
    
    return action, reason

def trade_management_loop(client):
    logger.info("Starting Trade Management Loop...")
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
                
                with locks['prices']: live_prices[symbol] = current_price

                sl = float(signal['stop_loss'])
                tp2 = float(signal['target_price_2'])
                
                exit_reason = None
                if current_price <= sl: exit_reason = "Stop Loss Hit"
                elif current_price >= tp2: exit_reason = "Target 2 Hit"
                
                if exit_reason:
                    close_trade(client, signal, current_price, exit_reason)
                    continue

                with locks['market']: regime = current_market_regime
                action, reason = reanalyze_open_position(symbol, signal, df, regime)
                
                if action == "EXIT_NOW":
                    close_trade(client, signal, current_price, f"Smart Exit: {reason}")
                elif action == "TIGHTEN_SL":
                    new_sl = current_price * 0.998
                    if new_sl > sl:
                        update_signal_sl(signal['id'], new_sl)
                        send_telegram_alert(f"Tightening SL {symbol}: {reason}")
                elif action == "TRAILING_SL":
                    new_sl = current_price - (df.iloc[-1]['atr'] * 1.5)
                    if new_sl > sl:
                        update_signal_sl(signal['id'], new_sl)
                        send_telegram_alert(f"Trailing SL {symbol}: {reason}")

            time.sleep(5)
        except Exception as e:
            logger.error(f"Trade Loop Error: {e}")
            time.sleep(10)

def close_trade(client, signal, price, reason):
    qty = float(signal['quantity'])
    order = execute_order(client, signal['symbol'], Client.SIDE_SELL, qty)
    
    if order:
        exec_price = float(order.get('price', price))
        entry = float(signal['entry_price'])
        pnl = (exec_price - entry) / entry * 100
        
        with locks['signals']:
            if signal['symbol'] in open_signals_cache: del open_signals_cache[signal['symbol']]
        
        strat = signal['strategy_name']
        with locks['market']:
            if strat not in strategy_performance:
                strategy_performance[strat] = {'total_trades': 0, 'wins': 0, 'profit': 0, 'loss': 0}
            perf = strategy_performance[strat]
            perf['total_trades'] += 1
            if pnl > 0:
                perf['wins'] += 1
                perf['profit'] += pnl
            else:
                perf['loss'] += abs(pnl)
            
            if redis_client:
                try: redis_client.set('strategy_performance', json.dumps(strategy_performance))
                except: pass
        
        send_telegram_alert(f"Closed {signal['symbol']} ({strat}): {pnl:.2f}% - {reason}")
        logger.info(f"Closed {signal['symbol']}: {pnl:.2f}%")
        
        if paper_trading_mode:
            with locks['balance']: usdt_balance += (exec_price * qty) * (1 - 0.001)

def update_signal_sl(id, new_sl):
    with locks['signals']:
        for s in open_signals_cache.values():
            if s['id'] == id: s['stop_loss'] = new_sl

# --- الحلقة الرئيسية ---
def main_bot_loop():
    logger.info("Starting Main Bot Loop...")
    global client_binance
    client_binance = Client(API_KEY, API_SECRET)
    Thread(target=trade_management_loop, args=(client_binance,), daemon=True).start()
    
    while True:
        try:
            if not is_trading_enabled: 
                time.sleep(10)
                continue
            
            analyze_market_structure(client_binance)
            
            tickers = client_binance.get_ticker()
            symbols = [t['symbol'] for t in tickers if t['symbol'].endswith('USDT') and float(t['quoteVolume']) > 20000000]
            random.shuffle(symbols)

            # تقليل عدد الرموز المفحوصة إذا كان السوق سيئاً لتوفير الموارد وتقليل المخاطر
            scan_limit = 20
            if "bullish" in current_market_regime: scan_limit = 30
            elif "bearish" in current_market_regime: scan_limit = 10 # بحث محدود جداً
            
            for symbol in symbols[:scan_limit]:
                with locks['signals']:
                    if symbol in open_signals_cache or len(open_signals_cache) >= max_open_trades:
                        continue
                
                df = fetch_historical_data(client_binance, symbol, '5m', 2)
                if df is None: continue
                df = calculate_features(df)
                
                with locks['market']: regime = current_market_regime
                strategy = get_signal_from_strategies(symbol, df, regime)
                
                if strategy:
                    current_price = df.iloc[-1]['close']
                    sl, tp1, tp2 = calculate_entry_params(df, strategy)
                    quantity = calculate_position_size(symbol, current_price, df.iloc[-1]['atr'], strategy)
                    
                    if quantity <= 0: continue
                    
                    order = execute_order(client_binance, symbol, Client.SIDE_BUY, quantity)
                    
                    if order:
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
                        
                        with locks['signals']: open_signals_cache[symbol] = signal_data
                        send_telegram_alert(f"New Signal ({strategy}) {symbol} @ {current_price}")

            # تحديث الرصيد
            if paper_trading_mode:
                with locks['balance']:
                    total = usdt_balance
                    for s in open_signals_cache.values():
                        total += live_prices.get(s['symbol'], s['entry_price']) * s['quantity']
                    if redis_client:
                        try: redis_client.set('paper_balance', json.dumps({'usdt': usdt_balance, 'total': total}))
                        except: pass

            time.sleep(60) # فحص دوري كل دقيقة
        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            time.sleep(60)

# --- الواجهة ---
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head><meta charset="utf-8"><title>Bot Dashboard</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--text:#f1f5f9;--accent:#3b82f6;--success:#22c55e;--danger:#ef4444}
body{font-family:sans-serif;background:var(--bg);color:var(--text);margin:0;padding:20px}
.container{max-width:1200px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:20px}
.card{background:var(--card);padding:20px;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,0.3)}
h1,h2{margin-top:0}
.btn{background:var(--accent);color:white;border:none;padding:10px 20px;border-radius:5px;cursor:pointer}
.btn.stop{background:var(--danger)}
table{width:100%;border-collapse:collapse;margin-top:10px}
th,td{text-align:right;padding:10px;border-bottom:1px solid #334155}
.pos{color:var(--success)}.neg{color:var(--danger)}
</style>
</head>
<body>
<div class="container">
    <header style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
        <h1>Smart Bot V2</h1>
        <button id="toggleBtn" class="btn" onclick="toggleTrading()">Start Trading</button>
    </header>
    <div class="grid">
        <div class="card">
            <h2>Market Status</h2>
            <div id="regimeText">Loading...</div>
            <div style="font-size:30px;font-weight:bold;margin-top:10px" id="scoreVal">--</div>
        </div>
        <div class="card">
            <h2>Performance</h2>
            <p>Balance: <span id="balanceVal">--</span> USDT</p>
            <p>Open Trades: <span id="countVal">0</span></p>
            <p>Win Rate: <span id="winRateVal">--</span>%</p>
        </div>
    </div>
    <div class="card">
        <h2>Active Trades</h2>
        <table id="tradesTable">
            <thead><tr><th>Symbol</th><th>Strategy</th><th>PnL</th><th>Status</th></tr></thead>
            <tbody></tbody>
        </table>
    </div>
</div>
<script>
function update() {
    fetch('/api/data').then(r=>r.json()).then(d=>{
        document.getElementById('regimeText').innerText = d.market_regime.toUpperCase();
        document.getElementById('scoreVal').innerText = d.market_score;
        document.getElementById('balanceVal').innerText = d.balance.toFixed(2);
        document.getElementById('countVal').innerText = d.open_signals.length;
        document.getElementById('winRateVal').innerText = (d.win_rate || 0).toFixed(1);
        
        const btn = document.getElementById('toggleBtn');
        btn.className = d.is_enabled ? "btn stop" : "btn";
        btn.innerText = d.is_enabled ? "Stop" : "Start";

        let html = '';
        d.open_signals.forEach(s => {
            const price = d.live_prices[s.symbol] || s.entry_price;
            const pnl = ((price - s.entry_price) / s.entry_price * 100);
            html += `<tr><td>${s.symbol}</td><td>${s.strategy_name}</td>
            <td class="${pnl>=0?'pos':'neg'}">${pnl.toFixed(2)}%</td><td>Open</td></tr>`;
        });
        document.querySelector('#tradesTable tbody').innerHTML = html || '<tr><td colspan="4">No trades</td></tr>';
    });
}
function toggle(){fetch('/api/toggle',{method:'POST'}).then(update)}
setInterval(update, 2000);
update();
</script>
</body>
</html>
"""

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/data')
def get_data():
    with locks['market']: reg, sc = current_market_regime, market_score
    with locks['signals']: sigs = list(open_signals_cache.values())
    with locks['prices']: prs = live_prices.copy()
    with locks['balance']: bal = usdt_balance
    
    # حساب نسبة الفوز
    total_trades = sum(p['total_trades'] for p in strategy_performance.values())
    wins = sum(p['wins'] for p in strategy_performance.values())
    wr = (wins/total_trades*100) if total_trades > 0 else 0

    return jsonify({
        "market_regime": reg, "market_score": sc,
        "open_signals": sigs, "live_prices": prs,
        "balance": bal, "is_enabled": is_trading_enabled,
        "win_rate": wr
    })

@app.route('/api/toggle', methods=['POST'])
def toggle():
    global is_trading_enabled
    is_trading_enabled = not is_trading_enabled
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    init_db()
    Thread(target=main_bot_loop, daemon=True).start()

    app.run(host='0.0.0.0', port=PORT, use_reloader=False)
