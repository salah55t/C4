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
from typing import Optional, Dict, Any, Tuple, List
from functools import wraps
from contextlib import contextmanager
import warnings

# --- 1. إعدادات النظام ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('smart_bot_v15.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SmartBot_Arab_V15')


class ConfigError(Exception):
    pass


def load_config() -> Dict[str, str]:
    required_keys = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'DATABASE_URL']
    optional_keys = {'TELEGRAM_BOT_TOKEN': '', 'TELEGRAM_CHAT_ID': ''}
    
    config_values = {}
    missing_keys = []
    
    for key in required_keys:
        try:
            config_values[key] = config(key)
        except Exception:
            missing_keys.append(key)
    
    if missing_keys:
        raise ConfigError(f"❌ الإعدادات المطلوبة غير موجودة: {', '.join(missing_keys)}")
    
    for key, default in optional_keys.items():
        config_values[key] = config(key, default=default)
    
    return config_values


try:
    CONFIG = load_config()
    API_KEY = CONFIG['BINANCE_API_KEY']
    API_SECRET = CONFIG['BINANCE_API_SECRET']
    DB_URL = CONFIG['DATABASE_URL']
    TELEGRAM_TOKEN = CONFIG['TELEGRAM_BOT_TOKEN']
    TELEGRAM_CHAT_ID = CONFIG['TELEGRAM_CHAT_ID']
except ConfigError as e:
    logger.critical(str(e))
    exit(1)


# --- 2. إعدادات التداول ---
class TradingSettings:
    _defaults = {
        "is_trading_enabled": False,
        "paper_trading_mode": True,
        "base_capital": 1000.0,
        "risk_per_trade_pct": 2.5,
        "max_open_trades": 5,
        "max_drawdown_protect": 8.0,
        "timeframe_analysis": "15m",
        "atr_sl_multiplier": 1.5,
        "atr_tp1_multiplier": 2.5,
        "atr_tp2_multiplier": 5.0,
        "max_position_pct": 0.25,
        "trailing_activation": 1.5,
        "trailing_stop_atr": 1.5
    }
    
    def __init__(self):
        self._settings = self._defaults.copy()
        self._lock = Lock()
    
    def get(self, key: str, default=None):
        with self._lock:
            return self._settings.get(key, default)
    
    def set(self, key: str, value):
        with self._lock:
            if key in self._settings:
                self._settings[key] = value
    
    def toggle_trading(self) -> bool:
        with self._lock:
            self._settings['is_trading_enabled'] = not self._settings['is_trading_enabled']
            return self._settings['is_trading_enabled']
    
    def to_dict(self) -> Dict:
        with self._lock:
            return self._settings.copy()


BOT_SETTINGS = TradingSettings()


# --- 3. حالة النظام ---
class SystemState:
    def __init__(self, base_capital: float):
        self._state = {
            "market_regime": "Analyzing...",
            "trend_strength": 0,
            "volatility_index": "Normal",
            "btc_trend": "Neutral",
            "last_update": None
        }
        self._lock = Lock()
    
    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if key in self._state:
                    self._state[key] = value
            self._state['last_update'] = datetime.now()
    
    def to_dict(self) -> Dict:
        with self._lock:
            return self._state.copy()


system_state = SystemState(BOT_SETTINGS.get('base_capital'))


# --- 4. Cache (تم الإصلاح هنا) ---
class ThreadSafeCache:
    def __init__(self, maxlen: int = 200):
        self._data: Dict[str, Any] = {}
        self._logs = deque(maxlen=maxlen)
        self._lock = Lock()
    
    def set(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value
    
    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)
    
    def delete(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._data.pop(key, None)
    
    def items(self) -> List[Tuple[str, Any]]:
        with self._lock:
            # نرجع نسخة لتجنب مشاكل التعديل أثناء الدوران
            return list(self._data.items())
    
    def add_log(self, log_entry: Dict):
        with self._lock:
            self._logs.appendleft(log_entry)
    
    def get_logs(self) -> List[Dict]:
        with self._lock:
            return list(self._logs)

    # --- الإصلاح: إضافة دالة __len__ ---
    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


signals_cache = ThreadSafeCache()
prices_cache = ThreadSafeCache()


# --- 5. قاعدة البيانات ---
class DatabaseManager:
    def __init__(self, db_url: str):
        self._db_url = db_url
        self._conn = None
        self._lock = Lock()
    
    def _ensure_connection(self):
        try:
            if self._conn is None or self._conn.closed != 0:
                self._conn = psycopg2.connect(self._db_url, cursor_factory=RealDictCursor)
                self._conn.autocommit = True
        except Exception as e:
            logger.error(f"DB Connection Error: {e}")
    
    @contextmanager
    def get_cursor(self):
        with self._lock:
            self._ensure_connection()
            if self._conn:
                cursor = self._conn.cursor()
                try:
                    yield cursor
                finally:
                    cursor.close()
            else:
                # في حالة فشل الاتصال، نقوم بتمرير yield وهمي لتجنب الانهيار الكامل
                logger.error("Database connection lost, skipping DB operation.")
                yield None
    
    def init_tables(self):
        # التأكد من صحة المؤشر قبل التنفيذ
        try:
            with self.get_cursor() as cur:
                if cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS trades_v15 (
                            id SERIAL PRIMARY KEY,
                            symbol TEXT NOT NULL,
                            entry_price DOUBLE PRECISION NOT NULL,
                            stop_loss DOUBLE PRECISION NOT NULL,
                            tp1 DOUBLE PRECISION NOT NULL,
                            tp2 DOUBLE PRECISION NOT NULL,
                            quantity DOUBLE PRECISION NOT NULL,
                            strategy_name TEXT NOT NULL,
                            market_regime TEXT,
                            status TEXT DEFAULT 'open',
                            mode TEXT NOT NULL,
                            entry_time TIMESTAMP DEFAULT NOW(),
                            closed_at TIMESTAMP,
                            closing_price DOUBLE PRECISION,
                            profit_abs DOUBLE PRECISION,
                            profit_pct DOUBLE PRECISION,
                            exit_reason TEXT
                        );
                    """)
                    logger.info("✅ قاعدة البيانات جاهزة (V15).")
        except Exception as e:
            logger.critical(f"Failed to init DB: {e}")
    
    def insert_trade(self, trade_data: Dict) -> int:
        with self.get_cursor() as cur:
            if cur:
                cur.execute("""
                    INSERT INTO trades_v15 
                    (symbol, entry_price, stop_loss, tp1, tp2, quantity, 
                     strategy_name, market_regime, status, mode, entry_time)
                    VALUES (%(symbol)s, %(entry_price)s, %(stop_loss)s, %(tp1)s, %(tp2)s, 
                            %(quantity)s, %(strategy)s, %(regime)s, 'open', %(mode)s, NOW())
                    RETURNING id
                """, trade_data)
                return cur.fetchone()['id']
        return 0
    
    def update_stop_loss(self, trade_id: int, new_sl: float):
        with self.get_cursor() as cur:
            if cur:
                cur.execute("UPDATE trades_v15 SET stop_loss = %s WHERE id = %s", (new_sl, trade_id))
    
    def close_trade(self, trade_id: int, closing_price: float, profit_pct: float, profit_abs: float, exit_reason: str):
        with self.get_cursor() as cur:
            if cur:
                cur.execute("""
                    UPDATE trades_v15 
                    SET status = 'closed', closed_at = NOW(), closing_price = %s,
                        profit_pct = %s, profit_abs = %s, exit_reason = %s
                    WHERE id = %s
                """, (closing_price, profit_pct, profit_abs, exit_reason, trade_id))
    
    def get_trade_statistics(self) -> Dict:
        with self.get_cursor() as cur:
            if not cur:
                 return {'win_rate': 0, 'profit_factor': 0, 'total_pnl_usd': 0, 'trade_count': 0, 'history': []}
            cur.execute("SELECT closed_at, profit_pct, profit_abs FROM trades_v15 WHERE status = 'closed' ORDER BY closed_at ASC")
            rows = cur.fetchall()
        
        if not rows:
            return {'win_rate': 0, 'profit_factor': 0, 'total_pnl_usd': 0, 'trade_count': 0, 'history': []}
        
        wins = sum(1 for r in rows if r['profit_pct'] > 0)
        gross_profit = sum(r['profit_abs'] for r in rows if r['profit_abs'] > 0)
        gross_loss = sum(abs(r['profit_abs']) for r in rows if r['profit_abs'] < 0)
        cum_pnl = 0.0
        history = []
        
        for row in rows:
            cum_pnl += row['profit_pct']
            history.append({'t': row['closed_at'].strftime('%d %H:%M'), 'v': round(cum_pnl, 2)})
        
        return {
            'trade_count': len(rows),
            'total_pnl_usd': round(gross_profit - gross_loss, 2),
            'win_rate': round((wins / len(rows)) * 100, 1),
            'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9,
            'history': history
        }


db = DatabaseManager(DB_URL)


# --- 6. نظام التنبيهات ---
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._enabled = bool(token and chat_id)
    
    def _send(self, text: str):
        if not self._enabled: return
        try:
            requests.post(self._base_url, data={"chat_id": self._chat_id, "text": text, "parse_mode": "Markdown"}, timeout=5)
        except: pass
    
    def notify_buy(self, p: Dict):
        m = f"🚀 *إشارة شراء قوية | {p['symbol']}*\n🔎 الاستراتيجية: `{p['strategy']}`\n💵 الدخول: `{p['price']}`\n🛑 الوقف: `{p['sl']}`\n🎯 أهداف: `{p['tp1']}` - `{p['tp2']}`"
        Thread(target=self._send, args=(m,), daemon=True).start()
    
    def notify_sell(self, p: Dict):
        e = "✅ ربح" if p['profit'] > 0 else "🔻 وقف"
        m = f"{e} *إغلاق صفقة | {p['symbol']}*\nسعر الخروج: `{p['price']}`\nالربح: `{p['profit']:.2f}%`\nالسبب: {p['reason']}"
        Thread(target=self._send, args=(m,), daemon=True).start()


notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)


# --- 7. التحليل الفني ---
class TechnicalAnalyzer:
    @staticmethod
    def fetch_data(client: Client, symbol: str, interval: str, limit: int = 150) -> Optional[pd.DataFrame]:
        try:
            klines = client.get_historical_klines(symbol, interval, limit=limit)
            if not klines or len(klines) < 50: return None
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qv', 't', 'tb', 'tq', 'ig'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e: 
            logger.error(f"Fetch Error ({symbol}): {e}")
            return None
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Slope
        df['slope_ema50'] = np.degrees(np.arctan(df['ema50'].diff() / df['ema50']))
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger
        df['bb_mid'] = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (2 * std)
        df['bb_lower'] = df['bb_mid'] - (2 * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # ATR & ADX
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        df['atr'] = df['tr'].rolling(14).mean()
        
        # ADX Simplified Logic
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr_s = df['tr'].rolling(14).sum()
        p_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr_s.replace(0, np.inf))
        m_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr_s.replace(0, np.inf))
        dx = 100 * abs(p_di - m_di) / (p_di + m_di).replace(0, np.inf)
        df['adx'] = dx.rolling(14).mean()

        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] > (df['vol_ma'] * 1.5)
        
        return df.fillna(0)


analyzer = TechnicalAnalyzer()


# --- 8. محلل بيئة السوق ---
class MarketRegimeAnalyzer:
    @staticmethod
    def analyze(client: Client) -> str:
        df = analyzer.fetch_data(client, 'BTCUSDT', '1h', 100)
        if df is None: return "Neutral"
        
        df = analyzer.calculate_indicators(df)
        last = df.iloc[-1]
        
        bull_trend = (last['close'] > last['ema200']) and (last['ema50'] > last['ema200'])
        strong_momentum = last['adx'] > 25
        high_volatility = (last['atr'] / last['close']) * 100 > 2.5
        
        if bull_trend and strong_momentum:
            regime = "BULL_STRONG"
        elif bull_trend and not strong_momentum:
            regime = "BULL_WEAK"
        elif not bull_trend and last['close'] < last['ema200'] and strong_momentum:
            regime = "BEAR_STRONG"
        elif high_volatility:
            regime = "HIGH_VOLATILITY"
        else:
            regime = "RANGING"
            
        system_state.update(
            market_regime=regime, 
            trend_strength=int(last['adx']),
            btc_trend="Bullish" if bull_trend else "Bearish"
        )
        return regime


# --- 9. مصنع الاستراتيجيات ---
class StrategyFactory:
    @staticmethod
    def get_signal(symbol: str, df: pd.DataFrame, regime: str) -> Tuple[Optional[str], str]:
        if len(df) < 50: return None, "بيانات غير كافية"
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. فلتر السيولة
        if curr['close'] * curr['volume'] < 50000:
            return None, "سيولة ضعيفة"

        # 2. Trend Master
        if regime in ['BULL_STRONG', 'BULL_WEAK']:
            if (curr['ema20'] > curr['ema50'] > curr['ema200']):
                if (curr['rsi'] > 50) and (curr['macd_hist'] > 0) and (curr['macd_hist'] > prev['macd_hist']):
                    if curr['adx'] > 20:
                        return "TREND_MASTER", "ترند قوي + زخم متصاعد"

        # 3. Volatility Breakout
        if curr['bb_width'] < 0.15:
            pass 
        elif (prev['bb_width'] < 0.15) and (curr['bb_width'] > prev['bb_width']):
            if (curr['close'] > curr['bb_upper']) and curr['vol_spike']:
                return "VOLATILITY_BREAKOUT", "اختراق بولنجر مع حجم تداول"

        # 4. Reversal Hunter
        if regime != 'BEAR_STRONG':
            if curr['rsi'] < 30:
                past_low = df['close'].iloc[-10:-2].min()
                past_rsi = df['rsi'].iloc[-10:-2].min()
                if (curr['close'] <= past_low) and (curr['rsi'] > past_rsi):
                    if curr['close'] > curr['open']: 
                        return "REVERSAL_HUNTER", "دايفرجنس إيجابي + تشبع بيعي"
        
        return None, "لا توجد فرصة مؤكدة"


# --- 10. مدير الصفقات ---
class TradeManager:
    @staticmethod
    def calculate_qty(entry: float, sl: float, capital: float) -> float:
        risk_amt = capital * (BOT_SETTINGS.get('risk_per_trade_pct') / 100)
        risk_per_share = abs(entry - sl)
        if risk_per_share == 0: return 0
        qty = risk_amt / risk_per_share
        
        max_pos_val = capital * BOT_SETTINGS.get('max_position_pct')
        if (qty * entry) > max_pos_val:
            qty = max_pos_val / entry
        return qty

    @staticmethod
    def manage(trade: Dict, df: pd.DataFrame) -> Tuple[str, float, str]:
        last = df.iloc[-1]
        curr = last['close']
        entry = trade['entry_price']
        sl = trade['stop_loss']
        
        profit_pct = (curr - entry) / entry * 100
        
        # 1. تأمين الصفقة
        if profit_pct >= BOT_SETTINGS.get('trailing_activation', 1.5):
            new_sl = entry * 1.002
            if sl < new_sl:
                return "UPDATE_SL", new_sl, "تأمين الصفقة (Breakeven)"
            
            # 2. ملاحقة الأرباح
            atr_trail = curr - (last['atr'] * BOT_SETTINGS.get('trailing_stop_atr', 1.5))
            if atr_trail > sl:
                 return "UPDATE_SL", atr_trail, "ملاحقة الأرباح"

        # 3. إغلاق الطوارئ
        if last['close'] < last['ema50'] and trade['strategy'] == "TREND_MASTER":
             if profit_pct > -1.0:
                return "CLOSE_NOW", curr, "كسر ترند (EMA50)"

        return "HOLD", 0, ""


trade_manager = TradeManager()


# --- 11. المحرك الرئيسي ---
class TradingEngine:
    def __init__(self):
        self._client = None
        self._symbols = []
        self._running = False
    
    def setup(self):
        self._client = Client(API_KEY, API_SECRET)
        try:
            tickers = self._client.get_ticker()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and float(t['quoteVolume']) > 10000000]
            usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            self._symbols = [x['symbol'] for x in usdt_pairs[:40]]
            logger.info(f"📋 تم تحميل {len(self._symbols)} عملة ذات سيولة عالية")
        except:
            self._symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']
            
    def run(self):
        self.setup()
        self._running = True
        logger.info("🚀 SmartBot V15 Engine Started")
        
        while self._running:
            try:
                if not BOT_SETTINGS.get('is_trading_enabled'):
                    time.sleep(5)
                    continue

                regime = MarketRegimeAnalyzer.analyze(self._client)
                paper = BOT_SETTINGS.get('paper_trading_mode')
                
                # 1. إدارة المفتوح (تم الإصلاح: استخدام items() لنسخ القائمة)
                # استخدام نسخة لتجنب RuntimeError عند الحذف
                active_trades_list = signals_cache.items()
                
                for sym, trade in active_trades_list:
                    df = analyzer.fetch_data(self._client, sym, '5m', 50)
                    if df is None: continue
                    
                    curr = df.iloc[-1]['close']
                    prices_cache.set(sym, curr)
                    
                    # فحص وقف الخسارة
                    if curr <= trade['stop_loss']:
                        trade_obj = signals_cache.delete(sym)
                        if trade_obj:
                            profit = (curr - trade_obj['entry_price']) / trade_obj['entry_price'] * 100
                            db.close_trade(trade_obj['id'], curr, profit, 0, "Stop Loss")
                            notifier.notify_sell({'symbol': sym, 'price': curr, 'profit': profit, 'reason': "Stop Loss 🛑"})
                        continue
                        
                    act, val, note = trade_manager.manage(trade, df)
                    
                    if act == "UPDATE_SL":
                        trade['stop_loss'] = val
                        signals_cache.set(sym, trade)
                        db.update_stop_loss(trade['id'], val)
                        notifier._send(f"🛡️ تحديث وقف {sym} إلى {val:.4f}")
                        
                    elif act == "CLOSE_NOW":
                        trade_obj = signals_cache.delete(sym)
                        if trade_obj:
                            profit = (val - trade_obj['entry_price']) / trade_obj['entry_price'] * 100
                            db.close_trade(trade_obj['id'], val, profit, 0, note)
                            notifier.notify_sell({'symbol': sym, 'price': val, 'profit': profit, 'reason': note})

                # 2. بحث جديد (تم إصلاح الخطأ: الآن len() تعمل)
                if len(signals_cache) < BOT_SETTINGS.get('max_open_trades'):
                    random.shuffle(self._symbols)
                    for sym in self._symbols[:15]:
                        if sym in signals_cache: continue
                        
                        df = analyzer.fetch_data(self._client, sym, BOT_SETTINGS.get('timeframe_analysis'))
                        if df is None: continue
                        
                        df = analyzer.calculate_indicators(df)
                        strat, reason = StrategyFactory.get_signal(sym, df, regime)
                        
                        if strat:
                            last = df.iloc[-1]
                            curr, atr = last['close'], last['atr']
                            sl = curr - (atr * BOT_SETTINGS.get('atr_sl_multiplier'))
                            tp1 = curr + (atr * BOT_SETTINGS.get('atr_tp1_multiplier'))
                            tp2 = curr + (atr * BOT_SETTINGS.get('atr_tp2_multiplier'))
                            qty = trade_manager.calculate_qty(curr, sl, BOT_SETTINGS.get('base_capital'))
                            
                            if qty > 0:
                                t_data = {
                                    'symbol': sym, 'entry_price': curr, 'stop_loss': sl,
                                    'tp1': tp1, 'tp2': tp2, 'quantity': qty,
                                    'strategy': strat, 'regime': regime, 
                                    'mode': 'PAPER' if paper else 'REAL'
                                }
                                tid = db.insert_trade(t_data)
                                if tid:
                                    t_data['id'] = tid
                                    t_data['entry_time'] = datetime.now()
                                    signals_cache.set(sym, t_data)
                                    notifier.notify_buy({'symbol': sym, 'strategy': strat, 'price': curr, 'sl': sl, 'tp1': tp1, 'tp2': tp2})
                                    signals_cache.add_log({'t': datetime.now().strftime('%H:%M'), 's': sym, 'st': 'دخول', 'r': strat})

                time.sleep(20)
            except Exception as e:
                logger.error(f"Engine Error: {e}")
                time.sleep(10)

engine = TradingEngine()


# --- 12. Web App ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def index(): return render_template_string(DASHBOARD_HTML)

@app.route('/api/analytics')
def analytics():
    return jsonify({
        "market": system_state.to_dict(),
        "signals": [v for k,v in signals_cache.items()],
        "prices": {k:prices_cache.get(k) for k,v in signals_cache.items()}, # تحسين جلب الأسعار
        "stats": db.get_trade_statistics(),
        "logs": signals_cache.get_logs(),
        "settings": BOT_SETTINGS.to_dict()
    })

@app.route('/api/toggle', methods=['POST'])
def toggle(): return jsonify({"is_trading_enabled": BOT_SETTINGS.toggle_trading()})

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot V15 - Pro Logic</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0b0e11; --panel: #151a1e; --border: #2b3139; --text: #eaecef; --green: #0ecb81; --red: #f6465d; --accent: #f0b90b; }
        body { background: var(--bg); color: var(--text); font-family: 'Tajawal'; padding: 20px; margin: 0; }
        .header { display: flex; justify-content: space-between; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 20px; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
        .btn { background: var(--accent); border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; }
        td, th { padding: 12px; border-bottom: 1px solid var(--border); text-align: right; }
        .col-2 { grid-column: span 2; } .col-4 { grid-column: span 4; }
        @media(max-width: 768px) { .grid { grid-template-columns: 1fr; } .col-2, .col-4 { grid-column: span 1; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>SmartBot <span style="color:var(--accent)">V15.1</span> <small style="font-size:14px; color:#666">Stable</small></h1>
        <button id="pwrBtn" class="btn" onclick="toggle()">تحميل...</button>
    </div>
    
    <div class="grid">
        <div class="card"><h3>حالة السوق</h3><h2 id="regime" style="color:var(--accent)">--</h2></div>
        <div class="card"><h3>الربح الصافي</h3><h2 id="pnl">--</h2></div>
        <div class="card"><h3>نسبة النجاح</h3><h2 id="winrate">--</h2></div>
        <div class="card"><h3>صفقات نشطة</h3><h2 id="active">--</h2></div>
    </div>

    <div class="grid">
        <div class="card col-2">
            <h3>📈 منحنى النمو</h3>
            <div style="height:250px"><canvas id="chart"></canvas></div>
        </div>
        <div class="card col-2">
            <h3>📋 الصفقات المفتوحة</h3>
            <div style="overflow:auto; height:250px">
                <table id="tbl"><thead><th>الزوج</th><th>الاستراتيجية</th><th>الربح %</th></thead><tbody></tbody></table>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h3>📝 سجل العمليات</h3>
        <div style="height:150px; overflow:auto"><table id="logtbl"><tbody></tbody></table></div>
    </div>

    <script>
        const stratMap = {
            "TREND_MASTER": "سيد الترند 🐂",
            "VOLATILITY_BREAKOUT": "اختراق انفجاري 💥",
            "REVERSAL_HUNTER": "قناص القيعان 🎣"
        };
        let chart;

        async function update() {
            try {
                const res = await fetch('/api/analytics');
                const d = await res.json();
                
                // Header
                const btn = document.getElementById('pwrBtn');
                btn.innerText = d.settings.is_trading_enabled ? "⏹ إيقاف" : "▶ تشغيل";
                btn.style.background = d.settings.is_trading_enabled ? "var(--red)" : "var(--green)";
                
                // Cards
                document.getElementById('regime').innerText = d.market.market_regime;
                document.getElementById('pnl').innerText = "$" + d.stats.total_pnl_usd;
                document.getElementById('pnl').style.color = d.stats.total_pnl_usd >= 0 ? "var(--green)" : "var(--red)";
                document.getElementById('winrate').innerText = d.stats.win_rate + "%";
                document.getElementById('active').innerText = d.signals.length;

                // Table
                document.querySelector('#tbl tbody').innerHTML = d.signals.map(s => {
                    const curr = d.prices[s.symbol] || s.entry_price;
                    const pnl = ((curr - s.entry_price)/s.entry_price*100).toFixed(2);
                    return `<tr><td>${s.symbol}</td><td>${stratMap[s.strategy]||s.strategy}</td><td style="color:${pnl>=0?'var(--green)':'var(--red)'}">${pnl}%</td></tr>`;
                }).join('');

                // Logs
                document.querySelector('#logtbl tbody').innerHTML = d.logs.map(l => 
                    `<tr><td style="color:#666">${l.t}</td><td>${l.s}</td><td>${l.r}</td></tr>`
                ).join('');

                // Chart
                if(!chart) {
                    chart = new Chart(document.getElementById('chart'), {
                        type: 'line', data: {labels:[], datasets:[{label:'USD', data:[], borderColor:'#f0b90b', tension:0.4}]},
                        options: {responsive:true, plugins:{legend:{display:false}}, scales:{x:{display:false}, y:{grid:{color:'#222'}}}}
                    });
                }
                if(d.stats.history.length > 0) {
                    chart.data.labels = d.stats.history.map(h=>h.t);
                    chart.data.datasets[0].data = d.stats.history.map(h=>h.v);
                    chart.update();
                }
            } catch (e) { console.error(e); }
        }
        
        async function toggle() { await fetch('/api/toggle', {method:'POST'}); update(); }
        setInterval(update, 2000);
        update();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    db.init_tables()
    Thread(target=engine.run, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)