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

# --- 1. إعدادات النظام المحسنة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('smart_bot_v14.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SmartBot_Arab_V14')


class ConfigError(Exception):
    """خطأ مخصص للإعدادات"""
    pass


def load_config() -> Dict[str, str]:
    """تحميل الإعدادات مع التحقق الصارم"""
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


# --- 2. إعدادات التداول المحسنة ---
class TradingSettings:
    """إدارة إعدادات التداول بشكل آمن"""
    
    _defaults = {
        "is_trading_enabled": False,
        "paper_trading_mode": True,
        "base_capital": 1000.0,
        "risk_per_trade_pct": 2.0,
        "max_open_trades": 6,
        "max_drawdown_protect": 10.0,
        "volume_lookback": 50,
        "timeframe_analysis": "15m",
        "timeframe_trend": "1h",
        "atr_sl_multiplier": 2.0,
        "atr_tp1_multiplier": 2.0,
        "atr_tp2_multiplier": 4.0,
        "max_position_pct": 0.2,
        "trailing_stop_atr": 2.0
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
LEADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']


# --- 3. حالة النظام المحسنة ---
class SystemState:
    """إدارة حالة النظام بشكل thread-safe"""
    
    def __init__(self, base_capital: float):
        self._state = {
            "market_regime": "Neutral",
            "trend_strength": 0,
            "volatility_index": "Low",
            "portfolio_value": base_capital,
            "last_update": None
        }
        self._lock = Lock()
    
    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if key in self._state:
                    self._state[key] = value
            self._state['last_update'] = datetime.now()
    
    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)
    
    def to_dict(self) -> Dict:
        with self._lock:
            return self._state.copy()


system_state = SystemState(BOT_SETTINGS.get('base_capital'))


# --- 4. Cache محسن مع Thread Safety ---
class ThreadSafeCache:
    """Cache آمن للخيوط"""
    
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
            return list(self._data.items())
    
    def values(self) -> List[Any]:
        with self._lock:
            return list(self._data.values())
    
    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
    
    def add_log(self, log_entry: Dict):
        with self._lock:
            self._logs.appendleft(log_entry)
    
    def get_logs(self) -> List[Dict]:
        with self._lock:
            return list(self._logs)


signals_cache = ThreadSafeCache()
prices_cache = ThreadSafeCache()


# --- 5. قاعدة البيانات المحسنة ---
class DatabaseManager:
    """مدير قاعدة البيانات مع Connection Pooling"""
    
    def __init__(self, db_url: str):
        self._db_url = db_url
        self._conn: Optional[psycopg2.extensions.connection] = None
        self._lock = Lock()
    
    def _ensure_connection(self):
        """التأكد من وجود اتصال صالح"""
        if self._conn is None or self._conn.closed != 0:
            self._conn = psycopg2.connect(
                self._db_url,
                cursor_factory=RealDictCursor
            )
            self._conn.autocommit = True
    
    @contextmanager
    def get_cursor(self):
        """Context manager للحصول على cursor"""
        with self._lock:
            self._ensure_connection()
            cursor = self._conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    def init_tables(self):
        """إنشاء الجداول"""
        with self.get_cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades_v14 (
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
                    exit_reason TEXT,
                    CONSTRAINT positive_quantity CHECK (quantity > 0),
                    CONSTRAINT valid_status CHECK (status IN ('open', 'closed'))
                );
                
                CREATE INDEX IF NOT EXISTS idx_trades_v14_status ON trades_v14(status);
                CREATE INDEX IF NOT EXISTS idx_trades_v14_symbol ON trades_v14(symbol);
            """)
        logger.info("✅ قاعدة البيانات جاهزة (V14).")
    
    def insert_trade(self, trade_data: Dict) -> int:
        """إدراج صفقة جديدة"""
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO trades_v14 
                (symbol, entry_price, stop_loss, tp1, tp2, quantity, 
                 strategy_name, market_regime, status, mode, entry_time)
                VALUES (%(symbol)s, %(entry_price)s, %(stop_loss)s, %(tp1)s, %(tp2)s, 
                        %(quantity)s, %(strategy)s, %(regime)s, 'open', %(mode)s, NOW())
                RETURNING id
            """, trade_data)
            return cur.fetchone()['id']
    
    def update_stop_loss(self, trade_id: int, new_sl: float):
        """تحديث وقف الخسارة"""
        with self.get_cursor() as cur:
            cur.execute(
                "UPDATE trades_v14 SET stop_loss = %s WHERE id = %s",
                (new_sl, trade_id)
            )
    
    def close_trade(self, trade_id: int, closing_price: float, 
                    profit_pct: float, profit_abs: float, exit_reason: str):
        """إغلاق الصفقة"""
        with self.get_cursor() as cur:
            cur.execute("""
                UPDATE trades_v14 
                SET status = 'closed', 
                    closed_at = NOW(), 
                    closing_price = %s,
                    profit_pct = %s, 
                    profit_abs = %s, 
                    exit_reason = %s
                WHERE id = %s
            """, (closing_price, profit_pct, profit_abs, exit_reason, trade_id))
    
    def get_trade_statistics(self) -> Dict:
        """الحصول على إحصائيات التداول"""
        with self.get_cursor() as cur:
            cur.execute("""
                SELECT 
                    closed_at, 
                    profit_pct, 
                    profit_abs
                FROM trades_v14 
                WHERE status = 'closed' 
                ORDER BY closed_at ASC
            """)
            rows = cur.fetchall()
        
        if not rows:
            return {
                'win_rate': 0, 
                'profit_factor': 0, 
                'total_pnl_usd': 0, 
                'trade_count': 0, 
                'history': []
            }
        
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        cum_pnl = 0.0
        history = []
        
        for row in rows:
            pnl_pct = row['profit_pct']
            pnl_abs = row['profit_abs'] or 0
            
            if pnl_pct > 0:
                wins += 1
                gross_profit += pnl_abs
            else:
                gross_loss += abs(pnl_abs)
            
            cum_pnl += pnl_pct
            history.append({
                't': row['closed_at'].strftime('%d %H:%M') if row['closed_at'] else '',
                'v': round(cum_pnl, 2)
            })
        
        return {
            'trade_count': len(rows),
            'total_pnl_usd': round(gross_profit - gross_loss, 2),
            'win_rate': round((wins / len(rows)) * 100, 1),
            'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9,
            'history': history
        }


db = DatabaseManager(DB_URL)


# --- 6. نظام التنبيهات المحسن ---
class TelegramNotifier:
    """مدير إشعارات تيليجرام"""
    
    def __init__(self, token: str, chat_id: str):
        self._token = token
        self._chat_id = chat_id
        self._enabled = bool(token and chat_id)
        self._base_url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._timeout = 10
    
    def _send(self, message: str):
        """إرسال رسالة"""
        if not self._enabled:
            return
        
        try:
            requests.post(
                self._base_url,
                data={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                },
                timeout=self._timeout
            )
        except requests.RequestException as e:
            logger.warning(f"فشل إرسال إشعار تيليجرام: {e}")
    
    def notify_buy(self, payload: Dict):
        """إشعار شراء"""
        mode_icon = "🧪 تجريبي" if payload.get('is_paper') else "💰 حقيقي"
        message = (
            f"🚀 *تنفيذ دخول استراتيجي | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"📊 الاستراتيجية: `{payload['strategy']}`\n"
            f"🌍 حالة السوق: {payload.get('regime', 'N/A')}\n"
            f"💵 السعر: `{payload['price']:.8g}`\n"
            f"🛑 الوقف: `{payload['sl']:.8g}`\n"
            f"🎯 الأهداف: `{payload['tp1']:.8g}` ➔ `{payload['tp2']:.8g}`\n"
            f"🕹️ الوضع: {mode_icon}"
        )
        Thread(target=self._send, args=(message,), daemon=True).start()
    
    def notify_sell(self, payload: Dict):
        """إشعار بيع"""
        pnl = payload['profit']
        emoji = "✅ ربح" if pnl > 0 else "🔻 خسارة"
        message = (
            f"{emoji} *إغلاق مركز | {payload['symbol']}*\n"
            f"ـــــــــــــــــــــــــــــــــــــــــــــــــــــ\n"
            f"📉 الخروج: `{payload['price']:.8g}`\n"
            f"💰 الصافي: `{pnl:.2f}%`\n"
            f"📝 السبب: _{payload['reason']}_\n"
            f"⏱️ المدة: {payload['duration']} دقيقة"
        )
        Thread(target=self._send, args=(message,), daemon=True).start()
    
    def notify_sl_update(self, payload: Dict):
        """إشعار تحديث وقف الخسارة"""
        message = (
            f"🛡️ *تحديث وقف الخسارة | {payload['symbol']}*\n"
            f"الوقف الجديد: `{payload['new_sl']:.8g}`\n"
            f"السبب: {payload['reason']}"
        )
        Thread(target=self._send, args=(message,), daemon=True).start()


notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)


# --- 7. محرك التحليل الفني المحسن ---
class TechnicalAnalyzer:
    """محرك التحليل الفني"""
    
    @staticmethod
    def fetch_data(client: Client, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """جلب بيانات السوق"""
        try:
            klines = client.get_historical_klines(symbol, interval, limit=limit)
            if not klines:
                return None
            
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 
                        'taker_buy_base', 'taker_buy_quote', 'ignore']
            )
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except Exception as e:
            logger.debug(f"خطأ في جلب البيانات لـ {symbol}: {e}")
            return None
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """حساب جميع المؤشرات الفنية"""
        df = df.copy()
        
        # EMAs
        for span in [9, 50, 200]:
            df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        rsi_min = df['rsi'].rolling(14).min()
        rsi_max = df['rsi'].rolling(14).max()
        rsi_range = rsi_max - rsi_min
        df['stoch_k'] = ((df['rsi'] - rsi_min) / rsi_range.replace(0, np.inf)) * 100
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR & True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        
        atr_safe = df['atr'].replace(0, np.inf)
        df['plus_di'] = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr_safe)
        df['minus_di'] = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr_safe)
        
        di_sum = df['plus_di'] + df['minus_di']
        di_diff = (df['plus_di'] - df['minus_di']).abs()
        df['dx'] = 100 * (di_diff / di_sum.replace(0, np.inf))
        df['adx'] = df['dx'].rolling(14).mean()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (2 * bb_std)
        df['bb_lower'] = df['bb_mid'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, np.inf)
        
        # Ichimoku Tenkan-sen
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        # Volume MA
        df['vol_ma'] = df['volume'].rolling(20).mean()
        
        # ملء القيم الفارغة
        return df.ffill().fillna(0)


analyzer = TechnicalAnalyzer()


# --- 8. محلل بيئة السوق المحسن ---
class MarketRegimeAnalyzer:
    """تحليل حالة السوق"""
    
    REGIMES = {
        'BULL_STRONG': 'Bull_Trend_Strong',
        'BULL_ACCUMULATION': 'Bull_Accumulation',
        'BEAR_STRONG': 'Bear_Trend_Strong',
        'HIGH_VOLATILITY': 'High_Volatility_Choppy',
        'RANGING': 'Ranging',
        'NEUTRAL': 'Neutral'
    }
    
    @classmethod
    def analyze(cls, client: Client) -> str:
        """تحليل حالة السوق الحالية"""
        df = analyzer.fetch_data(client, 'BTCUSDT', '4h', 100)
        if df is None:
            return cls.REGIMES['NEUTRAL']
        
        df = analyzer.calculate_indicators(df)
        last = df.iloc[-1]
        
        # حساب نقاط الاتجاه
        trend_score = sum([
            last['close'] > last['ema200'],
            last['ema50'] > last['ema200'],
            last['macd'] > last['macd_signal']
        ])
        
        adx = last['adx']
        atr_pct = (last['atr'] / last['close']) * 100 if last['close'] > 0 else 0
        
        # تحديد الحالة
        if trend_score == 3 and adx > 25:
            regime = cls.REGIMES['BULL_STRONG']
        elif trend_score >= 2 and adx < 20:
            regime = cls.REGIMES['BULL_ACCUMULATION']
        elif trend_score == 0 and adx > 25:
            regime = cls.REGIMES['BEAR_STRONG']
        elif atr_pct > 2.0:
            regime = cls.REGIMES['HIGH_VOLATILITY']
        elif adx < 20:
            regime = cls.REGIMES['RANGING']
        else:
            regime = cls.REGIMES['NEUTRAL']
        
        # تحديث الحالة
        system_state.update(
            market_regime=regime,
            trend_strength=int(adx),
            volatility_index="High" if atr_pct > 1.5 else "Normal"
        )
        
        logger.info(f"🧠 حالة السوق: {regime} | القوة: {int(adx)}")
        return regime


# --- 9. مصنع الاستراتيجيات المحسن ---
class StrategyFactory:
    """مصنع الاستراتيجيات"""
    
    STRATEGIES = {
        'TREND_PULLBACK': 'Trend_Pullback',
        'MOMENTUM_BREAKOUT': 'Momentum_Breakout',
        'SNIPER_REVERSION': 'Sniper_Reversion',
        'DEEP_VALUE_SCALP': 'Deep_Value_Scalp',
        'GOLDEN_CROSS': 'Golden_Cross'
    }
    
    @classmethod
    def get_signal(cls, symbol: str, df: pd.DataFrame, regime: str) -> Tuple[Optional[str], str]:
        """الحصول على إشارة التداول"""
        if len(df) < 3:
            return None, "بيانات غير كافية"
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # فلتر السيولة المتكيف
        vol_factor = 0.3 if "High_Volatility" in regime else 0.6
        vol_ma = last['vol_ma']
        
        if vol_ma > 0 and last['volume'] < vol_ma * vol_factor:
            bb_upper = last.get('bb_upper', 0)
            if bb_upper > 0 and last['close'] <= bb_upper:
                return None, "سيولة منخفضة"
        
        # استراتيجية سحابة الاتجاه
        if "Bull_Trend" in regime:
            if last['close'] > last['ema50'] and last['adx'] > 20:
                # إعادة دخول مع التصحيح
                if last['low'] <= last['tenkan_sen'] < last['close']:
                    return cls.STRATEGIES['TREND_PULLBACK'], "إعادة دخول (تصحيح)"
                
                # اختراق الزخم
                if last['close'] > prev['high'] and last['macd_hist'] > 0:
                    return cls.STRATEGIES['MOMENTUM_BREAKOUT'], "اختراق زخم"
        
        # استراتيجية القناص المرتد (للسوق العرضي)
        elif "Ranging" in regime or "Accumulation" in regime:
            bb_width = last.get('bb_width', 1)
            if bb_width < 0.10:  # انضغاط
                if last['rsi'] < 40 and last['stoch_k'] < 20:
                    if last['close'] > prev['close']:
                        return cls.STRATEGIES['SNIPER_REVERSION'], "ارتداد من القاع (تشبع)"
        
        # استراتيجية خطف السيولة (للسوق المتقلب)
        elif "High_Volatility" in regime:
            ema9 = last['ema9']
            if ema9 > 0:
                dist_ema = (last['close'] - ema9) / ema9 * 100
                if dist_ema < -3.0 and last['rsi'] < 25:
                    return cls.STRATEGIES['DEEP_VALUE_SCALP'], "انحراف سعري حاد"
        
        # التقاطع الذهبي (شامل)
        if last['ema50'] > last['ema200'] > 0 and prev['ema50'] <= prev['ema200']:
            return cls.STRATEGIES['GOLDEN_CROSS'], "تقاطع ذهبي طويل الأمد"
        
        return None, "لا توجد فرصة مناسبة"


# --- 10. مدير الصفقات المحسن ---
class TradeManager:
    """مدير الصفقات"""
    
    @staticmethod
    def calculate_position_size(entry: float, sl: float, capital: float, risk_pct: float, max_pos_pct: float) -> float:
        """حساب حجم المركز"""
        if entry <= 0 or sl <= 0:
            return 0
        
        risk_amount = capital * (risk_pct / 100)
        price_diff = abs(entry - sl)
        
        if price_diff <= 0:
            return 0
        
        qty = risk_amount / price_diff
        max_qty = (capital * max_pos_pct) / entry
        
        return min(qty, max_qty)
    
    @staticmethod
    def manage_active_trade(trade: Dict, df: pd.DataFrame) -> Tuple[str, float, str, str]:
        """إدارة الصفقة النشطة"""
        last = df.iloc[-1]
        curr = float(last['close'])
        entry = float(trade['entry_price'])
        tp1 = float(trade['tp1'])
        tp2 = float(trade['tp2'])
        sl = float(trade['stop_loss'])
        
        profit_pct = ((curr - entry) / entry * 100) if entry > 0 else 0
        duration_hours = (datetime.now() - trade['entry_time']).total_seconds() / 3600
        
        health_msg = "مستقر"
        
        # جني الأرباح المرحلي
        if curr >= tp2:
            if sl < tp1:
                return "UPDATE_SL", tp1, "تأمين ربح الهدف الأول", "ربح ممتاز 🟢"
        elif curr >= tp1:
            safe_sl = entry * 1.002
            if sl < safe_sl:
                return "UPDATE_SL", safe_sl, "صفقة خالية من المخاطر", "مؤمنة 🛡️"
        
        # وقف الخسارة المتحرك
        if profit_pct > 2.0:
            atr = last['atr']
            trailing_multiplier = BOT_SETTINGS.get('trailing_stop_atr', 2.0)
            atr_trail = curr - (atr * trailing_multiplier)
            
            if atr_trail > sl:
                return "UPDATE_SL", atr_trail, "ملاحقة الأرباح (ATR)", "منطلق 🏃"
        
        # وقف الوقت
        if duration_hours > 6 and abs(profit_pct) < 0.5:
            return "CLOSE_NOW", curr, "تجميد رأس المال (خروج زمني)", "راكد ⚠️"
        
        # الخروج الفني
        regime = trade.get('market_regime', '')
        if "Bull" in regime and curr < last['ema50']:
            return "CLOSE_NOW", curr, "كسر الاتجاه (EMA50)", "انعكاس 🔻"
        
        return "HOLD", 0, "", health_msg
    
    @classmethod
    def open_trade(cls, symbol: str, price: float, sl: float, tp1: float, tp2: float,
                   qty: float, strategy: str, regime: str, is_paper: bool):
        """فتح صفقة جديدة"""
        try:
            mode = 'PAPER' if is_paper else 'REAL'
            
            trade_data = {
                'symbol': symbol,
                'entry_price': float(price),
                'stop_loss': float(sl),
                'tp1': float(tp1),
                'tp2': float(tp2),
                'quantity': float(qty),
                'strategy': strategy,
                'regime': regime,
                'mode': mode
            }
            
            db_id = db.insert_trade(trade_data)
            
            trade = {
                'id': db_id,
                'symbol': symbol,
                'entry_price': float(price),
                'stop_loss': float(sl),
                'tp1': float(tp1),
                'tp2': float(tp2),
                'quantity': float(qty),
                'entry_time': datetime.now(),
                'strategy': strategy,
                'market_regime': regime,
                'is_paper': is_paper
            }
            
            signals_cache.set(symbol, trade)
            signals_cache.add_log({
                't': datetime.now().strftime('%H:%M'),
                's': symbol,
                'st': 'دخول',
                'r': strategy
            })
            
            notifier.notify_buy({
                **trade,
                'price': price,
                'sl': sl
            })
            
            logger.info(f"✅ صفقة جديدة: {symbol} | {strategy} | السعر: {price}")
            
        except Exception as e:
            logger.error(f"خطأ في فتح الصفقة: {e}")
    
    @classmethod
    def close_trade(cls, symbol: str, price: float, reason: str, is_paper: bool):
        """إغلاق الصفقة"""
        try:
            trade = signals_cache.delete(symbol)
            if not trade:
                return
            
            price = float(price)
            entry_price = trade['entry_price']
            
            profit_pct = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            profit_abs = (price - entry_price) * trade['quantity']
            duration = int((datetime.now() - trade['entry_time']).total_seconds() / 60)
            
            db.close_trade(
                trade['id'],
                price,
                profit_pct,
                profit_abs,
                reason
            )
            
            notifier.notify_sell({
                'symbol': symbol,
                'price': price,
                'profit': profit_pct,
                'reason': reason,
                'duration': duration
            })
            
            logger.info(f"🔒 إغلاق صفقة: {symbol} | {reason} | الربح: {profit_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"خطأ في إغلاق الصفقة: {e}")


trade_manager = TradeManager()


# --- 11. المحرك الرئيسي المحسن ---
class TradingEngine:
    """محرك التداول الرئيسي"""
    
    def __init__(self):
        self._client: Optional[Client] = None
        self._symbols: List[str] = []
        self._running = False
    
    def _init_client(self):
        """تهيئة عميل Binance"""
        self._client = Client(API_KEY, API_SECRET)
        logger.info("✅ تم الاتصال بـ Binance")
    
    def _load_symbols(self):
        """تحميل قائمة العملات"""
        try:
            with open('crypto_list.txt', 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
                self._symbols = [s if s.endswith('USDT') else f'{s}USDT' for s in symbols]
        except FileNotFoundError:
            self._symbols = [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
                'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT'
            ]
        logger.info(f"📋 تم تحميل {len(self._symbols)} عملة")
    
    def _manage_open_trades(self, regime: str, paper: bool):
        """إدارة الصفقات المفتوحة"""
        for symbol, trade in signals_cache.items():
            df = analyzer.fetch_data(self._client, symbol, '5m', 60)
            if df is None:
                continue
            
            df = analyzer.calculate_indicators(df)
            curr_price = float(df['close'].iloc[-1])
            prices_cache.set(symbol, curr_price)
            
            # فحص وقف الخسارة
            if curr_price <= trade['stop_loss']:
                trade_manager.close_trade(symbol, curr_price, "ضرب وقف الخسارة 🛑", paper)
                continue
            
            # إدارة الصفقة
            action, value, note, health = trade_manager.manage_active_trade(trade, df)
            
            if action == "UPDATE_SL":
                updated_trade = trade.copy()
                updated_trade['stop_loss'] = float(value)
                signals_cache.set(symbol, updated_trade)
                db.update_stop_loss(trade['id'], float(value))
                notifier.notify_sl_update({
                    'symbol': symbol,
                    'new_sl': value,
                    'reason': note
                })
            
            elif action == "CLOSE_NOW":
                trade_manager.close_trade(symbol, curr_price, f"خروج ذكي: {note}", paper)
    
    def _scan_for_opportunities(self, regime: str, paper: bool, max_trades: int):
        """البحث عن فرص جديدة"""
        if len(signals_cache) >= max_trades:
            return
        
        try:
            tickers = self._client.get_ticker()
            valid_tickers = [t for t in tickers if t['symbol'] in self._symbols]
            
            # ترتيب حسب السيولة والتغير
            valid_tickers.sort(
                key=lambda x: float(x['quoteVolume']) * abs(float(x['priceChangePercent'])),
                reverse=True
            )
            
            scanned = 0
            for ticker in valid_tickers[:30]:  # فحص أعلى 30
                if scanned >= 20:
                    break
                
                symbol = ticker['symbol']
                if symbol in signals_cache:
                    continue
                
                scanned += 1
                
                df = analyzer.fetch_data(
                    self._client,
                    symbol,
                    BOT_SETTINGS.get('timeframe_analysis', '15m'),
                    100
                )
                
                if df is None:
                    continue
                
                df = analyzer.calculate_indicators(df)
                strategy, reason = StrategyFactory.get_signal(symbol, df, regime)
                
                if strategy:
                    last = df.iloc[-1]
                    curr = float(last['close'])
                    atr = float(last['atr'])
                    
                    # حساب المستويات
                    sl_mult = BOT_SETTINGS.get('atr_sl_multiplier', 2.0)
                    tp1_mult = BOT_SETTINGS.get('atr_tp1_multiplier', 2.0)
                    tp2_mult = BOT_SETTINGS.get('atr_tp2_multiplier', 4.0)
                    
                    sl = curr - (atr * sl_mult)
                    tp1 = curr + (atr * tp1_mult)
                    tp2 = curr + (atr * tp2_mult)
                    
                    # حساب الكمية
                    qty = trade_manager.calculate_position_size(
                        curr, sl,
                        BOT_SETTINGS.get('base_capital', 1000),
                        BOT_SETTINGS.get('risk_per_trade_pct', 2.0),
                        BOT_SETTINGS.get('max_position_pct', 0.2)
                    )
                    
                    if qty > 0:
                        trade_manager.open_trade(
                            symbol, curr, sl, tp1, tp2,
                            qty, strategy, regime, paper
                        )
                        time.sleep(1)
                else:
                    # تسجيل عشوائي للفحص
                    if random.random() < 0.05:
                        signals_cache.add_log({
                            't': datetime.now().strftime('%H:%M'),
                            's': symbol,
                            'st': 'فحص',
                            'r': reason
                        })
                
                time.sleep(0.2)
                
        except Exception as e:
            logger.error(f"خطأ في البحث عن الفرص: {e}")
    
    def run(self):
        """تشغيل المحرك"""
        self._init_client()
        self._load_symbols()
        self._running = True
        
        logger.info("🚀 SmartBot V14 Engine Started")
        
        while self._running:
            try:
                enabled = BOT_SETTINGS.get('is_trading_enabled', False)
                paper = BOT_SETTINGS.get('paper_trading_mode', True)
                max_trades = BOT_SETTINGS.get('max_open_trades', 6)
                
                if not enabled:
                    time.sleep(5)
                    continue
                
                # تحليل السوق
                regime = MarketRegimeAnalyzer.analyze(self._client)
                
                # إدارة الصفقات
                self._manage_open_trades(regime, paper)
                
                # البحث عن فرص
                self._scan_for_opportunities(regime, paper, max_trades)
                
                time.sleep(15)
                
            except Exception as e:
                logger.error(f"خطأ في المحرك: {e}")
                time.sleep(10)
    
    def stop(self):
        """إيقاف المحرك"""
        self._running = False


engine = TradingEngine()


# --- 12. واجهة Flask المحسنة ---
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/analytics')
def analytics():
    """API للتحليلات"""
    try:
        market = system_state.to_dict()
        
        # تحويل التاريخ لنص
        if market.get('last_update'):
            market['last_update'] = market['last_update'].isoformat()
        
        signals = []
        for symbol, trade in signals_cache.items():
            trade_copy = {k: v for k, v in trade.items() if k != 'entry_time'}
            signals.append(trade_copy)
        
        prices = {}
        for symbol, _ in signals_cache.items():
            price = prices_cache.get(symbol)
            if price:
                prices[symbol] = price
        
        logs = signals_cache.get_logs()
        stats = db.get_trade_statistics()
        settings = BOT_SETTINGS.to_dict()
        
        return jsonify({
            "market": market,
            "signals": signals,
            "prices": prices,
            "stats": stats,
            "logs": logs,
            "settings": settings
        })
        
    except Exception as e:
        logger.error(f"خطأ في API analytics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/toggle', methods=['POST'])
def toggle():
    """تبديل حالة التداول"""
    new_state = BOT_SETTINGS.toggle_trading()
    logger.info(f"⚡ حالة التداول: {'مفعّل' if new_state else 'متوقف'}")
    return jsonify({"is_trading_enabled": new_state})


@app.route('/api/settings', methods=['GET', 'POST'])
def settings_api():
    """API للإعدادات"""
    if request.method == 'GET':
        return jsonify(BOT_SETTINGS.to_dict())
    
    elif request.method == 'POST':
        data = request.get_json()
        if data:
            for key, value in data.items():
                BOT_SETTINGS.set(key, value)
        return jsonify(BOT_SETTINGS.to_dict())


@app.route('/api/health')
def health():
    """فحص صحة النظام"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "open_trades": len(signals_cache),
        "trading_enabled": BOT_SETTINGS.get('is_trading_enabled')
    })


# --- واجهة HTML المحسنة ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartBot V14 - لوحة التحكم</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root { 
            --bg: #0b0e11; 
            --panel: #151a1e; 
            --border: #2b3139; 
            --text: #eaecef; 
            --green: #0ecb81; 
            --red: #f6465d; 
            --accent: #f0b90b; 
            --blue: #3b82f6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            background: var(--bg); 
            color: var(--text); 
            font-family: 'Tajawal', sans-serif; 
            padding: 20px; 
            font-size: 14px; 
            line-height: 1.6;
        }
        .header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 25px; 
            padding-bottom: 20px; 
            border-bottom: 1px solid var(--border); 
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(12, 1fr); 
            gap: 20px; 
            margin-bottom: 20px; 
        }
        .col-3 { grid-column: span 3; } 
        .col-4 { grid-column: span 4; } 
        .col-6 { grid-column: span 6; } 
        .col-8 { grid-column: span 8; } 
        .col-12 { grid-column: span 12; }
        .card { 
            background: var(--panel); 
            border: 1px solid var(--border); 
            border-radius: 12px; 
            padding: 20px; 
            transition: border-color 0.3s ease;
        }
        .card:hover { border-color: var(--accent); }
        .card h3 { 
            margin-bottom: 15px; 
            color: #848e9c; 
            font-size: 11px; 
            text-transform: uppercase; 
            letter-spacing: 1.5px; 
            font-weight: 700;
        }
        .big-num { 
            font-size: 32px; 
            font-weight: 900; 
            color: var(--text); 
            line-height: 1.2;
        }
        .sub-text { 
            color: #848e9c; 
            font-size: 12px; 
            margin-top: 8px;
        }
        .status-dot { 
            height: 10px; 
            width: 10px; 
            border-radius: 50%; 
            display: inline-block; 
            margin-left: 8px;
            background-color: #555;
        }
        .dot-green { 
            background-color: var(--green); 
            box-shadow: 0 0 10px var(--green), 0 0 20px var(--green); 
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .btn { 
            background: var(--accent); 
            color: #000; 
            border: none; 
            padding: 10px 24px; 
            border-radius: 8px; 
            font-weight: bold; 
            cursor: pointer; 
            transition: all 0.2s ease; 
            font-family: 'Tajawal', sans-serif;
            font-size: 14px;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(240, 185, 11, 0.3); }
        .btn:active { transform: translateY(0); }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: right; padding: 14px 12px; border-bottom: 1px solid var(--border); }
        th { 
            color: #848e9c; 
            font-size: 11px; 
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
        }
        .pnl-g { color: var(--green); font-weight: bold; }
        .pnl-r { color: var(--red); font-weight: bold; }
        .badge {
            background: var(--border);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
        
        @media(max-width: 1024px) { 
            .col-3, .col-4 { grid-column: span 6; } 
        }
        @media(max-width: 768px) { 
            .col-3, .col-4, .col-6, .col-8 { grid-column: span 12; }
            .header { flex-direction: column; gap: 15px; text-align: center; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1 style="font-size:28px; font-weight:900">
                SmartBot <span style="color:var(--accent)">V14</span>
                <span style="font-size:14px; color:#848e9c; font-weight:400">PRO</span>
            </h1>
            <span style="font-size:13px; color:#848e9c">نظام إدارة المحفظة الذكي - النسخة المحسنة</span>
        </div>
        <div style="display:flex; gap:20px; align-items:center">
            <div style="text-align:left">
                <span id="connectionStatus" class="status-dot"></span>
                <span style="font-size:12px; color:#848e9c">متصل</span>
            </div>
            <button id="powerBtn" class="btn" onclick="toggleBot()">جاري التحميل...</button>
        </div>
    </div>

    <!-- مؤشرات الأداء -->
    <div class="grid">
        <div class="card col-3">
            <h3>🌍 حالة السوق</h3>
            <div id="regime" class="big-num" style="color:var(--accent); font-size:18px">--</div>
            <div class="sub-text">قوة الاتجاه: <span id="trendStr" style="color:var(--text)">0</span></div>
        </div>
        <div class="card col-3">
            <h3>📊 نسبة النجاح</h3>
            <div class="big-num"><span id="winRate">0</span><small style="font-size:18px">%</small></div>
            <div class="sub-text">إجمالي الصفقات: <span id="tradeCount" style="color:var(--text)">0</span></div>
        </div>
        <div class="card col-3">
            <h3>💰 صافي الأرباح</h3>
            <div id="totalPnl" class="big-num">$0.00</div>
            <div class="sub-text">عامل الربح: <span id="profFact" style="color:var(--text)">0</span></div>
        </div>
        <div class="card col-3">
            <h3>⚡ المخاطرة الحالية</h3>
            <div class="big-num"><span id="openRisk">0</span><small style="font-size:18px">%</small></div>
            <div class="sub-text">صفقات مفتوحة: <span id="activeCount" style="color:var(--text)">0</span></div>
        </div>
    </div>

    <!-- الرسوم البيانية -->
    <div class="grid">
        <div class="card col-8">
            <h3>📈 نمو المحفظة (Equity Curve)</h3>
            <div style="height: 280px;"><canvas id="equityChart"></canvas></div>
        </div>
        <div class="card col-4">
            <h3>🎯 توزيع الصفقات</h3>
            <div style="height: 280px; position:relative">
                <canvas id="statsChart"></canvas>
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); text-align:center">
                    <span style="font-size:24px; font-weight:900" id="winRateCenter">0%</span><br>
                    <span style="font-size:11px; color:#848e9c">معدل الفوز</span>
                </div>
            </div>
        </div>
    </div>

    <!-- الجداول -->
    <div class="grid">
        <div class="card col-8">
            <h3>📋 المحفظة النشطة</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>العملة</th>
                            <th>الاستراتيجية</th>
                            <th>الدخول</th>
                            <th>السعر الحالي</th>
                            <th>الربح/الخسارة</th>
                            <th>الأهداف</th>
                        </tr>
                    </thead>
                    <tbody id="tradesBody"></tbody>
                </table>
            </div>
        </div>
        <div class="card col-4">
            <h3>📝 سجل النظام</h3>
            <div style="height: 320px; overflow-y: auto;">
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
            "Bull_Trend_Strong": "🐂 اتجاه صاعد قوي",
            "Bull_Accumulation": "📈 تجميع صاعد",
            "Bear_Trend_Strong": "🐻 اتجاه هابط قوي",
            "High_Volatility_Choppy": "⚡ تذبذب عالي",
            "Ranging": "🦀 عرضي مستقر",
            "Neutral": "⚖️ محايد"
        };

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
            gradient.addColorStop(0, 'rgba(240, 185, 11, 0.3)');
            gradient.addColorStop(1, 'rgba(240, 185, 11, 0)');

            equityChart = new Chart(ctx1, {
                type: 'line',
                data: { 
                    labels: [], 
                    datasets: [{ 
                        label: 'النمو %', 
                        data: [], 
                        borderColor: '#f0b90b', 
                        backgroundColor: gradient, 
                        borderWidth: 2, 
                        fill: true, 
                        tension: 0.4, 
                        pointRadius: 0,
                        pointHoverRadius: 6
                    }] 
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: false, 
                    interaction: { mode: 'index', intersect: false },
                    plugins: { 
                        legend: { display: false },
                        tooltip: { 
                            backgroundColor: '#1a1d21',
                            borderColor: '#f0b90b',
                            borderWidth: 1
                        }
                    }, 
                    scales: { 
                        x: { display: false }, 
                        y: { grid: { borderDash: [5, 5], color: '#2b3139' } } 
                    } 
                }
            });

            const ctx2 = document.getElementById('statsChart').getContext('2d');
            statsChart = new Chart(ctx2, {
                type: 'doughnut',
                data: { 
                    labels: ['ربح', 'خسارة'], 
                    datasets: [{ 
                        data: [50, 50], 
                        backgroundColor: ['#0ecb81', '#f6465d'], 
                        borderWidth: 0,
                        hoverOffset: 8
                    }] 
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: false, 
                    cutout: '75%', 
                    plugins: { 
                        legend: { 
                            position: 'bottom',
                            labels: { padding: 15, usePointStyle: true }
                        } 
                    } 
                }
            });
        }

        async function updateData() {
            try {
                const res = await fetch('/api/analytics');
                if (!res.ok) throw new Error('Network error');
                const d = await res.json();

                // تحديث الأزرار
                const btn = document.getElementById('powerBtn');
                document.getElementById('connectionStatus').className = "status-dot dot-green";
                
                if(d.settings.is_trading_enabled) {
                    btn.innerText = "⏹️ إيقاف البوت";
                    btn.style.background = "var(--red)";
                    btn.style.color = "#fff";
                } else {
                    btn.innerText = "▶️ تشغيل البوت";
                    btn.style.background = "var(--green)";
                    btn.style.color = "#fff";
                }

                // المؤشرات
                const regKey = d.market.market_regime;
                document.getElementById('regime').innerText = regimeMap[regKey] || regKey;
                document.getElementById('trendStr').innerText = d.market.trend_strength;
                
                document.getElementById('winRate').innerText = d.stats.win_rate.toFixed(1);
                document.getElementById('winRateCenter').innerText = d.stats.win_rate.toFixed(1) + "%";
                document.getElementById('tradeCount').innerText = d.stats.trade_count;
                
                const pnl = d.stats.total_pnl_usd;
                const pnlEl = document.getElementById('totalPnl');
                pnlEl.innerText = (pnl >= 0 ? "+" : "") + "$" + pnl.toFixed(2);
                pnlEl.style.color = pnl >= 0 ? "var(--green)" : "var(--red)";
                document.getElementById('profFact').innerText = d.stats.profit_factor.toFixed(2);

                document.getElementById('activeCount').innerText = d.signals.length;
                const riskPct = d.signals.length * (d.settings.risk_per_trade_pct || 2);
                document.getElementById('openRisk').innerText = riskPct.toFixed(1);

                // تحديث الشارت
                if(d.stats.history && d.stats.history.length > 0) {
                    equityChart.data.labels = d.stats.history.map(h => h.t);
                    equityChart.data.datasets[0].data = d.stats.history.map(h => h.v);
                    equityChart.update('none');
                    
                    const winRate = d.stats.win_rate;
                    statsChart.data.datasets[0].data = [winRate, 100 - winRate];
                    statsChart.update('none');
                }

                // جدول الصفقات
                const tb = document.getElementById('tradesBody');
                if (d.signals.length === 0) {
                    tb.innerHTML = `<tr><td colspan='6' style='text-align:center; padding:40px; color:#666'>
                        <div style="font-size:40px; margin-bottom:10px">📭</div>
                        لا توجد صفقات نشطة حالياً
                    </td></tr>`;
                } else {
                    tb.innerHTML = d.signals.map(s => {
                        const curr = d.prices[s.symbol] || s.entry_price;
                        const pnl = ((curr - s.entry_price) / s.entry_price) * 100;
                        const stratName = stratMap[s.strategy] || s.strategy;
                        return `
                        <tr>
                            <td style="font-weight:bold; color:var(--accent)">${s.symbol}</td>
                            <td><span class="badge">${stratName}</span></td>
                            <td>${Number(s.entry_price).toFixed(6)}</td>
                            <td>${Number(curr).toFixed(6)}</td>
                            <td class="${pnl>=0?'pnl-g':'pnl-r'}">${pnl>=0?'+':''}${pnl.toFixed(2)}%</td>
                            <td style="font-size:11px; color:#848e9c">${Number(s.tp1).toFixed(4)} ➔ ${Number(s.tp2).toFixed(4)}</td>
                        </tr>`;
                    }).join('');
                }

                // السجل
                const logsBody = document.getElementById('logsBody');
                if (d.logs && d.logs.length > 0) {
                    logsBody.innerHTML = d.logs.map(l => `
                        <tr>
                            <td style="color:#555; width:50px">${l.t}</td>
                            <td style="font-weight:bold; color:var(--accent)">${l.s}</td>
                            <td style="color:${l.st==='دخول'?'var(--green)':'#848e9c'}">${l.st}</td>
                            <td style="color:#848e9c">${l.r}</td>
                        </tr>
                    `).join('');
                } else {
                    logsBody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:#555; padding:20px">لا توجد سجلات</td></tr>`;
                }

            } catch(e) { 
                console.error('Update error:', e);
                document.getElementById('connectionStatus').className = "status-dot";
            }
        }

        async function toggleBot() { 
            try {
                await fetch('/api/toggle', {method:'POST'});
                await updateData();
            } catch(e) {
                console.error('Toggle error:', e);
            }
        }

        // التهيئة
        initCharts();
        updateData();
        setInterval(updateData, 2500);
    </script>
</body>
</html>
"""


# --- نقطة الدخول ---
if __name__ == "__main__":
    try:
        db.init_tables()
        Thread(target=engine.run, daemon=True).start()
        logger.info("🖥️ لوحة التحكم تعمل على المنفذ 5000")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        logger.info("👋 إيقاف البوت...")
        engine.stop()
