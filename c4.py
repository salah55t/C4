import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, request, Response
from flask_cors import CORS
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7 # We don't need 90 days of data for live prediction
MIN_VOLUME_24H_USDT: float = 10_000_000 # Filter for more liquid coins

# <-- اسم النموذج يجب أن يطابق الاسم المستخدم في التدريب
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V2'
MODEL_PREDICTION_THRESHOLD = 0.65 # Set a higher probability threshold for creating a signal

# Indicator Parameters
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BBANDS_PERIOD: int = 20
BBANDS_STD_DEV: float = 2.0
ATR_PERIOD: int = 14

# Global State
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
ticker_data: Dict[str, Dict[str, float]] = {} # Store price and 24h volume
ml_models_cache: Dict[str, Any] = {}

# ---------------------- Binance Client & DB Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    # DB initialization logic (same as ml.py, can be refactored into a common lib)
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                    target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL, 
                    status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                    profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB);
                CREATE TABLE IF NOT EXISTS ml_models (id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB);
            """)
            conn.commit()
            logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: exit(1)

def check_db_connection() -> bool:
    global conn
    try:
        if conn is None or conn.closed != 0: init_db()
        else: conn.cursor().execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError):
        try: init_db()
        except Exception: return False
        return True
    return False

# ---------------------- Indicator Calculation (must match ml.py) ----------------------
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Tuple[pd.Series, pd.Series]:
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(df: pd.DataFrame, period: int = BBANDS_PERIOD, std_dev: float = BBANDS_STD_DEV) -> Tuple[pd.Series, pd.Series]:
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    df['candle_body_size'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    return df

# ---------------------- Core Bot Logic ----------------------

def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        # logger.info(f"✅ [ML Model] تم تحميل حزمة نموذج '{model_name}' من الذاكرة المؤقتة.")
        return ml_models_cache[model_name]
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                # <-- Validate bundle contents
                if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                    ml_models_cache[model_name] = model_bundle
                    logger.info(f"✅ [ML Model] تم تحميل حزمة نموذج '{model_name}' من قاعدة البيانات.")
                    return model_bundle
                else:
                    logger.error(f"❌ [ML Model] حزمة النموذج '{model_name}' غير صالحة (مكونات مفقودة).")
                    return None
            logger.warning(f"⚠️ [ML Model] لم يتم العثور على نموذج لـ {symbol} بالاسم '{model_name}'.")
            return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ أثناء تحميل حزمة نموذج ML لـ {symbol}: {e}", exc_info=True)
        return None

def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    global ticker_data
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        for item in data:
            symbol = item.get('s')
            if symbol and 'USDT' in symbol:
                if symbol not in ticker_data: ticker_data[symbol] = {}
                ticker_data[symbol]['price'] = float(item.get('c', 0))
                ticker_data[symbol]['volume_24h_usdt'] = float(item.get('q', 0))
    except Exception as e: logger.error(f"❌ [WS] خطأ في معالجة رسالة المؤشر: {e}")

def run_websocket_manager() -> None:
    logger.info("ℹ️ [WS] بدء مدير WebSocket...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_ticker_socket(callback=handle_ticker_message) # Use full ticker for 24h volume
    twm.join()

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if model_bundle:
            self.ml_model = model_bundle.get('model')
            self.scaler = model_bundle.get('scaler')
            self.feature_names = model_bundle.get('feature_names') # <-- The crucial list
        else:
            self.ml_model = None
            self.scaler = None
            self.feature_names = None

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        df_calc = df.copy()
        df_calc['atr'] = calculate_atr(df_calc, ATR_PERIOD)
        df_calc['rsi'] = calculate_rsi(df_calc, RSI_PERIOD)
        df_calc['macd'], df_calc['macd_signal'] = calculate_macd(df_calc)
        df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
        df_calc['bb_upper'], df_calc['bb_lower'] = calculate_bollinger_bands(df_calc)
        df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['close']
        df_calc = calculate_candlestick_features(df_calc)
        df_calc['relative_volume'] = df_calc['volume'] / df_calc['volume'].rolling(window=30, min_periods=1).mean()
        return df_calc

    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        symbol_log_prefix = f"🔍 [Signal Gen {self.symbol}]"
        if df_processed is None or df_processed.empty: return None
        if not all([self.ml_model, self.scaler, self.feature_names]):
            # logger.info(f"{symbol_log_prefix} رفض: نموذج ML أو مكوناته غير محملة.")
            return None
        
        last_row = df_processed.iloc[-1]
        current_price = ticker_data.get(self.symbol, {}).get('price')
        if current_price is None: return None

        try:
            # <-- This is the fix for the UserWarning
            features_df = pd.DataFrame([last_row], columns=df_processed.columns)
            features_for_model = features_df[self.feature_names]

            if features_for_model.isnull().values.any():
                logger.warning(f"{symbol_log_prefix} رفض: قيم فارغة في الميزات قبل التنبؤ.")
                return None

            features_scaled = self.scaler.transform(features_for_model)
            prediction_proba = self.ml_model.predict_proba(features_scaled)[0][1]

            if prediction_proba < MODEL_PREDICTION_THRESHOLD:
                # logger.info(f"{symbol_log_prefix} رفض: احتمالية التنبؤ ({prediction_proba:.2f}) أقل من الحد الأدنى ({MODEL_PREDICTION_THRESHOLD}).")
                return None
            
        except Exception as e:
            logger.error(f"❌ {symbol_log_prefix} خطأ أثناء التنبؤ: {e}", exc_info=True)
            return None

        # If prediction is confident, define trade parameters
        atr_val = last_row.get('atr', 0)
        if atr_val == 0: return None
        
        target_price = current_price * 1.015 # 1.5% profit
        stop_loss = current_price * 0.99   # 1.0% stop-loss

        if stop_loss >= current_price or target_price <= current_price: return None

        return {
            'symbol': self.symbol, 
            'entry_price': current_price, 
            'target_price': target_price, 
            'stop_loss': stop_loss,
            'strategy_name': BASE_ML_MODEL_NAME, 
            'signal_details': {'ML_Probability': f"{prediction_proba:.2%}"}
        }

def send_telegram_alert(signal_data: Dict[str, Any]) -> None:
    symbol = signal_data['symbol'].replace('_', '\\_')
    entry = signal_data['entry_price']
    target = signal_data['target_price']
    sl = signal_data['stop_loss']
    profit_pct = ((target / entry) - 1) * 100
    
    message = (f"💡 *إشارة تداول جديدة ({BASE_ML_MODEL_NAME})* 💡\n--------------------\n"
               f"🪙 **الزوج:** `{symbol}`\n"
               f"📈 **النوع:** شراء\n"
               f"➡️ **الدخول:** `${entry:,.8g}`\n"
               f"🎯 **الهدف:** `${target:,.8g}` ({profit_pct:+.2f}%)\n"
               f"🛑 **وقف الخسارة:** `${sl:,.8g}`\n"
               f"🔍 **الثقة:** {signal_data['signal_details']['ML_Probability']}\n"
               f"--------------------")
    
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(CHAT_ID), 'text': message, 'parse_mode': 'Markdown', 'reply_markup': json.dumps(reply_markup)}
    try:
        requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    if not check_db_connection() or not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details) 
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (signal['symbol'], signal['entry_price'], signal['target_price'], signal['stop_loss'], 
                  signal.get('strategy_name'), json.dumps(signal.get('signal_details', {}))))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ أثناء إدراج الإشارة: {e}")
        if conn: conn.rollback()
        return False

def track_signals() -> None:
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات...")
    while True:
        try:
            if not check_db_connection() or not conn:
                time.sleep(15)
                continue
                
            with conn.cursor() as track_cur:
                track_cur.execute("SELECT id, symbol, entry_price, target_price, stop_loss FROM signals WHERE status = 'open';")
                open_signals = track_cur.fetchall()

            for signal in open_signals:
                price_info = ticker_data.get(signal['symbol'])
                if not price_info or 'price' not in price_info: continue
                
                price = price_info['price']
                status, closing_price = None, None

                if price >= signal['target_price']:
                    status, closing_price = 'target_hit', signal['target_price']
                elif price <= signal['stop_loss']:
                    status, closing_price = 'stop_loss_hit', signal['stop_loss']
                
                if status:
                    profit_pct = ((closing_price / signal['entry_price']) - 1) * 100
                    with conn.cursor() as update_cur:
                        update_cur.execute("""
                            UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s 
                            WHERE id = %s;
                        """, (status, closing_price, profit_pct, signal['id']))
                    conn.commit()
                    
                    alert_msg = (f"{'✅' if status == 'target_hit' else '🛑'} *{'Target Hit' if status == 'target_hit' else 'Stop Loss Hit'}*\n"
                                 f"`{signal['symbol'].replace('_', '\\_')}` | Profit: {profit_pct:+.2f}%")
                    send_telegram_alert({'symbol': alert_msg, 'entry_price': 0, 'target_price': 0, 'stop_loss': 0, 'signal_details': {'ML_Probability':''}})

            time.sleep(3) # Check prices every 3 seconds
        except Exception as e:
            logger.error(f"❌ [Tracker] خطأ في دورة التتبع: {e}")
            if conn: conn.rollback()
            time.sleep(30)

def main_loop():
    logger.info("✅ [Main] بدء دورة المسح الرئيسية.")
    # Wait for initial data to populate
    time.sleep(10) 
    
    while True:
        try:
            # Filter symbols by 24h volume
            liquid_symbols = [s for s, d in ticker_data.items() if d.get('volume_24h_usdt', 0) > MIN_VOLUME_24H_USDT]
            if not liquid_symbols:
                logger.warning("⚠️ [Main] لم يتم العثور على رموز ذات سيولة كافية. الانتظار...")
                time.sleep(60)
                continue

            if not check_db_connection() or not conn:
                time.sleep(60)
                continue
            
            with conn.cursor() as cur_check:
                cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE status = 'open';")
                open_count = cur_check.fetchone().get('count', 0)

            if open_count >= MAX_OPEN_TRADES:
                # logger.info(f"⚠️ [Main] الحد الأقصى للصفقات المفتوحة ({open_count}). الانتظار...")
                time.sleep(60)
                continue
            
            slots_available = MAX_OPEN_TRADES - open_count
            
            for symbol in liquid_symbols:
                if slots_available <= 0: break
                
                with conn.cursor() as symbol_cur:
                    symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND status = 'open' LIMIT 1;", (symbol,))
                    if symbol_cur.fetchone(): continue # Skip if already in an open trade

                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df_hist is None or df_hist.empty: continue
                
                strategy = TradingStrategy(symbol)
                if not strategy.ml_model: continue # Skip if model not loaded

                df_indicators = strategy.populate_indicators(df_hist)
                if df_indicators is None: continue
                
                potential_signal = strategy.generate_signal(df_indicators)
                if potential_signal:
                    logger.info(f"💰 [Main] تم العثور على إشارة صالحة لـ {symbol}. محاولة الحفظ...")
                    if insert_signal_into_db(potential_signal):
                        send_telegram_alert(potential_signal)
                        slots_available -= 1
            
            wait_time = 60 # Check every minute
            time.sleep(wait_time)

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع: {main_err}", exc_info=True)
            time.sleep(120)

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    app = Flask(__name__)
    CORS(app)
    @app.route('/')
    def home(): return "Trading Bot is running"
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port)

if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات تداول العملات الرقمية (V2)...")
    try:
        init_db()
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=track_signals, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        run_flask()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 [Main] طلب إيقاف...")
    finally:
        if conn: conn.close()
        logger.info("👋 [Main] تم إيقاف البوت.")
        os._exit(0)
