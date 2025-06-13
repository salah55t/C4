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
from threading import Thread, Lock
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO, # تم ضبط المستوى على INFO لإظهار كل الرسائل المطلوبة
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
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7
MIN_VOLUME_24H_USDT: float = 10_000_000

BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V3'
MODEL_PREDICTION_THRESHOLD = 0.65

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
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {} 
signal_cache_lock = Lock()

# ---------------------- Binance Client & DB Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    logger.info("✅ [Binance] تم الاتصال بمنصة Binance بنجاح.")
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

def init_db(retries: int = 5, delay: int = 5) -> None:
    # ... (code with existing logging is fine)
    global conn
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
    # ... (code with existing logging is fine)
    global conn
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] Connection lost. Re-initializing...")
            init_db()
        else:
            conn.cursor().execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError):
        try:
            logger.warning("⚠️ [DB] Connection failed check. Re-initializing...")
            init_db()
        except Exception:
            return False
        return True
    return False

# ---------------------- Symbol Validation ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    # ... (code with existing logging is fine)
    logger.info(f"ℹ️ [Validation] Reading symbols from '{filename}' and validating with Binance...")
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
            raw_symbols_from_file = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted_symbols = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols_from_file}
        exchange_info = client.get_exchange_info()
        active_binance_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        validated_symbols = sorted(list(formatted_symbols.intersection(active_binance_symbols)))
        logger.info(f"✅ [Validation] Bot will scan {len(validated_symbols)} validated symbols.")
        return validated_symbols
    except Exception as e:
        logger.error(f"❌ [Validation] An error occurred during symbol validation: {e}")
        return []

# --- Data Fetching and Indicator Calculation ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    # ... (code with existing logging is fine)
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

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (code is purely computational, no logging needed)
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd'] = ema_fast - ema_slow
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_upper'] = sma + (std * BBANDS_STD_DEV)
    df_calc['bb_lower'] = sma - (std * BBANDS_STD_DEV)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / sma
    df_calc['bb_pos'] = (df_calc['close'] - sma) / std.replace(0, np.nan)
    df_calc['candle_body_size'] = (df_calc['close'] - df_calc['open']).abs()
    df_calc['upper_wick'] = df_calc['high'] - df_calc[['open', 'close']].max(axis=1)
    df_calc['lower_wick'] = df_calc[['open', 'close']].min(axis=1) - df_calc['low']
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    return df_calc

# --- Model Loading and WebSocket ---
def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    # ... (code with existing logging is fine)
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache: return ml_models_cache[model_name]
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                    ml_models_cache[model_name] = model_bundle
                    logger.info(f"✅ [ML Model] Successfully loaded '{model_name}' from database.")
                    return model_bundle
            logger.warning(f"⚠️ [ML Model] Model '{model_name}' not found in the database for symbol {symbol}.")
            return None
    except Exception as e:
        logger.error(f"❌ [ML Model] Error loading ML model bundle for {symbol}: {e}", exc_info=True)
        return None

def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    # ... (code unchanged, logging here would be too verbose)
    global open_signals_cache
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        
        for item in data:
            symbol = item.get('s')
            if not symbol: continue

            with signal_cache_lock:
                if symbol in open_signals_cache:
                    price = float(item.get('c', 0))
                    if price == 0: continue
                    
                    signal = open_signals_cache[symbol]
                    status, closing_price = None, None

                    if price >= signal['target_price']:
                        status, closing_price = 'target_hit', signal['target_price']
                    elif price <= signal['stop_loss']:
                        status, closing_price = 'stop_loss_hit', signal['stop_loss']

                    if status:
                        # سجل مهم: تم العثور على حدث إغلاق صفقة في الوقت الفعلي
                        logger.info(f"⚡ [Real-time Track] Event '{status}' triggered for {symbol} at price {price:.8f}")
                        del open_signals_cache[symbol]
                        Thread(target=close_signal_in_db, args=(signal, status, closing_price)).start()

    except Exception as e:
        logger.error(f"❌ [WS Tracker] Error processing real-time ticker message: {e}")

def run_websocket_manager() -> None:
    # ... (code with existing logging is fine)
    logger.info("ℹ️ [WS] Starting WebSocket manager for real-time tracking...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_ticker_socket(callback=handle_ticker_message)
    logger.info("✅ [WS] WebSocket connected and listening for ticker updates.")
    twm.join()
    
# --- Trading Strategy and Signal Generation ---
class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_db(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        return calculate_features(df)

    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]):
            # سجل مهم: سبب رفض الإشارة
            logger.info(f"ℹ️ [Signal Reject] {self.symbol}: Signal rejected. Reason: ML model or scaler not loaded.")
            return None
        
        last_row = df_processed.iloc[-1]
        
        try:
            features_df = pd.DataFrame([last_row], columns=df_processed.columns)[self.feature_names]
            if features_df.isnull().values.any():
                # سجل مهم: سبب رفض الإشارة
                logger.info(f"ℹ️ [Signal Reject] {self.symbol}: Signal rejected. Reason: Null values found in feature data.")
                return None
            
            features_scaled = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_names)
            
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0][1]
            
            if prediction_proba < MODEL_PREDICTION_THRESHOLD:
                # سجل مهم: سبب رفض الإشارة
                logger.info(f"ℹ️ [Signal Reject] {self.symbol}: Signal rejected. Reason: Probability {prediction_proba:.2%} is below threshold {MODEL_PREDICTION_THRESHOLD:.2%}.")
                return None
            
            # سجل مهم: تم العثور على إشارة محتملة
            logger.info(f"✅ [Signal Found] {self.symbol}: Potential signal found with probability {prediction_proba:.2%}.")
            return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability': f"{prediction_proba:.2%}"}}

        except Exception as e:
            logger.warning(f"⚠️ [Signal Gen] {self.symbol}: Could not generate signal due to an error: {e}")
            return None

# --- Telegram and Database Functions ---
def send_telegram_message(target_chat_id: str, text: str):
    # ... (code with existing logging is fine)
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        logger.info(f"✉️ [Telegram] Successfully sent a message.")
    except Exception as e:
        logger.error(f"❌ [Telegram] Failed to send generic message: {e}")

def send_new_signal_alert(signal_data: Dict[str, Any]) -> None:
    # ... (code is fine, main logging is in `send_telegram_message`)
    safe_symbol = signal_data['symbol'].replace('_', '\\_')
    entry, target, sl = signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss']
    profit_pct = ((target / entry) - 1) * 100
    message = (f"💡 *New Trading Signal ({BASE_ML_MODEL_NAME})* 💡\n--------------------\n"
               f"🪙 **Symbol:** `{safe_symbol}`\n"
               f"📈 **Type:** LONG\n"
               f"➡️ **Entry:** `${entry:,.8g}`\n"
               f"🎯 **Target:** `${target:,.8g}` ({profit_pct:+.2f}%)\n"
               f"🛑 **Stop Loss:** `${sl:,.8g}`\n"
               f"🔍 **Confidence:** {signal_data['signal_details']['ML_Probability']}\n--------------------")
    reply_markup = {"inline_keyboard": [[{"text": "📊 Open Dashboard", "url": WEBHOOK_URL or '#'}]]}
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(CHAT_ID), 'text': message, 'parse_mode': 'Markdown', 'reply_markup': json.dumps(reply_markup)}
    try:
        requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e:
        logger.error(f"❌ [Telegram] Failed to send new signal alert: {e}")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;",
                (signal['symbol'], signal['entry_price'], signal['target_price'], signal['stop_loss'], signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})))
            )
            new_id = cur.fetchone()['id']
            signal['id'] = new_id
        conn.commit()
        
        with signal_cache_lock:
            open_signals_cache[signal['symbol']] = signal
        # سجل مهم: تأكيد إضافة الصفقة للتتبع
        logger.info(f"✅ [DB & Cache] Successfully inserted signal for {signal['symbol']} (ID: {new_id}) and added to real-time tracking cache.")
        return signal

    except Exception as e:
        logger.error(f"❌ [DB Insert] Error inserting signal for {signal['symbol']}: {e}")
        if conn: conn.rollback()
        return None

def close_signal_in_db(signal: Dict, status: str, closing_price: float):
    if not check_db_connection() or not conn: return
    try:
        profit_pct = ((closing_price / signal['entry_price']) - 1) * 100
        with conn.cursor() as update_cur:
            update_cur.execute(
                "UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;",
                (status, closing_price, profit_pct, signal['id'])
            )
        conn.commit()
        # سجل مهم: تأكيد إغلاق الصفقة
        logger.info(f"✅ [DB Close] Successfully closed signal {signal['id']} for {signal['symbol']} with status '{status}'. Profit: {profit_pct:.2f}%")
        
        safe_symbol = signal['symbol'].replace('_', '\\_')
        status_icon = '✅' if status == 'target_hit' else '🛑'
        status_text = 'Target Hit' if status == 'target_hit' else 'Stop Loss Hit'
        alert_msg = f"{status_icon} *{status_text}*\n`{safe_symbol}` | Profit: {profit_pct:+.2f}%"
        send_telegram_message(CHAT_ID, alert_msg)

    except Exception as e:
        logger.error(f"❌ [DB Close] Error closing signal {signal['id']} for {signal['symbol']}: {e}")
        if conn: conn.rollback()
        with signal_cache_lock:
            open_signals_cache[signal['symbol']] = signal
        logger.warning(f"⚠️ [Real-time Track] Re-added {signal['symbol']} to cache after a database closing failure.")

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [Cache Load] Loading previously open signals into tracking cache...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                for signal in open_signals:
                    open_signals_cache[signal['symbol']] = dict(signal)
            # سجل مهم: تأكيد عدد الصفقات التي يتم تتبعها
            logger.info(f"✅ [Cache Load] Loaded {len(open_signals)} open signals. Now tracking {len(open_signals_cache)} signals in real-time.")
    except Exception as e:
        logger.error(f"❌ [Cache Load] Failed to load open signals: {e}")

# --- Main Application Loops ---
def main_loop():
    global validated_symbols_to_scan
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ [Main] No validated symbols to scan. Bot will not proceed."); return
    
    logger.info(f"✅ [Main] Starting main scan loop for {len(validated_symbols_to_scan)} symbols.")
    time.sleep(10)

    while True:
        try:
            with signal_cache_lock:
                open_count = len(open_signals_cache)

            if open_count >= MAX_OPEN_TRADES:
                # سجل مهم: سبب توقف البحث عن إشارات جديدة
                logger.info(f"ℹ️ [Main Pause] Reached max open trades limit ({open_count}/{MAX_OPEN_TRADES}). Pausing new signal generation.")
                time.sleep(60)
                continue

            slots_available = MAX_OPEN_TRADES - open_count
            logger.info(f"ℹ️ [Main Scan] Starting new scan cycle. Open trades: {open_count}, Slots available: {slots_available}")
            
            for symbol in validated_symbols_to_scan:
                if slots_available <= 0: break
                
                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        # سجل مهم: سبب رفض الإشارة
                        logger.info(f"ℹ️ [Signal Reject] {symbol}: Skipped, an open trade already exists for this symbol.")
                        continue
                
                try:
                    latest_ticker = client.get_symbol_ticker(symbol=symbol)
                    current_price = float(latest_ticker['price'])
                except BinanceAPIException as e:
                    logger.warning(f"⚠️ [Price Fetch] {symbol}: Could not get latest price: {e}. Skipping.")
                    continue
                
                df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df_hist is None or df_hist.empty:
                    # سجل مهم: سبب رفض الإشارة
                    logger.info(f"ℹ️ [Signal Reject] {symbol}: Skipped, failed to fetch historical data.")
                    continue
                
                strategy = TradingStrategy(symbol)
                
                df_features = strategy.get_features(df_hist)
                if df_features is None:
                    logger.info(f"ℹ️ [Signal Reject] {symbol}: Skipped, feature calculation resulted in None.")
                    continue

                potential_signal = strategy.generate_signal(df_features)
                if potential_signal:
                    potential_signal['entry_price'] = current_price
                    potential_signal['target_price'] = current_price * 1.015
                    potential_signal['stop_loss'] = current_price * 0.99
                    
                    saved_signal = insert_signal_into_db(potential_signal)
                    if saved_signal:
                        send_new_signal_alert(saved_signal)
                        slots_available -= 1
            
            logger.info("ℹ️ [Main Scan] Scan cycle finished. Waiting for next cycle...")
            time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as main_err:
            logger.error(f"❌ [Main] Unexpected error in main loop: {main_err}", exc_info=True)
            time.sleep(120)

def run_flask():
    # ... (code with existing logging is fine)
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    app = Flask(__name__); CORS(app)
    @app.route('/')
    def home(): return "Trading Bot is running"
    logger.info(f"ℹ️ [Flask] Starting Flask app on {host}:{port}...")
    try: from waitress import serve; serve(app, host=host, port=port, threads=8)
    except ImportError: app.run(host=host, port=port)

if __name__ == "__main__":
    logger.info("🚀 Starting Crypto Trading Signal Bot (V4.1 - Detailed Logging)...")
    try:
        init_db()
        load_open_signals_to_cache()
        
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        
        run_flask()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 [Main] Shutdown requested...")
    finally:
        if conn:
            conn.close()
            logger.info("🔌 [DB] Database connection closed.")
        logger.info("👋 [Main] Bot has been shut down.")
        os._exit(0)
