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
ticker_data: Dict[str, Dict[str, float]] = {}
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []

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

# ---------------------- Symbol Validation ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
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
    """Calculates all technical indicators and features to match the training script."""
    df_calc = df.copy()
    
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df_calc['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd'] = ema_fast - ema_slow
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']

    # Bollinger Bands
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_upper'] = sma + (std * BBANDS_STD_DEV)
    df_calc['bb_lower'] = sma - (std * BBANDS_STD_DEV)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / sma
    df_calc['bb_pos'] = (df_calc['close'] - sma) / std.replace(0, np.nan)

    # Candlestick features
    df_calc['candle_body_size'] = (df_calc['close'] - df_calc['open']).abs()
    df_calc['upper_wick'] = df_calc['high'] - df_calc[['open', 'close']].max(axis=1)
    df_calc['lower_wick'] = df_calc[['open', 'close']].min(axis=1) - df_calc['low']

    # Relative volume
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)

    return df_calc

# --- Model Loading and WebSocket ---
def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
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
            logger.warning(f"⚠️ [ML Model] Model '{model_name}' not found in the database.")
            return None
    except Exception as e:
        logger.error(f"❌ [ML Model] Error loading ML model bundle for {symbol}: {e}", exc_info=True)
        return None

def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    global ticker_data
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        for item in data:
            symbol = item.get('s')
            if symbol and symbol in validated_symbols_to_scan:
                if symbol not in ticker_data: ticker_data[symbol] = {}
                ticker_data[symbol]['price'] = float(item.get('c', 0))
                ticker_data[symbol]['volume_24h_usdt'] = float(item.get('q', 0))
    except Exception as e: logger.error(f"❌ [WS] Error processing ticker message: {e}")

def run_websocket_manager() -> None:
    logger.info("ℹ️ [WS] Starting WebSocket manager...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_ticker_socket(callback=handle_ticker_message)
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
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        last_row = df_processed.iloc[-1]
        current_price = ticker_data.get(self.symbol, {}).get('price')
        if current_price is None: return None

        try:
            features_df = pd.DataFrame([last_row], columns=df_processed.columns)[self.feature_names]
            if features_df.isnull().values.any(): return None
            
            # --- ✅ FIX: Restore feature names after scaling to remove the UserWarning ---
            features_scaled = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_names)
            
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0][1]
            
            if prediction_proba < MODEL_PREDICTION_THRESHOLD: return None
        except Exception as e:
            logger.error(f"❌ [Signal Gen {self.symbol}] Error during prediction: {e}")
            return None

        target_price, stop_loss = current_price * 1.015, current_price * 0.99
        if stop_loss >= current_price or target_price <= current_price: return None
        return {'symbol': self.symbol, 'entry_price': current_price, 'target_price': target_price, 'stop_loss': stop_loss, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability': f"{prediction_proba:.2%}"}}

# --- Telegram and Database Functions ---
def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e:
        logger.error(f"❌ [Telegram] Failed to send generic message: {e}")

def send_new_signal_alert(signal_data: Dict[str, Any]) -> None:
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
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] Failed to send new signal alert: {e}")

def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    if not check_db_connection() or not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details) VALUES (%s, %s, %s, %s, %s, %s);", (signal['symbol'], signal['entry_price'], signal['target_price'], signal['stop_loss'], signal.get('strategy_name'), json.dumps(signal.get('signal_details', {}))))
        conn.commit(); return True
    except Exception as e:
        logger.error(f"❌ [DB Insert] Error inserting signal: {e}"); conn.rollback(); return False

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
            if not check_db_connection() or not conn: time.sleep(60); continue
            with conn.cursor() as cur_check: cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE status = 'open';"); open_count = cur_check.fetchone().get('count', 0)
            if open_count >= MAX_OPEN_TRADES: time.sleep(60); continue
            slots_available = MAX_OPEN_TRADES - open_count
            for symbol in validated_symbols_to_scan:
                if slots_available <= 0: break
                if ticker_data.get(symbol, {}).get('volume_24h_usdt', 0) < MIN_VOLUME_24H_USDT: continue
                with conn.cursor() as symbol_cur: symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND status = 'open' LIMIT 1;", (symbol,));
                if symbol_cur.fetchone(): continue
                df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df_hist is None or df_hist.empty: continue
                strategy = TradingStrategy(symbol)
                if not strategy.ml_model: continue
                
                df_features = strategy.get_features(df_hist)
                if df_features is None: continue

                potential_signal = strategy.generate_signal(df_features)
                if potential_signal:
                    logger.info(f"💰 [Main] Valid signal found for {symbol}. Attempting to save...")
                    if insert_signal_into_db(potential_signal):
                        logger.info(f"✅ [Main] Signal for {symbol} saved to DB and alert sent.")
                        send_new_signal_alert(potential_signal); slots_available -= 1
            time.sleep(60)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err: logger.error(f"❌ [Main] Unexpected error: {main_err}", exc_info=True); time.sleep(120)

def track_signals() -> None:
    logger.info("ℹ️ [Tracker] Starting signal tracking loop...")
    while True:
        try:
            if not check_db_connection() or not conn: time.sleep(15); continue
            with conn.cursor() as track_cur: track_cur.execute("SELECT id, symbol, entry_price, target_price, stop_loss FROM signals WHERE status = 'open';"); open_signals = track_cur.fetchall()
            for signal in open_signals:
                price_info = ticker_data.get(signal['symbol']);
                if not price_info or 'price' not in price_info: continue
                price = price_info['price']; status, closing_price = None, None
                if price >= signal['target_price']: status, closing_price = 'target_hit', signal['target_price']
                elif price <= signal['stop_loss']: status, closing_price = 'stop_loss_hit', signal['stop_loss']
                if status:
                    profit_pct = ((closing_price / signal['entry_price']) - 1) * 100
                    with conn.cursor() as update_cur: update_cur.execute("UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;",(status, closing_price, profit_pct, signal['id']))
                    conn.commit()
                    safe_symbol = signal['symbol'].replace('_', '\\_'); status_icon = '✅' if status == 'target_hit' else '🛑'; status_text = 'Target Hit' if status == 'target_hit' else 'Stop Loss Hit'
                    alert_msg = f"{status_icon} *{status_text}*\n`{safe_symbol}` | Profit: {profit_pct:+.2f}%"
                    send_telegram_message(CHAT_ID, alert_msg)
            time.sleep(3)
        except Exception as e: logger.error(f"❌ [Tracker] Error in tracking loop: {e}"); conn.rollback(); time.sleep(30)

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    app = Flask(__name__); CORS(app)
    @app.route('/')
    def home(): return "Trading Bot is running"
    logger.info(f"ℹ️ [Flask] Starting Flask app on {host}:{port}...")
    try: from waitress import serve; serve(app, host=host, port=port, threads=8)
    except ImportError: app.run(host=host, port=port)

if __name__ == "__main__":
    logger.info("🚀 Starting Crypto Trading Signal Bot (V3 Architecture)...")
    try:
        init_db()
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=track_signals, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        run_flask()
    except (KeyboardInterrupt, SystemExit): logger.info("🛑 [Main] Shutdown requested...")
    finally:
        if conn: conn.close(); logger.info("🔌 [DB] Database connection closed.")
        logger.info("👋 [Main] Bot has been shut down."); os._exit(0)
