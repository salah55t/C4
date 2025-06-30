import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import base64
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
import gc
from github import Github, GithubException, Repository

# --- إعدادات أساسية ---
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v6_with_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV6_With_SR')

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    GITHUB_TOKEN: Optional[str] = config('GITHUB_TOKEN', default=None)
    GITHUB_REPO: str = config('GITHUB_REPO')
    RESULTS_FOLDER: str = 'ml_results'
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# --- ثوابت الاستراتيجية ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6_With_SR'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BTC_CORR_PERIOD: int = 30
MODEL_CONFIDENCE_THRESHOLD = 0.70
MAX_OPEN_TRADES: int = 5
TRADE_AMOUNT_USDT: float = 10.0
USE_DYNAMIC_SL_TP = True
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
USE_BTC_TREND_FILTER = True
BTC_SYMBOL = 'BTCUSDT'
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 50

# --- متغيرات عامة ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
github_repo_obj: Optional[Repository] = None 
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
current_prices: Dict[str, float] = {}
prices_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()

def init_github_repo():
    """Initialize and return the GitHub repository object."""
    global github_repo_obj
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning("⚠️ [GitHub] GitHub token or repo not configured.")
        return
    try:
        g = Github(GITHUB_TOKEN)
        github_repo_obj = g.get_repo(GITHUB_REPO)
        logger.info(f"✅ [GitHub] Successfully connected to repository: {GITHUB_REPO}")
    except Exception as e:
        logger.error(f"❌ [GitHub] Failed to connect to GitHub repository: {e}")
        github_repo_obj = None

def load_ml_model_from_github(symbol: str) -> Optional[Dict[str, Any]]:
    """Loads a machine learning model for a given symbol from GitHub."""
    global ml_models_cache, github_repo_obj
    model_key = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_key in ml_models_cache:
        return ml_models_cache[model_key]
    if github_repo_obj is None:
        logger.warning("GitHub repository object not initialized.")
        return None

    model_filename = f"{RESULTS_FOLDER}/{symbol}_latest_model.pkl"
    logger.info(f"ℹ️ [GitHub Load] Attempting to load model for {symbol} from path: {model_filename}")
    try:
        file_content_object = github_repo_obj.get_contents(model_filename, ref="main")
        
        # ---
        # ** FIX START: The file content from GitHub is a base64 string. It must be decoded back into bytes before unpickling. **
        # ---
        base64_content = file_content_object.content
        if not base64_content:
            logger.warning(f"⚠️ [GitHub Load] Model file content for {symbol} is empty. It will be skipped.")
            return None
        
        model_bytes = base64.b64decode(base64_content)
        # ---
        # ** FIX END **
        # ---

        if not model_bytes:
            logger.warning(f"⚠️ [GitHub Load] After base64 decoding, model bytes for {symbol} are empty. Skipping.")
            return None

        # Unpickle the model from the decoded bytes
        model_bundle = pickle.loads(model_bytes)
        
        if 'model' in model_bundle and 'scaler' in model_bundle:
            ml_models_cache[model_key] = model_bundle
            logger.info(f"✅ [GitHub Load] Successfully loaded and cached model for {symbol}.")
            return model_bundle
        else:
            logger.warning(f"⚠️ [GitHub Load] Model bundle for {symbol} is invalid (missing 'model' or 'scaler').")
            return None
    except GithubException as e:
        if e.status == 404:
            logger.warning(f"⚠️ [GitHub Load] Model file not found for {symbol} at path '{model_filename}'.")
        else:
            logger.error(f"❌ [GitHub Load] A GitHub error occurred for {symbol}: {e}")
        return None
    except (pickle.UnpicklingError, TypeError, Exception) as e:
        logger.error(f"❌ [GitHub Load] Failed to decode or unpickle model for {symbol}. The file may be corrupt or in the wrong format. Error: {e}", exc_info=True)
        return None

def init_db(retries: int = 5, delay: int = 5):
    """Initializes the database connection with retry logic."""
    global conn
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False # Use transactions
            with conn.cursor() as cur:
                cur.execute("CREATE TABLE IF NOT EXISTS signals (id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price REAL, target_price REAL, stop_loss REAL, status TEXT, closing_price REAL, closed_at TIMESTAMP, profit_percentage REAL, strategy_name TEXT, signal_details JSONB, created_at TIMESTAMPTZ DEFAULT NOW());")
                cur.execute("CREATE TABLE IF NOT EXISTS notifications (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ DEFAULT NOW(), type TEXT, message TEXT, is_read BOOLEAN DEFAULT FALSE);")
                cur.execute("CREATE TABLE IF NOT EXISTS support_resistance_levels (id SERIAL PRIMARY KEY, symbol TEXT, level_price REAL, level_type TEXT, timeframe TEXT, strength NUMERIC, score NUMERIC, last_tested_at TIMESTAMPTZ, details TEXT, created_at TIMESTAMPTZ DEFAULT NOW(), CONSTRAINT uq_level UNIQUE (symbol, level_price, timeframe, level_type));")
            conn.commit()
            logger.info("✅ [DB] Database initialized successfully.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Connection attempt {attempt + 1} failed: {e}")
            if conn: conn.rollback()
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.critical("❌ [DB] All database connection attempts failed. Exiting.")
                exit(1)

def check_db_connection() -> bool:
    """Checks if the database connection is alive, reconnects if not."""
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("DB connection is closed or None. Attempting to reconnect...")
        init_db()
    try:
        if conn:
             with conn.cursor() as cur:
                cur.execute("SELECT 1;")
             return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"DB connection check failed: {e}. Attempting to reconnect...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except:
            return False
    return False

def log_and_notify(level: str, message: str, notification_type: str):
    """Logs a message and saves it as a notification in the database."""
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    
    if not check_db_connection() or not conn:
        logger.error("Cannot save notification, DB connection is not available.")
        return
    try:
        with notifications_lock:
            notifications_cache.appendleft({"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message})
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to insert notification into DB: {e}")
        if conn: conn.rollback()

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """Reads a list of symbols and validates them against Binance exchange info."""
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        exchange_info = client.get_exchange_info()
        active = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] Bot will monitor {len(validated)} symbols.")
        return validated
    except Exception as e:
        logger.error(f"Failed to get validated symbols: {e}")
        return []

# The rest of the functions (fetch_historical_data, TradingStrategy, etc.) remain the same
# as they do not relate to the GitHub loading issue. For brevity, they are not repeated here.
# Assume they are present from this point onwards.

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.astype({c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            return pd.DataFrame(cur.fetchall())
    except Exception as e:
        logger.error(f"Error fetching S/R levels from DB for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        for col in ['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']: df[col] = 0.0
        return df
    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False, na=False)]
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False, na=False)]
    support_levels, resistance_levels = supports['level_price'].to_numpy(), resistances['level_price'].to_numpy()
    support_scores, resistance_scores = pd.Series(supports['score'].values, index=supports['level_price']).to_dict(), pd.Series(resistances['score'].values, index=resistances['level_price']).to_dict()
    results = []
    for price in df['close']:
        dist_s, score_s, dist_r, score_r = 1.0, 0.0, 1.0, 0.0
        if support_levels.size > 0:
            diffs = price - support_levels
            below = diffs[diffs >= 0]
            if below.size > 0:
                near_s_p = support_levels[diffs == below.min()][0]
                dist_s = (price - near_s_p) / price if price > 0 else 0
                score_s = support_scores.get(near_s_p, 0)
        if resistance_levels.size > 0:
            diffs = resistance_levels - price
            above = diffs[diffs >= 0]
            if above.size > 0:
                near_r_p = resistance_levels[diffs == above.min()][0]
                dist_r = (near_r_p - price) / price if price > 0 else 0
                score_r = resistance_scores.get(near_r_p, 0)
        results.append((dist_s, score_s, dist_r, score_r))
    df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = results
    return df

def calculate_base_features(df: pd.DataFrame) -> pd.DataFrame:
    delta = df['close'].diff()
    gain, loss = delta.clip(lower=0).ewm(com=RSI_PERIOD-1, adjust=False).mean(), (-delta.clip(upper=0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    tr = pd.concat([df['high'] - df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    return df.astype('float32', errors='ignore')

def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]):
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        for item in data:
            symbol = item.get('s')
            if not symbol: continue
            price = float(item.get('c', 0))
            if price == 0: continue
            with prices_lock: current_prices[symbol] = price
            sig_proc, status, c_price = None, None, None
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    signal = open_signals_cache[symbol]
                    if price >= signal.get('target_price', float('inf')): status, c_price, sig_proc = 'target_hit', signal.get('target_price'), signal
                    elif price <= signal.get('stop_loss', float('-inf')): status, c_price, sig_proc = 'stop_loss_hit', signal.get('stop_loss'), signal
            if sig_proc and status:
                Thread(target=close_signal, args=(sig_proc, status, c_price, "auto")).start()
    except Exception as e:
        logger.warning(f"Error in ticker handler: {e}")

def run_websocket_manager():
    logger.info("Starting WebSocket manager...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_ticker_socket(callback=handle_ticker_message)
    logger.info("✅ WebSocket manager started.")
    twm.join()

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_from_github(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            df_featured = calculate_base_features(df_15m.copy())
            df_featured = calculate_sr_features(df_featured, sr_levels_df)
            df_featured['returns'] = df_featured['close'].pct_change()
            merged = df_featured.join(btc_df['btc_returns']).fillna(0)
            df_featured['btc_correlation'] = merged['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged['btc_returns'])
            df_featured['rsi_4h'] = calculate_base_features(df_4h.copy())['rsi']
            df_featured.fillna(method='ffill', inplace=True)
            df_featured.fillna(method='bfill', inplace=True)
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            return df_featured[self.feature_names].dropna()
        except Exception as e:
            logger.error(f"Error getting features for {self.symbol}: {e}", exc_info=True)
            return None

    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty:
            return None
        try:
            features_scaled = self.scaler.transform(df_features.iloc[[-1]])
            prediction = self.ml_model.predict(features_scaled)[0]
            proba = self.ml_model.predict_proba(features_scaled)[0]
            
            # Find the index for the "buy" class (1)
            buy_class_index = list(self.ml_model.classes_).index(1)
            prob_buy = proba[buy_class_index]

            if prediction == 1 and prob_buy >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [Signal Found] {self.symbol}: Buy signal with confidence {prob_buy:.2%}")
                return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability_Buy': f"{prob_buy:.2%}"}}
            return None
        except Exception as e:
            logger.error(f"Error generating signal for {self.symbol}: {e}", exc_info=True)
            return None

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except: pass

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, status) VALUES (%s, %s, %s, %s, %s, %s, 'open') RETURNING id;",
                        (signal['symbol'], signal['entry_price'], signal['target_price'], signal['stop_loss'], signal['strategy_name'], json.dumps(signal['signal_details'])))
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        return signal
    except Exception as e:
        logger.error(f"Error inserting signal into DB: {e}")
        if conn: conn.rollback()
        return None

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    symbol = signal['symbol']
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal['id']: return
    
    if not check_db_connection() or not conn: return
    try:
        profit_pct = ((closing_price / signal['entry_price']) - 1) * 100
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status=%s, closing_price=%s, closed_at=NOW(), profit_percentage=%s WHERE id=%s;", (status, closing_price, profit_pct, signal['id']))
        conn.commit()
        
        with signal_cache_lock: del open_signals_cache[symbol]
        
        status_map = {'target_hit': '✅ Target Hit', 'stop_loss_hit': '🛑 Stop Loss Hit', 'manual_close': '🖐️ Manual Close'}
        msg = f"*{status_map.get(status, status)}*\n`{symbol}` | Profit: `{profit_pct:+.2f}%`"
        send_telegram_message(msg)
        log_and_notify('info', f"{status_map.get(status, status)}: {symbol} | Profit: {profit_pct:+.2f}%", 'CLOSE_SIGNAL')
    except Exception as e:
        logger.error(f"Error closing signal in DB: {e}")
        if conn: conn.rollback()

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in cur.fetchall(): open_signals_cache[signal['symbol']] = dict(signal)
        logger.info(f"Loaded {len(open_signals_cache)} open signals into cache.")
    except Exception as e:
        logger.error(f"Error loading open signals to cache: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(cur.fetchall()): notifications_cache.appendleft({**n, 'timestamp': n['timestamp'].isoformat()})
    except Exception as e:
        logger.error(f"Error loading notifications to cache: {e}")

def get_btc_trend() -> Dict[str, Any]:
    if not client: return {"is_uptrend": False}
    try:
        klines = client.get_klines(symbol=BTC_SYMBOL, interval=BTC_TREND_TIMEFRAME, limit=BTC_TREND_EMA_PERIOD * 2)
        df = pd.DataFrame(klines, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'ct', 'qv', 'nt', 'tbb', 'tbq', 'ig'])
        df['c'] = pd.to_numeric(df['c'])
        ema, price = df['c'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean().iloc[-1], df['c'].iloc[-1]
        return {"is_uptrend": price > ema, "message": "Uptrend" if price > ema else "Downtrend"}
    except Exception as e:
        logger.warning(f"Could not get BTC trend: {e}")
        return {"is_uptrend": False}

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None:
        btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def main_loop():
    time.sleep(15) 
    if not validated_symbols_to_scan:
        log_and_notify("critical", "CRITICAL: No validated symbols to scan. Bot cannot run. Check crypto_list.txt and Binance connection.", "SYSTEM_ERROR"); return
    
    while True:
        try:
            if USE_BTC_TREND_FILTER and not get_btc_trend().get("is_uptrend"):
                logger.info("BTC trend is down. Pausing signal generation.")
                time.sleep(300); continue

            with signal_cache_lock: open_count = len(open_signals_cache)
            if open_count >= MAX_OPEN_TRADES:
                time.sleep(60); continue
            
            slots_available = MAX_OPEN_TRADES - open_count
            btc_data = get_btc_data_for_bot()
            if btc_data is None:
                logger.warning("Could not fetch BTC data for correlation. Retrying in 2 minutes.")
                time.sleep(120); continue
            
            for symbol in validated_symbols_to_scan:
                if slots_available <= 0: break
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                try:
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None or df_4h is None: continue
                    
                    strategy = TradingStrategy(symbol)
                    if not strategy.ml_model:
                        logger.debug(f"No model loaded for {symbol}, skipping.")
                        continue
                        
                    df_features = strategy.get_features(df_15m, df_4h, btc_data, fetch_sr_levels_from_db(symbol))
                    del df_15m, df_4h; gc.collect()
                    if df_features is None: continue
                    
                    potential_signal = strategy.generate_signal(df_features)
                    if potential_signal:
                        with prices_lock: current_price = current_prices.get(symbol)
                        if not current_price: continue
                        
                        potential_signal['entry_price'] = current_price
                        atr = df_features['atr'].iloc[-1]
                        potential_signal['stop_loss'] = current_price - (atr * ATR_SL_MULTIPLIER)
                        potential_signal['target_price'] = current_price + (atr * ATR_TP_MULTIPLIER)
                        
                        saved_signal = insert_signal_into_db(potential_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[symbol] = saved_signal
                            slots_available -= 1
                            log_and_notify('info', f"New BUY signal opened for {symbol} at {current_price}", "NEW_SIGNAL")
                            send_telegram_message(f"🚀 *New Signal: BUY {symbol}*\n`Entry:` {current_price}\n`TP:` {potential_signal['target_price']:.4f}\n`SL:` {potential_signal['stop_loss']:.4f}")

                except Exception as e:
                    logger.error(f"Error in main loop for symbol {symbol}: {e}", exc_info=True)
            time.sleep(60)
        except Exception as e:
            log_and_notify("error", f"Critical error in main loop: {e}", "SYSTEM_ERROR"); time.sleep(120)

app = Flask(__name__)
CORS(app)

@app.route('/api/status')
def get_status():
    with signal_cache_lock: s = list(open_signals_cache.values())
    for sig in s:
        with prices_lock: sig['current_price'] = current_prices.get(sig['symbol'])
        # ensure datetime is string serializable
        if 'created_at' in sig and hasattr(sig['created_at'], 'isoformat'):
            sig['created_at'] = sig['created_at'].isoformat()
    return jsonify({"open_signals": s, "max_trades": MAX_OPEN_TRADES})

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    from waitress import serve
    logger.info(f"Starting API server on http://{host}:{port}")
    serve(app, host=host, port=port, threads=8)

def initialize_bot_services():
    global client, validated_symbols_to_scan
    try:
        client = Client(API_KEY, API_SECRET)
        init_github_repo()
        init_db()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
    except Exception as e:
        log_and_notify("critical", f"Bot initialization failed: {e}", "SYSTEM_ERROR")

if __name__ == "__main__":
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
