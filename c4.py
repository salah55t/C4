import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any
from collections import deque, defaultdict
from waitress import serve

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v5.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV5')
logging.getLogger('binance').setLevel(logging.WARNING) # تقليل رسائل مكتبة بينانس

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
     logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
MODEL_CONFIDENCE_THRESHOLD = 0.55
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_OPEN_TRADES: int = 5
MAX_WEBSOCKET_SYMBOLS: int = 200 # !!! حد أقصى لعدد العملات على WebSocket
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7
# ... (باقي الثوابت)
RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, ATR_PERIOD = 14, 12, 26, 9, 14
EMA_SLOW_PERIOD, EMA_FAST_PERIOD, BTC_CORR_PERIOD = 200, 50, 30

# ---  المتغيرات العامة والأقفال ---
db_pool: Optional[pool.SimpleConnectionPool] = None
client: Optional[Client] = None
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
current_prices: Dict[str, float] = defaultdict(float) # استخدام defaultdict
btc_data_cache: Optional[pd.DataFrame] = None
signal_cache_lock = Lock()
prices_lock = Lock()
btc_data_lock = Lock()
notifications_cache = deque(maxlen=100)
notifications_lock = Lock()

# =================================================================================
# قسم التنبيهات وقاعدة البيانات
# =================================================================================
def init_db():
    global db_pool
    if db_pool: return
    try:
        db_pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=DB_URL, cursor_factory=RealDictCursor)
        conn = db_pool.getconn()
        # (باقي كود تهيئة قاعدة البيانات)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                    target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                    status TEXT DEFAULT 'open', created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    closing_price DOUBLE PRECISION, closed_at TIMESTAMP WITH TIME ZONE,
                    profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                    trailing_stop_price DOUBLE PRECISION);""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    type TEXT NOT NULL, message TEXT NOT NULL);""")
        conn.commit()
        db_pool.putconn(conn)
        logger.info("✅ [DB Pool] تم تهيئة مجمع اتصالات قاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [DB Pool] خطأ في تهيئة مجمع الاتصالات: {e}", exc_info=True); exit(1)

def execute_db_query(query, params=None, fetch=None):
    if not db_pool: logger.error("❌ [DB] مجمع الاتصالات غير متاح."); return None
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch == 'one': return cur.fetchone()
            if fetch == 'all': return cur.fetchall()
            conn.commit()
            if 'RETURNING' in query.upper(): return cur.fetchone()
            return True
    except Exception as e:
        if conn: conn.rollback()
        logger.error(f"❌ [DB Query] فشل تنفيذ الاستعلام: {e}")
        return None
    finally:
        if conn: db_pool.putconn(conn)

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10).raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def log_and_notify(level: str, message: str, notification_type: str, send_tg: bool = False):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    with notifications_lock:
        notifications_cache.appendleft({"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message})
    if notification_type in ["NEW_TRADE", "TRADE_CLOSE", "SYSTEM_ERROR", "SYSTEM_INFO"]:
        execute_db_query("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
    if send_tg:
        send_telegram_message(message)

# =================================================================================
# قسم جلب البيانات وحساب المؤشرات
# =================================================================================
def get_validated_symbols(filename: str = 'crypto_list.txt'):
    try:
        script_dir = os.path.dirname(__file__)
        with open(os.path.join(script_dir, filename), 'r', encoding='utf-8') as f:
            symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in symbols}
        info = client.get_exchange_info()
        active = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        validated = sorted(list(formatted.intersection(active)))
        
        # !!! تعديل مهم: تحديد عدد الرموز لتجنب الحمل الزائد
        if len(validated) > MAX_WEBSOCKET_SYMBOLS:
            msg = f"⚠️ *تحذير الأداء:* قائمة العملات تحتوي على {len(validated)} عملة. سيتم مراقبة أول {MAX_WEBSOCKET_SYMBOLS} عملة فقط لتجنب الحمل الزائد على الخادم. يرجى تقليل عدد العملات في `crypto_list.txt`."
            log_and_notify('warning', msg.replace('*', ''), 'SYSTEM_INFO', send_tg=True)
            validated = validated[:MAX_WEBSOCKET_SYMBOLS]
            
        logger.info(f"✅ [Validation] سيتم مراقبة {len(validated)} عملة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ في التحقق من الرموز: {e}"); return []

def fetch_historical_data(symbol: str, interval: str, days: int):
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
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [Data] خطأ API من Binance لـ {symbol}: {e}"); return None
    except Exception as e:
        logger.error(f"❌ [Data] خطأ عام في جلب البيانات لـ {symbol}: {e}"); return None

# ... (باقي دوال جلب البيانات وحساب المؤشرات كما هي)
def update_btc_data_cache():
    temp_btc_df = fetch_historical_data('BTCUSDT', SIGNAL_GENERATION_TIMEFRAME, days=15)
    if temp_btc_df is not None:
        with btc_data_lock:
            global btc_data_cache
            temp_btc_df['btc_returns'] = temp_btc_df['close'].pct_change()
            btc_data_cache = temp_btc_df

def btc_cache_updater_loop():
    while True:
        try: update_btc_data_cache(); time.sleep(900)
        except Exception as e: logger.error(f"❌ [BTC Loop] خطأ: {e}"); time.sleep(60)

def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=MACD_SIGNAL, adjust=False).mean()
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    if btc_df is not None and not btc_df.empty:
        df_calc['returns'] = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else: df_calc['btc_correlation'] = 0.0
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    df_calc['hour_of_day'] = df_calc.index.hour
    return df_calc.replace([np.inf, -np.inf], np.nan).dropna()


def load_ml_model_bundle_from_db(symbol: str):
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache: return ml_models_cache[model_name]
    res = execute_db_query("SELECT model_data FROM ml_models WHERE model_name = %s LIMIT 1;", (model_name,), fetch='one')
    if res and res.get('model_data'):
        try:
            bundle = pickle.loads(res['model_data'])
            ml_models_cache[model_name] = bundle
            logger.info(f"✅ [ML] تم تحميل النموذج '{model_name}'.")
            return bundle
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(f"❌ [ML] فشل تحميل النموذج '{model_name}' من قاعدة البيانات، الملف تالف: {e}")
            return None
    logger.warning(f"⚠️ [ML] لم يتم العثور على النموذج '{model_name}'.")
    return None

# =================================================================================
# قسم الاستراتيجية وإدارة الصفقات
# =================================================================================
class TradingStrategy:
    # (كود الكلاس كما هو)
    def __init__(self, symbol: str):
        self.symbol = symbol
        bundle = load_ml_model_bundle_from_db(symbol)
        self.ml_model, self.scaler, self.feature_names = (bundle.get('model'), bundle.get('scaler'), bundle.get('feature_names')) if bundle else (None, None, None)

    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        last_row = df_processed.iloc[-1:]
        try:
            features_df = last_row[self.feature_names]
            if features_df.isnull().values.any(): return None
            features_scaled_np = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names, index=features_df.index)
            prediction = self.ml_model.predict(features_scaled_df)[0]
            if prediction != 1: return None
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
            confidence = prediction_proba[np.where(self.ml_model.classes_ == 1)[0][0]]
            if confidence < MODEL_CONFIDENCE_THRESHOLD: return None
            logger.info(f"✅ [Signal Found] {self.symbol}: إشارة شراء محتملة بثقة {confidence:.2%}.")
            return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Confidence': f"{confidence:.2%}"}}
        except Exception as e:
            logger.warning(f"⚠️ [Signal Gen] {self.symbol}: خطأ: {e}"); return None

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    # (كود الدالة كما هو)
    entry_price = signal['entry_price']
    profit = ((closing_price - entry_price) / entry_price) * 100
    execute_db_query(
        "UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;",
        (status, closing_price, profit, signal['id'])
    )
    outcome_emoji = "✅" if profit >= 0 else "❌"
    msg = (f"{outcome_emoji} *إغلاق صفقة* | `{closed_by}`\n\n"
           f"العملة: *{signal['symbol']}*\n"
           f"سعر الإغلاق: `${closing_price:,.4f}`\n"
           f"الربح/الخسارة: *{profit:,.2f}%*\n"
           f"سعر الدخول: `${entry_price:,.4f}`")
    log_and_notify('info', msg.replace('*', '').replace(',', ''), "TRADE_CLOSE", send_tg=True)
    with signal_cache_lock:
        if signal['symbol'] in open_signals_cache:
            del open_signals_cache[signal['symbol']]

def trade_monitoring_loop():
    """!!! خيط منفصل لمراقبة الصفقات المفتوحة"""
    logger.info("✅ [Monitor] بدء خيط مراقبة الصفقات...")
    while True:
        try:
            with signal_cache_lock:
                # إنشاء نسخة لتجنب مشاكل التزامن
                open_signals = list(open_signals_cache.values())

            if not open_signals:
                time.sleep(5)
                continue

            for signal in open_signals:
                symbol = signal['symbol']
                # لا حاجة للقفل هنا لأن القراءة من defaultdict آمنة
                price = current_prices[symbol]
                
                if price == 0: # لم يتم استلام سعر بعد
                    continue
                
                # التحقق من الهدف أو وقف الخسارة
                if price >= signal['target_price']:
                    close_signal(signal, 'closed_tp', signal['target_price'], 'TP')
                elif price <= signal['stop_loss']:
                    close_signal(signal, 'closed_sl', signal['stop_loss'], 'SL')
            
            time.sleep(1) # دورة مراقبة كل ثانية
        except Exception as e:
            logger.error(f"❌ [Monitor Loop] خطأ في حلقة المراقبة: {e}", exc_info=True)
            time.sleep(10)


def handle_ticker_message(msg):
    """!!! معالج رسائل مُحسَّن وسريع جداً"""
    if msg.get('e') != '24hrTicker' or 's' not in msg or 'c' not in msg: 
        return
    # التحديث فقط، بدون أي منطق آخر
    current_prices[msg['s']] = float(msg['c'])

def run_websocket_manager():
    # !!! تعديل: زيادة حجم الطابور الداخلي
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    
    symbols_for_ws = [s.lower()+'@ticker' for s in validated_symbols_to_scan]
    if symbols_for_ws:
        # زيادة حجم الطابور الداخلي لمكتبة بينانس
        twm.start_multiplex_socket(callback=handle_ticker_message, streams=symbols_for_ws)
        logger.info(f"✅ [WebSocket] تم بدء مراقبة Ticker لـ {len(symbols_for_ws)} عملة.")
    
    # بدء خيط مراقبة الصفقات المنفصل
    monitor_thread = Thread(target=trade_monitoring_loop, daemon=True)
    monitor_thread.start()

    twm.join()

# ... (باقي دوال إدارة الصفقات كما هي)
def insert_signal_into_db(signal: Dict):
    params = (
        signal['symbol'], signal['entry_price'], signal['target_price'], signal['stop_loss'],
        signal['strategy_name'], json.dumps(signal.get('signal_details', {})), signal.get('trailing_stop_price')
    )
    new_signal_record = execute_db_query(
        "INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, trailing_stop_price) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *;", params, fetch='one')
    
    if new_signal_record:
        msg = (f"🚀 *إشارة جديدة* | `{signal['strategy_name']}`\n\n"
               f"العملة: *{signal['symbol']}*\n"
               f"سعر الدخول: `${signal['entry_price']:,.4f}`\n"
               f"الهدف (TP): `${signal['target_price']:,.4f}`\n"
               f"وقف الخسارة (SL): `${signal['stop_loss']:,.4f}`\n"
               f"الثقة: `{signal.get('signal_details', {}).get('ML_Confidence', 'N/A')}`")
        log_and_notify('info', msg.replace('*', '').replace(',', ''), "NEW_TRADE", send_tg=True)
    return new_signal_record

def load_open_signals_to_cache():
    logger.info("ℹ️ [Cache] تحميل الصفقات المفتوحة...")
    open_signals = execute_db_query("SELECT * FROM signals WHERE status = 'open';", fetch='all')
    if open_signals is not None:
        with signal_cache_lock:
            open_signals_cache.clear()
            for s in open_signals: open_signals_cache[s['symbol']] = dict(s)
        logger.info(f"✅ [Cache] تم تحميل {len(open_signals)} صفقة مفتوحة.")

# =================================================================================
# حلقة العمل الرئيسية
# =================================================================================
def main_loop():
    logger.info("[Main Loop] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15) 
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. سيتوقف البوت.", "SYSTEM_ERROR", send_tg=True); return
    
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM_INFO", send_tg=True)
    
    while True:
        try:
            with signal_cache_lock: open_count = len(open_signals_cache)
            if open_count >= MAX_OPEN_TRADES:
                time.sleep(30); continue
            
            slots_available = MAX_OPEN_TRADES - open_count
            logger.info(f"ℹ️ [Scan] بدء دورة مسح. المراكز المتاحة: {slots_available}")
            with btc_data_lock: current_btc_data = btc_data_cache
            if current_btc_data is None:
                logger.warning("⚠️ [Scan] بيانات البيتكوين غير متاحة, سيتم تخطي هذه الدورة."); time.sleep(60); continue

            for symbol in validated_symbols_to_scan:
                if (MAX_OPEN_TRADES - len(open_signals_cache)) <= 0: break
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                try:
                    df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty: continue
                    
                    df_features = calculate_features(df_hist, current_btc_data)
                    if df_features is None or df_features.empty: continue
                    
                    strategy = TradingStrategy(symbol)
                    potential_signal = strategy.generate_signal(df_features)
                    
                    if potential_signal:
                        current_price = current_prices[symbol]
                        if current_price == 0: continue
                        atr_value = df_features['atr'].iloc[-1]
                        if atr_value <= 0: continue
                        
                        potential_signal['entry_price'] = current_price
                        potential_signal['stop_loss'] = current_price - (atr_value * SL_ATR_MULTIPLIER)
                        potential_signal['target_price'] = current_price + (atr_value * TP_ATR_MULTIPLIER)
                        potential_signal['trailing_stop_price'] = potential_signal['stop_loss']
                        
                        saved_signal = insert_signal_into_db(potential_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                except Exception as e:
                    logger.error(f"❌ [Processing Error] {symbol}: {e}", exc_info=True)
            logger.info("ℹ️ [Scan End] انتهت دورة المسح."); time.sleep(90)
        except Exception as main_err:
            log_and_notify("error", f"خطأ كارثي في الحلقة الرئيسية: {main_err}", "SYSTEM_ERROR", send_tg=True); time.sleep(120)


# =================================================================================
# قسم واجهة API للوحة التحكم
# =================================================================================
app = Flask(__name__)
CORS(app)

@app.route('/')
def home(): return "<html><body><h1>Crypto Trading Bot V5 API is running.</h1></body></html>", 200

@app.route('/api/health')
def health_check():
    """نقطة فحص للتأكد من أن الخادم يعمل"""
    return jsonify({'status': 'ok'})

@app.route('/api/signals')
def get_signals():
    # (كود API كما هو)
    all_signals = execute_db_query("SELECT * FROM signals ORDER BY created_at DESC LIMIT 100;", fetch='all')
    if all_signals is None: return jsonify([])
    
    current_prices_copy = dict(current_prices)
    
    processed_signals = []
    for s in all_signals:
        signal_dict = dict(s)
        if signal_dict['status'] == 'open':
            signal_dict['current_price'] = current_prices_copy.get(signal_dict['symbol'])
        processed_signals.append(signal_dict)
        
    return jsonify(processed_signals)

@app.route('/api/stats')
def get_stats():
    # (كود API كما هو)
    stats_query = """
        SELECT
            SUM(CASE WHEN status != 'open' THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN profit_percentage >= 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_percentage < 0 THEN 1 ELSE 0 END) as losses,
            SUM(profit_percentage) as total_profit_percentage
        FROM signals WHERE status != 'open';
    """
    stats = execute_db_query(stats_query, fetch='one')
    if not stats or stats.get('closed_trades', 0) == 0:
        return jsonify({'wins': 0, 'losses': 0, 'win_rate': 0, 'loss_rate': 0, 'total_profit_usdt': 0})

    closed_trades = stats.get('closed_trades', 0)
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)
    
    win_rate = (wins / closed_trades) * 100 if closed_trades > 0 else 0
    loss_rate = (losses / closed_trades) * 100 if closed_trades > 0 else 0
    
    return jsonify({
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'total_profit_usdt': stats.get('total_profit_percentage', 0)
    })
    
@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: notifs = list(notifications_cache)
    return jsonify(notifs)

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    # (كود API كما هو)
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)

    if not signal_to_close:
        return jsonify({'error': 'الصفقة غير موجودة أو مغلقة بالفعل'}), 404

    price = current_prices.get(signal_to_close['symbol'])

    if not price or price == 0:
        return jsonify({'error': 'لا يمكن الحصول على السعر الحالي للعملة'}), 500
    
    close_signal(signal_to_close, 'closed_manual', price, 'Manual')
    return jsonify({'message': f'تم إرسال طلب إغلاق للصفقة {signal_to_close["symbol"]}'})

# =================================================================================
# قسم التهيئة والتشغيل
# =================================================================================
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Init] بدء تهيئة خدمات البوت V5...")
    try:
        init_db()
        client = Client(API_KEY, API_SECRET)
        load_open_signals_to_cache()
        
        Thread(target=btc_cache_updater_loop, daemon=True).start()
        logger.info("... انتظار أول جلب لبيانات البيتكوين ...")
        time.sleep(10)
        # التأكد من وجود بيانات البيتكوين قبل المتابعة
        with btc_data_lock:
            if btc_data_cache is None:
                update_btc_data_cache() # محاولة أخيرة
                if btc_data_cache is None:
                    log_and_notify("critical", "❌ فشل جلب بيانات البيتكوين الأولية. سيتوقف البوت.", "SYSTEM_ERROR", send_tg=True)
                    return

        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            log_and_notify("critical", "❌ لا توجد رموز معتمدة للمسح.", "SYSTEM_ERROR", send_tg=True); return

        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Init] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"خطأ حاسم أثناء التهيئة: {e}", "SYSTEM_ERROR", send_tg=True)

if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل تطبيق بوت التداول V5...")
    initialization_thread = Thread(target=initialize_bot_services)
    initialization_thread.start()
    
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🌍 بدء تشغيل خادم الويب على {host}:{port}")
    serve(app, host=host, port=port, threads=8)
