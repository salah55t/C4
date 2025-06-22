import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import lightgbm as lgb
import optuna
import warnings
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread, Event

# ---------------------- تجاهل التحذيرات المستقبلية من Pandas ----------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v5.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V5')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
    logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
    exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 120
HYPERPARAM_TUNING_TRIALS: int = 5
BTC_SYMBOL = 'BTCUSDT'
ADX_PERIOD: int = 14
BBANDS_PERIOD: int = 20
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
STOCH_RSI_PERIOD: int = 14
STOCH_K: int = 3
STOCH_D: int = 3
REL_VOL_PERIOD: int = 30
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30
STOCH_RSI_OVERBOUGHT: int = 80
STOCH_RSI_OVERSOLD: int = 20
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24
DB_KEEP_ALIVE_INTERVAL: int = 300 # بالثواني (5 دقائق)

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

# --- دوال الاتصال والتحقق ---

def get_db_connection(force_reconnect: bool = False) -> Optional[psycopg2.extensions.connection]:
    """
    يوفر اتصالاً صالحًا بقاعدة البيانات، ويعيد الاتصال إذا لزم الأمر.
    """
    global conn
    if force_reconnect or conn is None or conn.closed != 0:
        if conn and not conn.closed:
            try:
                conn.close()
            except psycopg2.Error:
                pass 

        try:
            logger.info("ℹ️ [DB] محاولة (إعادة) الاتصال بقاعدة البيانات...")
            conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                        model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
                """)
            conn.commit()
            logger.info("✅ [DB] تم إنشاء اتصال قاعدة البيانات والتحقق من الجدول بنجاح.")
        except psycopg2.OperationalError as e:
            logger.critical(f"❌ [DB] لا يمكن إنشاء اتصال بقاعدة البيانات: {e}")
            conn = None
        except Exception as e:
            logger.critical(f"❌ [DB] حدث خطأ غير متوقع أثناء تهيئة قاعدة البيانات: {e}")
            conn = None
    return conn

def db_keep_alive(stop_event: Event):
    """
    دالة تعمل في الخلفية لإرسال استعلام بسيط للحفاظ على الاتصال.
    """
    logger.info("💡 [Keep-Alive] تم بدء خدمة الحفاظ على اتصال قاعدة البيانات.")
    while not stop_event.is_set():
        try:
            local_conn = get_db_connection()
            if local_conn and not local_conn.closed:
                with local_conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                logger.info("💡 [Keep-Alive] تم إرسال نبضة للحفاظ على الاتصال.")
            else:
                logger.warning("💡 [Keep-Alive] محاولة إرسال نبضة ولكن الاتصال مغلق. سيتم إعادة الاتصال في الدورة التالية.")
        except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
            logger.error(f"❌ [Keep-Alive] فقدان الاتصال أثناء إرسال النبضة: {e}. سيتم فرض إعادة الاتصال.")
            get_db_connection(force_reconnect=True)
        except Exception as e:
            logger.error(f"❌ [Keep-Alive] خطأ غير متوقع: {e}")

        # انتظر الفترة المحددة أو حتى يتم طلب الإيقاف
        stop_event.wait(DB_KEEP_ALIVE_INTERVAL)
    logger.info("🛑 [Keep-Alive] تم إيقاف خدمة الحفاظ على اتصال قاعدة البيانات.")


def get_binance_client():
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل تهيئة عميل Binance: {e}"); exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    # ... (الكود بدون تغيير)
    if not client:
        logger.error("❌ [Validation] عميل Binance لم يتم تهيئته.")
        return []
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in symbols}
        info = client.get_exchange_info()
        active = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول.")
        return validated
    except FileNotFoundError:
        logger.error(f"❌ [Validation] ملف قائمة العملات '{filename}' غير موجود.")
        return []
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ في التحقق من الرموز: {e}"); return []
# --- باقي دوال معالجة البيانات والتدريب (بدون تغيير) ---

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol} على إطار {interval}: {e}"); return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change()

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    # ... (الكود بدون تغيير)
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    tr = pd.concat([high_low, (df_calc['high'] - df_calc['close'].shift()).abs(), (df_calc['low'] - df_calc['close'].shift()).abs()], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    plus_dm = ((df_calc['high'].diff() > -df_calc['low'].diff()) & (df_calc['high'].diff() > 0)).astype(int)
    minus_dm = ((-df_calc['low'].diff() > df_calc['high'].diff()) & (-df_calc['low'].diff() > 0)).astype(int)
    plus_di = 100 * (plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'])
    minus_di = 100 * (minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'])
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    # ... (rest of the function is the same)
    return df_calc

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    # ... (الكود بدون تغيير)
    labels = pd.Series(0, index=prices.index, dtype=int)
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0: continue
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        for j in range(1, MAX_HOLD_PERIOD + 1):
            if i + j >= len(prices): break
            if prices.iloc[i + j] >= upper_barrier:
                labels.iloc[i] = 1; break
            if prices.iloc[i + j] <= lower_barrier:
                labels.iloc[i] = -1; break
    return labels

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    # ... (الكود بدون تغيير)
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    df_featured = calculate_features(df_15m, btc_df)
    # ... rest of the function ...
    return df_featured[[]], df_featured['target'], []


def tune_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    # ... (الكود بدون تغيير)
    return None, None, None

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    local_conn = get_db_connection()
    if not local_conn:
        logger.error(f"❌ [DB Save] No database connection for '{model_name}'. Skipping save.")
        return

    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        with local_conn.cursor() as db_cur:
            db_cur.execute("""
                INSERT INTO ml_models (model_name, model_data, metrics) VALUES (%s, %s, %s) 
                ON CONFLICT (model_name) DO UPDATE SET model_data = EXCLUDED.model_data, 
                trained_at = NOW(), metrics = EXCLUDED.metrics;
            """, (model_name, pickle.dumps(model_bundle), json.dumps(metrics)))
        local_conn.commit()
        logger.info(f"✅ [DB Save] Model bundle '{model_name}' saved successfully.")
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.error(f"❌ [DB Save] Connection lost while saving '{model_name}': {e}. Forcing reconnect.")
        get_db_connection(force_reconnect=True)
    except Exception as e:
        logger.error(f"❌ [DB Save] Error saving model bundle '{model_name}': {e}")
        try:
            if not local_conn.closed:
                local_conn.rollback()
        except psycopg2.Error as db_err:
            logger.error(f"❌ [DB Save] Failed to rollback: {db_err}. Forcing reconnect.")
            get_db_connection(force_reconnect=True)

def check_if_model_exists(model_name: str) -> bool:
    local_conn = get_db_connection()
    if not local_conn:
        logger.error("❌ [DB Check] No DB connection. Assuming model doesn't exist.")
        return False
    try:
        with local_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM ml_models WHERE model_name = %s LIMIT 1;", (model_name,))
            return cur.fetchone() is not None
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.error(f"❌ [DB Check] Connection lost checking for '{model_name}': {e}. Forcing reconnect.")
        get_db_connection(force_reconnect=True)
        return False
    except Exception as e:
        logger.error(f"❌ [DB Check] Error checking for model '{model_name}': {e}")
        return False

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def run_training_job(stop_event: Event):
    """
    الدالة الرئيسية التي تقوم بتشغيل وظيفة التدريب.
    """
    try:
        logger.info(f"🚀 Starting ADVANCED ML model training job ({BASE_ML_MODEL_NAME})...")
        get_binance_client()
        fetch_and_cache_btc_data()
        symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
        if not symbols_to_train:
            logger.critical("❌ [Main] No valid symbols. Exiting."); return
            
        send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill process {len(symbols_to_train)} symbols.")
        successful_models, failed_models, skipped_models = 0, 0, 0
        
        for symbol in symbols_to_train:
            if stop_event.is_set():
                logger.warning("🛑 [Main] Received stop signal. Aborting training loop.")
                break
                
            model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            if check_if_model_exists(model_name):
                logger.info(f"⏭️ [Main] Model for {symbol} already exists. Skipping.")
                skipped_models += 1; continue
            
            logger.info(f"\n--- ⏳ [Main] Starting training for {symbol} ---")
            try:
                # ... (نفس منطق التدريب)
                pass
            except Exception as e:
                logger.critical(f"❌ [Main] Fatal error for {symbol}: {e}", exc_info=True); failed_models += 1
            time.sleep(1)

        completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                            f"- ✅ Successfully trained: {successful_models}\n"
                            f"- ❌ Failed/Discarded: {failed_models}\n"
                            f"- ⏭️ Already trained (Skipped): {skipped_models}\n"
                            f"- 📊 Total symbols processed: {len(symbols_to_train)}")
        send_telegram_message(completion_message)
        logger.info(completion_message)
    finally:
        global conn
        if conn and not conn.closed: 
            conn.close()
            logger.info("ℹ️ [Main] Database connection closed.")
        # إرسال إشارة إيقاف لجميع الخيوط الأخرى
        stop_event.set()
        logger.info("👋 [Main] ML training job finished.")

app = Flask(__name__)

@app.route('/')
def health_check():
    return "ML Trainer service is running and healthy.", 200

if __name__ == "__main__":
    # إنشاء حدث لإيقاف الخيوط بأمان
    stop_event = Event()
    
    # بدء خيط الحفاظ على الاتصال
    keep_alive_thread = Thread(target=db_keep_alive, args=(stop_event,), daemon=True)
    keep_alive_thread.start()
    
    # بدء خيط التدريب الرئيسي
    training_thread = Thread(target=run_training_job, args=(stop_event,), daemon=True)
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    
    # تشغيل خادم الويب في الخيط الرئيسي
    # سيبقى يعمل حتى يتم إيقاف العملية يدويًا
    app.run(host='0.0.0.0', port=port)

    # عند إيقاف خادم الويب (مثل Ctrl+C)، أرسل إشارة الإيقاف
    logger.info("👋 Shutting down... Sending stop signal to background threads.")
    stop_event.set()
    # انتظر الخيوط حتى تنتهي
    training_thread.join(timeout=10)
    keep_alive_thread.join(timeout=10)
    logger.info("👋 Shutdown complete.")
