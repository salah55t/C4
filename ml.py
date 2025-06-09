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
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
    handlers=[
        logging.FileHandler('ml_trainer_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainerEnhanced')

# ---------------------- تحميل المتغيرات البيئية ----------------------
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
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 120
BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V2_Enhanced'

# Indicator Parameters (matching c4.py)
RSI_PERIOD: int = 14
VOLUME_LOOKBACK_CANDLES: int = 2
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 3
ENTRY_ATR_PERIOD: int = 14
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
MACD_FAST_PERIOD: int = 12
MACD_SLOW_PERIOD: int = 26
MACD_SIGNAL_PERIOD: int = 9
BB_PERIOD: int = 20
BB_STD_DEV: int = 2
ADX_PERIOD: int = 14

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None

# --- Full implementation of all required functions ---

def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn, cur
    logger.info("[DB] تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL, current_target DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION, volume_15m DOUBLE PRECISION, achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION, closed_at TIMESTAMP, sent_at TIMESTAMP DEFAULT NOW(),
                    entry_time TIMESTAMP DEFAULT NOW(), time_to_target INTERVAL, profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT, signal_details JSONB, stop_loss DOUBLE PRECISION
                );
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE, model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB
                );
            """)
            conn.commit()
            logger.info("✅ [DB] جداول 'signals' و 'ml_models' جاهزة.")
            return
        except (OperationalError, Exception) as e:
            logger.error(f"❌ [DB] فشلت محاولة الاتصال {attempt + 1}: {e}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise e
            time.sleep(delay)

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client:
        logger.error("❌ [Data] عميل Binance غير مهيأ لجلب البيانات.")
        return None
    try:
        start_str_overall = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} من {start_str_overall}...")
        interval_map = {'15m': Client.KLINE_INTERVAL_15MINUTE}
        binance_interval = interval_map.get(interval)
        if not binance_interval:
            logger.error(f"❌ [Data] فترة زمنية غير مدعومة: {interval}")
            return None
        klines = client.get_historical_klines(symbol, binance_interval, start_str_overall)
        if not klines:
            logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) لـ {symbol}.")
            return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[numeric_cols].dropna()
        df.sort_index(inplace=True)
        logger.debug(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) لـ {symbol}.")
        return df
    except (BinanceAPIException, BinanceRequestException, Exception) as e:
        logger.error(f"❌ [Data] خطأ Binance أثناء جلب البيانات لـ {symbol}: {e}")
        return None

def calculate_ema(series: pd.Series, span: int) -> pd.Series: return series.ewm(span=span, adjust=False).mean()
def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    return df
def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df
def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    if 'atr' not in df.columns: df = calculate_atr_indicator(df, period)
    hl2 = (df['high'] + df['low']) / 2
    df['upper_band'] = hl2 + (multiplier * df['atr'])
    df['lower_band'] = hl2 - (multiplier * df['atr'])
    df['in_uptrend'] = True
    for current in range(1, len(df.index)):
        previous = current - 1
        if df['close'].iloc[current] > df['upper_band'].iloc[previous]: df.loc[df.index[current], 'in_uptrend'] = True
        elif df['close'].iloc[current] < df['lower_band'].iloc[previous]: df.loc[df.index[current], 'in_uptrend'] = False
        else:
            df.loc[df.index[current], 'in_uptrend'] = df['in_uptrend'].iloc[previous]
            if df['in_uptrend'].iloc[current] and df['lower_band'].iloc[current] < df['lower_band'].iloc[previous]: df.loc[df.index[current], 'lower_band'] = df['lower_band'].iloc[previous]
            if not df['in_uptrend'].iloc[current] and df['upper_band'].iloc[current] > df['upper_band'].iloc[previous]: df.loc[df.index[current], 'upper_band'] = df['upper_band'].iloc[previous]
    df['supertrend_direction'] = np.where(df['in_uptrend'], 1, -1)
    df.drop(['upper_band', 'lower_band', 'in_uptrend'], axis=1, inplace=True, errors='ignore')
    return df
def calculate_macd(df: pd.DataFrame, fast_period: int = MACD_FAST_PERIOD, slow_period: int = MACD_SLOW_PERIOD, signal_period: int = MACD_SIGNAL_PERIOD) -> pd.DataFrame:
    df['ema_fast'] = calculate_ema(df['close'], span=fast_period)
    df['ema_slow'] = calculate_ema(df['close'], span=slow_period)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = calculate_ema(df['macd'], span=signal_period)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True, errors='ignore')
    return df
def calculate_bollinger_bands(df: pd.DataFrame, period: int = BB_PERIOD, std_dev: int = BB_STD_DEV) -> pd.DataFrame:
    df['bb_ma'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_ma'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_ma'] - (df['bb_std'] * std_dev)
    df.drop(['bb_ma', 'bb_std'], axis=1, inplace=True, errors='ignore')
    return df
def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    if 'atr' not in df.columns: df = calculate_atr_indicator(df, period)
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = df['low'].diff().mul(-1)
    df['plus_dm'] = np.where((df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0), df['plus_dm'], 0)
    df['minus_dm'] = np.where((df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0), df['minus_dm'], 0)
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / df['atr'])
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1))
    df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
    df.drop(['plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx'], axis=1, inplace=True, errors='ignore')
    return df
def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    min_data_for_ema = 55
    if df_btc is None or len(df_btc) < min_data_for_ema: return None
    ema20 = calculate_ema(df_btc['close'], 20)
    ema50 = calculate_ema(df_btc['close'], 50)
    trend_series = pd.Series(index=ema20.index, data=0.0)
    trend_series[(df_btc['close'] > ema20) & (ema20 > ema50)] = 1.0
    trend_series[(df_btc['close'] < ema20) & (ema20 < ema50)] = -1.0
    return trend_series.reindex(df_btc.index).fillna(0.0)

# FIXED: More robust path handling
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Symbols] قراءة قائمة الرموز من '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename)
            if not os.path.exists(file_path):
                 logger.error(f"❌ [Symbols] الملف '{filename}' غير موجود في دليل السكريبت أو الدليل الحالي.")
                 return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT" for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"ℹ️ [Symbols] تم قراءة {len(raw_symbols)} رمزًا مبدئيًا. جاري التحقق من Binance...")
        if not client:
            logger.error("❌ [Symbols] عميل Binance غير جاهز للتحقق من الرموز.")
            return raw_symbols
        exchange_info = client.get_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('status') == 'TRADING' and s.get('isSpotTradingAllowed')}
        validated_symbols = [s for s in raw_symbols if s in valid_symbols]
        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0: logger.warning(f"⚠️ [Symbols] تم إزالة {removed_count} رمزًا غير صالح للتداول.")
        logger.info(f"✅ [Symbols] تم التحقق من {len(validated_symbols)} رمزًا صالحًا للتداول.")
        return validated_symbols
    except Exception as e:
        logger.error(f"❌ [Symbols] فشل قراءة أو التحقق من الرموز من '{filename}': {e}", exc_info=True)
        return []

def convert_np_values(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_np_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_np_values(item) for item in obj]
    if pd.isna(obj): return None
    return obj

def save_ml_model_to_db(model_bundle: Tuple, model_name: str, metrics: Dict[str, Any]) -> bool:
    if not conn: logger.error("❌ [DB Save] لا يمكن حفظ النموذج، لا يوجد اتصال بقاعدة البيانات."); return False
    try:
        model_binary = pickle.dumps(model_bundle)
        metrics_json = json.dumps(convert_np_values(metrics))
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT id FROM ml_models WHERE model_name = %s;", (model_name,))
            if db_cur.fetchone():
                db_cur.execute("UPDATE ml_models SET model_data = %s, trained_at = NOW(), metrics = %s WHERE model_name = %s;", (model_binary, metrics_json, model_name))
            else:
                db_cur.execute("INSERT INTO ml_models (model_name, model_data, trained_at, metrics) VALUES (%s, %s, NOW(), %s);", (model_name, model_binary, metrics_json))
        conn.commit()
        logger.info(f"✅ [DB Save] تم حفظ النموذج '{model_name}' بنجاح في قاعدة البيانات.")
        return True
    except (psycopg2.Error, pickle.PicklingError, Exception) as e:
        logger.error(f"❌ [DB Save] خطأ أثناء حفظ النموذج '{model_name}': {e}", exc_info=True)
        if conn: conn.rollback()
        return False

def send_telegram_message(target_chat_id: str, text: str, **kwargs):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown', **kwargs}
    try:
        requests.post(url, json=payload, timeout=20).raise_for_status()
    except (requests.exceptions.RequestException, Exception) as e:
        logger.error(f"❌ [Telegram] فشل إرسال رسالة: {e}")

# ---------------------- ML Data Preparation & Training (ENHANCED) ----------------------
def prepare_data_for_ml(df: pd.DataFrame, symbol: str, target_period: int = 8, profit_threshold: float = 0.015) -> Optional[pd.DataFrame]:
    logger.info(f"ℹ️ [ML Prep] تجهيز بيانات محسنة لـ {symbol}...")
    try:
        df_calc = df.copy()
        df_calc = calculate_rsi_indicator(df_calc)
        df_calc = calculate_atr_indicator(df_calc)
        df_calc = calculate_supertrend(df_calc)
        df_calc = calculate_macd(df_calc)
        df_calc = calculate_bollinger_bands(df_calc)
        df_calc = calculate_adx(df_calc)
        df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES).mean()
        df_calc['rsi_momentum_bullish'] = ((df_calc['rsi'].diff(RSI_MOMENTUM_LOOKBACK_CANDLES) > 0) & (df_calc['rsi'] > 50)).astype(int)
        df_calc['bb_upper_dist'] = (df_calc['bb_upper'] - df_calc['close']) / df_calc['close']
        df_calc['bb_lower_dist'] = (df_calc['close'] - df_calc['bb_lower']) / df_calc['close']

        btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
        if btc_df is not None:
            btc_trend_series = _calculate_btc_trend_feature(btc_df)
            df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'), left_index=True, right_index=True, how='left')
            df_calc['btc_trend_feature'].fillna(0.0, inplace=True)
        else:
            df_calc['btc_trend_feature'] = 0.0

        df_calc['future_high'] = df_calc['high'].shift(-target_period).rolling(window=target_period).max()
        df_calc['target'] = (df_calc['future_high'] >= df_calc['close'] * (1 + profit_threshold)).astype(int)
        
        feature_columns = ['volume_15m_avg', 'rsi_momentum_bullish', 'btc_trend_feature', 'supertrend_direction', 'macd_hist', 'bb_upper_dist', 'bb_lower_dist', 'adx']
        
        for col in feature_columns:
            if col not in df_calc.columns: df_calc[col] = np.nan
        
        df_cleaned = df_calc.dropna(subset=feature_columns + ['target']).copy()
        logger.info(f"✅ [ML Prep] تم تجهيز البيانات لـ {symbol}. عدد الصفوف: {len(df_cleaned)}, الأهداف الإيجابية: {df_cleaned['target'].sum()}")
        return df_cleaned[feature_columns + ['target']]
    except Exception as e:
        logger.error(f"❌ [ML Prep] خطأ في تجهيز البيانات لـ {symbol}: {e}", exc_info=True)
        return None

def train_and_evaluate_model(data: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    logger.info("ℹ️ [ML Train] بدء تدريب وتقييم النموذج باستخدام RandomForest...")
    X = data.drop('target', axis=1)
    y = data['target']

    if X.empty or y.empty or y.nunique() < 2:
        logger.error("❌ [ML Train] لا توجد بيانات كافية أو فئات هدف للتدريب.")
        return None, {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=150, max_depth=15, min_samples_leaf=5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'model_type': 'RandomForestClassifier',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'feature_names': X.columns.tolist()
    }
    
    logger.info(f"📊 [ML Train] مقاييس أداء النموذج:")
    for key, value in metrics.items():
        if isinstance(value, float): logger.info(f"  - {key.capitalize()}: {value:.4f}")
    
    return (model, scaler), metrics

# ---------------------- Main Training Script ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء سكريبت تدريب نموذج التعلم الآلي المحسن...")
    
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
    except Exception as startup_err:
        logger.critical(f"❌ [Main] فشل في تهيئة العميل أو قاعدة البيانات: {startup_err}")
        exit(1)

    symbols = get_crypto_symbols()
    if not symbols:
        logger.critical("❌ [Main] لا توجد رموز صالحة للتدريب. تحقق من 'crypto_list.txt'.")
        exit(1)

    overall_summary = []
    start_time = time.time()

    for symbol in symbols:
        model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
        logger.info(f"\n--- ⏳ [Main] بدء التدريب لـ {symbol} ({model_name}) ---")
        try:
            df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_hist is None or df_hist.empty:
                logger.warning(f"⚠️ [Main] تعذر جلب بيانات كافية لـ {symbol}. تخطي."); continue
            
            df_ml = prepare_data_for_ml(df_hist, symbol)
            if df_ml is None or df_ml.empty or df_ml['target'].sum() < 10:
                logger.warning(f"⚠️ [Main] لا توجد بيانات تدريب قابلة للاستخدام لـ {symbol} بعد الإعداد. تخطي."); continue

            model_bundle, model_metrics = train_and_evaluate_model(df_ml)
            if model_bundle is None:
                logger.error(f"❌ [Main] فشل تدريب النموذج لـ {symbol}."); continue

            if save_ml_model_to_db(model_bundle, model_name, model_metrics):
                summary = f"✅ {symbol}: Success | Precision: {model_metrics['precision']:.2f}, Recall: {model_metrics['recall']:.2f}"
                overall_summary.append(summary)
            else:
                overall_summary.append(f"❌ {symbol}: DB Save Failed")
        except Exception as e:
            logger.critical(f"❌ [Main] حدث خطأ فادح أثناء التدريب لـ {symbol}: {e}", exc_info=True)
            overall_summary.append(f"❌ {symbol}: Training Error")
        time.sleep(2)

    duration = time.time() - start_time
    summary_message = f"🤖 *اكتمل تدريب نموذج ML*\n*المدة:* {duration:.2f} ثانية\n\n" + "\n".join(overall_summary)
    send_telegram_message(CHAT_ID, summary_message)
    
    if conn: conn.close()
    logger.info("👋 [Main] انتهى سكريبت التدريب المحسن.")
