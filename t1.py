import time
import os
import json
import logging
import numpy as np
import pandas as pd
import psycopg2
import pickle
import redis
import re
import gc
from decimal import Decimal
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Set, Tuple
from sklearn.preprocessing import StandardScaler
import warnings

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_backtest_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotBacktest')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# ---------------------- ملفات الفلاتر الديناميكية (نسخة من c4.py) ----------------------
FILTER_PROFILES: Dict[str, Dict[str, Any]] = {
    "STRONG_UPTREND": {
        "description": "اتجاه صاعد قوي (نقاط 4+)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 25.0, "rel_vol": 0.2, "rsi_range": (50, 95), "roc": 0.0,
            "slope": 0.0, "min_rrr": 1.3, "min_volatility_pct": 0.25,
            "min_btc_correlation": -0.1, "min_bid_ask_ratio": 1.1 # min_bid_ask_ratio will be skipped in backtest
        }
    },
    "UPTREND": {
        "description": "اتجاه صاعد (نقاط 1-3)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 20.0,
            "rel_vol": 0.1,
            "rsi_range": (48, 90),
            "roc": -0.5,
            "slope": -0.05,
            "min_rrr": 1.4,
            "min_volatility_pct": 0.20,
            "min_btc_correlation": -0.2,
            "min_bid_ask_ratio": 1.1 # min_bid_ask_ratio will be skipped in backtest
        }
    },
    "RANGING": {
        "description": "اتجاه عرضي/محايد (نقاط 0)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 18.0, "rel_vol": 0.2, "rsi_range": (45, 75), "roc": 0.05,
            "slope": 0.0, "min_rrr": 1.5, "min_volatility_pct": 0.25,
            "min_btc_correlation": -0.2, "min_bid_ask_ratio": 1.2 # min_bid_ask_ratio will be skipped in backtest
        }
    },
    "DOWNTREND": {
        "description": "اتجاه هابط (مراقبة الانعكاس)",
        "strategy": "REVERSAL",
        "filters": {
            "min_rrr": 2.0, "min_volatility_pct": 0.5, "min_btc_correlation": -0.5,
            "min_relative_volume": 1.5, "min_bid_ask_ratio": 1.5 # min_bid_ask_ratio will be skipped in backtest
        }
    },
    "STRONG_DOWNTREND": { "description": "اتجاه هابط قوي (التداول متوقف)", "strategy": "DISABLED", "filters": {} },
    "WEEKEND": {
        "description": "سيولة منخفضة (عطلة نهاية الأسبوع)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 17.0, "rel_vol": 0.2, "rsi_range": (40, 70), "roc": 0.1,
            "slope": 0.0, "min_rrr": 1.5, "min_volatility_pct": 0.25,
            "min_btc_correlation": -0.4, "min_bid_ask_ratio": 1.4 # min_bid_ask_ratio will be skipped in backtest
        }
    }
}

# ---------------------- الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_ANALYSIS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30 # Used for fetching data for features
BTC_SYMBOL: str = 'BTCUSDT'
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_PERIODS: List[int] = [21, 50, 200]
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5
BUY_CONFIDENCE_THRESHOLD = 0.80
USE_PEAK_FILTER: bool = True
PEAK_LOOKBACK_PERIOD: int = 50
PEAK_DISTANCE_THRESHOLD_PCT: float = 0.995
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.2

# Global client for Binance API (for historical data)
client: Optional[Client] = None
conn: Optional[psycopg2.extensions.connection] = None
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {} # Will be loaded once

# ---------------------- دوال قاعدة البيانات ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] Initializing database connection for backtest...")
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        separator = '&' if '?' in db_url_to_use else '?'
        db_url_to_use += f"{separator}sslmode=require"
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False # Ensure transactions are managed
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_signals_data (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        signal_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        ml_confidence DOUBLE PRECISION,
                        market_trend_15m TEXT,
                        market_trend_1h TEXT,
                        market_trend_4h TEXT,
                        filter_profile_name TEXT,
                        filter_adx DOUBLE PRECISION,
                        filter_rel_vol DOUBLE PRECISION,
                        filter_rsi_range_min DOUBLE PRECISION,
                        filter_rsi_range_max DOUBLE PRECISION,
                        filter_roc DOUBLE PRECISION,
                        filter_slope DOUBLE PRECISION,
                        filter_min_rrr DOUBLE PRECISION,
                        filter_min_volatility_pct DOUBLE PRECISION,
                        filter_min_btc_correlation DOUBLE PRECISION,
                        filter_min_bid_ask_ratio DOUBLE PRECISION,
                        filter_min_relative_volume DOUBLE PRECISION,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_backtest_signals_symbol_timestamp ON backtest_signals_data (symbol, signal_timestamp);")
            conn.commit()
            logger.info("✅ [DB] Backtest database schema is up-to-date.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during backtest DB initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect to the backtest database.")
            exit(1) # Exit if DB connection fails critically

def insert_backtest_result(data: Dict[str, Any]) -> None:
    if not conn:
        logger.error("❌ [DB Insert] No database connection for backtest results.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO backtest_signals_data (
                    symbol, signal_timestamp, ml_confidence,
                    market_trend_15m, market_trend_1h, market_trend_4h,
                    filter_profile_name, filter_adx, filter_rel_vol,
                    filter_rsi_range_min, filter_rsi_range_max, filter_roc,
                    filter_slope, filter_min_rrr, filter_min_volatility_pct,
                    filter_min_btc_correlation, filter_min_bid_ask_ratio,
                    filter_min_relative_volume
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                data['symbol'], data['signal_timestamp'], data['ml_confidence'],
                data['market_trend_15m'], data['market_trend_1h'], data['market_trend_4h'],
                data['filter_profile_name'], data.get('filter_adx'), data.get('filter_rel_vol'),
                data.get('filter_rsi_range_min'), data.get('filter_rsi_range_max'), data.get('filter_roc'),
                data.get('filter_slope'), data.get('filter_min_rrr'), data.get('filter_min_volatility_pct'),
                data.get('filter_min_btc_correlation'), data.get('filter_min_bid_ask_ratio'),
                data.get('filter_min_relative_volume')
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] Saved backtest result for {data['symbol']} at {data['signal_timestamp']}")
    except Exception as e:
        logger.error(f"❌ [DB Insert] Error inserting backtest result for {data['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()

# ---------------------- دوال Binance والبيانات (معدلة للاختبار الخلفي) ----------------------
def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [Exchange Info] Fetching exchange trading rules...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [Exchange Info] Loaded rules for {len(exchange_info_map)} symbols.")
    except Exception as e:
        logger.error(f"❌ [Exchange Info] Could not fetch exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        if not exchange_info_map: get_exchange_info_map()

        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] Bot will monitor {len(validated)} symbols for backtest.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] Error during symbol validation: {e}", exc_info=True)
        return []

def fetch_historical_data_for_backtest(symbol: str, interval: str, end_time: datetime, lookback_days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        # Calculate start_str based on end_time and lookback_days
        start_time_str = (end_time - timedelta(days=lookback_days)).strftime("%d %b %Y %H:%M:%S")
        end_time_str = end_time.strftime("%d %b %Y %H:%M:%S")

        # Binance API limits historical data fetches, typically to 1000 candles per request.
        # We need enough data for indicator calculations (e.g., EMA_200, ADX_14, etc.)
        # For a 15m interval, 30 days is (30 * 24 * 4) = 2880 candles.
        # We might need multiple fetches or just rely on the API's internal limit handling.
        # For simplicity, we'll use `get_historical_klines` which handles `start_str` and `end_str`.
        klines = client.get_historical_klines(symbol, interval, start_time_str, end_time_str)
        
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching historical data for {symbol} up to {end_time}: {e}")
        return None

# ---------------------- دوال حساب الميزات وتحديد الاتجاه (نسخة من c4.py) ----------------------
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
    
    for period in EMA_PERIODS:
        df_calc[f'ema_{period}'] = df_calc['close'].ewm(span=period, adjust=False).mean()

    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['ema_50']) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['ema_200']) - 1
    
    if btc_df is not None and not btc_df.empty:
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = df_calc['close'].pct_change().rolling(window=30).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
        
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['roc_acceleration'] = df_calc[f'roc_{MOMENTUM_PERIOD}'].diff()
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    df_calc['hour_of_day'] = df_calc.index.hour
    
    return df_calc.astype('float32', errors='ignore')

def determine_market_trend_score_backtest(current_timestamp: datetime) -> Dict[str, Any]:
    """
    Determines market trend score for backtesting based on a specific timestamp.
    """
    logger.debug(f"🧠 [Market Score Backtest] Updating multi-timeframe trend score for {current_timestamp}...")
    total_score = 0
    details = {}
    tf_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.5}

    for tf in TIMEFRAMES_FOR_TREND_ANALYSIS:
        days_to_fetch = 5 if tf == '15m' else (15 if tf == '1h' else 50)
        df = fetch_historical_data_for_backtest(BTC_SYMBOL, tf, current_timestamp, days_to_fetch)
        if df is None or len(df) < EMA_PERIODS[-1]:
            details[tf] = {"score": 0, "label": "غير واضح", "reason": "بيانات غير كافية"}
            continue

        for period in EMA_PERIODS:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Ensure we are using the candle *at or before* current_timestamp
        df_filtered = df[df.index <= current_timestamp]
        if df_filtered.empty:
            details[tf] = {"score": 0, "label": "غير واضح", "reason": "لا توجد بيانات عند الطابع الزمني"}
            continue

        last_candle = df_filtered.iloc[-1]
        close = last_candle['close']
        ema21 = last_candle['ema_21']
        ema50 = last_candle['ema_50']
        ema200 = last_candle['ema_200']

        tf_score = 0
        if close > ema21: tf_score += 1
        elif close < ema21: tf_score -= 1
        
        if ema21 > ema50: tf_score += 1
        elif ema21 < ema50: tf_score -= 1

        if ema50 > ema200: tf_score += 1
        elif ema50 < ema200: tf_score -= 1

        label = "محايد"
        if tf_score >= 2: label = "صاعد"
        elif tf_score <= -2: label = "هابط"
        
        details[tf] = {"score": tf_score, "label": label, "reason": f"EMA21:{ema21:.2f}, EMA50:{ema50:.2f}, EMA200:{ema200:.2f}"}
        total_score += tf_score * tf_weights[tf]
    
    final_score = round(total_score)
    
    trend_label = "محايد"
    if final_score >= 4: trend_label = "صاعد قوي"
    elif final_score >= 1: trend_label = "صاعد"
    elif final_score <= -4: trend_label = "هابط قوي"
    elif final_score <= -1: trend_label = "هابط"

    return {
        "trend_score": final_score,
        "trend_label": trend_label,
        "details_by_tf": details,
        "last_updated": current_timestamp.isoformat()
    }

def get_session_state_backtest(current_timestamp: datetime) -> Tuple[List[str], str, str]:
    """
    Determines session state for backtesting based on a specific timestamp.
    """
    sessions = {"London": (8, 17), "New York": (13, 22), "Tokyo": (0, 9)}
    active_sessions = []
    current_hour = current_timestamp.hour
    
    # Check if it's a weekend based on the backtest timestamp
    if current_timestamp.weekday() >= 5: # Saturday (5) or Sunday (6)
        return [], "WEEKEND", "سيولة منخفضة جدا (عطلة نهاية الأسبوع)"
    
    for session, (start, end) in sessions.items():
        if start <= current_hour < end:
            active_sessions.append(session)
    
    if "London" in active_sessions and "New York" in active_sessions:
        return active_sessions, "HIGH_LIQUIDITY", "سيولة عالية (تداخل لندن/نيويورك)"
    elif len(active_sessions) >= 1:
        return active_sessions, "NORMAL_LIQUIDITY", f"سيولة عادية ({', '.join(active_sessions)})"
    else:
        return [], "LOW_LIQUIDITY", "سيولة منخفضة (خارج أوقات الذروة)"

def analyze_market_and_create_dynamic_profile_backtest(current_timestamp: datetime, force_momentum: bool = False) -> Dict[str, Any]:
    """
    Generates dynamic filter profile for backtesting.
    """
    logger.debug(f"🔬 [Dynamic Filter Backtest] Generating profile for {current_timestamp}...")
    
    if force_momentum:
        logger.debug(" [OVERRIDE] Manual momentum strategy is active for backtest.")
        base_profile = FILTER_PROFILES["UPTREND"].copy()
        liquidity_desc = "استراتيجية الزخم مفروضة يدوياً"
    else:
        active_sessions, liquidity_state, liquidity_desc = get_session_state_backtest(current_timestamp)
        market_state = determine_market_trend_score_backtest(current_timestamp)
        market_label = market_state.get("trend_label", "محايد")

        profile_key = "RANGING" # Default
        if "صاعد قوي" in market_label: profile_key = "STRONG_UPTREND"
        elif "صاعد" in market_label: profile_key = "UPTREND"
        elif "هابط قوي" in market_label: profile_key = "STRONG_DOWNTREND"
        elif "هابط" in market_label: profile_key = "DOWNTREND"

        if liquidity_state == "WEEKEND":
            base_profile = FILTER_PROFILES["WEEKEND"].copy()
        else:
            base_profile = FILTER_PROFILES.get(profile_key, FILTER_PROFILES["RANGING"]).copy()

    dynamic_filter_profile = {
        "name": base_profile['description'],
        "description": liquidity_desc,
        "strategy": base_profile.get("strategy", "DISABLED"),
        "filters": base_profile.get("filters", {}),
        "last_updated": current_timestamp.isoformat(),
    }
    
    logger.debug(f"✅ [Dynamic Filter Backtest] Profile: '{base_profile['description']}' | Strategy: '{base_profile['strategy']}'")
    return dynamic_filter_profile

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        return ml_models_cache[model_name]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
    model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        logger.debug(f"⚠️ [ML Model] Model file not found: '{model_path}'.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            return model_bundle
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] Error loading model for {symbol}: {e}", exc_info=True)
        return None

class TradingStrategyBacktest:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_features(df_15m, btc_df)
            df_4h_features = calculate_features(df_4h, None)
            df_4h_features = df_4h_features.rename(columns=lambda c: f"{c}_4h", inplace=False)
            required_4h_cols = ['rsi_4h', 'price_vs_ema50_4h']
            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='outer')
            df_featured.fillna(method='ffill', inplace=True)
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] Feature engineering failed: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled_np = self.scaler.transform(last_row_ordered_df)
            features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)
            prediction = self.ml_model.predict(features_scaled_df)[0]
            if prediction != 1: return None # Only interested in buy signals
            
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)
            confidence = float(np.max(prediction_proba[0]))
            logger.debug(f"ℹ️ [{self.symbol}] ML Model predicted 'BUY' with {confidence:.2%} confidence.")
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] ML Signal Generation Error: {e}")
            return None

def find_crazy_reversal_signal(df_featured: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if len(df_featured) < 30: return None
        
        last_candle = df_featured.iloc[-1]
        # Keltner Channel values are not calculated in this backtest script's calculate_features
        # For simplicity in backtesting, we'll remove the Keltner check or assume it passes
        # if last_candle['close'] <= last_candle['kc_upper']: return None
        
        lookback_period = 25
        relevant_data = df_featured.iloc[-lookback_period:-1]
        
        if relevant_data.empty: return None

        price_low_idx = relevant_data['low'].idxmin()
        price_low_val = relevant_data.loc[price_low_idx, 'low']
        rsi_at_price_low = relevant_data.loc[price_low_idx, 'rsi']

        current_price_low = last_candle['low']
        if current_price_low <= price_low_val:
            return None

        current_rsi = last_candle['rsi']
        if current_rsi >= rsi_at_price_low:
            return None

        if last_candle['relative_volume'] < 1.5:
            return None

        logger.debug(f"✅ [CRAZY REVERSAL] Signal detected for {df_featured.name}!")
        return {
            "signal_type": "CRAZY_REVERSAL",
            "reason": f"Hidden Divergence, Volume Spike (RelVol: {last_candle['relative_volume']:.2f})"
        }

    except Exception as e:
        symbol_name = df_featured.name if hasattr(df_featured, 'name') else 'Unknown'
        logger.error(f"❌ [{symbol_name}] Error in find_crazy_reversal_signal: {e}")
        return None

def calculate_tp_sl(entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    if last_atr <= 0:
        return None
    tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
    sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
    return {'target_price': tp, 'stop_loss': sl, 'source': 'ATR_Fallback'}


class BacktestingEngine:
    def __init__(self, start_date: datetime, end_date: datetime, batch_size: int = 3):
        global client
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.validated_symbols: List[str] = []
        self.client = Client(API_KEY, API_SECRET)
        client = self.client # Set global client for helper functions
        init_db() # Initialize backtest DB
        get_exchange_info_map() # Load exchange info once
        self.validated_symbols = get_validated_symbols()
        if not self.validated_symbols:
            logger.critical("❌ لا توجد رموز صالحة للاختبار الخلفي. سيتم الإغلاق.")
            exit(1)
        
        # Filter symbols to only include those with available ML models
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
        self.symbols_with_models = []
        for symbol in self.validated_symbols:
            if os.path.exists(os.path.join(model_dir_path, f"{BASE_ML_MODEL_NAME}_{symbol}.pkl")):
                self.symbols_with_models.append(symbol)
        
        if not self.symbols_with_models:
            logger.critical("❌ لا توجد رموز مع نماذج تعلم آلة متاحة للاختبار الخلفي. سيتم الإغلاق.")
            exit(1)
        logger.info(f"✅ تم العثور على {len(self.symbols_with_models)} رمزًا مع نماذج تعلم آلة للاختبار الخلفي.")

    def run_backtest(self):
        logger.info(f"🚀 بدء الاختبار الخلفي من {self.start_date} إلى {self.end_date}...")
        
        # Generate timestamps for each 15-minute interval
        current_timestamp = self.start_date
        interval_delta = timedelta(minutes=15)
        
        while current_timestamp <= self.end_date:
            logger.info(f"--- معالجة الطابع الزمني: {current_timestamp.isoformat()} ---")
            
            # 1. Determine market trend and dynamic filter profile for current timestamp
            market_state = determine_market_trend_score_backtest(current_timestamp)
            dynamic_filter_profile = analyze_market_and_create_dynamic_profile_backtest(current_timestamp)
            
            active_strategy_type = dynamic_filter_profile.get("strategy")
            if not active_strategy_type or active_strategy_type == "DISABLED":
                logger.warning(f"🛑 التداول معطل بواسطة ملف التعريف: '{dynamic_filter_profile.get('name')}' في هذا الطابع الزمني. تخطي.")
                current_timestamp += interval_delta
                continue

            btc_data_for_features = fetch_historical_data_for_backtest(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, current_timestamp, SIGNAL_GENERATION_LOOKBACK_DAYS)
            if btc_data_for_features is None or btc_data_for_features.empty:
                logger.warning(f"⚠️ بيانات BTC غير كافية للطابع الزمني {current_timestamp}. تخطي.")
                current_timestamp += interval_delta
                continue
            btc_data_for_features['btc_returns'] = btc_data_for_features['close'].pct_change()

            # Shuffle symbols to process in batches
            symbols_to_process_in_cycle = list(self.symbols_with_models)
            # random.shuffle(symbols_to_process_in_cycle) # Optional: shuffle for more realistic simulation

            # Process symbols in batches
            for i in range(0, len(symbols_to_process_in_cycle), self.batch_size):
                batch_symbols = symbols_to_process_in_cycle[i:i + self.batch_size]
                logger.info(f"📦 معالجة الدفعة: {', '.join(batch_symbols)}")

                for symbol in batch_symbols:
                    try:
                        # Fetch historical data for the symbol up to the current backtest timestamp
                        df_15m = fetch_historical_data_for_backtest(symbol, SIGNAL_GENERATION_TIMEFRAME, current_timestamp, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_15m.empty:
                            logger.debug(f"⚠️ [{symbol}] لا توجد بيانات 15 دقيقة كافية للطابع الزمني {current_timestamp}. تخطي.")
                            continue
                        df_15m.name = symbol # For logging/debugging

                        # Calculate features for the symbol
                        df_features_with_indicators = calculate_features(df_15m, btc_data_for_features)
                        if df_features_with_indicators is None or df_features_with_indicators.empty:
                            logger.debug(f"⚠️ [{symbol}] فشل حساب الميزات للطابع الزمني {current_timestamp}. تخطي.")
                            continue
                        
                        # Ensure df_features_with_indicators has enough rows for .iloc[-1]
                        if len(df_features_with_indicators) < 1:
                            logger.debug(f"⚠️ [{symbol}] بيانات الميزات غير كافية بعد الحساب. تخطي.")
                            continue

                        # Simulate ML model signal generation
                        strategy_backtest = TradingStrategyBacktest(symbol)
                        if not all([strategy_backtest.ml_model, strategy_backtest.scaler, strategy_backtest.feature_names]):
                            logger.debug(f"⚠️ [{symbol}] نموذج التعلم الآلي أو البيانات غير متوفرة. تخطي.")
                            continue
                        
                        df_4h = fetch_historical_data_for_backtest(symbol, '4h', current_timestamp, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_4h is None or df_4h.empty:
                            logger.debug(f"⚠️ [{symbol}] لا توجد بيانات 4 ساعات كافية للطابع الزمني {current_timestamp}. تخطي.")
                            continue

                        df_features = strategy_backtest.get_features(df_15m, df_4h, btc_data_for_features)
                        if df_features is None or df_features.empty or len(df_features) < 1:
                            logger.debug(f"⚠️ [{symbol}] فشل دمج الميزات أو لا توجد بيانات كافية. تخطي.")
                            continue

                        ml_signal = strategy_backtest.generate_buy_signal(df_features)
                        
                        # FIX: Handle formatting for 'N/A' correctly
                        confidence_display = f"{ml_signal['confidence']:.2%}" if ml_signal and 'confidence' in ml_signal else "N/A"
                        
                        if ml_signal and ml_signal['confidence'] >= BUY_CONFIDENCE_THRESHOLD:
                            logger.info(f"💡 [{symbol}] تم توليد إشارة شراء بواسطة ML بثقة {confidence_display} عند {current_timestamp}.")
                            
                            # Get the latest features for filter values
                            last_features = df_features.iloc[-1]
                            
                            # Get current market trends for all timeframes
                            market_trend_15m = market_state['details_by_tf'].get('15m', {}).get('label', 'غير واضح')
                            market_trend_1h = market_state['details_by_tf'].get('1h', {}).get('label', 'غير واضح')
                            market_trend_4h = market_state['details_by_tf'].get('4h', {}).get('label', 'غير واضح')

                            # Extract filter values from the dynamic profile
                            filters_data = dynamic_filter_profile.get("filters", {})
                            
                            # Calculate TP/SL to get RRR, even if not used for actual trading in backtest
                            # We need an "entry_price" for this. For backtest, we take the last close price.
                            entry_price_for_calc = df_15m['close'].iloc[-1]
                            last_atr = last_features.get('atr', 0)
                            tp_sl_data = calculate_tp_sl(entry_price_for_calc, last_atr)

                            # Prepare data to save
                            backtest_result = {
                                'symbol': symbol,
                                'signal_timestamp': current_timestamp,
                                'ml_confidence': ml_signal['confidence'],
                                'market_trend_15m': market_trend_15m,
                                'market_trend_1h': market_trend_1h,
                                'market_trend_4h': market_trend_4h,
                                'filter_profile_name': dynamic_filter_profile['name'],
                                'filter_adx': filters_data.get('adx'),
                                'filter_rel_vol': filters_data.get('rel_vol'),
                                'filter_rsi_range_min': filters_data.get('rsi_range', [None, None])[0],
                                'filter_rsi_range_max': filters_data.get('rsi_range', [None, None])[1],
                                'filter_roc': filters_data.get('roc'),
                                'filter_slope': filters_data.get('slope'),
                                'filter_min_rrr': filters_data.get('min_rrr'),
                                'filter_min_volatility_pct': filters_data.get('min_volatility_pct'),
                                'filter_min_btc_correlation': filters_data.get('min_btc_correlation'),
                                'filter_min_bid_ask_ratio': filters_data.get('min_bid_ask_ratio'),
                                'filter_min_relative_volume': filters_data.get('min_relative_volume') # For REVERSAL
                            }
                            insert_backtest_result(backtest_result)
                            
                        else:
                            # Use the pre-calculated confidence_display
                            logger.debug(f"[{symbol}] لم يتم توليد إشارة شراء أو الثقة منخفضة ({confidence_display}).")

                    except Exception as e:
                        logger.error(f"❌ [خطأ في معالجة الرمز] {symbol} في الطابع الزمني {current_timestamp}: {e}", exc_info=True)
                    finally:
                        # Explicitly delete large objects to aid garbage collection
                        del df_15m, df_features_with_indicators, df_4h, df_features, strategy_backtest
                        gc.collect() # Collect garbage after each symbol

                logger.info(f"🧹 تم الانتهاء من دفعة الرموز. تحرير الذاكرة...")
                gc.collect() # Collect garbage after each batch

            current_timestamp += interval_delta
            # Add a small sleep to avoid hammering the API if running locally and quickly
            # time.sleep(0.1) 
            
        logger.info("✅ اكتمل الاختبار الخلفي.")

if __name__ == "__main__":
    # Define backtest period (2 days)
    # Adjust these dates as needed for your backtesting period
    end_date = datetime.now(timezone.utc) - timedelta(days=2) # End 2 days ago
    start_date = end_date - timedelta(days=2) # Start 4 days ago (2 days duration)

    # Ensure start_date and end_date are aligned to 15-minute intervals
    # This helps in consistent candle fetching
    start_date = start_date.replace(second=0, microsecond=0)
    if start_date.minute % 15 != 0:
        start_date = start_date - timedelta(minutes=start_date.minute % 15)
    
    end_date = end_date.replace(second=0, microsecond=0)
    if end_date.minute % 15 != 0:
        end_date = end_date - timedelta(minutes=end_date.minute % 15)
    
    backtesting_engine = BacktestingEngine(start_date=start_date, end_date=end_date, batch_size=3)
    backtesting_engine.run_backtest()
    
    if conn:
        conn.close()
        logger.info("👋 تم إغلاق اتصال قاعدة البيانات للاختبار الخلفي.")
    logger.info("👋 اكتمل تشغيل ملف الاختبار الخلفي.")
