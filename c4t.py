import os
import logging
import pickle
import time
import json
import psycopg2
import numpy as np
import pandas as pd
from binance.client import Client
from decouple import config
from tqdm import tqdm
from flask import Flask
from threading import Thread
from datetime import datetime, timedelta, timezone
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Any

# ==================================================================================================
# -------------------------------- إعدادات الاختبار الخلفي (الإصدار السادس) --------------------------------
# ==================================================================================================
# الفترة الزمنية للاختبار بالأيام
BACKTEST_PERIOD_DAYS: int = 120
# الإطار الزمني للشموع (يجب أن يطابق إطار تدريب النموذج)
TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
# اسم النموذج الأساسي الذي سيتم اختباره
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6_With_SR'
# فترة جلب البيانات يجب أن تكون أطول لتغطية حساب المؤشرات
DATA_FETCH_LOOKBACK_DAYS: int = BACKTEST_PERIOD_DAYS + 90 # (e.g., 120 + 90)

# --- محاكاة التكاليف الواقعية ---
COMMISSION_PERCENT: float = 0.075 # النسبة المئوية للعمولة لكل صفقة
SLIPPAGE_PERCENT: float = 0.04 # النسبة المئوية للانزلاق السعري
INITIAL_TRADE_AMOUNT_USDT: float = 10.0 # حجم الصفقة الأولي بالدولار

# --- معلمات استراتيجية تعلم الآلة (ML) ---
USE_ML_STRATEGY: bool = True
MODEL_CONFIDENCE_THRESHOLD: float = 0.65 # الحد الأدنى لثقة النموذج لتوليد إشارة

# --- معلمات استراتيجية الدعم والمقاومة (S/R) ---
USE_SR_FIB_STRATEGY: bool = True
SR_PROXIMITY_PERCENT: float = 0.003 # 0.3% - مدى القرب من مستوى الدعم لتفعيله
MINIMUM_SR_SCORE_FOR_SIGNAL: int = 50 # الحد الأدنى لقوة المستوى لتوليد إشارة

# --- فلاتر الإشارات العامة ---
MINIMUM_PROFIT_PERCENTAGE: float = 0.8 # الحد الأدنى للربح المتوقع لفتح صفقة
MINIMUM_RISK_REWARD_RATIO: float = 1.2 # الحد الأدنى لنسبة المخاطرة إلى العائد
MINIMUM_15M_VOLUME_USDT: float = 200_000 # الحد الأدنى لحجم التداول في شمعة 15 دقيقة

# --- معلمات تحديد الهدف ووقف الخسارة (Fallback) ---
ATR_SL_MULTIPLIER: float = 2.0
ATR_TP_MULTIPLIER: float = 2.5

# --- معلمات المؤشرات (مطابقة لملفات c4.py و ml.py) ---
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
BTC_SYMBOL = 'BTCUSDT'

# ==================================================================================================
# --------------------------------- إعدادات النظام والاتصال -------------------------------------
# ==================================================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_v6_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BacktesterV6_SR')

app = Flask(__name__)
@app.route('/')
def health_check():
    return "Backtester V6 service is running."

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}"); exit(1)

# --- تهيئة الاتصالات ---
client: Optional[Client] = None
conn: Optional[psycopg2.extensions.connection] = None

def init_connections():
    global client, conn
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل الاتصال: {e}"); exit(1)

    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
        logger.info("✅ [PostgreSQL] تم الاتصال بقاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [PostgreSQL] فشل الاتصال بقاعدة البيانات: {e}"); exit(1)


# ==================================================================================================
# --------------------------------- دوال مساعدة لجلب البيانات والمعالجة ------------------------------
# ==================================================================================================

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [Validation] Reading symbols from '{filename}'...")
    if not client: logger.error("Binance client not initialized."); return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path): logger.error(f"File not found: {file_path}"); return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        validated = sorted(list(formatted.intersection(active_symbols)))
        logger.info(f"✅ [Validation] Found {len(validated)} symbols to backtest.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] Error: {e}", exc_info=True); return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: logger.warning(f"⚠️ No historical data for {symbol} for the period."); return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_localize('UTC') # Make index timezone-aware
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}"); return None

def fetch_sr_levels(symbol: str) -> pd.DataFrame:
    if not conn:
        logger.warning(f"⚠️ [{symbol}] Cannot fetch S/R levels, DB connection unavailable.")
        return pd.DataFrame()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s", (symbol,)
            )
            levels = cur.fetchall()
            return pd.DataFrame(levels) if levels else pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error fetching S/R levels: {e}"); conn.rollback(); return pd.DataFrame()

def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if not conn: logger.error("Cannot load model, DB connection is not available."); return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s;", (model_name,))
            record = cur.fetchone()
            if record and 'model_data' in record:
                model_bundle = pickle.loads(record['model_data'])
                if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                    return model_bundle
                else: logger.error(f"❌ [Model] Incomplete model bundle for '{model_name}'.")
            else: logger.warning(f"⚠️ [Model] Model '{model_name}' not found in DB.")
        return None
    except Exception as e:
        logger.error(f"❌ [Model] Error loading model '{model_name}' from DB: {e}"); conn.rollback(); return None

# ==================================================================================================
# ------------------------------- دوال هندسة الميزات (مطابقة للإصدار السادس) ----------------------------
# ==================================================================================================
def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        df['dist_to_support'] = 0.0; df['dist_to_resistance'] = 0.0
        df['score_of_support'] = 0.0; df['score_of_resistance'] = 0.0
        return df

    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence')]['level_price'].sort_values().to_numpy()
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence')]['level_price'].sort_values().to_numpy()
    support_scores = pd.Series(sr_levels_df['score'].values, index=sr_levels_df['level_price']).to_dict()
    resistance_scores = pd.Series(sr_levels_df['score'].values, index=sr_levels_df['level_price']).to_dict()

    def get_sr_info(price):
        dist_support, score_support, dist_resistance, score_resistance = 1.0, 0.0, 1.0, 0.0
        if supports.size > 0:
            idx = np.searchsorted(supports, price, side='right') - 1
            if idx >= 0:
                nearest_support_price = supports[idx]
                dist_support = (price - nearest_support_price) / price if price > 0 else 0
                score_support = support_scores.get(nearest_support_price, 0)
        if resistances.size > 0:
            idx = np.searchsorted(resistances, price, side='left')
            if idx < len(resistances):
                nearest_resistance_price = resistances[idx]
                dist_resistance = (nearest_resistance_price - price) / price if price > 0 else 0
                score_resistance = resistance_scores.get(nearest_resistance_price, 0)
        return dist_support, score_support, dist_resistance, score_resistance

    results = df['close'].apply(get_sr_info)
    df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = pd.DataFrame(results.tolist(), index=df.index)
    return df

def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    op, hi, lo, cl = df['open'], df['high'], df['low'], df['close']
    body = abs(cl - op); candle_range = hi - lo; candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    df['candlestick_pattern'] = 0
    df.loc[(body / candle_range) < 0.05, 'candlestick_pattern'] = 3
    df.loc[(body > candle_range * 0.1) & (lower_wick >= body * 2) & (upper_wick < body), 'candlestick_pattern'] = 2
    df.loc[(body > candle_range * 0.1) & (upper_wick >= body * 2) & (lower_wick < body), 'candlestick_pattern'] = -2
    df.loc[(cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1)), 'candlestick_pattern'] = 1
    df.loc[(cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1)), 'candlestick_pattern'] = -1
    df.loc[(cl > op) & (body / candle_range > 0.95), 'candlestick_pattern'] = 4
    df.loc[(op > cl) & (body / candle_range > 0.95), 'candlestick_pattern'] = -4
    return df

def prepare_data_for_backtest(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: return None
    df_calc = df_15m.copy()
    high_low = df_calc['high'] - df_calc['low']; high_close = (df_calc['high'] - df_calc['close'].shift()).abs(); low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean(); loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean(); ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0; df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1; df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean(); std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2); lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    rsi_val = df_calc['rsi']; min_rsi = rsi_val.rolling(window=STOCH_RSI_PERIOD).min(); max_rsi = rsi_val.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi_val - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100; df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['market_condition'] = 0; df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1; df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean(); ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1; df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc = calculate_candlestick_patterns(df_calc)
    df_calc = calculate_sr_features(df_calc, sr_levels) # إضافة ميزات الدعم والمقاومة
    delta_4h = df_4h['close'].diff(); gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean(); loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9)))); ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean(); df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
    mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
    df_featured = df_calc.join(mtf_features)
    df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].ffill()
    return df_featured.dropna()

# ==================================================================================================
# ----------------------------- محرك الاختبار الخلفي (مُحدَّث بالكامل) ------------------------------
# ==================================================================================================
def validate_and_filter_signal(signal: Dict, last_candle_data: pd.Series) -> Optional[Dict]:
    entry_price, target_price, stop_loss = signal['entry_price'], signal['target_price'], signal['stop_loss']
    
    last_15m_volume_usdt = last_candle_data['volume'] * last_candle_data['close']
    if last_15m_volume_usdt < MINIMUM_15M_VOLUME_USDT: return None # فلتر السيولة
        
    if not all([target_price, stop_loss]) or target_price <= entry_price or stop_loss >= entry_price: return None # فلتر منطقية الهدف والوقف

    potential_profit_pct = ((target_price / entry_price) - 1) * 100
    if potential_profit_pct < MINIMUM_PROFIT_PERCENTAGE: return None # فلتر الحد الأدنى للربح

    potential_risk = entry_price - stop_loss
    if potential_risk <= 0: return None
    
    risk_reward_ratio = (target_price - entry_price) / potential_risk
    if risk_reward_ratio < MINIMUM_RISK_REWARD_RATIO: return None # فلتر نسبة المخاطرة للعائد
    
    signal['signal_details']['risk_reward_ratio'] = f"{risk_reward_ratio:.2f}:1"
    signal['signal_details']['last_15m_volume_usdt'] = f"${last_15m_volume_usdt:,.0f}"
    return signal

def run_backtest_for_symbol(symbol: str, df_full: pd.DataFrame, sr_levels_df: pd.DataFrame, model_bundle: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trades = []
    
    if USE_ML_STRATEGY and model_bundle:
        model, scaler, feature_names = model_bundle['model'], model_bundle['scaler'], model_bundle['feature_names']
        missing = [col for col in feature_names if col not in df_full.columns]
        if missing: logger.error(f"Missing features {missing} for {symbol}. Skipping ML."); return []
        
        features_df = df_full[feature_names]
        # FIX: Recreate DataFrame after scaling to prevent UserWarning
        features_scaled_np = scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(features_scaled_np, columns=feature_names, index=features_df.index)
        
        try:
            class_1_index = list(model.classes_).index(1)
            predictions = model.predict_proba(features_scaled_df)[:, class_1_index]
            df_full['prediction'] = predictions
        except Exception as e:
            logger.error(f"Could not get predictions for {symbol}: {e}"); df_full['prediction'] = 0
    else:
        df_full['prediction'] = 0

    in_trade = False
    trade_details = {}
    
    for i in range(len(df_full)):
        current_candle = df_full.iloc[i]
        candle_time = current_candle.name

        # --- التحقق من إغلاق الصفقات المفتوحة ---
        if in_trade:
            exit_price, exit_reason = None, None
            if current_candle['high'] >= trade_details['tp']:
                exit_price, exit_reason = trade_details['tp'], 'TP Hit'
            elif current_candle['low'] <= trade_details['sl']:
                exit_price, exit_reason = trade_details['sl'], 'SL Hit'
            
            if exit_price:
                trade_details.update({
                    'exit_price': exit_price, 'exit_reason': exit_reason,
                    'exit_time': candle_time, 'duration_candles': i - trade_details['entry_index']
                })
                trades.append(trade_details)
                in_trade = False; trade_details = {}
            continue

        # --- التحقق من فتح صفقات جديدة ---
        final_signal = None
        
        # --- استراتيجية تعلم الآلة ---
        if USE_ML_STRATEGY and current_candle['prediction'] >= MODEL_CONFIDENCE_THRESHOLD:
            entry_price = current_candle['close']
            ml_signal = {'symbol': symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'entry_price': entry_price, 'signal_details': {'ML_Probability_Buy': f"{current_candle['prediction']:.2%}"}}
            
            new_target, new_stop_loss = None, None
            if not sr_levels_df.empty:
                supports = sr_levels_df[(sr_levels_df['level_type'].str.contains('support|poc')) & (sr_levels_df['level_price'] < entry_price)]
                if not supports.empty:
                    strongest_support = supports.loc[supports['score'].idxmax()]
                    new_stop_loss = strongest_support['level_price'] * 0.99
                    ml_signal['signal_details']['StopLoss_Reason'] = f"S/R Support (Score: {strongest_support['score']:.0f})"
                resistances = sr_levels_df[(sr_levels_df['level_type'].str.contains('resistance|poc')) & (sr_levels_df['level_price'] > entry_price)]
                if not resistances.empty:
                    closest_resistance = resistances.loc[resistances['level_price'].idxmin()]
                    new_target = closest_resistance['level_price'] * 0.998
                    ml_signal['signal_details']['Target_Reason'] = f"S/R Resistance (Score: {closest_resistance['score']:.0f})"

            atr_value = current_candle['atr']
            ml_signal['stop_loss'] = new_stop_loss if new_stop_loss else entry_price - (atr_value * ATR_SL_MULTIPLIER)
            ml_signal['target_price'] = new_target if new_target else entry_price + (atr_value * ATR_TP_MULTIPLIER)
            final_signal = validate_and_filter_signal(ml_signal, current_candle)

        # --- استراتيجية الدعم والمقاومة ---
        if not final_signal and USE_SR_FIB_STRATEGY and not sr_levels_df.empty:
            entry_price = current_candle['close']
            strong_supports = sr_levels_df[(sr_levels_df['score'] >= MINIMUM_SR_SCORE_FOR_SIGNAL) & (sr_levels_df['level_type'].str.contains('support|poc')) & (sr_levels_df['level_price'] < entry_price)]
            if not strong_supports.empty:
                # FIX: Use .loc instead of .iloc with the index label from idxmin()
                closest_support_idx = strong_supports['level_price'].sub(entry_price).abs().idxmin()
                closest_support = strong_supports.loc[closest_support_idx]
                
                if (entry_price - closest_support['level_price']) / closest_support['level_price'] <= SR_PROXIMITY_PERCENT:
                    resistances_above = sr_levels_df[(sr_levels_df['level_type'].str.contains('resistance|poc')) & (sr_levels_df['level_price'] > entry_price)]
                    if not resistances_above.empty:
                        closest_resistance = resistances_above.loc[resistances_above['level_price'].idxmin()]
                        sr_signal = {
                            'symbol': symbol, 'strategy_name': 'SR_Fib_Strategy', 'entry_price': entry_price,
                            'stop_loss': closest_support['level_price'] * 0.99, 'target_price': closest_resistance['level_price'] * 0.998,
                            'signal_details': {'trigger_level_info': f"Support at {closest_support['level_price']:.8g} (Score: {closest_support['score']:.0f})"}
                        }
                        final_signal = validate_and_filter_signal(sr_signal, current_candle)

        # --- إذا تم تأكيد الإشارة، افتح الصفقة ---
        if final_signal:
            in_trade = True
            trade_details = {
                'symbol': symbol, 'entry_time': candle_time, 'entry_price': final_signal['entry_price'],
                'entry_index': i, 'tp': final_signal['target_price'], 'sl': final_signal['stop_loss'],
                'strategy_type': final_signal['strategy_name'], 'details': final_signal['signal_details']
            }

    return trades

# ==================================================================================================
# ------------------------------------ التقرير والوظيفة الرئيسية -------------------------------------
# ==================================================================================================
def generate_report(all_trades: List[Dict[str, Any]]):
    if not all_trades: logger.warning("No trades were executed during the backtest."); return

    df_trades = pd.DataFrame(all_trades)
    df_trades['entry_price_adj'] = df_trades['entry_price'] * (1 + SLIPPAGE_PERCENT / 100)
    df_trades['exit_price_adj'] = df_trades['exit_price'] * (1 - SLIPPAGE_PERCENT / 100)
    df_trades['pnl_pct_raw'] = ((df_trades['exit_price_adj'] / df_trades['entry_price_adj']) - 1) * 100
    entry_cost = INITIAL_TRADE_AMOUNT_USDT
    exit_value = entry_cost * (1 + df_trades['pnl_pct_raw'] / 100)
    commission_entry = entry_cost * (COMMISSION_PERCENT / 100); commission_exit = exit_value * (COMMISSION_PERCENT / 100)
    df_trades['commission_total'] = commission_entry + commission_exit
    df_trades['pnl_usdt_net'] = (exit_value - entry_cost) - df_trades['commission_total']
    df_trades['pnl_pct_net'] = (df_trades['pnl_usdt_net'] / INITIAL_TRADE_AMOUNT_USDT) * 100

    total_trades = len(df_trades); winning_trades = df_trades[df_trades['pnl_usdt_net'] > 0]; losing_trades = df_trades[df_trades['pnl_usdt_net'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_net_pnl = df_trades['pnl_usdt_net'].sum()
    gross_profit = winning_trades['pnl_usdt_net'].sum(); gross_loss = abs(losing_trades['pnl_usdt_net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_win = winning_trades['pnl_usdt_net'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl_usdt_net'].mean()) if len(losing_trades) > 0 else 0
    risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
    
    report_str = f"""
================================================================================
📈 BACKTESTING REPORT: {BASE_ML_MODEL_NAME}
Period: Last {BACKTEST_PERIOD_DAYS} days ({TIMEFRAME} + {HIGHER_TIMEFRAME} MTF)
Costs: {COMMISSION_PERCENT}% commission/trade, {SLIPPAGE_PERCENT}% slippage
================================================================================
--- Net Performance (After Costs) ---
Total Net PnL: ${total_net_pnl:,.2f}
Total Trades: {total_trades}
Win Rate: {win_rate:.2f}%
Profit Factor: {profit_factor:.2f}
Average Risk/Reward Ratio: {risk_reward_ratio:.2f}:1

--- Averages (Net) ---
Average Winning Trade: ${avg_win:,.2f}
Average Losing Trade: -${avg_loss:,.2f}

--- Totals (Net) ---
Gross Profit: ${gross_profit:,.2f} ({len(winning_trades)} trades)
Gross Loss: -${gross_loss:,.2f} ({len(losing_trades)} trades)
Total Commissions Paid: ${df_trades['commission_total'].sum():,.2f}
================================================================================
"""
    logger.info(report_str)
    
    # --- إحصائيات حسب الاستراتيجية ---
    if 'strategy_type' in df_trades.columns:
        strategy_stats = df_trades.groupby('strategy_type')['pnl_usdt_net'].agg(['sum', 'count', lambda x: (x > 0).sum() / len(x) * 100])
        strategy_stats.columns = ['Total PnL ($)', 'Trade Count', 'Win Rate (%)']
        logger.info("\n--- Performance by Strategy ---\n" + strategy_stats.to_string(float_format="%.2f"))

    try:
        if not os.path.exists('reports'): os.makedirs('reports')
        report_filename = os.path.join('reports', f"backtest_report_{BASE_ML_MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_trades.to_csv(report_filename, index=False)
        logger.info(f"\n✅ Full trade log saved to: {report_filename}\n================================================================================\n")
    except Exception as e:
        logger.error(f"Could not save report to CSV: {e}")

def start_backtesting_job():
    logger.info("🚀 Starting V6 backtesting job...")
    init_connections()
    time.sleep(2)

    symbols_to_test = get_validated_symbols()
    if not symbols_to_test: logger.critical("❌ No valid symbols. Backtesting job aborted."); return

    all_trades = []

    logger.info(f"ℹ️ [BTC Data] Fetching historical data for {BTC_SYMBOL}...")
    btc_data = fetch_historical_data(BTC_SYMBOL, TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
    if btc_data is None: logger.critical("❌ Failed to fetch BTC data. Cannot proceed."); return
    btc_data['btc_returns'] = btc_data['close'].pct_change()
    logger.info("✅ [BTC Data] Successfully fetched and processed BTC data.")

    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        if symbol == BTC_SYMBOL: continue
        
        try:
            model_bundle = load_ml_model_bundle_from_db(symbol) if USE_ML_STRATEGY else None
            
            df_15m = fetch_historical_data(symbol, TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
            if df_15m is None or df_4h is None: continue

            sr_levels_df = fetch_sr_levels(symbol)
            
            df_featured = prepare_data_for_backtest(df_15m, df_4h, btc_data, sr_levels_df)
            if df_featured is None or df_featured.empty:
                logger.warning(f"⚠️ Could not generate features for {symbol}. Skipping."); continue

            backtest_start_date = datetime.now(timezone.utc) - timedelta(days=BACKTEST_PERIOD_DAYS)
            df_test_period = df_featured[df_featured.index >= backtest_start_date].copy()

            if df_test_period.empty: continue
            
            trades = run_backtest_for_symbol(symbol, df_test_period, sr_levels_df, model_bundle)
            if trades: all_trades.extend(trades)
        
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while processing {symbol}: {e}", exc_info=True)
            
        time.sleep(0.2) # To avoid hitting API limits aggressively

    generate_report(all_trades)
    
    if conn: conn.close()
    logger.info("👋 Backtesting job finished. The web service will remain active.")


# ==================================================================================================
# -------------------------------------------- التنفيذ ---------------------------------------------
# ==================================================================================================
if __name__ == "__main__":
    backtest_thread = Thread(target=start_backtesting_job)
    backtest_thread.daemon = True
    backtest_thread.start()

    port = int(os.environ.get("PORT", 10002))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
