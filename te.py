# -*- coding: utf-8 -*-
import os
import pickle
import warnings
import gc
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from binance.client import Client
from decouple import config

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعدادات الاختبار الخلفي ----------------------
print("--- بدء إعدادات الاختبار الخلفي (فترة اختبار محددة) ---")

# --- تحميل متغيرات البيئة ---
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    client = Client(API_KEY, API_SECRET)
    print("✅ تم تحميل متغيرات البيئة (API Keys, DB_URL) بنجاح.")
except Exception as e:
    print(f"❌ لم يتم العثور على المتغيرات المطلوبة (BINANCE_API_KEY, BINANCE_API_SECRET, DATABASE_URL) في ملف .env. الخطأ: {e}")
    exit()

# --- إعدادات المحاكاة ---
INITIAL_BALANCE_USDT = 1000.0
TRADE_AMOUNT_USDT = 100.0
FEE_PERCENT = 0.001
SLIPPAGE_PERCENT = 0.0005
MAX_OPEN_TRADES = 5

# --- إعدادات الفترة الزمنية ---
# فترة 90 يومًا للاختبار
BACKTEST_DAYS = 90
# فترة 90 يومًا "لتسخين" المؤشرات قبل بدء الاختبار
LOOKBACK_DAYS = 90

# --- إعدادات الاستراتيجية (يجب أن تتطابق مع ملف البوت) ---
BASE_ML_MODEL_NAME = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER = 'V7'
SIGNAL_GENERATION_TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '4h'
MODEL_CONFIDENCE_THRESHOLD = 0.70
USE_DYNAMIC_SL_TP = True
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
MIN_PROFIT_PERCENTAGE_FILTER = 1.0
BTC_SYMBOL = 'BTCUSDT'

# --- ثوابت المؤشرات (يجب أن تتطابق مع ملف البوت) ---
ADX_PERIOD = 14
BBANDS_PERIOD = 20
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
EMA_SLOW_PERIOD = 200
EMA_FAST_PERIOD = 50
BTC_CORR_PERIOD = 30
STOCH_RSI_PERIOD = 14
STOCH_K = 3
STOCH_D = 3
REL_VOL_PERIOD = 30

# ---------------------- دوال قاعدة البيانات ----------------------
conn = None

def init_db():
    global conn
    try:
        print("   - جاري الاتصال بقاعدة البيانات...")
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        print("   - ✅ تم الاتصال بقاعدة البيانات بنجاح.")
        return True
    except Exception as e:
        print(f"   - ❌ فشل الاتصال بقاعدة البيانات: {e}")
        return False

def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
            if not levels: return pd.DataFrame()
            return pd.DataFrame(levels)
    except Exception as e:
        print(f"   - ❌ [DB] لم يتمكن من جلب مستويات الدعم/المقاومة لـ {symbol}: {e}")
        return pd.DataFrame()

def fetch_ichimoku_features_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    if not conn: return pd.DataFrame()
    query = """
        SELECT timestamp, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        FROM ichimoku_features
        WHERE symbol = %s AND timeframe = %s
        ORDER BY timestamp;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol, timeframe))
            features = cur.fetchall()
            if not features: return pd.DataFrame()
            df_ichimoku = pd.DataFrame(features)
            df_ichimoku['timestamp'] = pd.to_datetime(df_ichimoku['timestamp'], utc=True)
            df_ichimoku.set_index('timestamp', inplace=True)
            return df_ichimoku
    except Exception as e:
        print(f"   - ❌ [DB] لم يتمكن من جلب بيانات إيشيموكو لـ {symbol}: {e}")
        return pd.DataFrame()

# ---------------------- دوال جلب البيانات وحساب المؤشرات ----------------------

def fetch_historical_data(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
    print(f"   - جاري جلب بيانات {symbol} على إطار {interval}...")
    try:
        klines = client.get_historical_klines(symbol, interval, start_dt.strftime("%d %b %Y %H:%M:%S"), end_dt.strftime("%d %b %Y %H:%M:%S"))
        if not klines:
            print(f"   - ⚠️ لم يتم العثور على بيانات لـ {symbol}.")
            return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        numeric_cols = {'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        print(f"   - ✅ تم جلب {len(df)} شمعة لـ {symbol}.")
        return df.dropna()
    except Exception as e:
        print(f"   - ❌ خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df_patterns = df.copy()
    op, hi, lo, cl = df_patterns['open'], df_patterns['high'], df_patterns['low'], df_patterns['close']
    body = abs(cl - op)
    candle_range = hi - lo
    candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    df_patterns['candlestick_pattern'] = 0
    is_bullish_marubozu = (cl > op) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1)
    is_bearish_marubozu = (op > cl) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1)
    is_bullish_engulfing = (cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1))
    is_bearish_engulfing = (cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1))
    is_hammer = (body > candle_range * 0.1) & (lower_wick >= body * 2) & (upper_wick < body)
    is_shooting_star = (body > candle_range * 0.1) & (upper_wick >= body * 2) & (lower_wick < body)
    is_doji = (body / candle_range) < 0.05
    df_patterns.loc[is_doji, 'candlestick_pattern'] = 3
    df_patterns.loc[is_hammer, 'candlestick_pattern'] = 2
    df_patterns.loc[is_shooting_star, 'candlestick_pattern'] = -2
    df_patterns.loc[is_bullish_engulfing, 'candlestick_pattern'] = 1
    df_patterns.loc[is_bearish_engulfing, 'candlestick_pattern'] = -1
    df_patterns.loc[is_bullish_marubozu, 'candlestick_pattern'] = 4
    df_patterns.loc[is_bearish_marubozu, 'candlestick_pattern'] = -4
    return df_patterns

def calculate_ichimoku_based_features(df: pd.DataFrame) -> pd.DataFrame:
    df['price_vs_tenkan'] = (df['close'] - df['tenkan_sen']) / df['tenkan_sen']
    df['price_vs_kijun'] = (df['close'] - df['kijun_sen']) / df['kijun_sen']
    df['tenkan_vs_kijun'] = (df['tenkan_sen'] - df['kijun_sen']) / df['kijun_sen']
    df['price_vs_kumo_a'] = (df['close'] - df['senkou_span_a']) / df['senkou_span_a']
    df['price_vs_kumo_b'] = (df['close'] - df['senkou_span_b']) / df['senkou_span_b']
    df['kumo_thickness'] = (df['senkou_span_a'] - df['senkou_span_b']).abs() / df['close']
    kumo_high = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    kumo_low = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    df['price_above_kumo'] = (df['close'] > kumo_high).astype(int)
    df['price_below_kumo'] = (df['close'] < kumo_low).astype(int)
    df['price_in_kumo'] = ((df['close'] >= kumo_low) & (df['close'] <= kumo_high)).astype(int)
    df['chikou_above_kumo'] = (df['chikou_span'] > kumo_high).astype(int)
    df['chikou_below_kumo'] = (df['chikou_span'] < kumo_low).astype(int)
    df['tenkan_kijun_cross'] = 0
    cross_up = (df['tenkan_sen'].shift(1) < df['kijun_sen'].shift(1)) & (df['tenkan_sen'] > df['kijun_sen'])
    cross_down = (df['tenkan_sen'].shift(1) > df['kijun_sen'].shift(1)) & (df['tenkan_sen'] < df['kijun_sen'])
    df.loc[cross_up, 'tenkan_kijun_cross'] = 1
    df.loc[cross_down, 'tenkan_kijun_cross'] = -1
    return df

def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        df['dist_to_support'] = 0.0; df['dist_to_resistance'] = 0.0
        df['score_of_support'] = 0.0; df['score_of_resistance'] = 0.0
        return df
    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    support_scores = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)].set_index('level_price')['score'].to_dict()
    resistance_scores = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)].set_index('level_price')['score'].to_dict()

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

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
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
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2); lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    rsi = df_calc['rsi']
    min_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).min(); max_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc = calculate_candlestick_patterns(df_calc)
    return df_calc.astype('float32', errors='ignore')

def load_ml_model_bundle_from_folder(symbol: str):
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        print(f"   - ⚠️ [نموذج تعلم الآلة] ملف النموذج '{model_path}' غير موجود للعملة {symbol}.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            print(f"   - ✅ [نموذج تعلم الآلة] تم تحميل النموذج '{model_name}' بنجاح.")
            return model_bundle
        else:
            print(f"   - ❌ [نموذج تعلم الآلة] حزمة النموذج في '{model_path}' غير مكتملة.")
            return None
    except Exception as e:
        print(f"   - ❌ [نموذج تعلم الآلة] خطأ في تحميل النموذج للعملة {symbol}: {e}")
        return None

# ---------------------- دالة جديدة لجلب العملات من ملف ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> list:
    """
    تقوم بقراءة قائمة رموز العملات المشفرة من ملف نصي، والتحقق من صحتها عبر Binance،
    وإرجاع قائمة بأزواج USDT القابلة للتداول.
    """
    print(f"--- ℹ️  جاري قراءة والتحقق من الرموز من '{filename}' ---")
    if not client:
        print("❌ كائن Binance client غير مهيأ. لا يمكن التحقق من الرموز.")
        return []
    try:
        # التأكد من أن مسار الملف صحيح، نسبة إلى موقع السكربت
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            print(f"❌ ملف الرموز غير موجود في '{file_path}'. الرجاء إنشاؤه.")
            # إنشاء ملف مثال
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# أضف رمز عملة واحد في كل سطر (مثال: BTC, ETH, SOL)\n")
                f.write("BTC\n")
                f.write("ETH\n")
                f.write("BNB\n")
                f.write("SOL\n")
                f.write("XRP\n")
                f.write("ADA\n")
            print(f"✅ تم إنشاء ملف مثال '{filename}' برموز افتراضية.")

        with open(file_path, 'r', encoding='utf-8') as f:
            # قراءة الرموز، تجاهل الأسطر الفارغة والتعليقات
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}

        # التأكد من أن جميع الرموز هي أزواج USDT
        formatted_symbols = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}

        # الحصول على جميع الرموز القابلة للتداول من Binance
        exchange_info = client.get_exchange_info()
        active_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}

        # إيجاد التقاطع بين الرموز من الملف والرموز النشطة على Binance
        validated = sorted(list(formatted_symbols.intersection(active_symbols)))

        if not validated:
            print(f"⚠️ لم يتم العثور على رموز صالحة وقابلة للتداول من '{filename}'.")
            return []

        print(f"✅ تم العثور على {len(validated)} عملة للاختبار الخلفي.")
        return validated
    except Exception as e:
        print(f"❌ حدث خطأ أثناء التحقق من الرموز: {e}")
        return []


# ---------------------- فئة الاستراتيجية والمنطق الرئيسي للاختبار ----------------------

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        if model_bundle:
            self.ml_model = model_bundle.get('model')
            self.scaler = model_bundle.get('scaler')
            self.feature_names = model_bundle.get('feature_names')
        else:
            self.ml_model, self.scaler, self.feature_names = None, None, None

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels_df: pd.DataFrame, ichimoku_df: pd.DataFrame) -> pd.DataFrame | None:
        try:
            df_featured = calculate_features(df_15m, btc_df)
            df_featured = calculate_sr_features(df_featured, sr_levels_df)
            
            if not ichimoku_df.empty:
                df_featured = df_featured.join(ichimoku_df, how='left')
                df_featured = calculate_ichimoku_based_features(df_featured)
            
            delta_4h = df_4h['close'].diff()
            gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
            ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
            df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
            mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
            df_featured = df_featured.join(mtf_features)
            df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
            
            for col in self.feature_names:
                if col not in df_featured.columns:
                    df_featured[col] = 0.0
            
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured[self.feature_names].dropna()

        except Exception as e:
            print(f"   - ❌ [{self.symbol}] فشل هندسة الميزات: {e}")
            return None

    def generate_signal(self, df_features: pd.DataFrame) -> dict | None:
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        if df_features.empty: return None
        
        last_row_df = df_features.iloc[[-1]]
        try:
            features_scaled = self.scaler.transform(last_row_df)
            prediction = self.ml_model.predict(features_scaled)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
            
            try:
                class_1_index = list(self.ml_model.classes_).index(1)
            except ValueError:
                return None
            
            prob_for_class_1 = prediction_proba[class_1_index]
            
            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                return {'signal': 'buy', 'confidence': prob_for_class_1}
            return None
        except Exception:
            return None

def run_backtest(symbol: str, start_date: datetime, end_date: datetime):
    print(f"\n{'='*20} بدء الاختبار الخلفي لـ {symbol} {'='*20}")
    
    # تاريخ بدء جلب البيانات = تاريخ بدء الاختبار - فترة الإحماء
    data_fetch_start = start_date - timedelta(days=LOOKBACK_DAYS)
    
    # 1. جلب البيانات
    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, data_fetch_start, end_date)
    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, data_fetch_start, end_date)
    btc_df = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, data_fetch_start, end_date)
    
    print("   - جاري جلب بيانات الدعم/المقاومة و إيشيموكو من قاعدة البيانات...")
    sr_levels = fetch_sr_levels_from_db(symbol)
    ichimoku_data = fetch_ichimoku_features_from_db(symbol, SIGNAL_GENERATION_TIMEFRAME)
    print(f"   - ✅ تم جلب {len(sr_levels)} مستوى دعم/مقاومة و {len(ichimoku_data)} سجل إيشيموكو.")

    if df_15m is None or df_4h is None or btc_df is None:
        print(f"❌ فشل جلب البيانات الأساسية لـ {symbol}. سيتم تخطي الاختبار.")
        return None

    btc_df['btc_returns'] = btc_df['close'].pct_change()
    
    # 2. إعداد الاستراتيجية وحساب المؤشرات
    strategy = TradingStrategy(symbol)
    if not strategy.ml_model:
        print(f"❌ لا يمكن المتابعة بدون نموذج تعلم آلة لـ {symbol}.")
        return None
        
    print("   - جاري حساب المؤشرات الفنية ودمج بيانات DB...")
    df_features = strategy.get_features(df_15m, df_4h, btc_df, sr_levels, ichimoku_data)
    if df_features is None:
        print(f"❌ فشل حساب المؤشرات لـ {symbol}.")
        return None
    print("   - ✅ تم حساب المؤشرات بنجاح.")

    df_main = df_15m.join(df_features, how='inner')
    
    # فلترة البيانات لتبدأ من تاريخ الاختبار الفعلي (بعد فترة الإحماء)
    df_main = df_main[df_main.index >= start_date]
    
    # 3. إعداد متغيرات المحاكاة
    balance = INITIAL_BALANCE_USDT
    open_trades = []
    trade_history = []
    
    # 4. بدء حلقة المحاكاة
    print(f"   - بدء المحاكاة على {len(df_main)} شمعة...")
    for i in range(len(df_main)):
        current_candle = df_main.iloc[i]
        current_price = current_candle['close']
        current_high = current_candle['high']
        current_low = current_candle['low']
        current_time = current_candle.name

        trades_to_close_indices = []
        for j, trade in enumerate(open_trades):
            if current_high >= trade['target_price']:
                exit_price = trade['target_price']
                exit_price_after_slippage = exit_price * (1 - SLIPPAGE_PERCENT)
                exit_fee = TRADE_AMOUNT_USDT * FEE_PERCENT
                profit = (exit_price_after_slippage - trade['entry_price_after_slippage']) * trade['amount_coins']
                balance += (TRADE_AMOUNT_USDT + profit - exit_fee)
                
                trade['exit_price'] = exit_price; trade['exit_time'] = current_time
                trade['profit_usdt'] = profit - exit_fee; trade['status'] = 'Target Hit'
                trade_history.append(trade); trades_to_close_indices.append(j)
                continue
            
            if current_low <= trade['stop_loss']:
                exit_price = trade['stop_loss']
                exit_price_after_slippage = exit_price * (1 - SLIPPAGE_PERCENT)
                exit_fee = TRADE_AMOUNT_USDT * FEE_PERCENT
                profit = (exit_price_after_slippage - trade['entry_price_after_slippage']) * trade['amount_coins']
                balance += (TRADE_AMOUNT_USDT + profit - exit_fee)

                trade['exit_price'] = exit_price; trade['exit_time'] = current_time
                trade['profit_usdt'] = profit - exit_fee; trade['status'] = 'Stop Loss'
                trade_history.append(trade); trades_to_close_indices.append(j)

        open_trades = [trade for j, trade in enumerate(open_trades) if j not in trades_to_close_indices]

        if len(open_trades) < MAX_OPEN_TRADES:
            signal = strategy.generate_signal(df_features.loc[[current_time]])
            
            if signal and signal['signal'] == 'buy':
                entry_price = current_price
                if USE_DYNAMIC_SL_TP:
                    atr_value = current_candle['atr']
                    stop_loss = entry_price - (atr_value * ATR_SL_MULTIPLIER)
                    target_price = entry_price + (atr_value * ATR_TP_MULTIPLIER)
                else:
                    stop_loss = entry_price * 0.985
                    target_price = entry_price * 1.02
                
                profit_percentage = ((target_price / entry_price) - 1) * 100
                if profit_percentage < MIN_PROFIT_PERCENTAGE_FILTER:
                    continue

                entry_price_after_slippage = entry_price * (1 + SLIPPAGE_PERCENT)
                entry_fee = TRADE_AMOUNT_USDT * FEE_PERCENT
                amount_in_coins = TRADE_AMOUNT_USDT / entry_price_after_slippage
                balance -= (TRADE_AMOUNT_USDT + entry_fee)

                new_trade = {
                    'symbol': symbol, 'entry_time': current_time, 'entry_price': entry_price,
                    'entry_price_after_slippage': entry_price_after_slippage, 'amount_coins': amount_in_coins,
                    'stop_loss': stop_loss, 'target_price': target_price,
                    'confidence': signal['confidence'], 'status': 'Open'
                }
                open_trades.append(new_trade)

    print("   - ✅ انتهت المحاكاة.")
    
    # 5. حساب وعرض النتائج
    final_balance = balance + (len(open_trades) * TRADE_AMOUNT_USDT)
    total_pnl = final_balance - INITIAL_BALANCE_USDT
    total_pnl_percent = (total_pnl / INITIAL_BALANCE_USDT) * 100
    wins = [t for t in trade_history if t['profit_usdt'] > 0]
    losses = [t for t in trade_history if t['profit_usdt'] <= 0]
    win_rate = (len(wins) / len(trade_history)) * 100 if trade_history else 0
    
    print("\n--- 📊 تقرير الأداء ---")
    print(f"العملة: {symbol}"); print(f"الفترة: من {start_date.date()} إلى {end_date.date()}"); print("-" * 30)
    print(f"الرصيد الأولي: ${INITIAL_BALANCE_USDT:,.2f}"); print(f"الرصيد النهائي: ${final_balance:,.2f}")
    print(f"إجمالي الربح/الخسارة: ${total_pnl:,.2f} ({total_pnl_percent:+.2f}%)"); print("-" * 30)
    print(f"إجمالي عدد الصفقات المغلقة: {len(trade_history)}"); print(f"الصفقات الرابحة: {len(wins)}")
    print(f"الصفقات الخاسرة: {len(losses)}"); print(f"نسبة النجاح (Win Rate): {win_rate:.2f}%")
    if wins: print(f"متوسط الربح للصفقة الرابحة: ${sum(t['profit_usdt'] for t in wins)/len(wins):.2f}")
    if losses: print(f"متوسط الخسارة للصفقة الخاسرة: ${sum(t['profit_usdt'] for t in losses)/len(losses):.2f}")
    print(f"عدد الصفقات التي لا تزال مفتوحة: {len(open_trades)}"); print("=" * 50)
    
    return trade_history

# ---------------------- نقطة انطلاق البرنامج ----------------------
if __name__ == "__main__":
    if not init_db():
        exit()

    # جلب الرموز من الملف المحلي بدلاً من قائمة ثابتة
    symbols_to_test = get_validated_symbols()

    if not symbols_to_test:
        print("❌ لا توجد عملات للاختبار. سيتم إنهاء الاختبار الخلفي.")
        exit()

    # --- تحديد التواريخ بدقة حسب طلبك ---
    # نهاية فترة الاختبار: قبل 90 يومًا من الآن
    end_date = datetime.now(timezone.utc) - timedelta(days=90)
    # بداية فترة الاختبار: قبل 180 يومًا من الآن
    start_date = datetime.now(timezone.utc) - timedelta(days=180)
    
    print(f"*** سيتم إجراء الاختبار على الفترة من {start_date.strftime('%Y-%m-%d')} إلى {end_date.strftime('%Y-%m-%d')} ***")
    
    all_results = {}
    for symbol in symbols_to_test:
        gc.collect()
        results = run_backtest(symbol, start_date, end_date)
        if results:
            all_results[symbol] = results
            
    if conn:
        conn.close()
        print("\n[DB] تم إغلاق الاتصال بقاعدة البيانات.")

    print("\n--- 🎉 انتهى الاختبار الخلفي لجميع العملات. ---")
