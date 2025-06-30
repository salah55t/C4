import os
import gc
import pickle
import logging
import warnings
import pandas as pd
import numpy as np
import psycopg2
from decouple import config
from binance.client import Client
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, timezone
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from tqdm import tqdm

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_v6_with_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester_V6_With_SR')

# ---------------------- إعداد متغيرات البيئة والثوابت ----------------------
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حرج في تحميل متغيرات البيئة: {e}")
    exit(1)

# --- إعدادات الاختبار الخلفي ---
INITIAL_CASH = 100.0  # القيمة الابتدائية للمحفظة في المحاكاة
TRADE_AMOUNT_USDT = 10.0 # مبلغ ثابت لكل صفقة
FEE = 0.001  # 0.1% رسوم التداول الفوري في Binance
SLIPPAGE = 0.0005 # 0.05% انزلاق سعري محاكى في الصفقات
COMMISSION = FEE + SLIPPAGE # العمولة المجمعة لمكتبة backtesting.py
BACKTEST_PERIOD_DAYS = 90 # فترة البيانات التاريخية للاختبار الخلفي

# --- التعديل الجديد: إزاحة فترة الاختبار الخلفي ---
# عدد الأيام لإزاحة فترة الاختبار لاختبار البيانات خارج العينة (out-of-sample).
# 0 = اختبار على أحدث البيانات (نفس بيانات التدريب).
# 90 = اختبار على البيانات من قبل 180 يومًا إلى قبل 90 يومًا (الفترة التي تسبق فترة التدريب).
OUT_OF_SAMPLE_OFFSET_DAYS = 90

# --- ثوابت الاستراتيجية والنموذج (يجب أن تتطابق مع البوت والمدرب) ---
BASE_ML_MODEL_NAME = 'LightGBM_Scalping_V6_With_SR'
SIGNAL_GENERATION_TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '4h'
BTC_SYMBOL = 'BTCUSDT'
MODEL_CONFIDENCE_THRESHOLD = 0.70 # عتبة الثقة للإشارة
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0

# --- إعدادات المؤشرات ---
ADX_PERIOD, BBANDS_PERIOD, RSI_PERIOD = 14, 20, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD, EMA_SLOW_PERIOD, EMA_FAST_PERIOD = 14, 200, 50
BTC_CORR_PERIOD, STOCH_RSI_PERIOD, STOCH_K, STOCH_D, REL_VOL_PERIOD = 30, 14, 3, 3, 30
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30
STOCH_RSI_OVERBOUGHT, STOCH_RSI_OVERSOLD = 80, 20

# متغيرات الاتصال العامة
conn = None
client = None

# ---------------------- دوال قاعدة البيانات ----------------------
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        logger.info("✅ [DB] تم تهيئة الاتصال بقاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}")
        exit(1)

def load_ml_model_bundle_from_db(symbol: str) -> dict | None:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if not conn:
        logger.error("[DB] الاتصال بقاعدة البيانات غير متاح.")
        return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                logger.info(f"✅ [ML Model] تم تحميل النموذج '{model_name}' للعملة {symbol} من قاعدة البيانات.")
                return model_bundle
        logger.warning(f"⚠️ [ML Model] النموذج '{model_name}' غير موجود في قاعدة البيانات للعملة {symbol}.")
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ في تحميل حزمة النموذج للعملة {symbol}: {e}")
        return None

def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        df = pd.read_sql(query, conn, params=(symbol,))
        if not df.empty:
            logger.info(f"✅ [S/R Levels] تم جلب {len(df)} من مستويات الدعم والمقاومة للعملة {symbol} من قاعدة البيانات.")
        return df
    except Exception as e:
        logger.error(f"❌ [S/R Levels] لا يمكن جلب مستويات الدعم والمقاومة للعملة {symbol}: {e}")
        return pd.DataFrame()

# ---------------------- جلب وإعداد البيانات ----------------------
def fetch_historical_data(symbol: str, interval: str, days: int, out_of_sample_period_days: int = 0) -> pd.DataFrame | None:
    """
    يجلب البيانات التاريخية.
    :param symbol: رمز العملة.
    :param interval: الإطار الزمني للشمعة (مثال: '15m').
    :param days: مدة فترة البيانات المراد جلبها (مثال: 90 يومًا).
    :param out_of_sample_period_days: منذ كم يوم يجب أن تنتهي فترة البيانات.
                                      0 يعني أنها تنتهي الآن (بيانات داخل العينة).
                                      90 يعني أنها تنتهي قبل 90 يومًا (بيانات خارج العينة).
    """
    if not client: return None
    try:
        now = datetime.now(timezone.utc)
        
        # حساب تاريخ نهاية فترة جلب البيانات
        end_dt = now - timedelta(days=out_of_sample_period_days)
        
        # حساب تاريخ بداية فترة جلب البيانات
        start_dt = end_dt - timedelta(days=days)

        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        # استخدام تاريخ البداية والنهاية في استدعاء الواجهة البرمجية
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        
        if not klines: return None
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # --- إعادة تسمية الأعمدة لتناسب backtesting.py ---
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية للعملة {symbol}: {e}")
        return None

# ---------------------- دوال هندسة الميزات (منسوخة من البوت/المدرب) ----------------------
def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    op, hi, lo, cl = df['Open'], df['High'], df['Low'], df['Close']
    body = abs(cl - op); candle_range = hi - lo
    candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    df['candlestick_pattern'] = 0
    is_bullish_engulfing = (cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1))
    is_bearish_engulfing = (cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1))
    df.loc[is_bullish_engulfing, 'candlestick_pattern'] = 1
    df.loc[is_bearish_engulfing, 'candlestick_pattern'] = -1
    return df

def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        df['dist_to_support'] = 0.0; df['dist_to_resistance'] = 0.0
        df['score_of_support'] = 0.0; df['score_of_resistance'] = 0.0
        return df
    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    support_scores = pd.Series(sr_levels_df['score'].values, index=sr_levels_df['level_price']).to_dict()
    def get_sr_info(price):
        dist_support, score_support, dist_resistance, score_resistance = 1.0, 0.0, 1.0, 0.0
        if supports.size > 0:
            idx = np.searchsorted(supports, price, side='right') - 1
            if idx >= 0:
                nearest_support = supports[idx]
                dist_support = (price - nearest_support) / price if price > 0 else 0
                score_support = support_scores.get(nearest_support, 0)
        if resistances.size > 0:
            idx = np.searchsorted(resistances, price, side='left')
            if idx < len(resistances):
                nearest_resistance = resistances[idx]
                dist_resistance = (nearest_resistance - price) / price if price > 0 else 0
                score_resistance = support_scores.get(nearest_resistance, 0)
        return dist_support, score_support, dist_resistance, score_resistance
    results = df['Close'].apply(get_sr_info)
    df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = pd.DataFrame(results.tolist(), index=df.index)
    return df

def create_all_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['High'] - df_calc['Low']; high_close = (df_calc['High'] - df_calc['Close'].shift()).abs(); low_close = (df_calc['Low'] - df_calc['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['High'].diff(); down_move = -df_calc['Low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['Close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['Close'].ewm(span=MACD_FAST, adjust=False).mean(); ema_slow = df_calc['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
    sma = df_calc['Close'].rolling(window=BBANDS_PERIOD).mean(); std_dev = df_calc['Close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2); lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    rsi_val = df_calc['rsi']
    min_rsi = rsi_val.rolling(window=STOCH_RSI_PERIOD).min(); max_rsi = rsi_val.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi_val - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['Volume'] / (df_calc['Volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['market_condition'] = 0
    df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1
    df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1
    ema_fast_trend = df_calc['Close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['Close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['Close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['Close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['Close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc = calculate_candlestick_patterns(df_calc)
    return df_calc

# ---------------------- فئة استراتيجية Backtesting.py ----------------------
class MLStrategy(Strategy):
    ml_model = None
    scaler = None
    feature_names = None

    def init(self):
        pass

    def next(self):
        if self.position:
            return

        try:
            features = self.data.df.loc[self.data.index[-1], self.feature_names]
            if features.isnull().any():
                return
        except (KeyError, IndexError):
            return

        features_df = pd.DataFrame([features])
        features_scaled_np = self.scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)
        
        prediction = self.ml_model.predict(features_scaled_df)[0]
        prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
        
        try:
            class_1_index = list(self.ml_model.classes_).index(1)
            prob_for_class_1 = prediction_proba[class_1_index]
        except ValueError:
            return

        if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
            current_atr = self.data.atr[-1]
            if pd.isna(current_atr) or current_atr == 0:
                return

            current_price = self.data.Close[-1]
            size_as_fraction = TRADE_AMOUNT_USDT / self.equity

            if size_as_fraction > 0 and size_as_fraction < 1:
                stop_loss_price = current_price - (current_atr * ATR_SL_MULTIPLIER)
                take_profit_price = current_price + (current_atr * ATR_TP_MULTIPLIER)
                self.buy(size=size_as_fraction, sl=stop_loss_price, tp=take_profit_price)

# ---------------------- كتلة التنفيذ الرئيسية ----------------------
def run_backtest():
    global client, conn
    logger.info("🚀 بدء الاختبار الخلفي المتقدم لاستراتيجية V6...")
    
    init_db()
    client = Client(API_KEY, API_SECRET)
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'crypto_list.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            symbols_to_test = [line.strip().upper() + "USDT" for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        logger.error("❌ ملف 'crypto_list.txt' غير موجود. سيتم الخروج.")
        return

    logger.info(f"ℹ️ جاري جلب بيانات BTC العالمية لفترة الاختبار الخلفي (Out-of-Sample: {OUT_OF_SAMPLE_OFFSET_DAYS} days)...")
    btc_df_full = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, BACKTEST_PERIOD_DAYS + 10, out_of_sample_period_days=OUT_OF_SAMPLE_OFFSET_DAYS)
    if btc_df_full is None:
        logger.critical("❌ فشل جلب بيانات BTC. لا يمكن المتابعة."); return
    btc_df_full['btc_returns'] = btc_df_full['Close'].pct_change()

    all_stats = []
    
    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        logger.info(f"\n--- ⏳ جاري معالجة الرمز: {symbol} ---")
        
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle:
            logger.warning(f"⚠️ تخطي {symbol}: لم يتم العثور على نموذج.")
            continue
        
        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, BACKTEST_PERIOD_DAYS, out_of_sample_period_days=OUT_OF_SAMPLE_OFFSET_DAYS)
        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, BACKTEST_PERIOD_DAYS * 5, out_of_sample_period_days=OUT_OF_SAMPLE_OFFSET_DAYS)
        
        if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty:
            logger.warning(f"⚠️ تخطي {symbol}: بيانات تاريخية غير كافية.")
            continue
            
        sr_levels = fetch_sr_levels_from_db(symbol)

        logger.info(f"هندسة الميزات لـ {symbol}...")
        
        data = create_all_features(df_15m, btc_df_full)
        
        delta_4h = df_4h['Close'].diff()
        gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
        ema_fast_4h = df_4h['Close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
        df_4h['price_vs_ema50_4h'] = (df_4h['Close'] / ema_fast_4h) - 1
        mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
        data = data.join(mtf_features, how='left').fillna(method='ffill')

        data = calculate_sr_features(data, sr_levels)
        
        data.dropna(inplace=True)
        
        if data.empty:
            logger.warning(f"⚠️ تخطي {symbol}: DataFrame فارغ بعد هندسة الميزات.")
            continue

        logger.info(f"جاري تشغيل الاختبار الخلفي لـ {symbol}...")
        bt = Backtest(
            data,
            MLStrategy,
            cash=INITIAL_CASH,
            commission=COMMISSION,
            exclusive_orders=True
        )
        
        stats = bt.run(
            ml_model=model_bundle['model'],
            scaler=model_bundle['scaler'],
            feature_names=model_bundle['feature_names']
        )
        
        logger.info(f"\n--- نتائج الاختبار الخلفي لـ {symbol} ---")
        print(stats)
        all_stats.append(stats)
        
        # bt.plot(filename=f"backtest_plot_{symbol}.html", open_browser=False)

        del data, df_15m, df_4h, sr_levels, model_bundle
        gc.collect()

    logger.info("\n\n--- 🏁 ملخص الاختبار الخلفي الشامل 🏁 ---")
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_df.index = [s['_strategy'] for s in all_stats] # Add symbol to index
        print(summary_df[[
            'Duration', 'Return [%]', 'Buy & Hold Return [%]', 'Win Rate [%]', 
            'Profit Factor', 'Sharpe Ratio', 'Sortino Ratio', '# Trades'
        ]])
        
        total_trades = summary_df['# Trades'].sum()
        total_profit = summary_df['Equity Final [$]'].sum() - summary_df['Equity Start [$]'].sum()
        avg_win_rate = summary_df['Win Rate [%]'].mean()
        avg_profit_factor = summary_df['Profit Factor'].mean()
        
        print("\n--- المقاييس المجمعة ---")
        print(f"إجمالي الرموز المختبرة: {len(summary_df)}")
        print(f"إجمالي عدد الصفقات: {total_trades}")
        print(f"إجمالي صافي الربح/الخسارة: ${total_profit:,.2f}")
        print(f"متوسط نسبة الربح: {avg_win_rate:.2f}%")
        print(f"متوسط عامل الربح: {avg_profit_factor:.2f}")
    else:
        print("لم يتم إكمال أي اختبار خلفي بنجاح.")
        
    if conn:
        conn.close()

if __name__ == "__main__":
    run_backtest()
