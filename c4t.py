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

# ---------------------- إعداد تسجيل الأحداث (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_v6_with_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester_V6_With_SR')

# ---------------------- إعداد البيئة والثوابت ----------------------
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حرج في تحميل متغيرات البيئة: {e}")
    exit(1)

# --- معلمات الاختبار الخلفي (Backtesting) ---
INITIAL_CASH = 100000.0  # القيمة الأولية للمحفظة للمحاكاة
TRADE_AMOUNT_USDT = 10.0 # مبلغ ثابت لكل صفقة
FEE = 0.001  # 0.1% رسوم التداول الفوري في بينانس
SLIPPAGE = 0.0005 # 0.05% انزلاق سعري محاكى في الصفقات
COMMISSION = FEE + SLIPPAGE # العمولة المجمعة لمكتبة backtesting.py
BACKTEST_PERIOD_DAYS = 90 # فترة البيانات التاريخية للاختبار الخلفي

# --- ثوابت الاستراتيجية والنموذج (يجب أن تتطابق مع البوت والمدرب) ---
BASE_ML_MODEL_NAME = 'LightGBM_Scalping_V6_With_SR'
SIGNAL_GENERATION_TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '4h'
BTC_SYMBOL = 'BTCUSDT'
MODEL_CONFIDENCE_THRESHOLD = 0.70 # عتبة ثقة الإشارة
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0

# --- معلمات المؤشرات ---
ADX_PERIOD, BBANDS_PERIOD, RSI_PERIOD = 14, 20, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD, EMA_SLOW_PERIOD, EMA_FAST_PERIOD = 14, 200, 50
BTC_CORR_PERIOD, STOCH_RSI_PERIOD, STOCH_K, STOCH_D, REL_VOL_PERIOD = 30, 14, 3, 3, 30
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30
STOCH_RSI_OVERBOUGHT, STOCH_RSI_OVERSOLD = 80, 20

# كائنات الاتصال العامة
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
                logger.info(f"✅ [ML Model] تم تحميل '{model_name}' للعملة {symbol} من قاعدة البيانات.")
                return model_bundle
        logger.warning(f"⚠️ [ML Model] لم يتم العثور على النموذج '{model_name}' في قاعدة البيانات للعملة {symbol}.")
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
        logger.error(f"❌ [S/R Levels] لم يتمكن من جلب مستويات الدعم والمقاومة للعملة {symbol}: {e}")
        return pd.DataFrame()

# ---------------------- جلب البيانات وإعدادها ----------------------
def fetch_historical_data(symbol: str, interval: str, days: int) -> pd.DataFrame | None:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # --- إعادة تسمية الأعمدة لتناسب مكتبة backtesting.py ---
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
    # تمرير المعلمات الخاصة بالاستراتيجية هنا
    ml_model = None
    scaler = None
    feature_names = None

    def init(self):
        # يتم استدعاء init مرة واحدة قبل بدء حلقة الاختبار الخلفي
        # نقوم بحساب المؤشرات مسبقًا، لذلك لا حاجة للكثير هنا.
        # البيانات التي يتم تمريرها إلى Backtest() ستحتوي بالفعل على جميع أعمدة الميزات.
        pass

    def next(self):
        # يتم استدعاء next لكل شمعة في البيانات التاريخية
        
        # إذا كانت هناك صفقة مفتوحة بالفعل، لا تفعل شيئًا. يتم التعامل مع SL/TP بواسطة الوسيط.
        if self.position:
            return

        # الحصول على قيم الميزات للشمعة الحالية
        # ملاحظة: self.data يحتوي على الميزات المحسوبة مسبقًا
        try:
            features = self.data.df.loc[self.data.index[-1], self.feature_names]
            if features.isnull().any():
                return # تخطي إذا كانت البيانات مفقودة
        except (KeyError, IndexError):
            return # تخطي إذا كانت الصفوف أو الأعمدة مفقودة

        # إعادة تشكيل للـ scaler والنموذج
        features_df = pd.DataFrame([features])
        
        # --- الإصلاح 1: تحويل مصفوفة numpy المقاسة مرة أخرى إلى DataFrame بأسماء الميزات ---
        features_scaled_np = self.scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)
        
        # الحصول على تنبؤ النموذج والاحتمالية باستخدام DataFrame مع أسماء الميزات
        prediction = self.ml_model.predict(features_scaled_df)[0]
        prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
        
        try:
            # العثور على احتمالية فئة "الشراء" (1)
            class_1_index = list(self.ml_model.classes_).index(1)
            prob_for_class_1 = prediction_proba[class_1_index]
        except ValueError:
            return # الفئة '1' غير موجودة في النموذج

        # --- منطق التداول ---
        if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
            # الحصول على ATR الحالي لـ SL/TP الديناميكي
            current_atr = self.data.atr[-1]
            if pd.isna(current_atr) or current_atr == 0:
                return # لا يمكن تعيين SL/TP بدون ATR

            current_price = self.data.Close[-1]
            
            # --- الإصلاح 2: حساب حجم الصفقة كجزء من رأس المال ---
            # تتطلب مكتبة الاختبار الخلفي أن يكون الحجم إما جزءًا من رأس المال (0 < size < 1)
            # أو عددًا صحيحًا من الوحدات (مثل 1، 2، 3).
            # حساب الوحدات مباشرة (مثل 10 / السعر) يمكن أن ينتج عنه رقم كسري
            # أكبر من 1 (مثل 1.25)، مما يسبب خطأ تأكيد.
            # النهج الصحيح هو حساب الكسر الذي يمثله مبلغ الصفقة المطلوب
            # من إجمالي رأس المال لدينا.
            size_as_fraction = TRADE_AMOUNT_USDT / self.equity

            # يجب أن نتأكد من أن الحجم المحسوب هو كسر صالح لطريقة `buy`.
            # يجب أن يكون > 0 و < 1. إذا لم يكن لدينا رأس مال كافٍ، فقد يكون >= 1.
            if size_as_fraction > 0 and size_as_fraction < 1:
                # تحديد مستويات وقف الخسارة وجني الأرباح
                stop_loss_price = current_price - (current_atr * ATR_SL_MULTIPLIER)
                take_profit_price = current_price + (current_atr * ATR_TP_MULTIPLIER)

                # وضع أمر الشراء مع SL و TP
                self.buy(size=size_as_fraction, sl=stop_loss_price, tp=take_profit_price)

# ---------------------- كتلة التنفيذ الرئيسية ----------------------
def run_backtest():
    global client, conn
    logger.info("🚀 بدء الاختبار الخلفي المتقدم لاستراتيجية V6...")
    
    # تهيئة الاتصالات
    init_db()
    client = Client(API_KEY, API_SECRET)
    
    # الحصول على الرموز المراد اختبارها
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'crypto_list.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            symbols_to_test = [line.strip().upper() + "USDT" for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        logger.error("❌ لم يتم العثور على 'crypto_list.txt'. سيتم الخروج.")
        return

    logger.info("ℹ️ جلب بيانات BTC العالمية لفترة الاختبار الخلفي...")
    btc_df_full = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, BACKTEST_PERIOD_DAYS + 10)
    if btc_df_full is None:
        logger.critical("❌ فشل في جلب بيانات BTC. لا يمكن المتابعة."); return
    btc_df_full['btc_returns'] = btc_df_full['Close'].pct_change()

    all_stats = []
    
    for symbol in tqdm(symbols_to_test, desc="اختبار العملات"):
        logger.info(f"\n--- ⏳ معالجة العملة: {symbol} ---")
        
        # 1. تحميل نموذج التعلم الآلي للعملة
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle:
            logger.warning(f"⚠️ تخطي {symbol}: لم يتم العثور على نموذج.")
            continue
        
        # 2. جلب جميع البيانات المطلوبة
        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, BACKTEST_PERIOD_DAYS)
        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, BACKTEST_PERIOD_DAYS * 5) # جلب المزيد من بيانات 4h لـ EMA
        
        if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty:
            logger.warning(f"⚠️ تخطي {symbol}: بيانات تاريخية غير كافية.")
            continue
            
        sr_levels = fetch_sr_levels_from_db(symbol)

        # 3. إعداد DataFrame الرئيسي مع جميع الميزات
        logger.info(f"هندسة الميزات للعملة {symbol}...")
        
        # إنشاء الميزات الأساسية
        data = create_all_features(df_15m, btc_df_full)
        
        # إنشاء ودمج ميزات MTF
        delta_4h = df_4h['Close'].diff()
        gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
        ema_fast_4h = df_4h['Close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
        df_4h['price_vs_ema50_4h'] = (df_4h['Close'] / ema_fast_4h) - 1
        mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
        data = data.join(mtf_features, how='left').fillna(method='ffill')

        # إنشاء ودمج ميزات S/R
        data = calculate_sr_features(data, sr_levels)
        
        data.dropna(inplace=True)
        
        if data.empty:
            logger.warning(f"⚠️ تخطي {symbol}: DataFrame فارغ بعد هندسة الميزات.")
            continue

        # 4. تشغيل الاختبار الخلفي للعملة
        logger.info(f"تشغيل الاختبار الخلفي للعملة {symbol}...")
        bt = Backtest(
            data,
            MLStrategy,
            cash=INITIAL_CASH,
            commission=COMMISSION, # يشمل الرسوم + الانزلاق
            exclusive_orders=True # منع أوامر متعددة في نفس الوقت
        )
        
        # تمرير النموذج المحمل و scaler إلى فئة الاستراتيجية
        stats = bt.run(
            ml_model=model_bundle['model'],
            scaler=model_bundle['scaler'],
            feature_names=model_bundle['feature_names']
        )
        
        # *** تعديل: إضافة اسم العملة إلى الإحصائيات لتسهيل التقرير ***
        stats['Symbol'] = symbol
        
        logger.info(f"\n--- نتائج الاختبار الخلفي للعملة {symbol} ---")
        print(stats)
        all_stats.append(stats)
        
        # اختياري: رسم النتائج
        # قم بإلغاء التعليق على السطر أدناه لإنشاء مخطط HTML لكل عملة
        # bt.plot(filename=f"backtest_plot_{symbol}.html", open_browser=False)

        del data, df_15m, df_4h, sr_levels, model_bundle
        gc.collect()

    # 5. *** جديد: عرض النتائج المجمعة وإنشاء تقرير شامل ***
    logger.info("\n\n--- 🏁 ملخص الاختبار الخلفي الشامل وإنشاء التقرير 🏁 ---")
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        if 'Symbol' in summary_df.columns:
            summary_df.set_index('Symbol', inplace=True)

        # --- حساب المقاييس المجمعة ---
        total_symbols_tested = len(summary_df)
        symbols_with_trades = summary_df[summary_df['# Trades'] > 0]
        total_symbols_with_trades = len(symbols_with_trades)

        total_trades = summary_df['# Trades'].sum()
        total_duration_days = BACKTEST_PERIOD_DAYS
        
        # حساب إجمالي الربح/الخسارة والعائد
        initial_portfolio_value = INITIAL_CASH * total_symbols_tested
        final_portfolio_value = summary_df['Equity Final [$]'].sum()
        total_net_profit_loss = final_portfolio_value - initial_portfolio_value
        total_return_pct = (total_net_profit_loss / initial_portfolio_value) * 100 if initial_portfolio_value > 0 else 0

        # حساب متوسطات المقاييس للعملات التي كان بها تداولات فقط
        if total_symbols_with_trades > 0:
            avg_win_rate = symbols_with_trades['Win Rate [%]'].mean()
            avg_profit_factor = symbols_with_trades['Profit Factor'].replace([np.inf, -np.inf], np.nan).mean()
            avg_sharpe = symbols_with_trades['Sharpe Ratio'].mean()
            avg_sortino = symbols_with_trades['Sortino Ratio'].mean()
            avg_max_drawdown = symbols_with_trades['Max. Drawdown [%]'].mean()
            best_performer = symbols_with_trades.sort_values(by='Return [%]', ascending=False).iloc[0]
            worst_performer = symbols_with_trades.sort_values(by='Return [%]', ascending=True).iloc[0]
        else:
            avg_win_rate, avg_profit_factor, avg_sharpe, avg_sortino, avg_max_drawdown = 0, 0, 0, 0, 0
            best_performer, worst_performer = None, None

        # --- إنشاء محتوى التقرير ---
        report_lines = [
            "======================================================",
            "=          تقرير الاختبار الخلفي الشامل              =",
            "======================================================",
            f"تاريخ الإنشاء: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"فترة الاختبار: {total_duration_days} يوم",
            f"الاستراتيجية: {BASE_ML_MODEL_NAME}",
            "------------------------------------------------------",
            "                     الأداء العام                      ",
            "------------------------------------------------------",
            f"إجمالي العملات المختبرة: {total_symbols_tested}",
            f"العملات التي تم التداول عليها: {total_symbols_with_trades}",
            f"إجمالي رأس المال الأولي: ${initial_portfolio_value:,.2f}",
            f"إجمالي رأس المال النهائي: ${final_portfolio_value:,.2f}",
            f"إجمالي صافي الربح/الخسارة: ${total_net_profit_loss:,.2f}",
            f"إجمالي العائد: {total_return_pct:.2f}%",
            f"إجمالي عدد الصفقات: {int(total_trades)}",
            "------------------------------------------------------",
            "               متوسط المقاييس (لكل عملة)             ",
            "------------------------------------------------------",
            f"متوسط نسبة الربح: {avg_win_rate:.2f}%",
            f"متوسط عامل الربح: {avg_profit_factor:.2f}",
            f"متوسط أقصى تراجع: {avg_max_drawdown:.2f}%",
            f"متوسط نسبة شارب: {avg_sharpe:.2f}",
            f"متوسط نسبة سورتينو: {avg_sortino:.2f}",
            "------------------------------------------------------"
        ]

        if best_performer is not None:
            report_lines.append("              أفضل وأسوأ العملات أداءً             ")
            report_lines.append("------------------------------------------------------")
            profit_best = best_performer['Equity Final [$]'] - best_performer['Start Equity [$]']
            report_lines.append(f"الأفضل أداءً: {best_performer.name}")
            report_lines.append(f"  - صافي الربح: ${profit_best:,.2f}")
            report_lines.append(f"  - العائد: {best_performer['Return [%]']:.2f}%")
            report_lines.append(f"  - نسبة الربح: {best_performer['Win Rate [%]']:.2f}%")
            report_lines.append(f"  - عدد الصفقات: {int(best_performer['# Trades'])}")
        
        if worst_performer is not None:
            profit_worst = worst_performer['Equity Final [$]'] - worst_performer['Start Equity [$]']
            report_lines.append(f"\nالأسوأ أداءً: {worst_performer.name}")
            report_lines.append(f"  - صافي الربح: ${profit_worst:,.2f}")
            report_lines.append(f"  - العائد: {worst_performer['Return [%]']:.2f}%")
            report_lines.append(f"  - نسبة الربح: {worst_performer['Win Rate [%]']:.2f}%")
            report_lines.append(f"  - عدد الصفقات: {int(worst_performer['# Trades'])}")
        
        report_lines.append("======================================================")

        final_report_str = "\n".join(report_lines)

        # --- طباعة التقرير في الطرفية ---
        print("\n\n" + final_report_str)

        # --- حفظ التقرير في ملفات ---
        report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = 'backtest_reports'
        os.makedirs(report_dir, exist_ok=True)

        # حفظ ملخص التقرير في ملف نصي
        summary_filename = os.path.join(report_dir, f'report_summary_{report_timestamp}.txt')
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(final_report_str)
        logger.info(f"✅ تم حفظ تقرير الملخص الشامل في: {summary_filename}")

        # حفظ الإحصائيات التفصيلية في ملف CSV
        details_filename = os.path.join(report_dir, f'report_details_{report_timestamp}.csv')
        summary_df.to_csv(details_filename)
        logger.info(f"✅ تم حفظ الإحصائيات التفصيلية لكل عملة في: {details_filename}")

    else:
        logger.warning("لم يتم إكمال أي اختبار خلفي بنجاح.")
        
    if conn:
        conn.close()

if __name__ == "__main__":
    run_backtest()
