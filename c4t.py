import os
import gc
import pickle
import logging
import warnings
import pandas as pd
import numpy as np
import requests
import json
from decouple import config
from binance.client import Client
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, timezone
from backtesting import Backtest, Strategy
from tqdm import tqdm
import threading
from flask import Flask

# --- إعداد خادم الويب ---
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Backtester service is running.", 200

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_v7_with_ichimoku.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester_V7_With_Ichimoku')

# ---------------------- إعداد متغيرات البيئة والثوابت ----------------------
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
    logger.critical(f"❌ فشل حرج في تحميل متغيرات البيئة: {e}")
    exit(1)

# --- إعدادات الاختبار الخلفي ---
INITIAL_CASH = 100.0
TRADE_AMOUNT_USDT = 10.0
FEE = 0.001
SLIPPAGE = 0.0005
COMMISSION = FEE + SLIPPAGE
BACKTEST_PERIOD_DAYS = 30
OUT_OF_SAMPLE_OFFSET_DAYS = 126

# --- ثوابت الاستراتيجية والنموذج ---
BASE_ML_MODEL_NAME = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER = 'V7'
SIGNAL_GENERATION_TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '4h'
BTC_SYMBOL = 'BTCUSDT'
MODEL_CONFIDENCE_THRESHOLD = 0.70
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0

# --- إعدادات المؤشرات ---
ADX_PERIOD, BBANDS_PERIOD, RSI_PERIOD = 14, 20, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD, EMA_SLOW_PERIOD, EMA_FAST_PERIOD = 14, 200, 50
BTC_CORR_PERIOD, STOCH_RSI_PERIOD, STOCH_K, STOCH_D, REL_VOL_PERIOD = 30, 14, 3, 3, 30

# متغيرات الاتصال العامة
conn = None
client = None

# ---------------------- دوال قاعدة البيانات ----------------------
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = False
        logger.info("✅ [DB] تم تهيئة الاتصال بقاعدة البيانات بنجاح.")
        create_backtest_results_table() # <-- ✨ جديد: التأكد من وجود جدول النتائج
    except Exception as e:
        logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}")
        conn = None

# --- ✨ جديد: دالة لإنشاء جدول نتائج الاختبار الخلفي ---
def create_backtest_results_table():
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    run_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    stats JSONB NOT NULL,
                    duration_days INT,
                    return_pct NUMERIC,
                    win_rate_pct NUMERIC,
                    profit_factor NUMERIC,
                    num_trades INT
                );
            """)
        conn.commit()
        logger.info("✅ [DB] تم التأكد من وجود جدول 'backtest_results'.")
    except Exception as e:
        logger.error(f"❌ [DB] فشل في إنشاء جدول 'backtest_results': {e}")
        conn.rollback()

# --- ✨ جديد: دالة لحفظ نتيجة اختبار فردي في قاعدة البيانات ---
def save_backtest_results(symbol, strategy_name, run_timestamp, stats):
    if not conn:
        logger.error("❌ [DB Save] لا يمكن حفظ النتائج، لا يوجد اتصال بقاعدة البيانات.")
        return

    # استخراج القيم الرئيسية لتسهيل الاستعلامات
    # استخدام .get() مع قيمة افتراضية لتجنب الأخطاء إذا كانت الإحصائيات غير موجودة
    duration_str = stats.get('Duration', '0 days').split()[0]
    duration_days = int(duration_str) if duration_str.isdigit() else 0
    return_pct = stats.get('Return [%]', 0.0)
    win_rate_pct = stats.get('Win Rate [%]', 0.0)
    profit_factor = stats.get('Profit Factor', 0.0)
    num_trades = stats.get('# Trades', 0)

    # تحويل كائن الإحصائيات إلى سلسلة JSON
    stats_json = json.dumps(stats, default=str) # استخدام default=str لمعالجة أي أنواع بيانات غير قابلة للتحويل

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO backtest_results
                (symbol, strategy_name, run_timestamp, stats, duration_days, return_pct, win_rate_pct, profit_factor, num_trades)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (symbol, strategy_name, run_timestamp, stats_json, duration_days, return_pct, win_rate_pct, profit_factor, num_trades))
        conn.commit()
        logger.info(f"✅ [DB Save] تم حفظ نتائج الاختبار الخلفي للعملة {symbol} بنجاح.")
    except Exception as e:
        logger.error(f"❌ [DB Save] فشل في حفظ نتائج {symbol}: {e}")
        conn.rollback()


def load_ml_model_bundle_from_db(symbol: str) -> dict | None:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                conn.commit()
                return model_bundle
        conn.commit()
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model DB] خطأ في تحميل حزمة النموذج للعملة {symbol}: {e}")
        if conn: conn.rollback()
        return None

def load_ml_model_bundle_from_folder(symbol: str) -> dict | None:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, MODEL_FOLDER, f"{model_name}.pkl")
        if not os.path.exists(model_path): return None
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            return model_bundle
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model File] خطأ في تحميل حزمة النموذج من الملف للعملة {symbol}: {e}")
        return None

# ... (بقية دوال جلب البيانات وهندسة الميزات تبقى كما هي) ...
def fetch_historical_data(symbol: str, interval: str, days: int, out_of_sample_period_days: int = 0) -> pd.DataFrame | None:
    global client
    if not client:
        try: client = Client(API_KEY, API_SECRET)
        except Exception as e: logger.error(f"❌ [Binance] فشل في تهيئة اتصال Binance: {e}"); return None
    try:
        now = datetime.now(timezone.utc)
        end_dt = now - timedelta(days=out_of_sample_period_days)
        start_dt = end_dt - timedelta(days=days)
        klines = client.get_historical_klines(symbol, interval, start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S"))
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية للعملة {symbol}: {e}")
        return None

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
    return df_calc

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("⚠️ [Telegram] Token أو Chat ID غير معرف. تم تخطي إرسال الرسالة.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(CHAT_ID), 'text': text, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        logger.info("✅ [Telegram] تم إرسال تقرير الملخص بنجاح.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# ---------------------- فئة استراتيجية Backtesting.py ----------------------
class MLStrategy(Strategy):
    ml_model = None
    scaler = None
    feature_names = None

    def init(self):
        pass

    def next(self):
        if self.position: return

        try:
            features = self.data.df.loc[self.data.index[-1], self.feature_names]
            if features.isnull().any(): return
        except (KeyError, IndexError): return

        features_df = pd.DataFrame([features])
        features_scaled_np = self.scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)
        
        prediction = self.ml_model.predict(features_scaled_df)[0]
        prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
        
        try:
            class_1_index = list(self.ml_model.classes_).index(1)
            prob_for_class_1 = prediction_proba[class_1_index]
        except ValueError: return

        if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
            current_atr = self.data.atr[-1]
            if pd.isna(current_atr) or current_atr == 0: return

            required_cash = TRADE_AMOUNT_USDT * (1 + COMMISSION)
            if self.equity < required_cash: return

            current_price = self.data.Close[-1]
            size_as_fraction = TRADE_AMOUNT_USDT / self.equity

            if size_as_fraction > 0:
                stop_loss_price = current_price - (current_atr * ATR_SL_MULTIPLIER)
                take_profit_price = current_price + (current_atr * ATR_TP_MULTIPLIER)
                self.buy(size=size_as_fraction, sl=stop_loss_price, tp=take_profit_price)

# --- ✨ جديد: دالة لتوليد التقرير النهائي من قاعدة البيانات ---
def generate_report_from_db(run_timestamp):
    if not conn:
        logger.error("❌ [Report] لا يمكن إنشاء التقرير، لا يوجد اتصال بقاعدة البيانات.")
        return

    logger.info("📊 [Report] جاري إنشاء التقرير النهائي من النتائج المحفوظة...")
    try:
        # استخدام SQL لتجميع البيانات مباشرة من قاعدة البيانات
        query = """
            SELECT
                COUNT(*) AS total_symbols,
                SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) AS profitable_symbols,
                SUM(num_trades) AS total_trades,
                AVG(win_rate_pct) AS avg_win_rate,
                AVG(profit_factor) FILTER (WHERE profit_factor > 0 AND profit_factor != 'Infinity'::numeric) AS avg_profit_factor,
                SUM(return_pct) AS total_return_pct,
                AVG(return_pct) AS avg_return_pct
            FROM backtest_results
            WHERE run_timestamp = %s;
        """
        with conn.cursor() as cur:
            cur.execute(query, (run_timestamp,))
            summary = cur.fetchone()
        conn.commit()

        if not summary or summary['total_symbols'] == 0:
            logger.warning("⚠️ [Report] لم يتم العثور على نتائج للاختبار الحالي.")
            send_telegram_message("🏁 انتهى الاختبار الخلفي ولكن لم يتم العثور على نتائج صالحة لإنشاء تقرير.")
            return

        report_title = f"📊 *ملخص الاختبار الخلفي - {BASE_ML_MODEL_NAME}*"
        report_date = f"*{run_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}*"
        
        total_symbols = summary.get('total_symbols', 0)
        profitable_symbols = summary.get('profitable_symbols', 0)
        
        report_body = (
            f"----------------------------------------\n"
            f"▫️ *إجمالي الرموز المختبرة:* `{total_symbols}`\n"
            f"📈 *رموز رابحة:* `{int(profitable_symbols)}`\n"
            f"📉 *رموز خاسرة:* `{int(total_symbols - profitable_symbols)}`\n"
            f"🔄 *إجمالي الصفقات:* `{int(summary.get('total_trades', 0))}`\n"
            f"----------------------------------------\n"
            f"🎯 *متوسط نسبة الربح:* `{summary.get('avg_win_rate', 0):.2f}%`\n"
            f"💰 *متوسط عامل الربح:* `{summary.get('avg_profit_factor', 0):.2f}`\n"
            f"📈 *متوسط العائد لكل رمز:* `{summary.get('avg_return_pct', 0):.2f}%`\n"
            f"📊 *إجمالي العائد (مجموع):* `{summary.get('total_return_pct', 0):.2f}%`\n"
            f"----------------------------------------"
        )
        
        final_report = f"{report_title}\n{report_date}\n\n{report_body}"
        send_telegram_message(final_report)

    except Exception as e:
        logger.error(f"❌ [Report] فشل في إنشاء التقرير من قاعدة البيانات: {e}")
        if conn: conn.rollback()
        send_telegram_message("❌ حدث خطأ أثناء إنشاء تقرير الاختبار الخلفي.")

# ---------------------- كتلة التنفيذ الرئيسية ----------------------
def run_backtest():
    global conn
    logger.info(f"🚀 بدء الاختبار الخلفي المتقدم لاستراتيجية {BASE_ML_MODEL_NAME}...")
    
    init_db()
    if not conn:
        logger.critical("❌ لا يمكن تشغيل الاختبار الخلفي بدون اتصال بقاعدة البيانات.")
        send_telegram_message("❌ فشل الاختبار الخلفي: لم يتمكن من الاتصال بقاعدة البيانات.")
        return

    # --- ✨ جديد: تحديد وقت بدء فريد لهذه الجولة من الاختبار ---
    run_timestamp = datetime.now(timezone.utc)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'crypto_list.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            symbols_to_test = [line.strip().upper() + "USDT" for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        logger.error("❌ ملف 'crypto_list.txt' غير موجود."); return

    btc_df_full = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, BACKTEST_PERIOD_DAYS + 10, out_of_sample_period_days=OUT_OF_SAMPLE_OFFSET_DAYS)
    if btc_df_full is None:
        logger.critical("❌ فشل جلب بيانات BTC. لا يمكن المتابعة."); return
    btc_df_full['btc_returns'] = btc_df_full['Close'].pct_change()
    
    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        logger.info(f"\n--- ⏳ جاري معالجة الرمز: {symbol} ---")
        
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle:
            model_bundle = load_ml_model_bundle_from_folder(symbol)

        if not model_bundle:
            logger.warning(f"⚠️ تخطي {symbol}: لم يتم العثور على نموذج."); continue
        
        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, BACKTEST_PERIOD_DAYS, out_of_sample_period_days=OUT_OF_SAMPLE_OFFSET_DAYS)
        if df_15m is None or df_15m.empty:
            logger.warning(f"⚠️ تخطي {symbol}: بيانات تاريخية غير كافية."); continue
            
        data = create_all_features(df_15m, btc_df_full)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        
        if data.empty:
            logger.warning(f"⚠️ تخطي {symbol}: DataFrame فارغ بعد هندسة الميزات."); continue

        try:
            bt = Backtest(data, MLStrategy, cash=INITIAL_CASH, commission=COMMISSION, exclusive_orders=True)
            stats = bt.run(
                ml_model=model_bundle['model'],
                scaler=model_bundle['scaler'],
                feature_names=model_bundle['feature_names']
            )
            
            # --- ✨ تعديل: حفظ النتيجة مباشرة في قاعدة البيانات ---
            save_backtest_results(symbol, BASE_ML_MODEL_NAME, run_timestamp, stats.to_dict())

        except Exception as e:
            logger.error(f"❌ [Backtest Run] فشل الاختبار الخلفي للعملة {symbol}: {e}")
        
        # --- ✨ جديد: تحرير الذاكرة بعد كل عملة ---
        del data, df_15m, model_bundle
        gc.collect()
        logger.info(f"🧠 [Memory] تم تحرير الذاكرة بعد معالجة {symbol}.")

    # --- ✨ تعديل: توليد التقرير النهائي بعد الانتهاء من جميع العملات ---
    generate_report_from_db(run_timestamp)
        
    if conn:
        conn.close()
    logger.info("✅ انتهى خيط الاختبار الخلفي.")


if __name__ == "__main__":
    backtest_thread = threading.Thread(target=run_backtest, name="run_backtest", daemon=True)
    backtest_thread.start()
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
