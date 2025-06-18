import os
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from threading import Thread

import numpy as np
import pandas as pd
import psycopg2
from binance.client import Client
from decouple import config
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
from flask import Flask

# ==============================================================================
# --------------------------- إعدادات الاختبار الخلفي ----------------------------
# ==============================================================================
# الفترة الزمنية للاختبار بالايام
BACKTEST_PERIOD_DAYS: int = 180
# الإطار الزمني للشموع (يجب أن يطابق إطار تدريب النموذج)
TIMEFRAME: str = '15m'
# اسم النموذج الأساسي الذي سيتم اختباره
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V4'

# --- معلمات الاستراتيجية (يجب أن تطابق إعدادات البوت c4.py) ---
MODEL_PREDICTION_THRESHOLD: float = 0.70
ATR_SL_MULTIPLIER: float = 2.0
ATR_TP_MULTIPLIER: float = 3.0
USE_TRAILING_STOP: bool = False
#TRAILING_STOP_ACTIVATE_PERCENT: float = 0.75
#TRAILING_STOP_DISTANCE_PERCENT: float = 1.0

# --- !!! تعديل: تم تغيير فلتر RSI إلى نطاق !!! ---
USE_RSI_FILTER: bool = True
RSI_LOWER_THRESHOLD: float = 40.0 # الحد الأدنى لفلتر RSI
RSI_UPPER_THRESHOLD: float = 69.0 # الحد الأعلى لفلتر RSI

# --- إعدادات فلتر MACD الجديدة ---
USE_MACD_FILTER: bool = True
MACD_SHORT_PERIOD: int = 12
MACD_LONG_PERIOD: int = 26
MACD_SIGNAL_PERIOD: int = 9
MACD_DIF_CROSSOVER_ONLY: bool = True # فلتر لتقاطع DIF صعوديا فوق DEA


# --- معلمات محاكاة التكاليف الواقعية ---
# العمولة لكل صفقة (شراء أو بيع). 0.1% هو المعدل القياسي في Binance
COMMISSION_PERCENT: float = 0.1
# الانزلاق السعري المتوقع. 0.05% هو تقدير معقول للصفقات السوقية
SLIPPAGE_PERCENT: float = 0.05

# مبلغ افتراضي لكل صفقة بالدولار لمحاكاة الربح
INITIAL_TRADE_AMOUNT_USDT: float = 10.0

# ==============================================================================
# ---------------------------- إعدادات النظام والاتصال -------------------------
# ==============================================================================

# إعداد التسجيل (Logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester')

# إعداد خادم الويب (للتوافق مع منصات مثل Render)
app = Flask(__name__)
@app.route('/')
def health_check():
    return "Backtester service is running and alive."

# تحميل متغيرات البيئة
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# إعداد عميل Binance
client: Optional[Client] = None
try:
    client = Client(API_KEY, API_SECRET)
    logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
except Exception as e:
    logger.critical(f"❌ [Binance] فشل الاتصال: {e}")
    exit(1)

# إعداد الاتصال بقاعدة البيانات
conn: Optional[psycopg2.extensions.connection] = None
try:
    conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
except Exception as e:
    logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}")
    exit(1)

# ==============================================================================
# ------------------- دوال مساعدة (منسوخة ومعدلة من ملفاتك) --------------------
# ==============================================================================

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    تقرأ قائمة العملات وتتحقق من وجودها وصلاحيتها للتداول على Binance.
    """
    logger.info(f"ℹ️ [Validation] Reading symbols from '{filename}'...")
    if not client:
        logger.error("Binance client not initialized.")
        return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        exchange_info = client.get_exchange_info()
        active_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
        
        validated = sorted(list(formatted.intersection(active_symbols)))
        logger.info(f"✅ [Validation] Found {len(validated)} symbols to backtest.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] Error: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """
    تجلب البيانات التاريخية من Binance.
    """
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines:
            logger.warning(f"⚠️ No historical data found for {symbol} for the given period.")
            return None
            
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}")
        return None

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    تحسب جميع المؤشرات والميزات المطلوبة للنموذج (نسخة مطابقة لما في ملف التدريب والاستراتيجية).
    """
    df_calc = df.copy()
    RSI_PERIOD, BBANDS_PERIOD, ATR_PERIOD = 14, 20, 14 # تم إزالة MACD_FAST, MACD_SLOW, MACD_SIGNAL من هنا لأننا سنستخدم MACD_SHORT/LONG/SIGNAL_PERIOD
    BBANDS_STD_DEV = 2.0
    
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df_calc['rsi'] = 100 - (100 / (1 + rs))

    # --- حسابات MACD: DIF و DEA (MACD Signal) و MACD Hist ---
    ema_fast = df_calc['close'].ewm(span=MACD_SHORT_PERIOD, adjust=False).mean() # DIF (MACD Line)
    ema_slow = df_calc['close'].ewm(span=MACD_LONG_PERIOD, adjust=False).mean()  # Slow EMA
    df_calc['macd_line'] = ema_fast - ema_slow # هذا هو DIF
    df_calc['macd_signal_line'] = df_calc['macd_line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean() # هذا هو DEA
    df_calc['macd_hist'] = df_calc['macd_line'] - df_calc['macd_signal_line'] # هذا هو MACD Hist

    # تحديد تقاطع MACD
    # DIF يخترق صعودياً DEA
    df_calc['macd_crossover_up'] = (df_calc['macd_line'] > df_calc['macd_signal_line']) & \
                                   (df_calc['macd_line'].shift(1) <= df_calc['macd_signal_line'].shift(1))


    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_upper'] = sma + (std * BBANDS_STD_DEV)
    df_calc['bb_lower'] = sma - (std * BBANDS_STD_DEV)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / sma.replace(0, np.nan)
    df_calc['bb_pos'] = (df_calc['close'] - sma) / std.replace(0, np.nan)

    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['hour_of_day'] = df_calc.index.hour
    
    df_calc['candle_body_size'] = (df_calc['close'] - df_calc['open']).abs()
    df_calc['upper_wick'] = df_calc['high'] - df_calc[['open', 'close']].max(axis=1)
    df_calc['lower_wick'] = df_calc[['open', 'close']].min(axis=1) - df_calc['low']
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)

    return df_calc.dropna()

def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    """
    تحمل حزمة النموذج (النموذج + المعاير + أسماء الميزات) من قاعدة البيانات.
    """
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result.get('model_data'):
                model_bundle = pickle.loads(result['model_data'])
                logger.info(f"✅ [Model] Successfully loaded model '{model_name}' for {symbol}.")
                return model_bundle
            logger.warning(f"⚠️ [Model] Model '{model_name}' not found in DB for {symbol}.")
            return None
    except Exception as e:
        logger.error(f"❌ [Model] Error loading model for {symbol}: {e}", exc_info=True)
        return None

# ==============================================================================
# ----------------------------- محرك الاختبار الخلفي ----------------------------
# ==============================================================================

def run_backtest_for_symbol(symbol: str, data: pd.DataFrame, model_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    تقوم بتنفيذ محاكاة التداول على البيانات التاريخية لعملة واحدة.
    الأسعار المسجلة هنا هي أسعار "مثالية" قبل تطبيق الانزلاق والعمولة.
    """
    trades = []
    
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_names = model_bundle['feature_names']
    
    df_featured = calculate_features(data)
    
    if not all(col in df_featured.columns for col in feature_names):
        missing = [col for col in feature_names if col not in df_featured.columns]
        logger.error(f"Missing features {missing} for {symbol}. Skipping.")
        return []

    features_df = df_featured[feature_names]
    features_scaled_np = scaler.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled_np, columns=feature_names, index=features_df.index)
    predictions = model.predict_proba(features_scaled_df)[:, 1]
    
    df_featured['prediction'] = predictions
    
    in_trade = False
    trade_details = {}

    for i in range(len(df_featured)):
        current_candle = df_featured.iloc[i]
        
        # --- Logic to manage an active trade ---
        if in_trade:
            # Check for TP hit: if the candle's high touches the take profit
            if current_candle['high'] >= trade_details['tp']:
                trade_details['exit_price'] = trade_details['tp']
                trade_details['exit_reason'] = 'TP Hit'
            # Check for SL hit: if the candle's low touches the stop loss
            elif current_candle['low'] <= trade_details['sl']:
                trade_details['exit_price'] = trade_details['sl']
                trade_details['exit_reason'] = 'SL Hit'
            
            # Trailing Stop Loss Logic (currently disabled in settings)
            elif USE_TRAILING_STOP:
                activation_price = trade_details['entry_price'] + (trade_details['tp'] - trade_details['entry_price']) * 0.75
                if not trade_details.get('tsl_active') and current_candle['high'] >= activation_price:
                    trade_details['tsl_active'] = True
                if trade_details.get('tsl_active'):
                    new_tsl = current_candle['close'] * (1 - (1.0 / 100))
                    if new_tsl > trade_details['sl']:
                        trade_details['sl'] = new_tsl
            
            if trade_details.get('exit_price'):
                trade_details['exit_time'] = current_candle.name
                trade_details['duration_candles'] = i - trade_details['entry_index']
                trades.append(trade_details)
                in_trade = False
                trade_details = {}
            continue

        # --- Logic to enter a new trade ---
        # --- !!! تعديل: تطبيق فلتر RSI ضمن النطاق المطلوب !!! ---
        passes_rsi_filter = True # الافتراضي هو أن الشرط متحقق إذا كان الفلتر معطلاً
        if USE_RSI_FILTER:
            # إذا كان الفلتر مفعّلاً، تحقق من أن RSI ضمن النطاق المطلوب
            current_rsi_value = current_candle.get('rsi', 0)
            if not (RSI_LOWER_THRESHOLD <= current_rsi_value <= RSI_UPPER_THRESHOLD):
                passes_rsi_filter = False

        # --- تطبيق فلتر MACD الجديد ---
        passes_macd_filter = True
        if USE_MACD_FILTER:
            # تحقق من وجود قيم MACD وتقاطع DIF صعوديا فوق DEA
            if 'macd_line' not in current_candle or 'macd_signal_line' not in current_candle or \
               'macd_crossover_up' not in current_candle:
                passes_macd_filter = False
            elif MACD_DIF_CROSSOVER_ONLY and not current_candle['macd_crossover_up']:
                passes_macd_filter = False
            elif not MACD_DIF_CROSSOVER_ONLY and not (current_candle['macd_line'] > current_candle['macd_signal_line']):
                passes_macd_filter = False


        # تحقق من فلتر RSI، فلتر MACD، ومن توقع النموذج قبل الدخول في صفقة
        if not in_trade and passes_rsi_filter and passes_macd_filter and \
           current_candle['prediction'] >= MODEL_PREDICTION_THRESHOLD:
            in_trade = True
            entry_price = current_candle['close']
            atr_value = current_candle['atr']
            
            stop_loss = entry_price - (atr_value * ATR_SL_MULTIPLIER)
            take_profit = entry_price + (atr_value * ATR_TP_MULTIPLIER)
            
            trade_details = {
                'symbol': symbol,
                'entry_time': current_candle.name,
                'entry_price': entry_price, # Ideal price before slippage
                'entry_index': i,
                'tp': take_profit,
                'sl': stop_loss,
                'initial_sl': stop_loss,
            }

    return trades

def generate_report(all_trades: List[Dict[str, Any]]):
    """
    تنشئ وتعرض تقريرًا مفصلاً بنتائج الاختبار الخلفي،
    مع تطبيق الانزلاق السعري والعمولة للحصول على نتائج واقعية.
    """
    if not all_trades:
        logger.warning("No trades were executed during the backtest.")
        return

    df_trades = pd.DataFrame(all_trades)
    
    # --- تطبيق الانزلاق السعري والعمولة ---
    df_trades['entry_price_adj'] = df_trades['entry_price'] * (1 + SLIPPAGE_PERCENT / 100)
    df_trades['exit_price_adj'] = df_trades['exit_price'] * (1 - SLIPPAGE_PERCENT / 100)
    df_trades['pnl_pct_raw'] = ((df_trades['exit_price_adj'] / df_trades['entry_price_adj']) - 1) * 100
    
    entry_cost = INITIAL_TRADE_AMOUNT_USDT
    exit_value = entry_cost * (1 + df_trades['pnl_pct_raw'] / 100)
    
    commission_entry = entry_cost * (COMMISSION_PERCENT / 100)
    commission_exit = exit_value * (COMMISSION_PERCENT / 100)
    
    df_trades['commission_total'] = commission_entry + commission_exit
    df_trades['pnl_usdt_net'] = (exit_value - entry_cost) - df_trades['commission_total']
    df_trades['pnl_pct_net'] = (df_trades['pnl_usdt_net'] / INITIAL_TRADE_AMOUNT_USDT) * 100

    # --- إعداد التقرير ---
    total_trades = len(df_trades)
    winning_trades = df_trades[df_trades['pnl_usdt_net'] > 0]
    losing_trades = df_trades[df_trades['pnl_usdt_net'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_net_pnl = df_trades['pnl_usdt_net'].sum()
    
    gross_profit = winning_trades['pnl_usdt_net'].sum()
    gross_loss = abs(losing_trades['pnl_usdt_net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = winning_trades['pnl_usdt_net'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl_usdt_net'].mean()) if len(losing_trades) > 0 else 0
    risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')

    # بناء نص التقرير
    report_header = f"BACKTESTING REPORT: {BASE_ML_MODEL_NAME}"
    if USE_RSI_FILTER:
        report_header += f" (with RSI Filter between {RSI_LOWER_THRESHOLD} and {RSI_UPPER_THRESHOLD})"
    if USE_MACD_FILTER:
        report_header += f" (with MACD Filter: Short {MACD_SHORT_PERIOD}, Long {MACD_LONG_PERIOD}, Signal {MACD_SIGNAL_PERIOD})"
        if MACD_DIF_CROSSOVER_ONLY:
            report_header += " (DIF Crossover UP Only)"
        
    report_str = f"""
================================================================================
📈 {report_header}
Period: Last {BACKTEST_PERIOD_DAYS} days ({TIMEFRAME})
Costs: {COMMISSION_PERCENT}% commission/trade, {SLIPPAGE_PERCENT}% slippage
================================================================================

--- Net Performance (After Costs) ---
Total Net PnL: ${total_net_pnl:,.2f}
Total Trades: {total_trades}
Win Rate: {win_rate:.2f}%
Profit Factor: {profit_factor:.2f}

--- Averages (Net) ---
Average Winning Trade: ${avg_win:,.2f}
Average Losing Trade: -${avg_loss:,.2f}
Average Risk/Reward Ratio: {risk_reward_ratio:.2f}:1

--- Totals (Net) ---
Gross Profit: ${gross_profit:,.2f} ({len(winning_trades)} trades)
Gross Loss: -${gross_loss:,.2f} ({len(losing_trades)} trades)
Total Commissions Paid: ${df_trades['commission_total'].sum():,.2f}
"""
    logger.info(report_str)
    
    try:
        if not os.path.exists('reports'):
            os.makedirs('reports')
        report_filename = os.path.join('reports', f"backtest_report_{BASE_ML_MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_trades.to_csv(report_filename, index=False)
        logger.info(f"\n================================================================================\n✅ Full trade log saved to: {report_filename}\n================================================================================\n")
    except Exception as e:
        logger.error(f"Could not save report to CSV: {e}")

# ==============================================================================
# ---------------------------- الوظيفة الرئيسية للاختبار ------------------------
# ==============================================================================

def start_backtesting_job():
    """
    هذه هي الوظيفة التي تقوم بتشغيل عملية الاختبار الخلفي بأكملها.
    """
    logger.info("🚀 Starting backtesting job...")
    time.sleep(2) 
    
    symbols_to_test = get_validated_symbols()
    
    if not symbols_to_test:
        logger.critical("❌ No valid symbols to test. Backtesting job will not run.")
        return
        
    all_trades = []
    
    data_fetch_days = BACKTEST_PERIOD_DAYS + 10
    
    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle:
            continue
            
        df_hist = fetch_historical_data(symbol, TIMEFRAME, data_fetch_days)
        if df_hist is None or df_hist.empty:
            continue
            
        backtest_start_date = datetime.utcnow() - timedelta(days=BACKTEST_PERIOD_DAYS)
        df_to_test = df_hist[df_hist.index >= backtest_start_date]

        trades = run_backtest_for_symbol(symbol, df_to_test, model_bundle)
        if trades:
            all_trades.extend(trades)
        
        time.sleep(0.5)

    generate_report(all_trades)
    
    if conn:
        conn.close()
        logger.info("✅ Database connection closed.")
        
    logger.info("👋 Backtesting job finished. The web service will remain active.")

# ==============================================================================
# --------------------------------- التنفيذ -----------------------------------
# ==============================================================================

if __name__ == "__main__":
    backtest_thread = Thread(target=start_backtesting_job)
    backtest_thread.daemon = True
    backtest_thread.start()

    port = int(os.environ.get("PORT", 10002))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
