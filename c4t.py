import os
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import psycopg2
from binance.client import Client
from decouple import config
from psycopg2.extras import RealDictCursor
from tqdm import tqdm

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
USE_TRAILING_STOP: bool = True
TRAILING_STOP_ACTIVATE_PERCENT: float = 0.75
TRAILING_STOP_DISTANCE_PERCENT: float = 1.0

# مبلغ افتراضي لكل صفقة بالدولار لمحاكاة الربح
INITIAL_TRADE_AMOUNT_USDT: float = 100.0

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

# تحميل متغيرات البيئة
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# إعداد عميل Binance
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
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)
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
    # Parameters from your scripts
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BBANDS_PERIOD, ATR_PERIOD = 14, 12, 26, 9, 20, 14
    BBANDS_STD_DEV = 2.0
    
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df_calc['rsi'] = 100 - (100 / (1 + rs))

    # MACD & MACD Cross
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd'] = ema_fast - ema_slow
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    macd_above = df_calc['macd'] > df_calc['macd_signal']
    macd_below = df_calc['macd'] < df_calc['macd_signal']
    df_calc['macd_cross'] = 0
    df_calc.loc[macd_above & macd_below.shift(1), 'macd_cross'] = 1
    df_calc.loc[macd_below & macd_above.shift(1), 'macd_cross'] = -1

    # Bollinger Bands
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_pos'] = (df_calc['close'] - sma) / std.replace(0, np.nan)

    # Time-based Features
    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['hour_of_day'] = df_calc.index.hour
    
    # Candlestick features
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
    """
    trades = []
    
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_names = model_bundle['feature_names']
    
    # تحضير البيانات
    df_featured = calculate_features(data)
    
    # التأكد من أن جميع الأعمدة المطلوبة موجودة
    for col in feature_names:
        if col not in df_featured.columns:
            logger.error(f"Missing feature '{col}' in data for {symbol}. Skipping backtest for this symbol.")
            return []

    features_scaled = scaler.transform(df_featured[feature_names])
    predictions = model.predict_proba(features_scaled)[:, 1]
    
    df_featured['prediction'] = predictions
    
    in_trade = False
    trade_details = {}

    # المرور على كل شمعة لمحاكاة التداول
    for i in range(len(df_featured)):
        current_candle = df_featured.iloc[i]
        
        # --- منطق إدارة الصفقة المفتوحة ---
        if in_trade:
            # التحقق من وصول السعر للهدف أو وقف الخسارة
            if current_candle['high'] >= trade_details['tp']:
                trade_details['exit_price'] = trade_details['tp']
                trade_details['exit_reason'] = 'TP Hit'
            elif current_candle['low'] <= trade_details['sl']:
                trade_details['exit_price'] = trade_details['sl']
                trade_details['exit_reason'] = 'SL Hit'
            
            # منطق الوقف المتحرك
            elif USE_TRAILING_STOP:
                activation_price = trade_details['entry_price'] * (1 + (TRAILING_STOP_ACTIVATE_PERCENT / 100))
                # إذا تم تفعيل الوقف المتحرك
                if not trade_details.get('tsl_active') and current_candle['high'] >= activation_price:
                    trade_details['tsl_active'] = True
                
                if trade_details.get('tsl_active'):
                    new_tsl = current_candle['close'] * (1 - (TRAILING_STOP_DISTANCE_PERCENT / 100))
                    # تحديث وقف الخسارة إذا كان الوقف الجديد أعلى
                    if new_tsl > trade_details['sl']:
                        trade_details['sl'] = new_tsl
            
            # إذا تم إغلاق الصفقة
            if trade_details.get('exit_price'):
                trade_details['exit_time'] = current_candle.name
                trade_details['duration_candles'] = i - trade_details['entry_index']
                trades.append(trade_details)
                in_trade = False
                trade_details = {}
            continue

        # --- منطق فتح صفقة جديدة ---
        if not in_trade and current_candle['prediction'] >= MODEL_PREDICTION_THRESHOLD:
            in_trade = True
            entry_price = current_candle['close']
            atr_value = current_candle['atr']
            
            stop_loss = entry_price - (atr_value * ATR_SL_MULTIPLIER)
            take_profit = entry_price + (atr_value * ATR_TP_MULTIPLIER)
            
            trade_details = {
                'symbol': symbol,
                'entry_time': current_candle.name,
                'entry_price': entry_price,
                'entry_index': i,
                'tp': take_profit,
                'sl': stop_loss,
                'initial_sl': stop_loss, # للحفاظ على وقف الخسارة الأولي
            }

    return trades

def generate_report(all_trades: List[Dict[str, Any]]):
    """
    تنشئ وتعرض تقريرًا مفصلاً بنتائج الاختبار الخلفي.
    """
    if not all_trades:
        logger.warning("No trades were executed during the backtest.")
        return

    df_trades = pd.DataFrame(all_trades)
    
    # حساب الربح والخسارة لكل صفقة
    df_trades['pnl_pct'] = ((df_trades['exit_price'] / df_trades['entry_price']) - 1) * 100
    df_trades['pnl_usdt'] = df_trades['pnl_pct'] / 100 * INITIAL_TRADE_AMOUNT_USDT

    # حساب الإحصائيات العامة
    total_trades = len(df_trades)
    winning_trades = df_trades[df_trades['pnl_usdt'] > 0]
    losing_trades = df_trades[df_trades['pnl_usdt'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = df_trades['pnl_usdt'].sum()
    
    gross_profit = winning_trades['pnl_usdt'].sum()
    gross_loss = abs(losing_trades['pnl_usdt'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = winning_trades['pnl_usdt'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_usdt'].mean() if len(losing_trades) > 0 else 0
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # عرض التقرير
    print("\n" + "="*80)
    print(f"📈 BACKTESTING REPORT: {BASE_ML_MODEL_NAME}")
    print(f"Period: Last {BACKTEST_PERIOD_DAYS} days ({TIMEFRAME} timeframe)")
    print("="*80)
    print("\n--- General Performance ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Net PnL: ${total_pnl:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")

    print("\n--- Averages ---")
    print(f"Average Winning Trade: ${avg_win:,.2f}")
    print(f"Average Losing Trade: ${avg_loss:,.2f}")
    print(f"Average Risk/Reward Ratio: {risk_reward_ratio:.2f}:1")
    
    print("\n--- Totals ---")
    print(f"Gross Profit: ${gross_profit:,.2f} ({len(winning_trades)} trades)")
    print(f"Gross Loss: ${gross_loss:,.2f} ({len(losing_trades)} trades)")
    
    # حفظ التقرير في ملف CSV
    try:
        report_filename = f"backtest_report_{BASE_ML_MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_trades.to_csv(report_filename)
        print("\n" + "="*80)
        print(f"✅ Full trade log saved to: {report_filename}")
        print("="*80 + "\n")
    except Exception as e:
        logger.error(f"Could not save report to CSV: {e}")

# ==============================================================================
# --------------------------------- التنفيذ -----------------------------------
# ==============================================================================

if __name__ == "__main__":
    logger.info("🚀 Starting backtester...")
    symbols_to_test = get_validated_symbols()
    
    if not symbols_to_test:
        logger.critical("❌ No valid symbols to test. Exiting.")
        exit(1)
        
    all_trades = []
    
    # استخدام tqdm لإظهار شريط التقدم
    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        # 1. تحميل النموذج
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle:
            continue
            
        # 2. جلب البيانات التاريخية
        # نضيف 30 يومًا إضافية لتكون كافية لحساب المؤشرات الأولية دون فقدان بيانات
        df_hist = fetch_historical_data(symbol, TIMEFRAME, BACKTEST_PERIOD_DAYS + 30)
        if df_hist is None or df_hist.empty:
            continue
            
        # 3. تنفيذ الاختبار
        trades = run_backtest_for_symbol(symbol, df_hist, model_bundle)
        if trades:
            all_trades.extend(trades)
        
        time.sleep(1) # لتجنب إغراق واجهة Binance بالطلبات

    # 4. إنشاء التقرير النهائي
    generate_report(all_trades)
    
    # إغلاق الاتصال بقاعدة البيانات
    if conn:
        conn.close()
        
    logger.info("👋 Backtester finished.")
