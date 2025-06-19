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
MODEL_PREDICTION_THRESHOLD: float = 0.85 # تم تحديثه ليطابق البوت
ATR_SL_MULTIPLIER: float = 2.0
ATR_TP_MULTIPLIER: float = 3.0
USE_TRAILING_STOP: bool = False # البوت لا يستخدمه حالياً

# --- !!! جديد: فلاتر الاستراتيجية المحدثة لتطابق البوت c4.py !!! ---
# تم تحديث جميع المعلمات والفلاتر لتعكس بدقة منطق البوت الرئيسي
USE_RSI_FILTER: bool = True
RSI_LOWER_THRESHOLD: float = 45.0
RSI_UPPER_THRESHOLD: float = 65.0

USE_MACD_CROSS_FILTER: bool = True  # فلتر لتقاطع MACD الصعودي فقط

USE_STOCH_RSI_FILTER: bool = True
STOCH_RSI_LOWER_THRESHOLD: float = 25.0
STOCH_RSI_UPPER_THRESHOLD: float = 75.0

MIN_RELATIVE_VOLUME: float = 2.0  # فلتر جديد لحجم التداول النسبي

# ملاحظة: فلتر اتجاه البيتكوين (USE_BTC_TREND_FILTER) الموجود في البوت الرئيسي
# لم يتم تضمينه هنا لأنه يتطلب تحميل بيانات عملتين في نفس الوقت،
# مما يعقد عملية الاختبار الخلفي المبسطة. نتائج الاختبار قد تكون أكثر تفاؤلاً
# من أداء البوت الفعلي الذي يتوقف أثناء اتجاه البيتكوين الهابط.

# --- معلمات محاكاة التكاليف الواقعية ---
COMMISSION_PERCENT: float = 0.1
SLIPPAGE_PERCENT: float = 0.05
INITIAL_TRADE_AMOUNT_USDT: float = 10.0

# ==============================================================================
# ---------------------------- إعدادات النظام والاتصال -------------------------
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester')

app = Flask(__name__)
@app.route('/')
def health_check():
    return "Backtester service is running and alive."

try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

client: Optional[Client] = None
try:
    client = Client(API_KEY, API_SECRET)
    logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
except Exception as e:
    logger.critical(f"❌ [Binance] فشل الاتصال: {e}")
    exit(1)

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
    logger.info(f"ℹ️ [Validation] Reading symbols from '{filename}'...")
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path): logger.error(f"File not found: {file_path}"); return []
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
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: logger.warning(f"⚠️ No historical data for {symbol}"); return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}")
        return None

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    *** محدّثة بالكامل ***
    هذه الدالة الآن نسخة طبق الأصل من دالة حساب الميزات في البوت الرئيسي c4.py
    لضمان أن الاختبار الخلفي يستخدم نفس البيانات تمامًا.
    """
    df_calc = df.copy()
    # معلمات المؤشرات مطابقة للبوت الرئيسي
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BBANDS_PERIOD, ATR_PERIOD = 14, 12, 26, 10, 20, 14
    STOCH_RSI_PERIOD, STOCH_RSI_SMA_PERIOD, STOCH_RSI_K_PERIOD, STOCH_RSI_D_PERIOD = 14, 14, 3, 3
    BBANDS_STD_DEV: float = 2.0

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
    df_calc["rsi"] = 100 - (100 / (1 + rs))

    # Stochastic RSI
    stoch_rsi = 100 * ((df_calc['rsi'] - df_calc['rsi'].rolling(window=STOCH_RSI_PERIOD).min()) / \
                       (df_calc['rsi'].rolling(window=STOCH_RSI_PERIOD).max() - df_calc['rsi'].rolling(window=STOCH_RSI_PERIOD).min()).replace(0, np.nan))
    df_calc['stoch_rsi_k'] = stoch_rsi.rolling(window=STOCH_RSI_K_PERIOD).mean()
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_RSI_D_PERIOD).mean()

    # MACD and Cross
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd'] = ema_fast - ema_slow
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    macd_above = df_calc['macd'] > df_calc['macd_signal']
    macd_below = df_calc['macd'] < df_calc['macd_signal']
    df_calc['macd_cross'] = 0
    df_calc.loc[macd_above & macd_below.shift(1), 'macd_cross'] = 1  # تقاطع صعودي
    df_calc.loc[macd_below & macd_above.shift(1), 'macd_cross'] = -1 # تقاطع هبوطي

    # Bollinger Bands
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_upper'] = sma + (std * BBANDS_STD_DEV)
    df_calc['bb_lower'] = sma - (std * BBANDS_STD_DEV)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / sma
    df_calc['bb_pos'] = (df_calc['close'] - sma) / std.replace(0, np.nan)

    # Other Features
    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc['candle_body_size'] = (df_calc['close'] - df_calc['open']).abs()
    df_calc['upper_wick'] = df_calc['high'] - df_calc[['open', 'close']].max(axis=1)
    df_calc['lower_wick'] = df_calc[['open', 'close']].min(axis=1) - df_calc['low']
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    
    return df_calc.dropna()

def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result.get('model_data'):
                model_bundle = pickle.loads(result['model_data'])
                logger.info(f"✅ [Model] Loaded model '{model_name}' for {symbol}.")
                return model_bundle
            logger.warning(f"⚠️ [Model] Model '{model_name}' not found for {symbol}.")
            return None
    except Exception as e:
        logger.error(f"❌ [Model] Error loading model for {symbol}: {e}", exc_info=True)
        return None

# ==============================================================================
# ----------------------------- محرك الاختبار الخلفي ----------------------------
# ==============================================================================

def run_backtest_for_symbol(symbol: str, data: pd.DataFrame, model_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    *** محدّثة بالكامل ***
    تنفيذ محاكاة التداول مع تطبيق جميع الفلاتر الجديدة من البوت الرئيسي.
    """
    trades = []
    model, scaler, feature_names = model_bundle['model'], model_bundle['scaler'], model_bundle['feature_names']
    
    df_featured = calculate_features(data)
    
    missing_features = [col for col in feature_names if col not in df_featured.columns]
    if missing_features:
        logger.error(f"[{symbol}] Missing features required by model: {missing_features}. Skipping symbol.")
        return []

    features_df = df_featured[feature_names]
    features_scaled = scaler.transform(features_df)
    predictions = model.predict_proba(features_scaled)[:, 1]
    df_featured['prediction'] = predictions
    
    in_trade = False
    trade_details = {}

    for i in range(len(df_featured)):
        current_candle = df_featured.iloc[i]
        
        if in_trade:
            if current_candle['high'] >= trade_details['tp']:
                trade_details.update({'exit_price': trade_details['tp'], 'exit_reason': 'TP Hit'})
            elif current_candle['low'] <= trade_details['sl']:
                trade_details.update({'exit_price': trade_details['sl'], 'exit_reason': 'SL Hit'})
            
            if 'exit_price' in trade_details:
                trade_details.update({
                    'exit_time': current_candle.name,
                    'duration_candles': i - trade_details['entry_index']
                })
                trades.append(trade_details)
                in_trade, trade_details = False, {}
            continue

        # --- !!! جديد: منطق الدخول المحدث بالكامل ليطابق البوت c4.py !!! ---
        # 1. فلتر توقع النموذج
        if current_candle['prediction'] < MODEL_PREDICTION_THRESHOLD: continue
        
        # 2. فلتر تقاطع MACD
        if USE_MACD_CROSS_FILTER and current_candle.get('macd_cross') != 1: continue
        
        # 3. فلتر RSI
        if USE_RSI_FILTER:
            current_rsi = current_candle.get('rsi')
            if current_rsi is None or not (RSI_LOWER_THRESHOLD <= current_rsi <= RSI_UPPER_THRESHOLD): continue
            
        # 4. فلتر Stochastic RSI
        if USE_STOCH_RSI_FILTER:
            k, d = current_candle.get('stoch_rsi_k'), current_candle.get('stoch_rsi_d')
            if k is None or d is None or not \
               (STOCH_RSI_LOWER_THRESHOLD <= k <= STOCH_RSI_UPPER_THRESHOLD and 
                STOCH_RSI_LOWER_THRESHOLD <= d <= STOCH_RSI_UPPER_THRESHOLD): continue

        # 5. فلتر حجم التداول النسبي
        rel_vol = current_candle.get('relative_volume')
        if rel_vol is None or rel_vol < MIN_RELATIVE_VOLUME: continue

        # إذا مرت جميع الفلاتر، يمكن الدخول في صفقة
        if not in_trade:
            in_trade = True
            entry_price = current_candle['close']
            atr_value = current_candle['atr']
            stop_loss = entry_price - (atr_value * ATR_SL_MULTIPLIER)
            take_profit = entry_price + (atr_value * ATR_TP_MULTIPLIER)
            
            trade_details = {
                'symbol': symbol, 'entry_time': current_candle.name, 'entry_price': entry_price,
                'entry_index': i, 'tp': take_profit, 'sl': stop_loss,
            }
            # إضافة تفاصيل الفلاتر عند الدخول للصفقة للتحليل
            trade_details['debug_info'] = {
                'prediction': round(current_candle['prediction'], 4),
                'rsi': round(current_candle.get('rsi', -1), 2),
                'stoch_k': round(current_candle.get('stoch_rsi_k', -1), 2),
                'rel_volume': round(current_candle.get('relative_volume', -1), 2)
            }

    return trades

def generate_report(all_trades: List[Dict[str, Any]]):
    if not all_trades:
        logger.warning("No trades were executed during the backtest.")
        return

    df_trades = pd.DataFrame(all_trades)
    
    # تطبيق الانزلاق السعري والعمولة
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

    # إعداد التقرير
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

    # --- !!! جديد: بناء عنوان التقرير بشكل ديناميكي بناءً على الفلاتر المستخدمة !!! ---
    report_header = f"BACKTESTING REPORT: {BASE_ML_MODEL_NAME}"
    active_filters = []
    if USE_RSI_FILTER: active_filters.append(f"RSI ({RSI_LOWER_THRESHOLD}-{RSI_UPPER_THRESHOLD})")
    if USE_MACD_CROSS_FILTER: active_filters.append("MACD Cross")
    if USE_STOCH_RSI_FILTER: active_filters.append(f"StochRSI ({STOCH_RSI_LOWER_THRESHOLD}-{STOCH_RSI_UPPER_THRESHOLD})")
    if MIN_RELATIVE_VOLUME > 0: active_filters.append(f"RelVol (>{MIN_RELATIVE_VOLUME})")
    
    if active_filters:
        report_header += " | Filters: " + ", ".join(active_filters)
        
    report_str = f"""
================================================================================
📈 {report_header}
Period: Last {BACKTEST_PERIOD_DAYS} days ({TIMEFRAME}) | Model Threshold: {MODEL_PREDICTION_THRESHOLD}
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
        if not os.path.exists('reports'): os.makedirs('reports')
        report_filename = os.path.join('reports', f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # فك تفاصيل التصحيح إلى أعمدة منفصلة
        debug_df = pd.json_normalize(df_trades['debug_info']).fillna(-1)
        report_df = pd.concat([df_trades.drop(columns=['debug_info']), debug_df], axis=1)

        report_df.to_csv(report_filename, index=False, encoding='utf-8-sig')
        logger.info(f"\n================================================================================\n✅ Full trade log saved to: {report_filename}\n================================================================================\n")
    except Exception as e:
        logger.error(f"Could not save report to CSV: {e}")

# ==============================================================================
# ---------------------------- الوظيفة الرئيسية للاختبار ------------------------
# ==============================================================================

def start_backtesting_job():
    logger.info("🚀 Starting synchronized backtesting job...")
    time.sleep(2) 
    
    symbols_to_test = get_validated_symbols()
    if not symbols_to_test:
        logger.critical("❌ No valid symbols. Backtesting job will not run.")
        return
        
    all_trades = []
    data_fetch_days = BACKTEST_PERIOD_DAYS + 15 # أيام إضافية لحساب المؤشرات
    
    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle: continue
            
        df_hist = fetch_historical_data(symbol, TIMEFRAME, data_fetch_days)
        if df_hist is None or df_hist.empty: continue
            
        backtest_start_date = datetime.utcnow() - timedelta(days=BACKTEST_PERIOD_DAYS)
        df_to_test = df_hist[df_hist.index >= backtest_start_date.strftime('%Y-%m-%d')]

        if df_to_test.empty:
            logger.warning(f"[{symbol}] No data available for the backtest period. Skipping.")
            continue

        trades = run_backtest_for_symbol(symbol, df_to_test, model_bundle)
        if trades: all_trades.extend(trades)
        
        time.sleep(0.2) # إيقاف مؤقت طفيف لتجنب استهلاك الموارد

    generate_report(all_trades)
    
    if conn: conn.close(); logger.info("✅ Database connection closed.")
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
    
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=4)
    except ImportError:
        logger.warning("Waitress not found, using Flask's development server.")
        app.run(host='0.0.0.0', port=port)
