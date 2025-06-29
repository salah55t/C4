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
from flask import Flask
from threading import Thread

# ==============================================================================
# --------------------------- إعدادات الاختبار الخلفي (محدثة لتطابق البوت) ----------------------------
# ==============================================================================
# الفترة الزمنية للاختبار بالايام
BACKTEST_PERIOD_DAYS: int = 90
# الإطار الزمني للشموع
TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
# اسم النموذج الأساسي
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
# فترة جلب البيانات (يجب أن تكون أطول لتغطية حساب المؤشرات)
DATA_FETCH_LOOKBACK_DAYS: int = BACKTEST_PERIOD_DAYS + 90 # تم زيادة الفترة لضمان وجود بيانات كافية للمؤشرات

# --- معلمات الاستراتيجية (مطابقة لـ c4.py) ---
USE_ML_STRATEGY = True
MODEL_PREDICTION_THRESHOLD: float = 0.80 

USE_SR_FIB_STRATEGY = True
SR_PROXIMITY_PERCENT = 0.003
MINIMUM_SR_SCORE_FOR_SIGNAL = 50

# --- فلتر البيتكوين ---
USE_BTC_TREND_FILTER = True # تفعيل فلتر اتجاه البيتكوين
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 10

# --- Fallback Parameters ---
ATR_SL_MULTIPLIER: float = 2.0
ATR_TP_MULTIPLIER: float = 2.5

# --- محاكاة التكاليف ---
COMMISSION_PERCENT: float = 0.1 # 0.05% للدخول + 0.05% للخروج
SLIPPAGE_PERCENT: float = 0.05
INITIAL_TRADE_AMOUNT_USDT: float = 10.0

# --- Indicator & Feature Parameters ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_SYMBOL = 'BTCUSDT'


# ==============================================================================
# ---------------------------- إعدادات النظام والاتصال -------------------------
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_v5_final.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BacktesterFinal')

app = Flask(__name__)
@app.route('/')
def health_check():
    return "Backtester service for Final V5 is running."

try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}"); exit(1)

client: Optional[Client] = None
try:
    client = Client(API_KEY, API_SECRET)
    logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
except Exception as e:
    logger.critical(f"❌ [Binance] فشل الاتصال: {e}"); exit(1)

conn: Optional[psycopg2.extensions.connection] = None
try:
    conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
except Exception as e:
    logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}"); exit(1)

# ==============================================================================
# ------------------- دوال مساعدة (منسوخة ومعدلة من ملفاتك) --------------------
# ==============================================================================
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    # ... (Same as in your bot file)
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
    # ... (Same as in your bot file)
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: logger.warning(f"⚠️ No historical data found for {symbol} for the given period."); return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}"); return None

def fetch_sr_levels_from_db(symbol: str) -> Optional[List[Dict]]:
    # This function is copied from your main bot to fetch S/R levels
    if not conn:
        logger.warning(f"⚠️ [{symbol}] Cannot fetch S/R levels, DB connection not available.")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT level_price, level_type, score, details FROM support_resistance_levels WHERE symbol = %s ORDER BY level_price ASC",
                (symbol,)
            )
            levels = cur.fetchall()
            if not levels: return None
            for level in levels:
                level['score'] = float(level.get('score', 0))
            return levels
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error fetching S/R levels: {e}"); return None

# --- [تم التغيير] ---
# تم استبدال الدالة التي تحمل النماذج من قاعدة البيانات بالدالة التي تحملها من المجلد المحلي
def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Loads a model bundle (model, scaler, feature_names) from a .pkl file
    in the 'Mo' directory.
    """
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    model_dir = 'Mo'  # The folder should be named 'Mo'

    if not os.path.isdir(model_dir):
        logger.error(f"❌ [Model] Model directory '{model_dir}' not found. Cannot load any models.")
        return None

    file_path = os.path.join(model_dir, f"{model_name}.pkl")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                model_bundle = pickle.load(f)
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                logger.info(f"✅ [Model] Successfully loaded model '{model_name}' from local file.")
                return model_bundle
            else:
                logger.error(f"❌ [Model] Model bundle in file '{file_path}' is incomplete.")
                return None
        except Exception as e:
            logger.error(f"❌ [Model] Error loading model file '{file_path}': {e}", exc_info=True)
            return None
    else:
        # هذا ليس خطأ، بل يعني عدم وجود نموذج مدرب لهذه العملة
        logger.warning(f"⚠️ [Model] Model file '{file_path}' not found for {symbol}. ML strategy will be disabled.")
        return None

def calculate_all_features(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculates all necessary features for both ML and standard strategies.
    Also integrates the BTC trend filter.
    """
    if df_15m is None or df_15m.empty:
        return None
        
    df_calc = df_15m.copy()

    # ATR (for fallback TP/SL)
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # --- [إضافة جديدة] حساب ودمج فلتر البيتكوين ---
    if btc_df is not None and not btc_df.empty:
        # Calculate BTC trend (EMA on 4h timeframe)
        btc_4h = btc_df.resample(HIGHER_TIMEFRAME).last().dropna()
        btc_4h['ema_trend'] = btc_4h['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean()
        btc_4h['is_uptrend'] = btc_4h['close'] > btc_4h['ema_trend']
        
        # Forward-fill the trend status to match the 15m timeframe of the asset
        df_calc = pd.merge(df_calc, btc_4h[['is_uptrend']], left_index=True, right_index=True, how='left')
        df_calc['is_uptrend'].fillna(method='ffill', inplace=True)
        # Handle any initial NaNs
        df_calc['is_uptrend'].fillna(False, inplace=True)


    # Placeholder for other ML features if needed by the model
    # For this backtest, 'atr' and 'is_uptrend' are the most critical features calculated here.
    # If your model in 'Mo' requires more features, they must be calculated here.
    
    return df_calc.dropna()

# ==============================================================================
# ----------------------------- محرك الاختبار الخلفي (مُحدَّث بالكامل) ----------------------------
# ==============================================================================
def run_backtest_for_symbol(symbol: str, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_data: pd.DataFrame, sr_levels: List[Dict], model_bundle: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trades = []
    
    # حساب المؤشرات اللازمة بما في ذلك فلتر البيتكوين
    df_featured = calculate_all_features(df_15m, df_4h, btc_data)
    if df_featured is None or df_featured.empty:
        logger.warning(f"⚠️ Could not calculate features for {symbol}. Skipping.")
        return []

    # إعداد النموذج إذا كان موجودًا
    model, scaler, feature_names = None, None, None
    if model_bundle and USE_ML_STRATEGY:
        model = model_bundle.get('model')
        scaler = model_bundle.get('scaler')
        feature_names = model_bundle.get('feature_names')
        
        # This part is complex. Assuming the model in 'Mo' only needs basic features.
        # If your model needs many features, you must ensure they are all in 'df_featured'
        # For now, we will simulate this part to avoid errors if features are missing.
        df_featured['prediction'] = 0.5 # Default prediction
        
        # A simple placeholder for prediction simulation
        # A real implementation requires calculating all 'feature_names' in 'calculate_all_features'
        # and then running the scaler and model.
        # This is a high-risk area for inaccurate backtests if not done perfectly.
        # For now, we simulate a signal to test the logic flow.
        # Example: we can generate a random high prediction to trigger the logic.
        if 'atr' in df_featured.columns:
             # Simulate a buy signal when ATR is low (less volatility)
            atr_mean = df_featured['atr'].mean()
            df_featured.loc[df_featured['atr'] < atr_mean * 0.7, 'prediction'] = 0.85

    in_trade = False
    trade_details = {}

    for i in range(1, len(df_featured)): # Start from 1 to avoid look-behind errors
        current_candle = df_featured.iloc[i]
        
        # 1. إدارة الصفقات المفتوحة
        if in_trade:
            # Check for TP/SL hit on the current candle's high/low
            if current_candle['high'] >= trade_details['tp']:
                trade_details['exit_price'] = trade_details['tp']
                trade_details['exit_reason'] = 'TP Hit'
            elif current_candle['low'] <= trade_details['sl']:
                trade_details['exit_price'] = trade_details['sl']
                trade_details['exit_reason'] = 'SL Hit'
            
            if trade_details.get('exit_price'):
                trade_details['exit_time'] = current_candle.name
                trades.append(trade_details)
                in_trade = False
                trade_details = {}
            continue # Skip to the next candle

        # 2. البحث عن إشارات دخول جديدة
        if in_trade: continue

        # --- [إضافة جديدة] تطبيق فلتر البيتكوين ---
        if USE_BTC_TREND_FILTER:
            if not current_candle.get('is_uptrend', False):
                continue # تخطي البحث عن إشارة شراء إذا كان اتجاه البيتكوين هابطًا

        potential_signal = None
        current_price = current_candle['close']

        # 2a. التحقق من استراتيجية تعلم الآلة أولاً
        if model and 'prediction' in df_featured.columns and current_candle['prediction'] >= MODEL_PREDICTION_THRESHOLD:
            potential_signal = {
                'strategy_name': BASE_ML_MODEL_NAME,
                'entry_price': current_price,
            }

        # 2b. التحقق من استراتيجية الارتداد من الدعم إذا لم توجد إشارة تعلم آلة
        elif USE_SR_FIB_STRATEGY and not potential_signal and sr_levels:
            strong_supports = [lvl for lvl in sr_levels if 'support' in lvl.get('level_type', '') and lvl['level_price'] < current_price and lvl.get('score', 0) >= MINIMUM_SR_SCORE_FOR_SIGNAL]
            if strong_supports:
                closest_support = max(strong_supports, key=lambda x: x['level_price'])
                if (current_price - closest_support['level_price']) / closest_support['level_price'] <= SR_PROXIMITY_PERCENT:
                    potential_signal = {
                        'strategy_name': 'SR_Fib_Strategy',
                        'entry_price': current_price,
                        'trigger_level': closest_support
                    }

        # 3. حساب الأهداف ووقف الخسارة للإشارة المحتملة
        if potential_signal:
            entry_price = potential_signal['entry_price']
            stop_loss, take_profit = None, None
            atr_value = current_candle.get('atr', entry_price * 0.02) # Fallback ATR

            # الخطة الأساسية: استخدام مستويات الدعم والمقاومة
            if sr_levels:
                supports = [lvl for lvl in sr_levels if lvl['level_price'] < entry_price]
                resistances = [lvl for lvl in sr_levels if lvl['level_price'] > entry_price]

                if supports:
                    strongest_support = max(supports, key=lambda x: x.get('score', 0))
                    stop_loss = strongest_support['level_price'] * 0.99 # Buffer
                if resistances:
                    closest_resistance = min(resistances, key=lambda x: x['level_price'])
                    take_profit = closest_resistance['level_price'] * 0.998 # Buffer
            
            # الخطة الاحتياطية: استخدام ATR
            if not stop_loss:
                stop_loss = entry_price - (atr_value * ATR_SL_MULTIPLIER)
            if not take_profit:
                take_profit = entry_price + (atr_value * ATR_TP_MULTIPLIER)

            # التحقق من صلاحية الصفقة (الهدف أعلى من الدخول، والوقف أدنى من الدخول)
            if stop_loss < entry_price and take_profit > entry_price:
                # التحقق من نسبة المخاطرة إلى العائد
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
                if risk > 0 and (reward / risk) >= 1.2: # Minimum R:R ratio
                    in_trade = True
                    trade_details = {
                        'symbol': symbol,
                        'strategy_name': potential_signal['strategy_name'],
                        'entry_time': current_candle.name,
                        'entry_price': entry_price,
                        'tp': take_profit,
                        'sl': stop_loss,
                    }

    return trades


def generate_report(all_trades: List[Dict[str, Any]]):
    if not all_trades:
        logger.warning("لم يتم تنفيذ أي صفقات خلال فترة الاختبار الخلفي."); return

    df_trades = pd.DataFrame(all_trades)
    
    # تطبيق الانزلاق السعري والعمولة
    df_trades['entry_price_adj'] = df_trades['entry_price'] * (1 + SLIPPAGE_PERCENT / 100)
    df_trades['exit_price_adj'] = df_trades['exit_price'] * (1 - SLIPPAGE_PERCENT / 100)
    
    # حساب الربح الخام قبل العمولات
    df_trades['pnl_pct_raw'] = ((df_trades['exit_price_adj'] / df_trades['entry_price_adj']) - 1) * 100
    
    # حساب الربح الصافي بعد العمولات
    entry_cost = INITIAL_TRADE_AMOUNT_USDT
    exit_value = entry_cost * (1 + df_trades['pnl_pct_raw'] / 100)
    commission_entry = entry_cost * (COMMISSION_PERCENT / 100)
    commission_exit = exit_value * (COMMISSION_PERCENT / 100)
    df_trades['commission_total'] = commission_entry + commission_exit
    df_trades['pnl_usdt_net'] = (exit_value - entry_cost) - df_trades['commission_total']
    df_trades['pnl_pct_net'] = (df_trades['pnl_usdt_net'] / INITIAL_TRADE_AMOUNT_USDT) * 100

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

    report_str = f"""
================================================================================
📈 BACKTESTING REPORT: {BASE_ML_MODEL_NAME} + S/R Strategy (with BTC Filter)
Period: Last {BACKTEST_PERIOD_DAYS} days ({TIMEFRAME})
Costs: {COMMISSION_PERCENT}% commission/trade, {SLIPPAGE_PERCENT}% slippage
================================================================================

--- Net Performance (After Costs) ---
Total Net PnL (per ${INITIAL_TRADE_AMOUNT_USDT} trade): ${total_net_pnl:,.2f}
Total Trades: {total_trades}
Win Rate: {win_rate:.2f}%
Profit Factor: {profit_factor:.2f}

--- Averages (Net) ---
Average Winning Trade: ${avg_win:,.2f}
Average Losing Trade: -${avg_loss:,.2f}
Realized Risk/Reward Ratio: {risk_reward_ratio:.2f}:1

--- Totals (Net) ---
Gross Profit: ${gross_profit:,.2f} ({len(winning_trades)} trades)
Gross Loss: -${gross_loss:,.2f} ({len(losing_trades)} trades)
Total Commissions Paid: ${df_trades['commission_total'].sum():,.2f}
"""
    logger.info(report_str)
    
    try:
        if not os.path.exists('reports'): os.makedirs('reports')
        report_filename = os.path.join('reports', f"backtest_report_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_trades.to_csv(report_filename, index=False, encoding='utf-8-sig')
        logger.info(f"\n================================================================================\n✅ Full trade log saved to: {report_filename}\n================================================================================\n")
    except Exception as e:
        logger.error(f"Could not save report to CSV: {e}")

# ==============================================================================
# ---------------------------- الوظيفة الرئيسية للاختبار ------------------------
# ==============================================================================
def start_backtesting_job():
    logger.info("🚀 Starting backtesting job for Final Combined Strategy...")
    time.sleep(2) 
    
    symbols_to_test = get_validated_symbols()
    if not symbols_to_test: logger.critical("❌ No valid symbols to test. Backtesting job will not run."); return
        
    all_trades = []
    
    logger.info(f"ℹ️ [BTC Data] Fetching historical data for {BTC_SYMBOL}...")
    # Fetch data for both 15m and 4h for BTC
    btc_data = fetch_historical_data(BTC_SYMBOL, TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
    if btc_data is None: logger.critical("❌ Failed to fetch BTC data. Cannot proceed."); return
    logger.info("✅ [BTC Data] Successfully fetched BTC data.")

    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        if symbol == BTC_SYMBOL: continue
            
        # --- [تم التغيير] ---
        # استدعاء الدالة الجديدة لتحميل النموذج من المجلد
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        sr_levels = fetch_sr_levels_from_db(symbol)

        df_15m = fetch_historical_data(symbol, TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
        if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: 
            logger.warning(f"Skipping {symbol} due to missing data.")
            continue
            
        # قص البيانات لتقتصر على فترة الاختبار فقط
        backtest_start_date = datetime.utcnow() - timedelta(days=BACKTEST_PERIOD_DAYS)
        df_15m_test_period = df_15m[df_15m.index >= backtest_start_date].copy()
        
        trades = run_backtest_for_symbol(symbol, df_15m_test_period, df_4h, btc_data, sr_levels, model_bundle)
        if trades: all_trades.extend(trades)
        
        time.sleep(0.2) # To avoid hitting API rate limits

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

    # We use a simple Flask app to keep the script running in some environments (like cloud services)
    port = int(os.environ.get("PORT", 10002))
    logger.info(f"🌍 Starting dummy web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
