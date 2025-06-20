import os
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import psycopg2
import pandas_ta as ta
from binance.client import Client
from decouple import config
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
from flask import Flask

# ==============================================================================
# --------------------------- إعدادات الاختبار الخلفي (محدثة لـ V6) ----------------------------
# ==============================================================================
# الفترة الزمنية للاختبار بالايام
BACKTEST_PERIOD_DAYS: int = 45
# الإطار الزمني للشموع (يجب أن يطابق إطار تدريب النموذج)
TIMEFRAME: str = '15m'
# اسم النموذج الأساسي الذي سيتم اختباره (تم التحديث إلى V6)
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6'

# --- معلمات الاستراتيجية (تم تحديثها لتطابق إعدادات البوت c4.py) ---
MODEL_PREDICTION_THRESHOLD: float = 0.70
ATR_SL_MULTIPLIER: float = 1.5
ATR_TP_MULTIPLIER: float = 2.0
USE_TRAILING_STOP: bool = False

# --- محاكاة التكاليف الواقعية ---
COMMISSION_PERCENT: float = 0.1
SLIPPAGE_PERCENT: float = 0.05
INITIAL_TRADE_AMOUNT_USDT: float = 10.0

# --- معلمات المؤشرات (من c4.py V6) ---
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
BOLLINGER_PERIOD: int = 20
STDEV_PERIOD: int = 20
ADX_PERIOD: int = 14
ROC_PERIOD: int = 10
MFI_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
BTC_SYMBOL = 'BTCUSDT'

# ==============================================================================
# ---------------------------- إعدادات النظام والاتصال -------------------------
# ==============================================================================

# إعداد التسجيل (Logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_v6.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BacktesterV6')

# إعداد خادم الويب
app = Flask(__name__)
@app.route('/')
def health_check():
    return "Backtester service for V6 is running."

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
        active_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        
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

# ---!!! تحديث: V6 Feature Engineering ---
def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    V6 Feature Engineering. Uses pandas_ta for consistency and robustness.
    """
    df_calc = df.copy()
    
    # 1. Historical Price Data Features
    df_calc['returns'] = df_calc.ta.percent_return(close=df_calc['close'], append=True)
    df_calc['log_returns'] = df_calc.ta.log_return(close=df_calc['close'], append=True)
    
    # Moving Averages
    df_calc.ta.ema(length=EMA_FAST_PERIOD, append=True)
    df_calc.ta.ema(length=EMA_SLOW_PERIOD, append=True)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc[f'EMA_{EMA_FAST_PERIOD}']) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc[f'EMA_{EMA_SLOW_PERIOD}']) - 1
    
    # Volatility Features
    df_calc.ta.atr(length=ATR_PERIOD, append=True)
    bbands = df_calc.ta.bbands(length=BOLLINGER_PERIOD, append=True)
    df_calc['bollinger_width'] = bbands[f'BBB_{BOLLINGER_PERIOD}_2.0']
    df_calc['return_std_dev'] = df_calc['returns'].rolling(window=STDEV_PERIOD).std()
    
    # 2. Technical Indicator Features
    # Momentum Indicators
    df_calc.ta.rsi(length=RSI_PERIOD, append=True)
    df_calc.ta.roc(length=ROC_PERIOD, append=True)
    df_calc.ta.mfi(length=MFI_PERIOD, append=True)
    macd = df_calc.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True)
    
    # Volume Indicators
    df_calc.ta.obv(append=True)
    df_calc.ta.ad(append=True)
    
    # Trend Indicators
    df_calc.ta.adx(length=ADX_PERIOD, append=True)

    # 3. Broader Market Features
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = df_calc['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    
    # 4. Time and Date Features
    df_calc['day_of_week'] = df_calc.index.dayofweek # Monday=0, Sunday=6
    df_calc['hour_of_day'] = df_calc.index.hour
    
    # --- Placeholders for future advanced features ---
    # 5. Order Book Data (Requires WebSocket connection to order book stream)
    # df_calc['bid_ask_spread'] = np.nan # Example: best_ask - best_bid
    # df_calc['order_book_imbalance'] = np.nan # Example: (sum_bid_vol - sum_ask_vol) / (sum_bid_vol + sum_ask_vol)
    
    # 6. Sentiment Features (Requires API connection to news/social media data providers)
    # df_calc['news_sentiment'] = np.nan # Example: score from -1 to 1
    # df_calc['twitter_velocity'] = np.nan # Example: change in mention count

    # Clean up column names generated by pandas_ta
    df_calc.columns = [col.upper() for col in df_calc.columns]
    
    return df_calc.dropna()


def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    """
    تحمل حزمة النموذج (النموذج + المعاير + أسماء الميزات) من قاعدة البيانات.
    """
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
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

def run_backtest_for_symbol(symbol: str, data: pd.DataFrame, btc_data: pd.DataFrame, model_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    تقوم بتنفيذ محاكاة التداول على البيانات التاريخية لعملة واحدة.
    """
    trades = []
    
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_names = model_bundle['feature_names']
    
    df_featured = calculate_features(data, btc_data)
    
    # التأكد من أن جميع الأعمدة المطلوبة موجودة
    missing = [col for col in feature_names if col not in df_featured.columns]
    if missing:
        logger.error(f"Missing features {missing} for {symbol}. Skipping.")
        return []

    features_df = df_featured[feature_names]
    features_scaled_np = scaler.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled_np, columns=feature_names, index=features_df.index)
    
    try:
        # For binary classification, predict_proba returns shape (n_samples, 2)
        # We want the probability of the positive class (1)
        class_1_index = list(model.classes_).index(1)
        predictions = model.predict_proba(features_scaled_df)[:, class_1_index]
    except (ValueError, IndexError) as e:
        logger.error(f"Could not get probability for class '1' in model for {symbol}: {e}. Skipping.")
        return []
    
    df_featured['prediction'] = predictions
    
    in_trade = False
    trade_details = {}

    for i in range(len(df_featured)):
        current_candle = df_featured.iloc[i]
        
        if in_trade:
            # Check for TP/SL hit within the current candle's high/low
            if current_candle['HIGH'] >= trade_details['tp']:
                trade_details['exit_price'] = trade_details['tp']
                trade_details['exit_reason'] = 'TP Hit'
            elif current_candle['LOW'] <= trade_details['sl']:
                trade_details['exit_price'] = trade_details['sl']
                trade_details['exit_reason'] = 'SL Hit'
            
            if trade_details.get('exit_price'):
                trade_details['exit_time'] = current_candle.name
                trade_details['duration_candles'] = i - trade_details['entry_index']
                trades.append(trade_details)
                in_trade = False
                trade_details = {}
            continue

        if not in_trade and current_candle['PREDICTION'] >= MODEL_PREDICTION_THRESHOLD:
            in_trade = True
            entry_price = current_candle['CLOSE']
            atr_series_name = f'ATRr_{ATR_PERIOD}'.upper()
            atr_value = current_candle[atr_series_name]
            
            stop_loss = entry_price - (atr_value * ATR_SL_MULTIPLIER)
            take_profit = entry_price + (atr_value * ATR_TP_MULTIPLIER)
            
            trade_details = {
                'symbol': symbol,
                'entry_time': current_candle.name,
                'entry_price': entry_price,
                'entry_index': i,
                'tp': take_profit,
                'sl': stop_loss,
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
    
    # تطبيق الانزلاق السعري (Slippage)
    df_trades['entry_price_adj'] = df_trades['entry_price'] * (1 + SLIPPAGE_PERCENT / 100)
    df_trades['exit_price_adj'] = df_trades['exit_price'] * (1 - SLIPPAGE_PERCENT / 100)
    
    # حساب الربح/الخسارة قبل العمولات
    df_trades['pnl_pct_raw'] = ((df_trades['exit_price_adj'] / df_trades['entry_price_adj']) - 1) * 100
    
    # حساب العمولات والربح الصافي
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
📈 BACKTESTING REPORT: {BASE_ML_MODEL_NAME}
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
    logger.info(f"🚀 Starting backtesting job for {BASE_ML_MODEL_NAME} Strategy...")
    time.sleep(2) 
    
    symbols_to_test = get_validated_symbols()
    
    if not symbols_to_test:
        logger.critical("❌ No valid symbols to test. Backtesting job will not run.")
        return
        
    all_trades = []
    
    data_fetch_days = BACKTEST_PERIOD_DAYS + 50 # Fetch more data for indicator warmup
    
    logger.info(f"ℹ️ [BTC Data] Fetching historical data for {BTC_SYMBOL}...")
    btc_data = fetch_historical_data(BTC_SYMBOL, TIMEFRAME, data_fetch_days)
    if btc_data is None:
        logger.critical("❌ Failed to fetch BTC data. Cannot proceed with backtest.")
        return
    btc_data['btc_returns'] = btc_data['close'].pct_change()
    logger.info("✅ [BTC Data] Successfully fetched and processed BTC data.")

    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols", ncols=100):
        if symbol == BTC_SYMBOL:
            continue
            
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle:
            continue
            
        df_hist = fetch_historical_data(symbol, TIMEFRAME, data_fetch_days)
        if df_hist is None or df_hist.empty:
            continue
            
        # Filter data for the actual backtest period after calculating features
        backtest_start_date = datetime.utcnow() - timedelta(days=BACKTEST_PERIOD_DAYS)
        df_with_features = calculate_features(df_hist, btc_data)
        
        df_to_test = df_with_features[df_with_features.index >= backtest_start_date]
        if df_to_test.empty:
            continue

        # For backtesting, we pass the full featured df to ensure context, but simulate on the filtered period
        trades = run_backtest_for_symbol(symbol, df_to_test, btc_data, model_bundle)
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
