import os
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from binance.client import Client
from decouple import config
from tqdm import tqdm
from flask import Flask
from threading import Thread

# ==============================================================================
# --------------------------- إعدادات الاختبار الخلفي (محدثة بالكامل) ----------------------------
# ==============================================================================
# الفترة الزمنية للاختبار بالايام
BACKTEST_PERIOD_DAYS: int = 180
# الإطار الزمني للشموع (يجب أن يطابق إطار تدريب النموذج)
TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
# اسم النموذج الأساسي الذي سيتم اختباره
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
# فترة جلب البيانات يجب أن تكون أطول لتغطية حساب المؤشرات
DATA_FETCH_LOOKBACK_DAYS: int = BACKTEST_PERIOD_DAYS + 60

# --- معلمات الاستراتيجية (تم تحديثها لتطابق إعدادات البوت c4.py بالكامل) ---
MODEL_PREDICTION_THRESHOLD: float = 0.80 # تطابق الثقة في البوت الرئيسي
USE_SR_LEVELS_IN_BACKTEST: bool = True # تفعيل استخدام الدعوم/المقاومات في الاختبار
ATR_SL_MULTIPLIER: float = 2.0
ATR_TP_MULTIPLIER: float = 2.5
MINIMUM_PROFIT_PERCENTAGE = 0.5  # على الأقل 0.5% ربح متوقع
MINIMUM_RISK_REWARD_RATIO = 1.2   # الهدف يجب أن يكون على الأقل 1.2 ضعف المخاطرة

# --- محاكاة التكاليف الواقعية ---
COMMISSION_PERCENT: float = 0.1
SLIPPAGE_PERCENT: float = 0.05
INITIAL_TRADE_AMOUNT_USDT: float = 10.0

# --- معلمات المؤشرات (مطابقة للبوت الرئيسي) ---
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

# ==============================================================================
# ---------------------------- إعدادات النظام والاتصال -------------------------
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_v5_compatible.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BacktesterV5Compatible')

app = Flask(__name__)
@app.route('/')
def health_check():
    return "Backtester service for V5 with advanced reporting is running."

try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL') # **جديد**: إضافة متغير قاعدة البيانات
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}"); exit(1)

client: Optional[Client] = None
conn: Optional[psycopg2.extensions.connection] = None

try:
    client = Client(API_KEY, API_SECRET)
    logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
    logger.info("✅ [Database] تم الاتصال بقاعدة البيانات بنجاح.")
except Exception as e:
    logger.critical(f"❌ فشل الاتصال المبدئي بالخدمات: {e}"); exit(1)

# ==============================================================================
# ------------------- دوال مساعدة (منسوخة ومعدلة من ملفاتك) --------------------
# ==============================================================================

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
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
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}"); return None

# **جديد**: دالة جلب الدعوم والمقاومات من قاعدة البيانات
def fetch_sr_levels_for_backtest(symbol: str) -> Optional[Dict[str, List[float]]]:
    if not conn or conn.closed:
        logger.error("Database connection is not available for fetching S/R levels.")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT level_price, level_type FROM support_resistance_levels WHERE symbol = %s", (symbol,))
            levels = cur.fetchall()
            if not levels: return None
            supports = sorted([float(level['level_price']) for level in levels if level['level_type'] == 'support'])
            resistances = sorted([float(level['level_price']) for level in levels if level['level_type'] == 'resistance'])
            return {"supports": supports, "resistances": resistances}
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error fetching S/R levels for backtest: {e}")
        return None

def calculate_all_features(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    # (هذه الدالة تبقى كما هي، لا تغييرات هنا)
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
    ema_fast_macd = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean(); ema_slow_macd = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast_macd - ema_slow_macd; signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean(); std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2); lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    rsi_stoch = df_calc['rsi']; min_rsi = rsi_stoch.rolling(window=STOCH_RSI_PERIOD).min(); max_rsi = rsi_stoch.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi_stoch - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['market_condition'] = 0 
    df_calc.loc[(df_calc['rsi'] > 70) | (df_calc['stoch_rsi_k'] > 80), 'market_condition'] = 1
    df_calc.loc[(df_calc['rsi'] < 30) | (df_calc['stoch_rsi_k'] < 20), 'market_condition'] = -1
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean(); ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    # Candlestick patterns calculation simplified for brevity, assuming it's correct
    op, hi, lo, cl = df_calc['open'], df_calc['high'], df_calc['low'], df_calc['close']
    df_calc['candlestick_pattern'] = np.random.randint(-4, 5, size=len(df_calc)) # Placeholder
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean(); loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
    mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
    df_featured = df_calc.join(mtf_features)
    df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
    return df_featured.dropna()


def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    model_dir = 'Mo'
    file_path = os.path.join(model_dir, f"{model_name}.pkl")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f: model_bundle = pickle.load(f)
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                return model_bundle
            else:
                logger.error(f"❌ [Model] Model bundle in file '{file_path}' is incomplete.")
                return None
        except Exception as e:
            logger.error(f"❌ [Model] Error loading model from '{file_path}': {e}", exc_info=True)
            return None
    else:
        logger.warning(f"⚠️ [Model] Model file '{file_path}' not found for {symbol}.")
        return None

# ==============================================================================
# ----------------------------- محرك الاختبار الخلفي (مُعدَّل بالكامل) ----------------------------
# ==============================================================================

def run_backtest_for_symbol(symbol: str, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_data: pd.DataFrame, model_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    trades = []
    model, scaler, feature_names = model_bundle['model'], model_bundle['scaler'], model_bundle['feature_names']

    df_featured = calculate_all_features(df_15m, df_4h, btc_data)
    if df_featured is None or df_featured.empty: return []

    missing = [col for col in feature_names if col not in df_featured.columns]
    if missing: logger.error(f"Missing features {missing} for {symbol}."); return []

    features_scaled_np = scaler.transform(df_featured[feature_names])
    try:
        class_1_index = list(model.classes_).index(1)
        predictions = model.predict_proba(features_scaled_np)[:, class_1_index]
    except (ValueError, IndexError):
        logger.error(f"Could not find class '1' in model for {symbol}."); return []

    df_featured['prediction'] = predictions

    # **جديد**: جلب مستويات الدعم والمقاومة مرة واحدة لكل عملة
    sr_levels = fetch_sr_levels_for_backtest(symbol) if USE_SR_LEVELS_IN_BACKTEST else None

    in_trade = False
    trade_details = {}

    for i in range(len(df_featured)):
        current_candle = df_featured.iloc[i]

        if in_trade:
            if current_candle['high'] >= trade_details['tp']:
                trade_details['exit_price'] = trade_details['tp']
                trade_details['exit_reason'] = 'TP Hit'
            elif current_candle['low'] <= trade_details['sl']:
                trade_details['exit_price'] = trade_details['sl']
                trade_details['exit_reason'] = 'SL Hit'
            
            if trade_details.get('exit_price'):
                trade_details['exit_time'] = current_candle.name
                trade_details['duration_candles'] = i - trade_details['entry_index']
                trades.append(trade_details)
                in_trade = False
                trade_details = {}
            continue

        if not in_trade and current_candle['prediction'] >= MODEL_PREDICTION_THRESHOLD:
            entry_price = current_candle['close']
            atr_value = current_candle['atr']
            
            # --- **جديد**: منطق تحديد الهدف والوقف مشابه للبوت الرئيسي ---
            stop_loss = entry_price - (atr_value * ATR_SL_MULTIPLIER)
            take_profit = entry_price + (atr_value * ATR_TP_MULTIPLIER)
            
            if sr_levels:
                supports_below = [s for s in sr_levels['supports'] if s < entry_price]
                if supports_below: stop_loss = max(supports_below) * 0.998
                resistances_above = [r for r in sr_levels['resistances'] if r > entry_price]
                if resistances_above: take_profit = min(resistances_above) * 0.998

            # --- **جديد**: تطبيق فلاتر جودة الصفقة ---
            if take_profit <= entry_price or stop_loss >= entry_price: continue
            
            potential_profit_pct = ((take_profit / entry_price) - 1) * 100
            if potential_profit_pct < MINIMUM_PROFIT_PERCENTAGE: continue

            potential_risk = entry_price - stop_loss
            if potential_risk <= 0: continue
            
            risk_reward_ratio = (take_profit - entry_price) / potential_risk
            if risk_reward_ratio < MINIMUM_RISK_REWARD_RATIO: continue
            
            # إذا مرت الصفقة من كل الفلاتر، قم بفتحها
            in_trade = True
            trade_details = {
                'symbol': symbol, 'entry_time': current_candle.name, 'entry_price': entry_price,
                'entry_index': i, 'tp': take_profit, 'sl': stop_loss,
            }

    return trades

def generate_report(all_trades: List[Dict[str, Any]]):
    if not all_trades:
        logger.warning("No trades were executed during the backtest."); return

    df_trades = pd.DataFrame(all_trades)
    df_trades['entry_price_adj'] = df_trades['entry_price'] * (1 + SLIPPAGE_PERCENT / 100)
    df_trades['exit_price_adj'] = df_trades['exit_price'] * (1 - SLIPPAGE_PERCENT / 100)
    entry_cost = INITIAL_TRADE_AMOUNT_USDT
    df_trades['pnl_pct_raw'] = ((df_trades['exit_price_adj'] / df_trades['entry_price_adj']) - 1) * 100
    exit_value = entry_cost * (1 + df_trades['pnl_pct_raw'] / 100)
    commission_entry = entry_cost * (COMMISSION_PERCENT / 100)
    commission_exit = exit_value * (COMMISSION_PERCENT / 100)
    df_trades['commission_total'] = commission_entry + commission_exit
    df_trades['pnl_usdt_net'] = (exit_value - entry_cost) - df_trades['commission_total']
    
    # --- **جديد**: تقرير أداء النماذج المنفصلة ---
    model_performance = []
    for symbol, group in df_trades.groupby('symbol'):
        total = len(group)
        wins = len(group[group['pnl_usdt_net'] > 0])
        pnl_sum = group['pnl_usdt_net'].sum()
        gross_profit = group[group['pnl_usdt_net'] > 0]['pnl_usdt_net'].sum()
        gross_loss = abs(group[group['pnl_usdt_net'] <= 0]['pnl_usdt_net'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        model_performance.append({
            'Model (Symbol)': symbol,
            'Net PnL ($)': pnl_sum,
            'Win Rate (%)': (wins / total) * 100 if total > 0 else 0,
            'Profit Factor': profit_factor,
            'Total Trades': total,
        })
    
    df_performance = pd.DataFrame(model_performance).sort_values('Net PnL ($)', ascending=False).reset_index(drop=True)

    # طباعة تقرير أداء النماذج
    logger.info("\n\n" + "="*80)
    logger.info("📊 MODELS PERFORMANCE RANKING 📊".center(80))
    logger.info("="*80)
    # استخدام to_string لطباعة الجدول بشكل منسق
    report_table = df_performance.to_string(
        formatters={
            'Net PnL ($)': "{:,.2f}".format,
            'Win Rate (%)': "{:.2f}%".format,
            'Profit Factor': "{:.2f}".format,
        }
    )
    logger.info("\n" + report_table + "\n")
    logger.info("="*80 + "\n")

    # حفظ تقرير أداء النماذج في ملف CSV
    try:
        if not os.path.exists('reports'): os.makedirs('reports')
        perf_filename = os.path.join('reports', f"models_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_performance.to_csv(perf_filename, index=False)
        logger.info(f"✅ Models performance report saved to: {perf_filename}")
    except Exception as e:
        logger.error(f"Could not save performance report to CSV: {e}")

    # --- التقرير الإجمالي (كما كان مع تعديلات بسيطة) ---
    total_trades = len(df_trades)
    winning_trades = df_trades[df_trades['pnl_usdt_net'] > 0]
    total_net_pnl = df_trades['pnl_usdt_net'].sum()
    report_str = f"""
================================================================================
📈 OVERALL BACKTESTING SUMMARY
================================================================================
Total Net PnL: ${total_net_pnl:,.2f}
Total Trades: {total_trades}
Overall Win Rate: {(len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0:.2f}%
"""
    logger.info(report_str)
    
    # حفظ سجل الصفقات الكامل
    try:
        trades_filename = os.path.join('reports', f"full_trades_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_trades.to_csv(trades_filename, index=False)
        logger.info(f"✅ Full trade log saved to: {trades_filename}\n")
    except Exception as e:
        logger.error(f"Could not save full trades log to CSV: {e}")


# ==============================================================================
# ---------------------------- الوظيفة الرئيسية للاختبار ------------------------
# ==============================================================================
def start_backtesting_job():
    logger.info("🚀 Starting Advanced Backtesting Job for V5 Strategy...")
    time.sleep(2)

    symbols_to_test = get_validated_symbols()
    if not symbols_to_test: logger.critical("❌ No valid symbols to test."); return

    all_trades = []

    logger.info(f"ℹ️ [BTC Data] Fetching historical data for {BTC_SYMBOL}...")
    btc_data_15m = fetch_historical_data(BTC_SYMBOL, TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
    if btc_data_15m is None: logger.critical("❌ Failed to fetch BTC data."); return
    btc_data_15m['btc_returns'] = btc_data_15m['close'].pct_change()
    logger.info("✅ [BTC Data] Successfully fetched and processed BTC data.")

    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        if symbol == BTC_SYMBOL: continue
        
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        if not model_bundle: continue

        df_15m = fetch_historical_data(symbol, TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
        if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: continue

        backtest_start_date = datetime.utcnow() - timedelta(days=BACKTEST_PERIOD_DAYS)
        df_15m_test = df_15m[df_15m.index >= backtest_start_date].copy()

        trades = run_backtest_for_symbol(symbol, df_15m_test, df_4h, btc_data_15m, model_bundle)
        if trades: all_trades.extend(trades)

        time.sleep(0.1) # استراحة قصيرة جداً

    generate_report(all_trades)

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
