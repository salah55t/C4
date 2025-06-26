import os
import logging
import pickle
import time
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
from binance.client import Client
from decouple import config
from tqdm import tqdm
from flask import Flask
from threading import Thread

# ==============================================================================
# --------------------------- إعدادات الاختبار الخلفي ----------------------------
# ==============================================================================
# الفترة الزمنية للاختبار بالايام
BACKTEST_PERIOD_DAYS: int = 90
# الإطار الزمني للشموع (يجب أن يطابق إطار تدريب النموذج)
TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
# اسم النموذج الأساسي الذي سيتم اختباره
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
# فترة جلب البيانات يجب أن تكون أطول لتغطية حساب المؤشرات
DATA_FETCH_LOOKBACK_DAYS: int = BACKTEST_PERIOD_DAYS + 60

# --- معلمات الاستراتيجية (تم تحديثها لتطابق إعدادات البوت c4.py) ---
MODEL_PREDICTION_THRESHOLD: float = 0.80 # عتبة ثقة النموذج لتوليد إشارة أولية

# *** جديد: معلمات استراتيجية التوصيات المعلقة ***
USE_PENDING_RECOMMENDATION_STRATEGY: bool = True # تفعيل/إلغاء الاستراتيجية الجديدة
# وقف الخسارة الأولي (يستخدم كسعر تفعيل)
INITIAL_SL_ATR_MULTIPLIER: float = 1.5
# الهدف الأولي (يستخدم كهدف ثاني في الصفقة المفعلة)
INITIAL_TP_ATR_MULTIPLIER: float = 2.5
# وقف الخسارة للصفقة المفعلة (بعد التفعيل)
ACTIVATED_SL_ATR_MULTIPLIER: float = 2.0

# --- محاكاة التكاليف الواقعية ---
COMMISSION_PERCENT: float = 0.1
SLIPPAGE_PERCENT: float = 0.05
INITIAL_TRADE_AMOUNT_USDT: float = 10.0

# --- معلمات المؤشرات (مطابقة لملف c4.py) ---
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
        logging.FileHandler('backtester_v5_pending_strat.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BacktesterV5_PendingStrat')

app = Flask(__name__)
@app.route('/')
def health_check():
    return "Backtester service for V5 Pending Strategy is running."

try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}"); exit(1)

client: Optional[Client] = None
try:
    client = Client(API_KEY, API_SECRET)
    logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
except Exception as e:
    logger.critical(f"❌ [Binance] فشل الاتصال: {e}"); exit(1)

# ==============================================================================
# ------------------- دوال مساعدة (منسوخة ومعدلة من ملفاتك) --------------------
# ==============================================================================
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
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
    if not client: return None
    try:
        start_str = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
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

def calculate_all_features(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df_calc = df_15m.copy()
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
    ema_fast_macd = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow_macd = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast_macd - ema_slow_macd
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    rsi_stoch = df_calc['rsi']
    min_rsi = rsi_stoch.rolling(window=STOCH_RSI_PERIOD).min()
    max_rsi = rsi_stoch.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi_stoch - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['market_condition'] = 0
    df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1
    df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc = calculate_candlestick_patterns(df_calc)
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
    mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
    df_featured = df_calc.join(mtf_features)
    df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].ffill()
    return df_featured.dropna()

def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df_patterns = df.copy()
    op, hi, lo, cl = df_patterns['open'], df_patterns['high'], df_patterns['low'], df_patterns['close']
    body = abs(cl - op); candle_range = hi - lo; candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    df_patterns['candlestick_pattern'] = 0
    df_patterns.loc[(body / candle_range) < 0.05, 'candlestick_pattern'] = 3
    df_patterns.loc[(body > candle_range * 0.1) & (lower_wick >= body * 2) & (upper_wick < body), 'candlestick_pattern'] = 2
    df_patterns.loc[(body > candle_range * 0.1) & (upper_wick >= body * 2) & (lower_wick < body), 'candlestick_pattern'] = -2
    df_patterns.loc[(cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1)), 'candlestick_pattern'] = 1
    df_patterns.loc[(cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1)), 'candlestick_pattern'] = -1
    df_patterns.loc[(cl > op) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1), 'candlestick_pattern'] = 4
    df_patterns.loc[(op > cl) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1), 'candlestick_pattern'] = -4
    return df_patterns

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    model_dir = 'Mo'
    file_path = os.path.join(model_dir, f"{model_name}.pkl")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                model_bundle = pickle.load(f)
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                logger.info(f"✅ [Model] Successfully loaded model '{model_name}' for {symbol} from file: {file_path}")
                return model_bundle
            else:
                logger.error(f"❌ [Model] Model bundle in file '{file_path}' is incomplete.")
                return None
        except Exception as e:
            logger.error(f"❌ [Model] Unexpected error loading model from '{file_path}': {e}", exc_info=True)
            return None
    else:
        logger.warning(f"⚠️ [Model] Model file '{file_path}' not found for {symbol}.")
        if not os.path.isdir(model_dir):
            logger.warning(f"⚠️ [Model] The model directory '{model_dir}' does not exist.")
        return None

# ==============================================================================
# ----------------------------- محرك الاختبار الخلفي (مُعدَّل) ----------------------------
# ==============================================================================

def run_backtest_for_symbol(symbol: str, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_data: pd.DataFrame, model_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    trades = []
    model, scaler, feature_names = model_bundle['model'], model_bundle['scaler'], model_bundle['feature_names']

    df_featured = calculate_all_features(df_15m, df_4h, btc_data)
    if df_featured is None or df_featured.empty:
        logger.warning(f"⚠️ Could not calculate features for {symbol}. Skipping."); return []

    missing = [col for col in feature_names if col not in df_featured.columns]
    if missing: logger.error(f"Missing features {missing} for {symbol}. Skipping."); return []

    features_df = df_featured[feature_names]
    features_scaled_np = scaler.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled_np, columns=feature_names, index=features_df.index)

    try:
        class_1_index = list(model.classes_).index(1)
        predictions = model.predict_proba(features_scaled_df)[:, class_1_index]
    except (ValueError, IndexError):
        logger.error(f"Could not find class '1' in model for {symbol}. Skipping."); return []

    df_featured['prediction'] = predictions

    # --- متغيرات محاكاة الاستراتيجية الجديدة ---
    in_trade = False
    trade_details = {}
    pending_recommendation: Optional[Dict[str, Any]] = None

    for i in range(len(df_featured)):
        current_candle = df_featured.iloc[i]
        candle_time = current_candle.name

        # --- 1. إدارة الصفقة النشطة ---
        if in_trade:
            # التحقق من إغلاق الصفقة (هدف أو وقف)
            exit_price = None
            exit_reason = None
            if current_candle['high'] >= trade_details['tp']:
                exit_price = trade_details['tp']
                exit_reason = 'TP Hit'
            elif current_candle['low'] <= trade_details['sl']:
                exit_price = trade_details['sl']
                exit_reason = 'SL Hit'

            if exit_price:
                trade_details.update({
                    'exit_price': exit_price, 'exit_reason': exit_reason,
                    'exit_time': candle_time, 'duration_candles': i - trade_details['entry_index']
                })
                trades.append(trade_details)
                in_trade = False
                trade_details = {}
            continue # الانتقال للشمعة التالية بعد إدارة الصفقة

        # --- 2. إدارة التوصية المعلقة (فقط إذا لم نكن في صفقة) ---
        if pending_recommendation and USE_PENDING_RECOMMENDATION_STRATEGY:
            if current_candle['low'] <= pending_recommendation['trigger_price']:
                # *** تفعيل التوصية المعلقة ***
                in_trade = True
                entry_price = pending_recommendation['trigger_price'] # الدخول عند سعر التفعيل
                atr_value = pending_recommendation['atr_at_creation'] # استخدام ATR الأصلي للثبات

                # حساب الأهداف الجديدة ووقف الخسارة
                tp1 = pending_recommendation['original_entry_price']
                tp2 = pending_recommendation['original_target_price']
                new_stop_loss = entry_price - (atr_value * ACTIVATED_SL_ATR_MULTIPLIER)

                trade_details = {
                    'symbol': symbol, 'entry_time': candle_time, 'entry_price': entry_price,
                    'entry_index': i, 'tp': tp2, # الهدف النهائي هو الهدف الثاني
                    'sl': new_stop_loss, 'strategy_type': 'Pending-Triggered',
                    'details': {'TP1': tp1, 'TP2': tp2}
                }
                pending_recommendation = None # استهلاك التوصية
                continue # الانتقال للشمعة التالية بعد فتح الصفقة

        # --- 3. توليد إشارة جديدة (توصية معلقة أو صفقة مباشرة) ---
        if current_candle['prediction'] >= MODEL_PREDICTION_THRESHOLD:
            if USE_PENDING_RECOMMENDATION_STRATEGY:
                # *** إنشاء توصية معلقة جديدة (أو تحديثها) ***
                initial_entry = current_candle['close']
                atr_value = current_candle['atr']
                initial_sl = initial_entry - (atr_value * INITIAL_SL_ATR_MULTIPLIER)
                initial_tp = initial_entry + (atr_value * INITIAL_TP_ATR_MULTIPLIER)

                pending_recommendation = {
                    'symbol': symbol,
                    'creation_time': candle_time,
                    'original_entry_price': initial_entry,
                    'original_target_price': initial_tp,
                    'trigger_price': initial_sl,
                    'atr_at_creation': atr_value
                }
            else:
                # *** فتح صفقة مباشرة (الاستراتيجية القديمة) ***
                in_trade = True
                entry_price = current_candle['close']
                atr_value = current_candle['atr']
                stop_loss = entry_price - (atr_value * INITIAL_SL_ATR_MULTIPLIER)
                take_profit = entry_price + (atr_value * INITIAL_TP_ATR_MULTIPLIER)

                trade_details = {
                    'symbol': symbol, 'entry_time': candle_time, 'entry_price': entry_price,
                    'entry_index': i, 'tp': take_profit, 'sl': stop_loss,
                    'strategy_type': 'Direct-Entry', 'details': {}
                }

    return trades

def generate_report(all_trades: List[Dict[str, Any]]):
    if not all_trades:
        logger.warning("No trades were executed during the backtest."); return

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
    
    strategy_name = "Pending Recommendation Strategy" if USE_PENDING_RECOMMENDATION_STRATEGY else "Direct Entry Strategy"

    report_str = f"""
================================================================================
📈 BACKTESTING REPORT: {BASE_ML_MODEL_NAME}
Strategy: {strategy_name}
Period: Last {BACKTEST_PERIOD_DAYS} days ({TIMEFRAME} + {HIGHER_TIMEFRAME} MTF)
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
        report_filename = os.path.join('reports', f"backtest_report_{BASE_ML_MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_trades.to_csv(report_filename, index=False)
        logger.info(f"\n================================================================================\n✅ Full trade log saved to: {report_filename}\n================================================================================\n")
    except Exception as e:
        logger.error(f"Could not save report to CSV: {e}")

# ==============================================================================
# ---------------------------- الوظيفة الرئيسية للاختبار ------------------------
# ==============================================================================
def start_backtesting_job():
    logger.info("🚀 Starting backtesting job for V5 Strategy...")
    time.sleep(2)

    symbols_to_test = get_validated_symbols()
    if not symbols_to_test: logger.critical("❌ No valid symbols to test. Backtesting job will not run."); return

    all_trades = []

    logger.info(f"ℹ️ [BTC Data] Fetching historical data for {BTC_SYMBOL}...")
    btc_data_15m = fetch_historical_data(BTC_SYMBOL, TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
    if btc_data_15m is None: logger.critical("❌ Failed to fetch BTC data. Cannot proceed."); return
    btc_data_15m['btc_returns'] = btc_data_15m['close'].pct_change()
    logger.info("✅ [BTC Data] Successfully fetched and processed BTC data.")

    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        if symbol == BTC_SYMBOL: continue
        
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        if not model_bundle: continue

        df_15m = fetch_historical_data(symbol, TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
        if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: continue

        backtest_start_date = datetime.now(UTC) - timedelta(days=BACKTEST_PERIOD_DAYS)
        df_15m_test = df_15m[df_15m.index >= backtest_start_date].copy()

        trades = run_backtest_for_symbol(symbol, df_15m_test, df_4h, btc_data_15m, model_bundle)
        if trades: all_trades.extend(trades)

        time.sleep(0.5)

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
