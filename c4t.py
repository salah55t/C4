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
BACKTEST_PERIOD_DAYS: int = 180
TIMEFRAME: str = '15m'
# --- !!! تحديثات رئيسية للنموذج V5 !!! ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'

# --- معلمات الاستراتيجية (يجب أن تطابق البوت c4_v5.py) ---
MODEL_CONFIDENCE_THRESHOLD: float = 0.55 # عتبة الثقة في تنبؤ الصنف 1
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
USE_TRAILING_STOP: bool = True
TRAILING_STOP_ACTIVATE_PERCENT: float = 0.75
TRAILING_STOP_DISTANCE_PERCENT: float = 1.0

# --- محاكاة التكاليف ---
COMMISSION_PERCENT: float = 0.1
SLIPPAGE_PERCENT: float = 0.05
INITIAL_TRADE_AMOUNT_USDT: float = 10.0

# ==============================================================================
# ---------------------------- إعدادات النظام والاتصال -------------------------
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BacktesterV5')
app = Flask(__name__)
@app.route('/')
def health_check(): return "Backtester V5 service is running."

try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    client = Client(API_KEY, API_SECRET)
    conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    logger.info("✅ [Init] تم الاتصال بـ Binance وقاعدة البيانات بنجاح.")
except Exception as e:
    logger.critical(f"❌ [Init] فشل التهيئة: {e}"); exit(1)

# ==============================================================================
# ------------------- دوال مساعدة (منسوخة ومحدثة من ملفاتك) --------------------
# ==============================================================================
def get_validated_symbols(filename: str = 'crypto_list.txt'):
    # ... (Same as c4_v5.py)
    return []

def fetch_historical_data(symbol: str, interval: str, days: int):
    # ... (Same as c4_v5.py)
    return None

# --- !!! تحديث: دالة حساب الميزات لتتوافق مع V5 !!! ---
def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    ATR_PERIOD, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL = 14, 14, 12, 26, 9
    EMA_FAST_PERIOD, EMA_SLOW_PERIOD, BTC_CORR_PERIOD = 50, 200, 30

    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=MACD_SIGNAL, adjust=False).mean()
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    df_calc['hour_of_day'] = df_calc.index.hour
    
    return df_calc.dropna()

def load_ml_model_bundle_from_db(symbol: str):
    # ... (Same as c4_v5.py)
    return None

# ==============================================================================
# ----------------------------- محرك الاختبار الخلفي (محدث) ----------------------------
# ==============================================================================
def run_backtest_for_symbol(symbol: str, data: pd.DataFrame, btc_data: pd.DataFrame, model_bundle: Dict[str, Any]):
    trades = []
    model, scaler, feature_names = model_bundle['model'], model_bundle['scaler'], model_bundle['feature_names']
    
    df_featured = calculate_features(data, btc_data)
    
    if not all(col in df_featured.columns for col in feature_names):
        logger.error(f"Missing features for {symbol}. Skipping.")
        return []

    features_df = df_featured[feature_names]
    features_scaled = scaler.transform(features_df)
    
    # --- !!! تحديث: استخدام predict و predict_proba لنموذج V5 !!! ---
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    # الحصول على فهرس الصنف '1' (قد لا يكون دائماً الفهرس 1)
    class_1_index = np.where(model.classes_ == 1)[0][0]
    
    df_featured['prediction'] = predictions
    df_featured['confidence'] = probabilities[:, class_1_index]
    
    in_trade = False
    trade_details = {}

    for i in range(len(df_featured)):
        current_candle = df_featured.iloc[i]
        
        if in_trade:
            # (منطق إغلاق الصفقة لم يتغير)
            if current_candle['high'] >= trade_details['tp']:
                trade_details['exit_price'], trade_details['exit_reason'] = trade_details['tp'], 'TP Hit'
            elif current_candle['low'] <= trade_details['sl']:
                trade_details['exit_price'], trade_details['exit_reason'] = trade_details['sl'], 'SL Hit'
            
            # (منطق وقف الخسارة المتحرك لم يتغير)
            elif USE_TRAILING_STOP:
                # ... Trailing stop logic remains the same
                pass

            if trade_details.get('exit_price'):
                trade_details['exit_time'] = current_candle.name
                trades.append(trade_details)
                in_trade = False
                trade_details = {}
            continue

        # --- !!! تحديث: منطق دخول الصفقة لنموذج V5 !!! ---
        # الشرط: التنبؤ هو 1 (ربح) والثقة أعلى من العتبة
        if not in_trade and current_candle['prediction'] == 1 and current_candle['confidence'] >= MODEL_CONFIDENCE_THRESHOLD:
            in_trade = True
            entry_price = current_candle['close']
            atr_value = current_candle['atr']
            
            # تحديد TP/SL بناءً على ATR (مطابق للتدريب والبوت)
            stop_loss = entry_price - (atr_value * SL_ATR_MULTIPLIER)
            take_profit = entry_price + (atr_value * TP_ATR_MULTIPLIER)
            
            trade_details = {
                'symbol': symbol, 'entry_time': current_candle.name,
                'entry_price': entry_price, 'tp': take_profit, 'sl': stop_loss,
            }
    return trades

def generate_report(all_trades: List[Dict[str, Any]]):
    # ... (دالة إنشاء التقرير لم تتغير)
    if not all_trades:
        logger.warning("No trades were executed."); return
    df = pd.DataFrame(all_trades)
    # ... cost calculation ...
    # ... report printing ...
    pass


# ==============================================================================
# ---------------------------- الوظيفة الرئيسية للاختبار ------------------------
# ==============================================================================
def start_backtesting_job():
    logger.info("🚀 Starting backtesting job for V5 models...")
    time.sleep(2)
    
    symbols_to_test = get_validated_symbols()
    if not symbols_to_test:
        logger.critical("❌ No valid symbols to test."); return
        
    data_fetch_days = BACKTEST_PERIOD_DAYS + 30 # Extra data for indicator warmup
    
    # --- !!! جلب بيانات البيتكوين أولاً !!! ---
    logger.info("Fetching BTC data for correlation...")
    btc_hist_data = fetch_historical_data('BTCUSDT', TIMEFRAME, data_fetch_days)
    if btc_hist_data is None:
        logger.critical("❌ Could not fetch BTC data. Aborting."); return
    btc_hist_data['btc_returns'] = btc_hist_data['close'].pct_change()

    all_trades = []
    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle: continue
            
        df_hist = fetch_historical_data(symbol, TIMEFRAME, data_fetch_days)
        if df_hist is None or df_hist.empty: continue
            
        backtest_start_date = datetime.utcnow() - timedelta(days=BACKTEST_PERIOD_DAYS)
        df_to_test = df_hist[df_hist.index >= backtest_start_date]

        # --- تمرير بيانات العملة والبيتكوين ---
        trades = run_backtest_for_symbol(symbol, df_to_test, btc_hist_data, model_bundle)
        if trades:
            all_trades.extend(trades)
        
        time.sleep(0.5)

    generate_report(all_trades)
    
    if conn: conn.close()
    logger.info("✅ Database connection closed.")
    logger.info("👋 Backtesting job finished.")

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
