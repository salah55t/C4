import os
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from threading import Thread, Lock
import traceback

import numpy as np
import pandas as pd
import psycopg2
from binance.client import Client
from decouple import config
from psycopg2.extras import RealDictCursor
from flask import Flask, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

# ==============================================================================
# --------------------------- إعدادات الاختبار الخلفي ----------------------------
# ==============================================================================
BACKTEST_PERIOD_DAYS: int = 180
TIMEFRAME: str = '15m'
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V4'
MODEL_PREDICTION_THRESHOLD: float = 0.70
ATR_SL_MULTIPLIER: float = 1.5
ATR_TP_MULTIPLIER: float = 3.5
USE_RSI_FILTER: bool = True
RSI_LOWER_THRESHOLD: float = 40.0
RSI_UPPER_THRESHOLD: float = 69.0
COMMISSION_PERCENT: float = 0.1
SLIPPAGE_PERCENT: float = 0.05
INITIAL_TRADE_AMOUNT_USDT: float = 10.0

# ==============================================================================
# ---------------------------- إعدادات النظام والاتصال -------------------------
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('backtester.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger('Backtester')

# --- !!! جديد: إعداد Flask و الحالة العامة !!! ---
app = Flask(__name__, static_folder='reports')
CORS(app)

# الحالة العامة لمشاركة البيانات بين الخيوط (threads)
job_status = {
    "status": "IDLE",  # IDLE, RUNNING, COMPLETED, ERROR
    "progress": 0,
    "message": "لم يبدأ الاختبار بعد.",
    "current_symbol": "",
    "total_symbols": 0,
    "results": None
}
status_lock = Lock()
backtest_thread: Optional[Thread] = None


# تحميل متغيرات البيئة
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

client: Optional[Client] = None
conn: Optional[psycopg2.extensions.connection] = None

def initialize_connections():
    global client, conn
    try:
        if not client:
            client = Client(API_KEY, API_SECRET)
            logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        if not conn or conn.closed:
            conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ فشل الاتصال: {e}")
        with status_lock:
            job_status.update({"status": "ERROR", "message": f"فشل في تهيئة الاتصالات: {e}"})

# ==============================================================================
# ------------------- دوال مساعدة (منسوخة من السكريبت الأصلي) -------------------
# ==============================================================================
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [Validation] Reading symbols from '{filename}'...")
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}"); return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
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

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BBANDS_PERIOD, ATR_PERIOD = 14, 12, 26, 9, 20, 14
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
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd'] = ema_fast - ema_slow
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['hour_of_day'] = df_calc.index.hour
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
                logger.info(f"✅ [Model] Loaded '{model_name}' for {symbol}.")
                return model_bundle
            logger.warning(f"⚠️ [Model] Model '{model_name}' not found for {symbol}.")
            return None
    except Exception as e:
        logger.error(f"❌ [Model] Error loading model for {symbol}: {e}", exc_info=True); return None

# ==============================================================================
# ----------------------------- محرك الاختبار الخلفي ----------------------------
# ==============================================================================

def run_backtest_for_symbol(symbol: str, data: pd.DataFrame, model_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    trades = []
    model, scaler, feature_names = model_bundle['model'], model_bundle['scaler'], model_bundle['feature_names']
    df_featured = calculate_features(data.copy())
    if not all(col in df_featured.columns for col in feature_names): return []
    features_df = df_featured[feature_names]
    features_scaled_np = scaler.transform(features_df)
    df_featured['prediction'] = model.predict_proba(features_scaled_np)[:, 1]
    
    in_trade = False
    trade_details = {}
    for i in range(len(df_featured)):
        candle = df_featured.iloc[i]
        if in_trade:
            if candle['high'] >= trade_details['tp']: trade_details.update({'exit_price': trade_details['tp'], 'exit_reason': 'TP Hit'})
            elif candle['low'] <= trade_details['sl']: trade_details.update({'exit_price': trade_details['sl'], 'exit_reason': 'SL Hit'})
            if trade_details.get('exit_price'):
                trade_details.update({'exit_time': candle.name, 'duration_candles': i - trade_details['entry_index']})
                trades.append(trade_details)
                in_trade, trade_details = False, {}
            continue
        
        passes_rsi = not USE_RSI_FILTER or (RSI_LOWER_THRESHOLD <= candle.get('rsi', 0) <= RSI_UPPER_THRESHOLD)
        if not in_trade and passes_rsi and candle['prediction'] >= MODEL_PREDICTION_THRESHOLD:
            in_trade = True
            entry_price = candle['close']
            atr = candle['atr']
            sl, tp = entry_price - (atr * ATR_SL_MULTIPLIER), entry_price + (atr * ATR_TP_MULTIPLIER)
            trade_details = {'symbol': symbol, 'entry_time': candle.name, 'entry_price': entry_price, 'entry_index': i, 'tp': tp, 'sl': sl}
    return trades

def generate_report(all_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not all_trades: return {"error": "No trades were executed."}
    df = pd.DataFrame(all_trades)
    df['entry_price_adj'] = df['entry_price'] * (1 + SLIPPAGE_PERCENT / 100)
    df['exit_price_adj'] = df['exit_price'] * (1 - SLIPPAGE_PERCENT / 100)
    df['pnl_pct_raw'] = ((df['exit_price_adj'] / df['entry_price_adj']) - 1) * 100
    entry_cost = INITIAL_TRADE_AMOUNT_USDT
    exit_value = entry_cost * (1 + df['pnl_pct_raw'] / 100)
    commission_entry = entry_cost * (COMMISSION_PERCENT / 100)
    commission_exit = exit_value * (COMMISSION_PERCENT / 100)
    df['commission_total'] = commission_entry + commission_exit
    df['pnl_usdt_net'] = (exit_value - entry_cost) - df['commission_total']
    df['pnl_pct_net'] = (df['pnl_usdt_net'] / INITIAL_TRADE_AMOUNT_USDT) * 100

    wins = df[df['pnl_usdt_net'] > 0]
    losses = df[df['pnl_usdt_net'] <= 0]
    total_trades = len(df)
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_net_pnl = df['pnl_usdt_net'].sum()
    gross_profit = wins['pnl_usdt_net'].sum()
    gross_loss = abs(losses['pnl_usdt_net'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_win = wins['pnl_usdt_net'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl_usdt_net'].mean()) if len(losses) > 0 else 0

    # --- حفظ التقرير ---
    report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        if not os.path.exists('reports'): os.makedirs('reports')
        df.to_csv(os.path.join('reports', report_filename), index=False)
        logger.info(f"✅ Full trade log saved to: reports/{report_filename}")
    except Exception as e:
        logger.error(f"Could not save report CSV: {e}")
        report_filename = None

    report_data = {
        "total_net_pnl": total_net_pnl, "total_trades": total_trades, "win_rate": win_rate,
        "profit_factor": profit_factor, "avg_win": avg_win, "avg_loss": avg_loss,
        "total_commission": df['commission_total'].sum(),
        "report_filename": report_filename
    }
    logger.info(f"📊 Backtest Report Generated: {report_data}")
    return report_data


# ==============================================================================
# ---------------------------- الوظيفة الرئيسية للاختبار ------------------------
# ==============================================================================

def start_backtesting_job():
    global job_status
    try:
        initialize_connections()
        logger.info("🚀 Starting backtesting job...")
        
        symbols_to_test = get_validated_symbols()
        if not symbols_to_test:
            raise ValueError("No valid symbols to test.")
        
        with status_lock:
            job_status.update({
                "status": "RUNNING", "message": "Fetching symbols...", 
                "total_symbols": len(symbols_to_test), "progress": 0
            })
        
        all_trades = []
        data_fetch_days = BACKTEST_PERIOD_DAYS + 10
        
        for i, symbol in enumerate(symbols_to_test):
            with status_lock:
                job_status.update({
                    "progress": (i / len(symbols_to_test)) * 100,
                    "message": f"({i+1}/{len(symbols_to_test)}) Testing {symbol}...",
                    "current_symbol": symbol
                })

            model_bundle = load_ml_model_bundle_from_db(symbol)
            if not model_bundle: continue
            df_hist = fetch_historical_data(symbol, TIMEFRAME, data_fetch_days)
            if df_hist is None or df_hist.empty: continue
            
            backtest_start_date = datetime.utcnow() - timedelta(days=BACKTEST_PERIOD_DAYS)
            df_to_test = df_hist[df_hist.index >= backtest_start_date]
            trades = run_backtest_for_symbol(symbol, df_to_test, model_bundle)
            if trades: all_trades.extend(trades)
            time.sleep(0.1) # لمنع استهلاك الموارد بشكل مفرط

        with status_lock:
            job_status.update({"status": "RUNNING", "message": "Generating final report..."})
        
        final_results = generate_report(all_trades)
        
        with status_lock:
            job_status.update({"status": "COMPLETED", "progress": 100, "message": "اكتمل الاختبار بنجاح!", "results": final_results})

    except Exception as e:
        logger.error(f"❌ An error occurred during the backtest job: {e}", exc_info=True)
        with status_lock:
            job_status.update({"status": "ERROR", "message": f"خطأ: {e}\n{traceback.format_exc()}"})
    finally:
        if conn:
            conn.close()
            logger.info("🔌 Database connection closed.")

# ==============================================================================
# --------------------------------- واجهة API (Flask) ---------------------------
# ==============================================================================

@app.route('/')
def home():
    try:
        return render_template_string(open('trainer_dashboard.html', encoding='utf-8').read())
    except FileNotFoundError:
        return "Error: 'trainer_dashboard.html' not found.", 404

@app.route('/api/status')
def get_status():
    with status_lock:
        return jsonify(job_status)

@app.route('/api/start', methods=['POST'])
def start_job():
    global backtest_thread
    with status_lock:
        if backtest_thread and backtest_thread.is_alive():
            return jsonify({"error": "الاختبار قيد التشغيل بالفعل."}), 409
        
        # إعادة تعيين الحالة قبل البدء
        job_status.update({"status": "IDLE", "progress": 0, "message": "التحضير لبدء الاختبار...", "results": None})
        
        backtest_thread = Thread(target=start_backtesting_job)
        backtest_thread.daemon = True
        backtest_thread.start()
        return jsonify({"message": "تم بدء الاختبار بنجاح."})

@app.route('/reports/<path:filename>')
def download_report(filename):
    return send_from_directory('reports', filename, as_attachment=True)


# ==============================================================================
# --------------------------------- التنفيذ -----------------------------------
# ==============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10002))
    logger.info(f"🌍 Starting web server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
