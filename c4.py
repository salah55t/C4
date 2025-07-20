# ملف c4_complete_v9_3_dynamic_tp_sl.py - نسخة مع أهداف ديناميكية وإدارة ذاكرة
# تم التحديث بواسطة Gemini
import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import redis
import re
import gc
import random
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Set, Tuple
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v9_dynamic_tp_sl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV9_DynamicTPSL')

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
    TELEGRAM_BOT_TOKEN: str = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID: str = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# --- متغيرات عامة وإعدادات البوت ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
are_filters_disabled: bool = False
filters_disabled_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v9"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 10.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.85
SYMBOL_PROCESSING_BATCH_SIZE: int = 20
BATCH_PROCESSING_SLEEP_SECONDS: int = 10

# --- إعدادات المؤشرات الفنية ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5

# --- إعدادات الفلاتر المتقدمة ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
USE_PEAK_FILTER: bool = True
PEAK_CHECK_PERIOD: int = 50
PULLBACK_THRESHOLD_PCT: float = 0.988
BREAKOUT_ALLOWANCE_PCT: float = 1.003
DYNAMIC_FILTER_ANALYSIS_INTERVAL: int = 300
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_WALL_MULTIPLIER: float = 10.0
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.02

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": {"key": "INITIALIZING", "ar": "تهيئة..."}, "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
dynamic_filter_profile_cache: Dict[str, Any] = {}
last_dynamic_filter_analysis_time: float = 0
dynamic_filter_lock = Lock()
last_market_state_check = 0

# --- قواميس الترجمة ---
REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة", "Low Volatility": "تقلب منخفض جداً",
    "BTC Correlation": "ارتباط ضعيف بالبيتكوين", "RRR Filter": "نسبة المخاطرة/العائد غير كافية",
    "Momentum/Strength Filter": "فلتر الزخم والقوة", "Peak/Pullback Filter": "فلتر القمة/التصحيح",
    "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف", "ML Model Rejected Signal": "نموذج التعلم الآلي رفض الإشارة",
    "Invalid Position Size": "حجم الصفقة غير صالح", "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى", "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات", "Order Book Imbalance": "اختلال توازن دفتر الطلبات",
    "Large Sell Wall Detected": "تم كشف جدار بيع ضخم",
    "Dynamic TP/SL Calculation Failed": "فشل حساب الأهداف الديناميكية"
}
TREND_TRANSLATIONS = {
    "STRONG_UPTREND": "اتجاه صاعد قوي", "UPTREND": "اتجاه صاعد",
    "STRONG_DOWNTREND": "اتجاه هابط قوي", "DOWNTREND": "اتجاه هابط",
    "RANGING": "متذبذب (تجميع)", "UNCERTAIN": "غير واضح", "INITIALIZING": "تهيئة..."
}

# --- دالة إرسال رسائل تليجرام ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10).raise_for_status()
        logger.info("✅ [Telegram] تم إرسال الرسالة بنجاح.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# --- دوال تهيئة الخدمات ---
# ... (init_db, check_db_connection, log_and_notify, log_rejection, init_redis, get_exchange_info_map, get_validated_symbols)
# هذه الدوال تبقى كما هي من الإصدار السابق

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f'Memory usage for DF reduced from {start_mem:.2f}MB to {end_mem:.2f}MB')
    return df

# --- دوال جلب البيانات وحساب الميزات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[cols]
        for col in cols:
            if col != 'timestamp': df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return reduce_mem_usage(df.dropna())
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

# *** دالة جديدة: حساب TP/SL الديناميكي ***
def calculate_dynamic_tp_sl(
    df: pd.DataFrame,
    entry_price: float,
    atr_multiplier_sl: float = 1.5,
    atr_multiplier_tp: float = 2.5,
    rrr_min: float = 1.5,
    pivot_lookback: int = 20
) -> Tuple[float, float]:
    """
    حساب TP/SL ديناميكي باستخدام ATR + Pivots + RRR
    """
    df_copy = df.copy()
    # 1. حساب ATR
    df_copy['H-L'] = df_copy['high'] - df_copy['low']
    df_copy['H-PC'] = abs(df_copy['high'] - df_copy['close'].shift(1))
    df_copy['L-PC'] = abs(df_copy['low'] - df_copy['close'].shift(1))
    df_copy['TR'] = df_copy[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_copy['ATR'] = df_copy['TR'].ewm(span=14, adjust=False).mean() # Using EWM for smoother ATR
    atr = df_copy['ATR'].iloc[-1]

    if atr == 0: raise ValueError("ATR is zero, cannot calculate TP/SL.")

    # 2. حساب النقاط المحورية
    recent_df = df_copy.iloc[-pivot_lookback:]
    pivot_high = recent_df['high'].max()
    pivot_low = recent_df['low'].min()

    # 3. حساب SL بناءً على أقرب دعم
    sl_from_atr = entry_price - (atr * atr_multiplier_sl)
    sl_from_pivot = pivot_low - (atr * 0.2) # Small buffer below pivot
    sl_price = min(sl_from_atr, sl_from_pivot) # Use the lower (safer) stop loss

    # 4. حساب TP بناءً على RRR الديناميكي
    risk = entry_price - sl_price
    if risk <= 0: raise ValueError("Risk is zero or negative, cannot calculate TP.")
    
    tp_from_rrr = entry_price + (risk * rrr_min)
    tp_from_atr = entry_price + (atr * atr_multiplier_tp)
    tp_from_pivot = pivot_high + (atr * 0.2) # Small buffer above pivot

    # اختيار أقرب TP يحقق RRR
    valid_tp_candidates = [tp for tp in [tp_from_rrr, tp_from_atr, tp_from_pivot] if (tp - entry_price) / risk >= rrr_min]
    if not valid_tp_candidates:
        # If no candidate meets RRR, use the lowest one that's still profitable
        tp_price = tp_from_rrr
    else:
        tp_price = min(valid_tp_candidates)

    # 5. تحسين SL للحد من الانزلاق (تقريب لأقرب سنت)
    sl_price = round(sl_price, 4)
    tp_price = round(tp_price, 4)

    return tp_price, sl_price

# ... (Rest of the functions like calculate_all_features, trading strategy, filters, etc. remain the same)
# ... The key change is in the main_loop_enhanced

# ---------------------- واجهة Flask (معدلة) ----------------------
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V9.3</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; --accent-gray: #484F58;}
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .trend-light { width: 1rem; height: 1rem; border-radius: 9999px; border: 2px solid #30363D; transition: all 0.5s ease; }
        .light-green { background-color: var(--accent-green); box-shadow: 0 0 8px 1px var(--accent-green); }
        .light-red { background-color: var(--accent-red); box-shadow: 0 0 8px 1px var(--accent-red); }
        .light-yellow { background-color: var(--accent-yellow); box-shadow: 0 0 8px 1px var(--accent-yellow); }
        .light-gray { background-color: var(--accent-gray); }
        .light-green-strong { background-color: var(--accent-green); animation: pulse-green 1.5s infinite; }
        .light-red-strong { background-color: var(--accent-red); animation: pulse-red 1.5s infinite; }
        @keyframes pulse-green { 0%, 100% { box-shadow: 0 0 10px 3px var(--accent-green); } 50% { box-shadow: 0 0 4px 1px var(--accent-green); } }
        @keyframes pulse-red { 0%, 100% { box-shadow: 0 0 10px 3px var(--accent-red); } 50% { box-shadow: 0 0 4px 1px var(--accent-red); } }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V9.3</span></h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        <!-- Rest of dashboard HTML and JavaScript is identical to V9.2 -->
    </div>
<script>
// All JavaScript remains the same as the previous version
</script>
</body></html>
"""
# All Flask routes remain the same as the previous version.
# ...

# ---------------------- حلقات النظام (معدلة) ----------------------
def main_loop_enhanced():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan: 
        log_and_notify("critical", "لا توجد عملات صالحة للمسح.", "SYSTEM")
        return
    
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 بدء دورة مسح جديدة...")
            gc.collect()
            determine_market_state_enhanced()
            
            with dynamic_filter_lock: filter_profile = dynamic_filter_profile_cache
            if not filter_profile: 
                logger.warning("🛑 لم يتم تحميل ملف الفلاتر. سيتم المحاولة مجدداً.")
                time.sleep(60)
                continue

            btc_data = fetch_historical_data(BTC_SYMBOL, '1h', 10) # Fetch less data for BTC trend
            
            shuffled_symbols = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for i in range(0, len(shuffled_symbols), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = shuffled_symbols[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                logger.info(f"⚙️ Processing batch {i//SYMBOL_PROCESSING_BATCH_SIZE + 1} with {len(batch)} symbols...")

                for symbol in batch:
                    try:
                        with signal_cache_lock:
                            if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                                continue
                        
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 5) # Fetch less data for signal
                        if df_15m is None or len(df_15m) < 50: continue # Ensure enough data for pivots
                        
                        # The ML model part would go here (omitted for clarity of TP/SL change)
                        # For now, let's assume a signal is found and we need to set TP/SL
                        
                        entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])

                        # *** استبدال الآلية القديمة بالجديدة ***
                        try:
                            tp_price, sl_price = calculate_dynamic_tp_sl(df_15m, entry_price)
                            tp_sl_data = {'target_price': tp_price, 'stop_loss': sl_price, 'source': 'Dynamic_Pivot_ATR'}
                        except Exception as e:
                            log_rejection(symbol, "Dynamic TP/SL Calculation Failed", {"error": str(e)})
                            continue
                        
                        # The rest of the logic (filters, order placement) would follow...
                        # For example:
                        # if not passes_filters(symbol, last_features, filter_profile, entry_price, tp_sl_data, df_15m): continue
                        # ... create signal, place order, etc. ...
                        
                        # This is a placeholder to show the data is used
                        logger.info(f"✅ [{symbol}] Dynamic levels calculated: TP={tp_sl_data['target_price']}, SL={tp_sl_data['stop_loss']}")


                    except Exception as e:
                        logger.error(f"❌ [Processing Error] للعملة {symbol}: {e}", exc_info=True)
                    finally:
                        if 'df_15m' in locals(): del df_15m
                        gc.collect()

                logger.info(f"Batch {i//SYMBOL_PROCESSING_BATCH_SIZE + 1} processed. Sleeping for {BATCH_PROCESSING_SLEEP_SECONDS}s...")
                time.sleep(BATCH_PROCESSING_SLEEP_SECONDS)

            if 'btc_data' in locals(): del btc_data
            gc.collect()

            logger.info("✅ [End of Cycle] انتهت دورة المسح الكاملة. الانتظار 120 ثانية...")
            time.sleep(120)
            
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# --- نقطة الانطلاق ---
if __name__ == "__main__":
    # The startup logic remains the same
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V9.3 - أهداف ديناميكية) 🚀")
    # ... (omitted for brevity)
