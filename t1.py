import time
import os
import json
import logging
import numpy as np
import pandas as pd
import pickle
import re
from binance.client import Client
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler
import warnings
import threading
import http.server
import socketserver

# --- تجاهل التحذيرات ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# --- إعداد التسجيل ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('CryptoBacktesterDetailed')

# --- تحميل متغيرات البيئة (اختياري للاختبار) ---
try:
    API_KEY: str = config('BINANCE_API_KEY', default='')
    API_SECRET: str = config('BINANCE_API_SECRET', default='')
except Exception as e:
    logger.warning(f"لم يتم تحميل مفاتيح API من متغيرات البيئة: {e}")
    API_KEY, API_SECRET = '', ''

# ---------------------- إعدادات الاختبار التاريخي ----------------------
BACKTEST_DAYS: int = 10
INITIAL_CAPITAL: float = 10000.0
RISK_PER_TRADE_PERCENT: float = 1.0
MAX_OPEN_TRADES: int = 30
BUY_CONFIDENCE_THRESHOLD: float = 0.70
TRADING_FEE_PERCENT: float = 0.1

# ---------------------- ثوابت الاستراتيجية ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_ANALYSIS: List[str] = ['15m', '1h', '4h']
BTC_SYMBOL: str = 'BTCUSDT'
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
EMA_PERIODS: List[int] = [21, 50, 200]
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.2

# --- متغيرات عامة ---
client: Optional[Client] = None
exchange_info_map: Dict[str, Any] = {}
ml_models_cache: Dict[str, Any] = {}

# ---------------------- خادم الويب البسيط ----------------------
def start_web_server():
    """
    يبدأ خادم ويب بسيط في خيط منفصل للاستجابة لطلبات HTTP.
    هذا ضروري لمنصات مثل Render لمنع توقف الخدمة.
    """
    # احصل على المنفذ من متغيرات البيئة أو استخدم 8080 كقيمة افتراضية
    PORT = int(os.environ.get('PORT', 8080))
    
    # معالج طلبات بسيط
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write("الخادم يعمل بشكل سليم.".encode('utf-8'))

    # ابدأ الخادم في خيط منفصل
    def run_server():
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            logger.info(f"خادم الويب يعمل على المنفذ {PORT}")
            httpd.serve_forever()

    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True  # اسمح للبرنامج الرئيسي بالخروج حتى لو كان الخيط يعمل
    server_thread.start()

# ---------------------- دوال مساعدة (مقتبسة من البوت) ----------------------

def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    logger.info("جاري جلب قواعد التداول من المنصة...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"تم تحميل القواعد لـ {len(exchange_info_map)} عملة.")
    except Exception as e:
        logger.error(f"فشل جلب معلومات المنصة: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        if not os.path.exists(filename):
            logger.error(f"ملف العملات '{filename}' غير موجود. يرجى إنشاء الملف وإضافة العملات.")
            return []
        with open(filename, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"سيتم اختبار {len(validated)} عملة.")
        return validated
    except Exception as e:
        logger.error(f"خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, start_date_str: str) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        klines = client.get_historical_klines(symbol, interval, start_str=start_date_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

# --- [مُصحح] --- تم تعديل الدالة لتقبل اسم العملة
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
    df_calc = df.copy()
    for period in EMA_PERIODS:
        df_calc[f'ema_{period}'] = df_calc['close'].ewm(span=period, adjust=False).mean()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['ema_50']) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['ema_200']) - 1
    
    # --- [مُصحح] --- منطق جديد لحساب ارتباط البيتكوين
    if symbol == BTC_SYMBOL:
        df_calc['btc_correlation'] = 1.0
    elif btc_df is not None and not btc_df.empty:
        if 'btc_returns' not in btc_df.columns:
            raise ValueError("btc_df is missing the 'btc_returns' column")
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = df_calc['close'].pct_change().rolling(window=30).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0

    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['roc_acceleration'] = df_calc[f'roc_{MOMENTUM_PERIOD}'].diff()
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    df_calc['hour_of_day'] = df_calc.index.hour
    return df_calc.astype('float32', errors='ignore')

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache: return ml_models_cache[model_name]
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        logger.debug(f"⚠️ نموذج التعلم الآلي غير موجود: '{model_path}'.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            return model_bundle
        return None
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل النموذج لـ {symbol}: {e}", exc_info=True)
        return None

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    # --- [مُصحح] --- تم تعديل الدالة لتمرير اسم العملة
    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_features(df_15m, btc_df, self.symbol)
            df_4h_features = calculate_features(df_4h, None, self.symbol)
            df_4h_features = df_4h_features.rename(columns=lambda c: f"{c}_4h", inplace=False)
            required_4h_cols = ['rsi_4h', 'price_vs_ema50_4h']
            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='outer')
            df_featured.fillna(method='ffill', inplace=True)
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, features_row: pd.Series) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        try:
            features_df = pd.DataFrame([features_row], columns=self.feature_names)
            features_scaled_np = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)
            prediction = self.ml_model.predict(features_scaled_df)[0]
            if prediction != 1: return None
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)
            confidence = float(np.max(prediction_proba[0]))
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] خطأ في توليد إشارة النموذج: {e}")
            return None

def calculate_tp_sl(entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    if last_atr <= 0: return None
    tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
    sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
    return {'target_price': tp, 'stop_loss': sl}

def determine_market_trend_at_time(timestamp: pd.Timestamp, btc_data_all_tf: Dict[str, pd.DataFrame]) -> Dict:
    details = {}
    total_score = 0
    tf_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.5}
    
    for tf, df in btc_data_all_tf.items():
        if timestamp not in df.index:
            details[tf] = {"score": 0, "label": "غير واضح", "reason": "بيانات غير متوفرة في هذه النقطة الزمنية"}
            continue
        
        last_candle = df.loc[timestamp]
        close, ema21, ema50, ema200 = last_candle['close'], last_candle['ema_21'], last_candle['ema_50'], last_candle['ema_200']
        
        tf_score = 0
        if close > ema21: tf_score += 1
        elif close < ema21: tf_score -= 1
        if ema21 > ema50: tf_score += 1
        elif ema21 < ema50: tf_score -= 1
        if ema50 > ema200: tf_score += 1
        elif ema50 < ema200: tf_score -= 1
        
        label = "محايد"
        if tf_score >= 2: label = "صاعد"
        elif tf_score <= -2: label = "هابط"
        details[tf] = {"score": tf_score, "label": label}

        total_score += tf_score * tf_weights[tf]

    final_score = round(total_score)
    trend_label = "محايد"
    if final_score >= 4: trend_label = "صاعد قوي"
    elif final_score >= 1: trend_label = "صاعد"
    elif final_score <= -4: trend_label = "هابط قوي"
    elif final_score <= -1: trend_label = "هابط"
    
    return {"trend_score": final_score, "trend_label": trend_label, "details_by_tf": details}

def capture_trade_details(features_row: pd.Series, market_trend: Dict, confidence: float) -> Dict:
    details = {
        "market_trend": market_trend,
        "ml_confidence": f"{confidence:.2%}",
        "filters": {
            "adx": f"{features_row.get('adx', 0):.2f}",
            "rsi": f"{features_row.get('rsi', 0):.2f}",
            "relative_volume": f"{features_row.get('relative_volume', 0):.2f}",
            f"roc_{MOMENTUM_PERIOD}": f"{features_row.get(f'roc_{MOMENTUM_PERIOD}', 0):.2f}",
            f"ema_slope_{EMA_SLOPE_PERIOD}": f"{features_row.get(f'ema_slope_{EMA_SLOPE_PERIOD}', 0):.6f}",
            "btc_correlation": f"{features_row.get('btc_correlation', 0):.2f}",
            "price_vs_ema50": f"{features_row.get('price_vs_ema50', 0):.4f}",
            "price_vs_ema200": f"{features_row.get('price_vs_ema200', 0):.4f}",
            "atr": f"{features_row.get('atr', 0):.8f}"
        }
    }
    return details

def generate_detailed_report(closed_trades: List[Dict], initial_capital: float, final_capital: float):
    logger.info("\n" + "="*80)
    logger.info("📊 تقرير الاختبار التاريخي المفصل 📊")
    logger.info("="*80)

    if not closed_trades:
        logger.warning("لم يتم إغلاق أي صفقات خلال فترة الاختبار.")
        return

    total_trades = len(closed_trades)
    wins = [t for t in closed_trades if t['pnl_usdt'] > 0]
    losses = [t for t in closed_trades if t['pnl_usdt'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl_usdt = final_capital - initial_capital
    total_pnl_pct = (total_pnl_usdt / initial_capital) * 100
    gross_profit = sum(t['pnl_usdt'] for t in wins)
    gross_loss = abs(sum(t['pnl_usdt'] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_win_pct = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss_pct = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    
    logger.info(f"فترة الاختبار: {BACKTEST_DAYS} أيام")
    logger.info(f"رأس المال الأولي: ${initial_capital:,.2f}")
    logger.info(f"رأس المال النهائي: ${final_capital:,.2f}")
    logger.info("-" * 30)
    logger.info(f"إجمالي الربح/الخسارة: ${total_pnl_usdt:,.2f} ({total_pnl_pct:.2f}%)")
    logger.info(f"إجمالي عدد الصفقات: {total_trades}")
    logger.info(f"الصفقات الرابحة: {len(wins)} | الصفقات الخاسرة: {len(losses)}")
    logger.info(f"نسبة النجاح: {win_rate:.2f}%")
    logger.info(f"عامل الربح: {profit_factor:.2f}")
    logger.info(f"متوسط ربح الصفقة الرابحة: {avg_win_pct:.2f}%")
    logger.info(f"متوسط خسارة الصفقة الخاسرة: {avg_loss_pct:.2f}%")
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ قائمة الصفقات الرابحة ({len(wins)} صفقة)")
    logger.info("="*80)
    if not wins:
        logger.info("لا توجد صفقات رابحة.")
    else:
        for trade in wins:
            print_trade_details(trade)

    logger.info("\n" + "="*80)
    logger.info(f"❌ قائمة الصفقات الخاسرة ({len(losses)} صفقة)")
    logger.info("="*80)
    if not losses:
        logger.info("لا توجد صفقات خاسرة.")
    else:
        for trade in losses:
            print_trade_details(trade)

def print_trade_details(trade: Dict):
    details_json = json.dumps(trade['details'], indent=2, ensure_ascii=False)
    logger.info(
        f"\n--- العملة: {trade['symbol']} | الربح: {trade['pnl_pct']:.2f}% (${trade['pnl_usdt']:.2f}) ---"
        f"\nوقت الدخول: {trade['entry_time']} | سعر الدخول: {trade['entry_price']:.5f}"
        f"\nوقت الخروج: {trade['close_time']} | سعر الخروج: {trade['closing_price']:.5f} | سبب الخروج: {trade['close_reason']}"
        f"\nتفاصيل وقت الدخول:\n{details_json}"
    )

# ---------------------- المحرك الرئيسي للاختبار التاريخي ----------------------
def run_backtest():
    global client
    logger.info(f"🚀 بدء الاختبار التاريخي لمدة {BACKTEST_DAYS} أيام...")
    
    client = Client(API_KEY, API_SECRET)
    
    symbols_to_test = get_validated_symbols()
    if not symbols_to_test:
        logger.critical("لا توجد عملات صالحة للاختبار. الخروج.")
        return

    start_date = datetime.now(timezone.utc) - timedelta(days=BACKTEST_DAYS)
    start_date_str = start_date.strftime("%d %b %Y %H:%M:%S")
    
    all_data = {}
    logger.info(f"جاري تحميل البيانات التاريخية لـ {len(symbols_to_test)} عملة...")
    for symbol in symbols_to_test + [BTC_SYMBOL]:
        df = fetch_historical_data(symbol, TIMEFRAME, start_date_str)
        if df is not None and not df.empty:
            all_data[symbol] = df
        else:
            logger.warning(f"لم يتم العثور على بيانات لـ {symbol}، سيتم تجاهلها.")
        time.sleep(0.5)
    
    symbols_to_test = [s for s in symbols_to_test if s in all_data]
    if not symbols_to_test:
        logger.critical("لا توجد بيانات لأي من العملات المحددة. الخروج.")
        return

    logger.info("جاري حساب المؤشرات الفنية والميزات...")
    btc_data_15m = all_data[BTC_SYMBOL]
    btc_data_15m['btc_returns'] = btc_data_15m['close'].pct_change()
    
    btc_data_all_tf = {}
    for tf in TIMEFRAMES_FOR_TREND_ANALYSIS:
        df_btc_tf = fetch_historical_data(BTC_SYMBOL, tf, start_date_str)
        for period in EMA_PERIODS:
            df_btc_tf[f'ema_{period}'] = df_btc_tf['close'].ewm(span=period, adjust=False).mean()
        btc_data_all_tf[tf] = df_btc_tf.dropna()

    all_features = {}
    for symbol in symbols_to_test:
        df_15m = all_data[symbol]
        df_4h = fetch_historical_data(symbol, '4h', start_date_str)
        if df_4h is None: 
            logger.warning(f"لا توجد بيانات 4h لـ {symbol}, سيتم تجاهلها.")
            continue
        strategy = TradingStrategy(symbol)
        if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]):
            logger.warning(f"لا يوجد نموذج لـ {symbol}, سيتم تجاهلها.")
            continue
        features = strategy.get_features(df_15m, df_4h, btc_data_15m)
        if features is not None and not features.empty:
            all_features[symbol] = features
    
    symbols_to_test = [s for s in symbols_to_test if s in all_features]
    if not symbols_to_test:
        logger.critical("فشل حساب الميزات لجميع العملات. الخروج.")
        return

    main_df = pd.concat([df['close'].rename(f"{symbol}_close") for symbol, df in all_data.items()], axis=1)
    main_df.dropna(inplace=True)
    
    capital = INITIAL_CAPITAL
    open_trades: List[Dict] = []
    closed_trades: List[Dict] = []
    
    logger.info(f"▶️ بدء محاكاة التداول من {main_df.index[0]} إلى {main_df.index[-1]}...")
    
    for timestamp, row in main_df.iterrows():
        trades_to_close_indices = []
        for i, trade in enumerate(open_trades):
            symbol = trade['symbol']
            if timestamp not in all_data[symbol].index: continue
            current_candle = all_data[symbol].loc[timestamp]
            
            if current_candle['high'] >= trade['target_price']:
                closing_price, close_reason = trade['target_price'], 'target_hit'
            elif current_candle['low'] <= trade['stop_loss']:
                closing_price, close_reason = trade['stop_loss'], 'stop_loss_hit'
            else:
                continue
            
            pnl_usdt = (closing_price - trade['entry_price']) * trade['quantity']
            fee = (trade['notional_value'] * (TRADING_FEE_PERCENT / 100)) + \
                  (closing_price * trade['quantity'] * (TRADING_FEE_PERCENT / 100))
            net_pnl_usdt = pnl_usdt - fee
            pnl_pct = (net_pnl_usdt / trade['notional_value']) * 100
            capital += trade['notional_value'] + net_pnl_usdt
            
            trade.update({
                'closing_price': closing_price, 'close_time': timestamp,
                'pnl_usdt': net_pnl_usdt, 'pnl_pct': pnl_pct, 'close_reason': close_reason
            })
            closed_trades.append(trade)
            trades_to_close_indices.append(i)

        for i in sorted(trades_to_close_indices, reverse=True):
            del open_trades[i]
            
        if len(open_trades) >= MAX_OPEN_TRADES: continue

        for symbol in symbols_to_test:
            if any(t['symbol'] == symbol for t in open_trades): continue
            if symbol not in all_features or timestamp not in all_features[symbol].index: continue
            
            features_row = all_features[symbol].loc[timestamp]
            if features_row.isnull().any(): continue
            
            strategy = TradingStrategy(symbol)
            ml_signal = strategy.generate_buy_signal(features_row)
            
            if ml_signal and ml_signal['confidence'] >= BUY_CONFIDENCE_THRESHOLD:
                entry_price = features_row['close']
                last_atr = features_row['atr']
                tp_sl = calculate_tp_sl(entry_price, last_atr)
                if not tp_sl: continue
                
                risk_amount_usdt = capital * (RISK_PER_TRADE_PERCENT / 100)
                risk_per_coin = entry_price - tp_sl['stop_loss']
                if risk_per_coin <= 0: continue
                quantity = risk_amount_usdt / risk_per_coin
                notional_value = quantity * entry_price
                
                if capital < notional_value: continue
                
                market_trend = determine_market_trend_at_time(timestamp, btc_data_all_tf)
                trade_details = capture_trade_details(features_row, market_trend, ml_signal['confidence'])
                
                capital -= notional_value
                
                new_trade = {
                    'symbol': symbol, 'entry_time': timestamp, 'entry_price': entry_price,
                    'quantity': quantity, 'notional_value': notional_value,
                    'target_price': tp_sl['target_price'], 'stop_loss': tp_sl['stop_loss'],
                    'details': trade_details
                }
                open_trades.append(new_trade)
                logger.info(f"🟢 فتح صفقة: {symbol} @ ${entry_price:.4f} في {timestamp}")

    if open_trades:
        logger.info("إغلاق الصفقات المتبقية في نهاية فترة الاختبار...")
        last_timestamp = main_df.index[-1]
        for trade in open_trades:
            closing_price = main_df.loc[last_timestamp, f"{trade['symbol']}_close"]
            pnl_usdt = (closing_price - trade['entry_price']) * trade['quantity']
            fee = (trade['notional_value'] * (TRADING_FEE_PERCENT / 100)) * 2
            net_pnl_usdt = pnl_usdt - fee
            pnl_pct = (net_pnl_usdt / trade['notional_value']) * 100
            capital += trade['notional_value'] + net_pnl_usdt
            trade.update({
                'closing_price': closing_price, 'close_time': last_timestamp,
                'pnl_usdt': net_pnl_usdt, 'pnl_pct': pnl_pct, 'close_reason': 'end_of_backtest'
            })
            closed_trades.append(trade)

    generate_detailed_report(closed_trades, INITIAL_CAPITAL, capital)

if __name__ == "__main__":
    # --- [جديد] --- بدء تشغيل خادم الويب
    start_web_server()
    
    if not os.path.exists(MODEL_FOLDER):
        logger.critical(f"مجلد النماذج '{MODEL_FOLDER}' غير موجود. لا يمكن المتابعة.")
    else:
        run_backtest()

