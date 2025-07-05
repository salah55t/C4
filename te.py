# -*- coding: utf-8 -*-
import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import pickle
import argparse
from datetime import datetime, timedelta
from decouple import config
from binance.client import Client
from binance.exceptions import BinanceAPIException
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import warnings
import gc

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_backtester.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('UltimateStrategyBacktester')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY = config('BINANCE_API_KEY', default=None)
    API_SECRET = config('BINANCE_API_SECRET', default=None)
    TELEGRAM_BOT_TOKEN = config('TELEGRAM_BOT_TOKEN', default="PLEASE_FILL_YOUR_TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default="PLEASE_FILL_YOUR_TELEGRAM_CHAT_ID")
except Exception:
    API_KEY, API_SECRET = None, None
    TELEGRAM_BOT_TOKEN = "PLEASE_FILL_YOUR_TELEGRAM_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "PLEASE_FILL_YOUR_TELEGRAM_CHAT_ID"

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
COMMISSION_RATE = 0.001
SLIPPAGE_PERCENT = 0.0005
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '4h'
MAX_OPEN_TRADES = 10
MODEL_CONFIDENCE_THRESHOLD = 0.70
BASE_ML_MODEL_NAME = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER = 'V7'
BTC_SYMBOL = 'BTCUSDT'

# ---------------------- دوال مساعدة ----------------------

def send_telegram_report(report_text: str):
    """يرسل التقرير النهائي إلى تيليجرام."""
    if TELEGRAM_BOT_TOKEN.startswith("PLEASE_FILL") or TELEGRAM_CHAT_ID.startswith("PLEASE_FILL"):
        logger.error("❌ لم يتم تكوين توكن تيليجرام أو معرف الدردشة. سيتم طباعة التقرير هنا.")
        print("\n" + "="*50 + "\n--- التقرير النهائي ---\n" + "="*50 + "\n" + report_text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': report_text, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=20).raise_for_status()
        logger.info("✅ تم إرسال تقرير الاختبار الخلفي إلى تيليجرام بنجاح.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ فشل إرسال رسالة تيليجرام: {e}")
        print("\n--- التقرير النهائي (فشل الإرسال عبر تيليجرام) ---\n" + report_text)

def get_validated_symbols(client: Client, filename: str = 'crypto_list.txt') -> list[str]:
    """يقرأ قائمة الرموز من ملف ويتحقق منها مع Binance."""
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}'...")
    try:
        if not os.path.exists(filename):
            logger.error(f"❌ ملف العملات '{filename}' غير موجود.")
            return []
        with open(filename, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [التحقق] سيتم تحليل {len(validated)} عملة معتمدة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق] حدث خطأ أثناء التحقق من الرموز: {e}", exc_info=True)
        return []

def get_historical_data(client: Client, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    """يجلب البيانات التاريخية من Binance."""
    logger.info(f"⏳ جاري جلب البيانات التاريخية لـ {symbol} ({interval}) من {start_date} إلى {end_date}...")
    try:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date)
        if not klines: return pd.DataFrame()
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except BinanceAPIException as e:
        logger.error(f"❌ خطأ API من Binance أثناء جلب بيانات {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ خطأ عام أثناء جلب بيانات {symbol}: {e}")
        return pd.DataFrame()

def load_ml_model_bundle_from_folder(symbol: str) -> dict | None:
    """يحمل حزمة النموذج (النموذج + المُعدِّل + أسماء الميزات) من ملف pkl."""
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ [نموذج تعلم الآلة] ملف النموذج '{model_path}' غير موجود للعملة {symbol}.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            logger.info(f"✅ [نموذج تعلم الآلة] تم تحميل النموذج '{model_name}' بنجاح.")
            return model_bundle
        else:
            logger.error(f"❌ حزمة النموذج {model_name} غير مكتملة.")
            return None
    except Exception as e:
        logger.error(f"❌ [نموذج تعلم الآلة] خطأ في تحميل النموذج للعملة {symbol}: {e}", exc_info=True)
        return None

# ---------------------- دالة حساب الميزات الكاملة ----------------------
def calculate_all_features(df: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """يحسب جميع المؤشرات والميزات المطلوبة للنموذج."""
    if df.empty:
        return df
        
    df_calc = df.copy()

    # --- المؤشرات الأساسية ---
    df_calc['atr'] = (df_calc['high'] - df_calc['low']).rolling(window=14).mean() # ATR
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=14 - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=14 - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

    # --- MACD ---
    ema12 = df_calc['close'].ewm(span=12, adjust=False).mean()
    ema26 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = ema12 - ema26
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    df_calc['macd_cross'] = np.where(df_calc['macd'] > df_calc['macd_signal'], 1, -1)

    # --- Stochastic RSI ---
    rsi_val = df_calc['rsi']
    min_rsi = rsi_val.rolling(window=14).min()
    max_rsi = rsi_val.rolling(window=14).max()
    stoch_rsi_val = (rsi_val - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=3).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=3).mean()
    
    # --- EMAs ---
    df_calc['ema50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['ema50']) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['ema200']) - 1

    # --- Ichimoku Cloud ---
    high9 = df_calc['high'].rolling(window=9).max()
    low9 = df_calc['low'].rolling(window=9).min()
    df_calc['tenkan_sen'] = (high9 + low9) / 2
    high26 = df_calc['high'].rolling(window=26).max()
    low26 = df_calc['low'].rolling(window=26).min()
    df_calc['kijun_sen'] = (high26 + low26) / 2
    df_calc['senkou_span_a'] = ((df_calc['tenkan_sen'] + df_calc['kijun_sen']) / 2).shift(26)
    high52 = df_calc['high'].rolling(window=52).max()
    low52 = df_calc['low'].rolling(window=52).min()
    df_calc['senkou_span_b'] = ((high52 + low52) / 2).shift(26)
    df_calc['chikou_span'] = df_calc['close'].shift(-26)

    # --- ميزات إضافية من Ichimoku ---
    df_calc['price_vs_tenkan'] = (df_calc['close'] / df_calc['tenkan_sen']) - 1
    df_calc['price_vs_kijun'] = (df_calc['close'] / df_calc['kijun_sen']) - 1
    df_calc['tenkan_vs_kijun'] = (df_calc['tenkan_sen'] / df_calc['kijun_sen']) - 1
    df_calc['tenkan_kijun_cross'] = np.where(df_calc['tenkan_sen'] > df_calc['kijun_sen'], 1, -1)
    df_calc['price_vs_kumo_a'] = (df_calc['close'] / df_calc['senkou_span_a']) - 1
    df_calc['price_vs_kumo_b'] = (df_calc['close'] / df_calc['senkou_span_b']) - 1
    df_calc['price_above_kumo'] = np.where((df_calc['close'] > df_calc['senkou_span_a']) & (df_calc['close'] > df_calc['senkou_span_b']), 1, 0)
    df_calc['price_below_kumo'] = np.where((df_calc['close'] < df_calc['senkou_span_a']) & (df_calc['close'] < df_calc['senkou_span_b']), 1, 0)
    df_calc['price_in_kumo'] = np.where(
        ((df_calc['close'] > df_calc['senkou_span_a']) & (df_calc['close'] < df_calc['senkou_span_b'])) |
        ((df_calc['close'] < df_calc['senkou_span_a']) & (df_calc['close'] > df_calc['senkou_span_b'])), 1, 0)
    df_calc['kumo_thickness'] = np.abs(df_calc['senkou_span_a'] - df_calc['senkou_span_b']) / df_calc['close']
    df_calc['chikou_above_kumo'] = np.where((df_calc['chikou_span'] > df_calc['senkou_span_a'].shift(-26)) & (df_calc['chikou_span'] > df_calc['senkou_span_b'].shift(-26)), 1, 0)
    df_calc['chikou_below_kumo'] = np.where((df_calc['chikou_span'] < df_calc['senkou_span_a'].shift(-26)) & (df_calc['chikou_span'] < df_calc['senkou_span_b'].shift(-26)), 1, 0)

    # --- ميزات الدعم والمقاومة ---
    avg_atr = df_calc['atr'].mean()
    prominence = avg_atr * 0.6
    if prominence > 0:
        support_indices, _ = find_peaks(-df_calc['low'], prominence=prominence, width=5)
        resistance_indices, _ = find_peaks(df_calc['high'], prominence=prominence, width=5)
        supports = df_calc['low'].iloc[support_indices]
        resistances = df_calc['high'].iloc[resistance_indices]
        
        df_calc['dist_to_support'] = df_calc.apply(lambda row: (np.abs(supports[supports.index < row.name] - row['close']) / row['close']).min() if not supports[supports.index < row.name].empty else 1.0, axis=1)
        df_calc['dist_to_resistance'] = df_calc.apply(lambda row: (np.abs(resistances[resistances.index < row.name] - row['close']) / row['close']).min() if not resistances[resistances.index < row.name].empty else 1.0, axis=1)
        df_calc['score_of_support'] = 1 / (1 + df_calc['dist_to_support'] * 100)
        df_calc['score_of_resistance'] = 1 / (1 + df_calc['dist_to_resistance'] * 100)
    else:
        df_calc[['dist_to_support', 'dist_to_resistance', 'score_of_support', 'score_of_resistance']] = 1.0

    # --- ميزات أخرى ---
    df_calc['hour_of_day'] = df_calc.index.hour
    body_size = abs(df_calc['close'] - df_calc['open'])
    df_calc['candlestick_pattern'] = np.select(
        [body_size / (df_calc['high'] - df_calc['low']).replace(0, 1) < 0.1,  # Doji
         (df_calc['close'] > df_calc['open']) & (body_size / (df_calc['high'] - df_calc['low']).replace(0, 1) > 0.8), # Marubozu Bullish
         (df_calc['close'] < df_calc['open']) & (body_size / (df_calc['high'] - df_calc['low']).replace(0, 1) > 0.8)], # Marubozu Bearish
        [0, 1, -1], default=0.5) # Neutral

    # --- دمج ميزات الإطار الزمني الأعلى (4 ساعات) ---
    if not df_4h.empty:
        df_4h_calc = df_4h.copy()
        delta_4h = df_4h_calc['close'].diff()
        gain_4h = delta_4h.clip(lower=0).ewm(com=14 - 1, adjust=False).mean()
        loss_4h = -delta_4h.clip(upper=0).ewm(com=14 - 1, adjust=False).mean()
        df_4h_calc['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
        df_4h_calc['ema50_4h'] = df_4h_calc['close'].ewm(span=50, adjust=False).mean()
        
        # دمج باستخدام merge_asof
        df_calc = pd.merge_asof(df_calc.sort_index(), df_4h_calc[['rsi_4h', 'ema50_4h']].sort_index(), 
                                left_index=True, right_index=True, direction='backward')
        df_calc['price_vs_ema50_4h'] = (df_calc['close'] / df_calc['ema50_4h']) - 1
    else:
        df_calc[['rsi_4h', 'ema50_4h', 'price_vs_ema50_4h']] = np.nan

    # --- تحديد حالة السوق ---
    df_calc['market_condition'] = np.select(
        [df_calc['close'] > df_calc['ema200'],
         df_calc['close'] < df_calc['ema200']],
        [1, -1], default=0) # 1: Uptrend, -1: Downtrend, 0: Sideways

    return df_calc

# ---------------------- محرك الاختبار الخلفي ----------------------
def run_backtest(client: Client, start_date: str, end_date: str, trade_amount_usdt: float):
    """الدالة الرئيسية لتشغيل الاختبار الخلفي."""
    symbols = get_validated_symbols(client)
    if not symbols: return

    models = {}
    data_frames = {}

    for symbol in symbols:
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        if not model_bundle: continue
        
        df = get_historical_data(client, symbol, TIMEFRAME, start_date, end_date)
        df_4h = get_historical_data(client, symbol, HIGHER_TIMEFRAME, start_date, end_date)
        if df.empty: continue
            
        logger.info(f"⚙️ جاري حساب الميزات للعملة {symbol}...")
        df_featured = calculate_all_features(df, df_4h)
        
        feature_names = model_bundle['feature_names']
        missing_features = set(feature_names) - set(df_featured.columns)
        if missing_features:
            logger.warning(f"❌ ميزات ناقصة لـ {symbol} بعد الحساب: {missing_features}. سيتم تخطي العملة.")
            continue
        
        df_featured.dropna(inplace=True)
        if df_featured.empty:
            logger.warning(f"⚠️ لا توجد بيانات متبقية لـ {symbol} بعد إزالة القيم الفارغة.")
            continue
        
        try:
            features_to_scale = df_featured[feature_names]
            df_featured.loc[:, feature_names] = model_bundle['scaler'].transform(features_to_scale)
        except Exception as e:
            logger.error(f"❌ خطأ في تحجيم الميزات لـ {symbol}: {e}. سيتم تخطي العملة.")
            continue

        models[symbol] = model_bundle['model']
        data_frames[symbol] = df_featured
        gc.collect()

    if not data_frames:
        logger.critical("❌ لا توجد بيانات أو نماذج صالحة لإجراء الاختبار بعد معالجة جميع العملات."); return

    logger.info("🚀 بدء محاكاة التداول...")
    balance = trade_amount_usdt
    open_trades = []
    all_closed_trades = []
    
    # توحيد المؤشر الزمني لجميع البيانات
    common_index = None
    for df in data_frames.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    common_index = sorted(list(common_index))

    if not common_index:
        logger.critical("❌ لا يمكن إنشاء مؤشر زمني مشترك بين العملات."); return

    for timestamp in common_index:
        # إغلاق الصفقات
        for trade in open_trades[:]:
            symbol = trade['symbol']
            current_price = data_frames[symbol].loc[timestamp]['close']
            
            if current_price <= trade['stop_loss'] or current_price >= trade['target_price']:
                exit_price = trade['stop_loss'] if current_price <= trade['stop_loss'] else trade['target_price']
                exit_price_with_slippage = exit_price * (1 - SLIPPAGE_PERCENT)
                pnl = (exit_price_with_slippage - trade['entry_price_with_slippage']) * trade['quantity']
                
                trade.update({
                    'exit_price': exit_price_with_slippage, 'exit_time': timestamp,
                    'pnl': pnl, 'status': 'Stop Loss' if current_price <= trade['stop_loss'] else 'Take Profit'
                })
                balance += pnl
                all_closed_trades.append(trade)
                open_trades.remove(trade)

        # فتح صفقات جديدة
        if len(open_trades) < MAX_OPEN_TRADES:
            for symbol, model in models.items():
                if len(open_trades) >= MAX_OPEN_TRADES: break
                if any(t['symbol'] == symbol for t in open_trades): continue
                if timestamp not in data_frames[symbol].index: continue

                current_data = data_frames[symbol].loc[timestamp]
                
                features_scaled = current_data[model.feature_name_].to_frame().T
                prediction = model.predict(features_scaled)[0]
                prob_for_class_1 = model.predict_proba(features_scaled)[0][list(model.classes_).index(1)]

                if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                    entry_price = current_data['close']
                    entry_price_with_slippage = entry_price * (1 + SLIPPAGE_PERCENT)
                    quantity = (trade_amount_usdt / entry_price_with_slippage) * (1 - COMMISSION_RATE)
                    
                    atr_value = current_data['atr']
                    stop_loss = entry_price_with_slippage - (atr_value * ATR_SL_MULTIPLIER)
                    target_price = entry_price_with_slippage + (atr_value * ATR_TP_MULTIPLIER)

                    open_trades.append({
                        'symbol': symbol, 'entry_time': timestamp, 'entry_price': entry_price,
                        'entry_price_with_slippage': entry_price_with_slippage, 'quantity': quantity,
                        'stop_loss': stop_loss, 'target_price': target_price
                    })

    # --- حساب الإحصائيات النهائية وإنشاء التقرير ---
    logger.info("✅ اكتملت المحاكاة. جاري حساب الإحصائيات النهائية...")
    total_trades = len(all_closed_trades)
    if total_trades == 0:
        report = "*📊 تقرير الاختبار الخلفي*\n\n*📉 النتائج:*\nلم يتم تنفيذ أي صفقات. قد تكون الفلاتر صارمة جداً أو أن ظروف السوق لم تتوافق مع الاستراتيجية."
        send_telegram_report(report)
        return

    winning_trades = [t for t in all_closed_trades if t['pnl'] > 0]
    losing_trades = [t for t in all_closed_trades if t['pnl'] < 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = sum(t['pnl'] for t in all_closed_trades)
    
    total_profit = sum(t['pnl'] for t in winning_trades)
    total_loss = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    report = f"""
*📊 تقرير الاختبار الخلفي (محاكاة نهائية)*
--------------------------------------
*الفترة:* من `{start_date}` إلى `{end_date}`
*قائمة العملات:* `crypto_list.txt`
*المبلغ لكل صفقة:* `${trade_amount_usdt:,.2f}`
--------------------------------------
*📈 ملخص الأداء الإجمالي:*
*إجمالي الربح/الخسارة (PnL):* `${total_pnl:,.2f}`
*عامل الربح (Profit Factor):* `{profit_factor:.2f}`

*⚙️ إحصائيات الصفقات:*
*إجمالي عدد الصفقات:* `{total_trades}`
*الصفقات الرابحة:* `{len(winning_trades)}`
*الصفقات الخاسرة:* `{len(losing_trades)}`
*نسبة النجاح (Win Rate):* `{win_rate:.2f}%`
--------------------------------------
*ملاحظة: النتائج لا تضمن الأداء المستقبلي.*
"""
    send_telegram_report(report)

def main():
    """Main function to run the script."""
    logger.info("🚀 بدء تشغيل سكريبت الاختبار الخلفي...")
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ فشل الاتصال بـ Binance. يرجى التحقق من مفاتيح API. الخطأ: {e}")
        exit(1)

    parser = argparse.ArgumentParser(description="Run a crypto backtesting strategy.")
    end_date_dt = datetime.now()
    start_date_dt = end_date_dt - timedelta(days=30)
    start_date_default = start_date_dt.strftime("%Y-%m-%d")
    end_date_default = end_date_dt.strftime("%Y-%m-%d")

    parser.add_argument('--start-date', type=str, default=start_date_default, help=f'Start date (YYYY-MM-DD). Default: {start_date_default}')
    parser.add_argument('--end-date', type=str, default=end_date_default, help=f'End date (YYYY-MM-DD). Default: {end_date_default}')
    parser.add_argument('--amount', type=float, default=100.0, help='Amount per trade in USDT. Default: 100.0')

    args = parser.parse_args()

    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
        
        logger.info(f"🗓️ تشغيل الاختبار من {args.start_date} إلى {args.end_date} بمبلغ ${args.amount} لكل صفقة.")
        run_backtest(client, args.start_date, args.end_date, args.amount)

    except ValueError:
        logger.error("❌ صيغة التاريخ غير صحيحة. يرجى استخدام YYYY-MM-DD.")
    except Exception as e:
        logger.error(f"❌ حدث خطأ غير متوقع: {e}", exc_info=True)

    logger.info("👋 انتهى عمل السكريبت. وداعاً!")

# ---------------------- نقطة انطلاق البرنامج ----------------------
if __name__ == "__main__":
    main()
