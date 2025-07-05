# -*- coding: utf-8 -*-
import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import pickle
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

# ---------------------- إعداد الثوابت والمتغيرات العامة (مطابقة لـ c4.py) ----------------------
# --- إعدادات التداول والمحاكاة ---
COMMISSION_RATE = 0.001
SLIPPAGE_PERCENT = 0.0005
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
TIMEFRAME = '15m'
MAX_OPEN_TRADES = 10
MODEL_CONFIDENCE_THRESHOLD = 0.70

# --- إعدادات النموذج والميزات ---
BASE_ML_MODEL_NAME = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER = 'V7'
BTC_SYMBOL = 'BTCUSDT'

# --- إعدادات الفلاتر (جديد) ---
USE_BTC_TREND_FILTER = True
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 50

USE_SPEED_FILTER = True
SPEED_FILTER_ADX_THRESHOLD = 20.0
SPEED_FILTER_REL_VOL_THRESHOLD = 1.2
SPEED_FILTER_RSI_MIN = 45.0
SPEED_FILTER_RSI_MAX = 70.0

# --- إعدادات المؤشرات الفنية ---
ADX_PERIOD, RSI_PERIOD, ATR_PERIOD = 14, 14, 14
BBANDS_PERIOD, REL_VOL_PERIOD, BTC_CORR_PERIOD = 20, 30, 30
STOCH_RSI_PERIOD, STOCH_K, STOCH_D = 14, 3, 3
ICHIMOKU_TENKAN_PERIOD, ICHIMOKU_KIJUN_PERIOD, ICHIMOKU_SENKOU_B_PERIOD = 9, 26, 52
ICHIMOKU_CHIKOU_SHIFT, ICHIMOKU_SENKOU_SHIFT = -26, 26
SR_PEAK_WIDTH, SR_PEAK_PROMINENCE_MULTIPLIER = 5, 0.6

# ---------------------- دوال مساعدة ومنطق c4.py ----------------------

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
            logger.error(f"❌ ملف العملات '{filename}' غير موجود. يرجى إنشاء الملف ووضع الرموز فيه.")
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
    except Exception as e:
        logger.error(f"❌ خطأ أثناء جلب بيانات {symbol}: {e}")
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
        return None
    except Exception as e:
        logger.error(f"❌ [نموذج تعلم الآلة] خطأ في تحميل النموذج للعملة {symbol}: {e}", exc_info=True)
        return None

# ---------------------- دوال حساب المؤشرات والميزات (منطق c4 مدمج) ----------------------

def calculate_all_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """يحسب جميع المؤشرات والميزات المطلوبة للنموذج والفلاتر."""
    df_calc = df.copy()
    
    # ATR, ADX
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()

    # RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

    # BBands Width
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_width'] = ( (sma + std_dev * 2) - (sma - std_dev * 2) ) / sma.replace(0, 1e-9)

    # Stochastic RSI
    rsi_val = df_calc['rsi']
    min_rsi = rsi_val.rolling(window=STOCH_RSI_PERIOD).min()
    max_rsi = rsi_val.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi_val - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    
    # Relative Volume
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)

    # BTC Correlation
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    
    # Ichimoku
    high, low, close = df_calc['high'], df_calc['low'], df_calc['close']
    df_calc['tenkan_sen'] = (high.rolling(window=ICHIMOKU_TENKAN_PERIOD).max() + low.rolling(window=ICHIMOKU_TENKAN_PERIOD).min()) / 2
    df_calc['kijun_sen'] = (high.rolling(window=ICHIMOKU_KIJUN_PERIOD).max() + low.rolling(window=ICHIMOKU_KIJUN_PERIOD).min()) / 2
    df_calc['senkou_span_a'] = ((df_calc['tenkan_sen'] + df_calc['kijun_sen']) / 2).shift(ICHIMOKU_SENKOU_SHIFT)
    df_calc['senkou_span_b'] = ((high.rolling(window=ICHIMOKU_SENKOU_B_PERIOD).max() + low.rolling(window=ICHIMOKU_SENKOU_B_PERIOD).min()) / 2).shift(ICHIMOKU_SENKOU_SHIFT)
    
    # S/R features
    avg_atr = df_calc['atr'].mean()
    prominence = avg_atr * SR_PEAK_PROMINENCE_MULTIPLIER
    supports = pd.Series(dtype=float)
    resistances = pd.Series(dtype=float)
    if prominence > 0:
        support_indices, _ = find_peaks(-df_calc['low'], prominence=prominence, width=SR_PEAK_WIDTH)
        resistance_indices, _ = find_peaks(df_calc['high'], prominence=prominence, width=SR_PEAK_WIDTH)
        if len(support_indices) > 0: supports = df_calc['low'].iloc[support_indices]
        if len(resistance_indices) > 0: resistances = df_calc['high'].iloc[resistance_indices]

    df_calc['dist_to_support'] = df_calc['close'].apply(lambda p: (np.abs(supports - p) / p).min() if not supports.empty else 1.0)
    df_calc['dist_to_resistance'] = df_calc['close'].apply(lambda p: (np.abs(resistances - p) / p).min() if not resistances.empty else 1.0)

    return df_calc

# ---------------------- محرك الاختبار الخلفي النهائي ----------------------

def run_backtest(client: Client, start_date: str, end_date: str, trade_amount_usdt: float):
    """الدالة الرئيسية لتشغيل الاختبار الخلفي المتقدم مع الفلاتر."""
    
    # --- 1. التحضير والتهيئة ---
    symbols = get_validated_symbols(client)
    if not symbols: return

    models = {}
    data_frames = {}

    # تحميل بيانات البيتكوين للمؤشرات والفلتر
    btc_df = get_historical_data(client, BTC_SYMBOL, TIMEFRAME, start_date, end_date)
    if btc_df.empty:
        logger.critical("❌ لا يمكن المتابعة بدون بيانات البيتكوين."); return
    btc_df['btc_returns'] = btc_df['close'].pct_change()

    btc_trend_df = get_historical_data(client, BTC_SYMBOL, BTC_TREND_TIMEFRAME, start_date, end_date)
    use_btc_filter = USE_BTC_TREND_FILTER
    if btc_trend_df.empty:
        logger.warning(f"⚠️ لا يمكن تحميل بيانات اتجاه البيتكوين ({BTC_TREND_TIMEFRAME})، سيتم تعطيل الفلتر.");
        use_btc_filter = False
    else:
        btc_trend_df['ema_trend'] = btc_trend_df['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean()

    # تحميل النماذج والبيانات لكل عملة
    for symbol in symbols:
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        if not model_bundle: continue
        
        df = get_historical_data(client, symbol, TIMEFRAME, start_date, end_date)
        if df.empty: continue
            
        logger.info(f"جاري حساب الميزات للعملة {symbol}...")
        df_featured = calculate_all_features(df, btc_df)
        
        # دمج بيانات فلتر الاتجاه
        if use_btc_filter:
            # نستخدم merge_asof لدمج الإطار الزمني الأعلى مع الأقل
            df_featured = pd.merge_asof(df_featured.sort_index(), btc_trend_df[['close', 'ema_trend']].sort_index(), 
                                        left_index=True, right_index=True, direction='forward', 
                                        suffixes=('', '_trend'))
            df_featured['btc_is_uptrend'] = df_featured['close_trend'] > df_featured['ema_trend']
        else:
            df_featured['btc_is_uptrend'] = True # تعطيل الفلتر

        # الاحتفاظ بالقيم غير المحجمة للفلاتر
        df_featured['adx_unscaled'] = df_featured['adx']
        df_featured['rsi_unscaled'] = df_featured['rsi']
        df_featured['relative_volume_unscaled'] = df_featured['relative_volume']
        
        # التأكد من وجود جميع الميزات المطلوبة للنموذج
        feature_names = model_bundle['feature_names']
        missing_features = set(feature_names) - set(df_featured.columns)
        if missing_features:
            logger.warning(f"ميزات ناقصة لـ {symbol}: {missing_features}. سيتم تخطي العملة.")
            continue
        
        # تحجيم الميزات المطلوبة للنموذج فقط
        df_featured.dropna(inplace=True)
        if df_featured.empty: continue
        
        features_to_scale = df_featured[feature_names]
        df_featured.loc[:, feature_names] = model_bundle['scaler'].transform(features_to_scale)

        models[symbol] = model_bundle['model']
        data_frames[symbol] = df_featured
        gc.collect()

    if not data_frames:
        logger.critical("❌ لا توجد بيانات أو نماذج صالحة لإجراء الاختبار."); return

    # --- 2. إعداد متغيرات وحلقة التداول ---
    logger.info("🚀 بدء محاكاة التداول المتزامنة لجميع العملات مع الفلاتر...")
    balance = trade_amount_usdt
    open_trades = []
    all_closed_trades = []
    
    common_index = None
    for df in data_frames.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    
    # --- 3. حلقة التداول الرئيسية ---
    for timestamp in common_index:
        # إغلاق الصفقات
        for trade in open_trades[:]:
            symbol = trade['symbol']
            current_price = data_frames[symbol].loc[timestamp]['close_unscaled'] if 'close_unscaled' in data_frames[symbol].columns else data_frames[symbol].loc[timestamp]['close']
            
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
                
                # --- تطبيق الفلاتر قبل التنبؤ ---
                if use_btc_filter and not current_data['btc_is_uptrend']:
                    continue

                if USE_SPEED_FILTER:
                    if not (current_data['adx_unscaled'] >= SPEED_FILTER_ADX_THRESHOLD and
                            current_data['relative_volume_unscaled'] >= SPEED_FILTER_REL_VOL_THRESHOLD and
                            SPEED_FILTER_RSI_MIN <= current_data['rsi_unscaled'] < SPEED_FILTER_RSI_MAX):
                        continue
                
                # --- التنبؤ باستخدام النموذج ---
                features_scaled = current_data[model.feature_names_].to_frame().T
                prediction = model.predict(features_scaled)[0]
                prob_for_class_1 = model.predict_proba(features_scaled)[0][list(model.classes_).index(1)]

                if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                    entry_price = current_data['close_unscaled'] if 'close_unscaled' in current_data else current_data['close']
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

    # --- 4. حساب الإحصائيات النهائية وإنشاء التقرير ---
    logger.info("✅ اكتملت المحاكاة. جاري حساب الإحصائيات النهائية...")
    total_trades = len(all_closed_trades)
    if total_trades == 0:
        report = "*📊 تقرير الاختبار الخلفي*\n\n*📉 النتائج:*\nلم يتم تنفيذ أي صفقات. قد تكون الفلاتر صارمة جداً أو أن ظروف السوق لم تتوافق مع الاستراتيجية."
        send_telegram_report(report)
        return

    winning_trades = [t for t in all_closed_trades if t['pnl'] > 0]
    win_rate = (len(winning_trades) / total_trades) * 100
    total_pnl = sum(t['pnl'] for t in all_closed_trades)
    profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in (t for t in all_closed_trades if t['pnl'] < 0))) if any(t['pnl'] < 0 for t in all_closed_trades) else float('inf')

    report = f"""
*📊 تقرير الاختبار الخلفي (محاكاة نهائية مع الفلاتر)*
--------------------------------------
*الفترة:* من `{start_date}` إلى `{end_date}`
*قائمة العملات:* `crypto_list.txt`
*المبلغ لكل صفقة:* `${trade_amount_usdt:,.2f}`
*الحد الأقصى للصفقات:* `{MAX_OPEN_TRADES}`
*الفلاتر المفعلة:* `اتجاه BTC`, `السرعة`
--------------------------------------
*📈 ملخص الأداء الإجمالي:*
*إجمالي الربح/الخسارة (PnL):* `${total_pnl:,.2f}`
*عامل الربح (Profit Factor):* `{profit_factor:.2f}`

*⚙️ إحصائيات الصفقات:*
*إجمالي عدد الصفقات:* `{total_trades}`
*الصفقات الرابحة:* `{len(winning_trades)}`
*الصفقات الخاسرة:* `{len(all_closed_trades) - len(winning_trades)}`
*نسبة النجاح (Win Rate):* `{win_rate:.2f}%`
--------------------------------------
*ملاحظة: النتائج لا تضمن الأداء المستقبلي.*
"""
    send_telegram_report(report)

# ---------------------- نقطة انطلاق البرنامج ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل سكريبت الاختبار الخلفي النهائي...")
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ فشل الاتصال بـ Binance. يرجى التحقق من مفاتيح API. الخطأ: {e}")
        exit(1)

    try:
        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=30)
        start_date_default = start_date_dt.strftime("%Y-%m-%d")
        end_date_default = end_date_dt.strftime("%Y-%m-%d")

        start_date_input = input(f"Enter start date (YYYY-MM-DD) [Default: {start_date_default}]: ") or start_date_default
        end_date_input = input(f"Enter end date (YYYY-MM-DD) [Default: {end_date_default}]: ") or end_date_default
        datetime.strptime(start_date_input, "%Y-%m-%d")
        datetime.strptime(end_date_input, "%Y-%m-%d")
        trade_amount_input = float(input("Enter initial amount per trade in USDT [Default: 100]: ") or 100)

        run_backtest(client, start_date_input, end_date_input, trade_amount_input)

    except ValueError:
        logger.error("❌ خطأ في الإدخال. يرجى التأكد من إدخال مبلغ صحيح وتنسيق التاريخ (YYYY-MM-DD).")
    except Exception as e:
        logger.error(f"❌ حدث خطأ غير متوقع: {e}", exc_info=True)

    logger.info("👋 انتهى عمل السكريبت. وداعاً!")
