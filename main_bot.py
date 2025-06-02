import os # تم إضافة هذا الاستيراد
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
from threading import Thread
from flask import Flask, request, Response
from typing import List, Dict, Optional, Tuple, Any, Union

# استيراد الدوال المشتركة والثوابت من ملف المرافق
import utils # استيراد utils كـ module
from utils import (
    init_db, check_db_connection, initialize_binance_client,
    fetch_historical_data, calculate_rsi_indicator, calculate_atr_indicator,
    get_btc_trend_4h, fetch_recent_volume, get_crypto_symbols,
    load_ml_model_from_db, convert_np_values, get_interval_minutes,
    get_fear_greed_index, logger, client, conn, cur,
    RSI_PERIOD, VOLUME_LOOKBACK_CANDLES, RSI_MOMENTUM_LOOKBACK_CANDLES,
    ENTRY_ATR_PERIOD, BASE_ML_MODEL_NAME, TELEGRAM_TOKEN, CHAT_ID, WEBHOOK_URL
)

# استيراد وظيفة التدريب من train_models.py
from train_models import run_training_for_all_symbols

# ---------------------- ثوابت ومتغيرات عامة خاصة بالبوت ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 10
SIGNAL_GENERATION_TIMEFRAME: str = '5m' # تم التغيير إلى 5 دقائق
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3 # أيام البحث عن البيانات لتوليد الإشارة
MIN_PROFIT_MARGIN_PCT: float = 1.0
MIN_VOLUME_15M_USDT: float = 250000.0
ENTRY_ATR_MULTIPLIER: float = 1.5
TARGET_APPROACH_THRESHOLD_PCT: float = 0.005
BINANCE_FEE_RATE: float = 0.001

# Global variable for ticker data
ticker_data: Dict[str, float] = {}

# ---------------------- WebSocket Management for Ticker Prices ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Handles incoming WebSocket messages for mini-ticker prices."""
    global ticker_data
    try:
        if isinstance(msg, list):
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c')
                if symbol and 'USDT' in symbol and price_str:
                    try:
                        ticker_data[symbol] = float(price_str)
                    except ValueError:
                         logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol}: '{price_str}'")
        elif isinstance(msg, dict):
             if msg.get('e') == 'error':
                 logger.error(f"❌ [WS] رسالة خطأ من WebSocket: {msg.get('m', 'لا توجد تفاصيل خطأ')}")
             elif msg.get('stream') and msg.get('data'):
                 for ticker_item in msg.get('data', []):
                    symbol = ticker_item.get('s')
                    price_str = ticker_item.get('c')
                    if symbol and 'USDT' in symbol and price_str:
                        try:
                            ticker_data[symbol] = float(price_str)
                        except ValueError:
                             logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol} في البث المجمع: '{price_str}'")
        else:
             logger.warning(f"⚠️ [WS] تم استلام رسالة WebSocket بتنسيق غير متوقع: {type(msg)}")

    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة التيكر: {e}", exc_info=True)


def run_ticker_socket_manager() -> None:
    """Runs and manages the WebSocket connection for mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] بدء إدارة WebSocket لأسعار التيكر...")
            twm = ThreadedWebsocketManager(api_key=utils.API_KEY, api_secret=utils.API_SECRET)
            twm.start()

            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء بث WebSocket: {stream_name}")

            twm.join()
            logger.warning("⚠️ [WS] توقفت إدارة WebSocket. إعادة التشغيل...")

        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في إدارة WebSocket: {e}. إعادة التشغيل في 15 ثانية...", exc_info=True)

        time.sleep(15)

# ---------------------- Trading Strategy (ML-Only) -------------------

class ScalpingTradingStrategy:
    """Encapsulates the trading strategy logic, now relying solely on ML model prediction for buy signals."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model = load_ml_model_from_db(symbol)
        if self.ml_model is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] لم يتم تحميل نموذج تعلم الآلة لـ {symbol}. لن تتمكن الإستراتيجية من توليد إشارات.")

        self.feature_columns_for_ml = [ # Features expected by the ML model
            'volume_15m_avg',
            'rsi_momentum_bullish',
            'btc_trend_encoded'
        ]

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates only the required indicators for the ML model's features."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] حساب المؤشرات لنموذج ML...")
        min_len_required = max(RSI_PERIOD, RSI_MOMENTUM_LOOKBACK_CANDLES, VOLUME_LOOKBACK_CANDLES) + 5

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame قصير جدًا ({len(df)} < {min_len_required}) لحساب مؤشرات ML.")
            return None

        try:
            df_calc = df.copy()
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)

            df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()

            df_calc['rsi_momentum_bullish'] = 0
            if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
                for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
                    rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
                    if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                        df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1

            btc_trend = get_btc_trend_4h()
            df_calc['btc_trend_encoded'] = 0
            if "صعود" in btc_trend:
                df_calc['btc_trend_encoded'] = 1
            elif "هبوط" in btc_trend:
                df_calc['btc_trend_encoded'] = -1

            for col in self.feature_columns_for_ml:
                if col not in df_calc.columns:
                    logger.warning(f"⚠️ [Strategy {self.symbol}] عمود الميزة المفقود لنموذج ML: {col}")
                    df_calc[col] = np.nan
                else:
                    df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            initial_len = len(df_calc)
            all_required_cols = list(set(self.feature_columns_for_ml + [
                'open', 'high', 'low', 'close', 'volume', 'atr'
            ]))
            df_cleaned = df_calc.dropna(subset=all_required_cols).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] تم إسقاط {dropped_count} صفًا بسبب قيم NaN في المؤشرات.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ بعد إزالة قيم NaN للمؤشرات.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] تم حساب مؤشرات ML. أحدث حجم 15 دقيقة: {latest.get('volume_15m_avg', np.nan):.2f}, RSI Momentum: {latest.get('rsi_momentum_bullish', np.nan)}, ATR: {latest.get('atr', np.nan):.4f}, BTC Trend: {latest.get('btc_trend_encoded', np.nan)}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] خطأ: لم يتم العثور على عمود مطلوب أثناء حساب المؤشر: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ غير متوقع أثناء حساب المؤشر: {e}", exc_info=True)
            return None


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal based solely on the ML model's bullish prediction,
        followed by essential filters (BTC trend, volume, profit margin).
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] إنشاء إشارة شراء (تعتمد على ML فقط)...")

        min_signal_data_len = max(VOLUME_LOOKBACK_CANDLES, RSI_MOMENTUM_LOOKBACK_CANDLES) + 1
        if df_processed is None or df_processed.empty or len(df_processed) < min_signal_data_len:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ أو قصير جدًا (<{min_signal_data_len})، لا يمكن إنشاء إشارة.")
            return None

        required_cols_for_signal = list(set(self.feature_columns_for_ml + [
            'close', 'atr'
        ]))
        missing_cols = [col for col in required_cols_for_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame يفتقد أعمدة مطلوبة للإشارة: {missing_cols}.")
            return None

        last_row = df_processed.iloc[-1]

        current_price = ticker_data.get(self.symbol)
        if current_price is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] السعر الحالي غير متاح من بيانات التيكر. لا يمكن إنشاء إشارة.")
            return None

        if last_row[self.feature_columns_for_ml].isnull().values.any() or pd.isna(last_row.get('atr')):
             logger.warning(f"⚠️ [Strategy {self.symbol}] البيانات التاريخية تحتوي على قيم NaN في أعمدة المؤشرات المطلوبة. لا يمكن إنشاء إشارة.")
             return None

        signal_details = {}

        btc_trend = get_btc_trend_4h()
        if "هبوط" in btc_trend:
            logger.info(f"ℹ️ [Strategy {self.symbol}] التداول متوقف بسبب اتجاه البيتكوين الهابط ({btc_trend}).")
            signal_details['BTC_Trend'] = f'هبوط ({btc_trend}) ❌'
            return None
        elif "N/A" in btc_trend:
             logger.warning(f"⚠️ [Strategy {self.symbol}] لا يمكن تحديد اتجاه البيتكوين، سيتم تجاهل هذا الشرط.")
             signal_details['BTC_Trend'] = 'غير متاح (تجاهل)'
        else:
             signal_details['BTC_Trend'] = f'صعود أو استقرار ({btc_trend}) ✅'

        ml_prediction_result_text = "N/A (نموذج غير محمل)"
        ml_is_bullish = False

        if self.ml_model:
            try:
                features_for_prediction = pd.DataFrame([last_row[self.feature_columns_for_ml].values], columns=self.feature_columns_for_ml)
                if hasattr(self.ml_model, 'scaler') and self.ml_model.scaler is not None:
                    features_for_prediction = self.ml_model.scaler.transform(features_for_prediction)

                ml_pred = self.ml_model.predict(features_for_prediction)[0]
                if ml_pred == 1:
                    ml_is_bullish = True
                    ml_prediction_result_text = 'صعودي ✅'
                    logger.info(f"✨ [Strategy {self.symbol}] تنبؤ نموذج ML صعودي.")
                else:
                    ml_prediction_result_text = 'هابط ❌'
                    logger.info(f"ℹ️ [Strategy {self.symbol}] تنبؤ نموذج ML هابط. تم رفض الإشارة.")
            except Exception as ml_err:
                logger.error(f"❌ [Strategy {self.symbol}] خطأ في تنبؤ نموذج ML: {ml_err}", exc_info=True)
                ml_prediction_result_text = "خطأ في التنبؤ (0)"
        
        signal_details['ML_Prediction'] = ml_prediction_result_text

        if not ml_is_bullish:
            return None
        
        volume_recent = fetch_recent_volume(self.symbol, SIGNAL_GENERATION_TIMEFRAME, VOLUME_LOOKBACK_CANDLES)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى المطلوب ({MIN_VOLUME_15M_USDT:,.0f} USDT). تم رفض الإشارة.")
            signal_details['Volume_Check'] = f'فشل: سيولة غير كافية ({volume_recent:,.0f} USDT)'
            return None
        else:
            signal_details['Volume_Check'] = f'نجاح: سيولة كافية ({volume_recent:,.0f} USDT)'

        current_atr = last_row.get('atr')

        if pd.isna(current_atr) or current_atr <= 0:
             logger.warning(f"⚠️ [Strategy {self.symbol}] قيمة ATR غير صالحة ({current_atr}) لحساب الهدف. لا يمكن إنشاء إشارة.")
             return None

        target_multiplier = ENTRY_ATR_MULTIPLIER
        initial_target = current_price + (target_multiplier * current_atr)

        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] هامش الربح ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%). تم رفض الإشارة.")
            signal_details['Profit_Margin_Check'] = f'فشل: هامش ربح غير كافٍ ({profit_margin_pct:.2f}%)'
            return None
        else:
            signal_details['Profit_Margin_Check'] = f'نجاح: هامش ربح كافٍ ({profit_margin_pct:.2f}%)'


        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'current_target': float(f"{initial_target:.8g}"),
            'r2_score': 1.0, # Placeholder score as it's ML-driven now
            'strategy_name': 'Scalping_ML_Only', # Updated strategy name
            'signal_details': signal_details,
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': 1.0 # Placeholder
        }

        logger.info(f"✅ [Strategy {self.symbol}] تم تأكيد إشارة الشراء (ML فقط). السعر: {current_price:.6f}, ATR: {current_atr:.6f}, الحجم: {volume_recent:,.0f}, تنبؤ ML: {ml_prediction_result_text}")
        return signal_output


# ---------------------- Telegram Functions ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Sends a message via Telegram Bot API with improved error handling."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': str(target_chat_id),
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': disable_web_page_preview
    }
    if reply_markup:
        try:
            payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
        except (TypeError, ValueError) as json_err:
             logger.error(f"❌ [Telegram] فشل تحويل reply_markup إلى JSON: {json_err} - Markup: {reply_markup}")
             return None

    logger.debug(f"ℹ️ [Telegram] إرسال رسالة إلى {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم إرسال الرسالة بنجاح إلى {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (مهلة).")
         return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ HTTP: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] تفاصيل خطأ API: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] تعذر فك تشفير استجابة الخطأ: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ في الطلب): {req_err}")
        return None
    except Exception as e:
         logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء إرسال الرسالة: {e}", exc_info=True)
         return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends a new trading signal alert to Telegram in Arabic, displaying the ML prediction."""
    logger.debug(f"ℹ️ [Telegram Alert] تنسيق وإرسال تنبيه للإشارة: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A')
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        signal_details = signal_data.get('signal_details', {})

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        entry_fee = trade_value_signal * BINANCE_FEE_RATE
        exit_value = trade_value_signal * (1 + profit_pct / 100.0)
        exit_fee = exit_value * BINANCE_FEE_RATE
        total_trade_fees = entry_fee + exit_fee

        profit_usdt_gross = trade_value_signal * (profit_pct / 100)
        profit_usdt_net = profit_usdt_gross - total_trade_fees

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        btc_trend = signal_details.get('BTC_Trend', 'N/A')
        ml_prediction_status = signal_details.get('ML_Prediction', 'N/A')

        message = (
            f"💡 *إشارة تداول جديدة (تعتمد على ML فقط)* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (طويل)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            f"💧 **السيولة (آخر 15 دقيقة):** {volume_15m:,.0f} USDT\n"
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🎯 **الهدف الأولي:** `${target_price:,.8g}`\n"
            f"💰 **الربح المتوقع (إجمالي):** ({profit_pct:+.2f}% / ≈ ${profit_usdt_gross:+.2f})\n"
            f"💸 **الرسوم المتوقعة:** ${total_trade_fees:,.2f}\n"
            f"📈 **الربح الصافي المتوقع:** ${profit_usdt_net:+.2f}\n"
            f"——————————————\n"
            f"🤖 *تنبؤ نموذج ML:* *{ml_prediction_status}*\n"
            f"✅ *الشروط الإضافية المحققة:*\n"
            f"  - فحص السيولة: {signal_details.get('Volume_Check', 'N/A')}\n"
            f"  - فحص هامش الربح: {signal_details.get('Profit_Margin_Check', 'N/A')}\n"
            f"——————————————\n"
            f"😨/🤑 **مؤشر الخوف والجشع:** {fear_greed}\n"
            f"₿ **اتجاه البيتكوين (4 ساعات):** {btc_trend}\n"
            f"——————————————\n"
            f"⏰ {timestamp_str}"
        )

        reply_markup = {
            "inline_keyboard": [
                [{"text": "📊 عرض تقرير الأداء", "callback_data": "get_report"}]
            ]
        }

        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')

    except KeyError as ke:
        logger.error(f"❌ [Telegram Alert] بيانات الإشارة غير مكتملة للرمز {signal_data.get('symbol', 'N/A')}: مفتاح مفقود {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة للرمز {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

def send_tracking_notification(details: Dict[str, Any]) -> None:
    """Formats and sends enhanced Telegram notifications for tracking events in Arabic."""
    symbol = details.get('symbol', 'N/A')
    signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown')
    message = ""
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
    closing_price = details.get('closing_price', 0.0)
    profit_pct = details.get('profit_pct', 0.0)
    current_price = details.get('current_price', 0.0)
    time_to_target = details.get('time_to_target', 'N/A')
    old_target = details.get('old_target', 0.0)
    new_target = details.get('new_target', 0.0)


    logger.debug(f"ℹ️ [Notification] تنسيق إشعار التتبع: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    if notification_type == 'target_hit':
        message = (
            f"✅ *تم الوصول إلى الهدف (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🎯 **سعر الإغلاق (الهدف):** `${closing_price:,.8g}`\n"
            f"💰 **الربح المحقق:** {profit_pct:+.2f}%\n"
            f"⏱️ **الوقت المستغرق:** {time_to_target}"
        )
    elif notification_type == 'target_updated':
         message = (
             f"↗️ *تم تحديث الهدف (ID: {signal_id})*\n"
             f"——————————————\n"
             f"🪙 **الزوج:** `{safe_symbol}`\n"
             f"📈 **السعر الحالي:** `${current_price:,.8g}`\n"
             f"🎯 **الهدف السابق:** `${old_target:,.8g}`\n"
             f"🎯 **الهدف الجديد:** `${new_target:,.8g}`\n"
             f"ℹ️ *تم التحديث بناءً على استمرار الزخم الصعودي.*"
         )
    else:
        logger.warning(f"⚠️ [Notification] نوع إشعار غير معروف: {notification_type} للتفاصيل: {details}")
        return

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table with the weighted score and entry time."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة {signal.get('symbol', 'N/A')} بسبب مشكلة في اتصال قاعدة البيانات.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] محاولة إدراج إشارة لـ {symbol}...")
    try:
        signal_prepared = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, current_target,
                 r2_score, strategy_name, signal_details, volume_15m, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW());
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                signal_prepared['initial_target'],
                signal_prepared['current_target'],
                signal_prepared.get('r2_score'),
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                signal_prepared.get('volume_15m')
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] تم إدراج إشارة لـ {symbol} في قاعدة البيانات (الدرجة: {signal_prepared.get('r2_score')}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] خطأ في قاعدة البيانات أثناء إدراج إشارة لـ {symbol}: {db_err}")
        if conn: conn.rollback()
        return False
    except (TypeError, ValueError) as convert_err:
         logger.error(f"❌ [DB Insert] خطأ في تحويل بيانات الإشارة قبل الإدراج لـ {symbol}: {convert_err} - بيانات الإشارة: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ غير متوقع أثناء إدراج إشارة لـ {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function ----------------------
def track_signals() -> None:
    """Tracks open signals and checks targets. Calculates time to target upon hit."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة في اتصال قاعدة البيانات.")
                time.sleep(15)
                continue

            with conn.cursor() as track_cur:
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_target, current_target, entry_time
                    FROM signals
                    WHERE achieved_target = FALSE;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                time.sleep(10)
                continue

            logger.debug(f"ℹ️ [Tracker] تتبع {len(open_signals)} إشارة مفتوحة...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False

                try:
                    entry_price = float(signal_row['entry_price'])
                    entry_time = signal_row['entry_time']
                    current_target = float(signal_row['current_target'])

                    current_price = ticker_data.get(symbol)

                    if current_price is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): السعر الحالي غير متاح في بيانات التيكر.")
                         continue

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f}")

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id, 'current_price': current_price}


                    # --- Check and Update Logic ---
                    # 1. Check for Target Hit
                    if current_price >= current_target:
                        profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                        closed_at = datetime.now()
                        time_to_target_duration = closed_at - entry_time if entry_time else timedelta(0)
                        time_to_target_str = str(time_to_target_duration)

                        update_query = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s WHERE id = %s;")
                        update_params = (current_target, closed_at, profit_pct, time_to_target_duration, signal_id)
                        log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): تم الوصول إلى الهدف عند {current_target:.8g} (الربح: {profit_pct:+.2f}%, الوقت: {time_to_target_str})."
                        notification_details.update({'type': 'target_hit', 'closing_price': current_target, 'profit_pct': profit_pct, 'time_to_target': time_to_target_str})
                        update_executed = True

                    # 2. Check for Target Extension (Only if Target not hit)
                    if not update_executed:
                        if current_price >= current_target * (1 - TARGET_APPROACH_THRESHOLD_PCT):
                             logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر قريب من الهدف ({current_price:.8g} مقابل {current_target:.8g}). التحقق من إشارة الاستمرار...")

                             df_continuation = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)

                             if df_continuation is not None and not df_continuation.empty:
                                 continuation_strategy = ScalpingTradingStrategy(symbol)
                                 df_continuation_indicators = continuation_strategy.populate_indicators(df_continuation)

                                 if df_continuation_indicators is not None:
                                     continuation_signal = continuation_strategy.generate_buy_signal(df_continuation_indicators)

                                     if continuation_signal:
                                         latest_row = df_continuation_indicators.iloc[-1]
                                         current_atr_for_new_target = latest_row.get('atr')

                                         if pd.notna(current_atr_for_new_target) and current_atr_for_new_target > 0:
                                             potential_new_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr_for_new_target)

                                             if potential_new_target > current_target:
                                                 old_target = current_target
                                                 new_target = potential_new_target
                                                 update_query = sql.SQL("UPDATE signals SET current_target = %s WHERE id = %s;")
                                                 update_params = (new_target, signal_id)
                                                 log_message = f"↗️ [Tracker] {symbol}(ID:{signal_id}): تم تحديث الهدف من {old_target:.8g} إلى {new_target:.8g} بناءً على إشارة الاستمرار."
                                                 notification_details.update({'type': 'target_updated', 'old_target': old_target, 'new_target': new_target})
                                                 update_executed = True
                                             else:
                                                 logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): تم اكتشاف إشارة استمرار، لكن الهدف الجديد ({potential_new_target:.8g}) ليس أعلى من الهدف الحالي ({current_target:.8g}). عدم تحديث الهدف.")
                                         else:
                                             logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب الهدف الجديد بسبب ATR غير صالح ({current_atr_for_new_target}) من بيانات الاستمرار.")
                                     else:
                                         logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر قريب من الهدف، ولكن لم يتم إنشاء إشارة استمرار.")
                                 else:
                                     logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): فشل في ملء المؤشرات للتحقق من الاستمرار.")
                             else:
                                 logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن جلب البيانات التاريخية للتحقق من الاستمرار.")


                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في قاعدة البيانات أثناء التحديث: {db_err}")
                            if conn: conn.rollback()
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تنفيذ التحديث/الإشعار: {exec_err}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err:
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في تحويل قيم الإشارة الأولية: {convert_err}")
                    continue
                except Exception as inner_loop_err:
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء معالجة الإشارة: {inner_loop_err}", exc_info=True)
                     continue

            if active_signals_summary:
                logger.debug(f"ℹ️ [Tracker] نهاية حالة الدورة ({processed_in_cycle} معالجة): {'; '.join(active_signals_summary)}")

            time.sleep(3)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] خطأ في قاعدة البيانات في دورة التتبع الرئيسية: {db_cycle_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(30)
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارة: {cycle_err}", exc_info=True)
            time.sleep(30)

# ---------------------- Comprehensive Performance Report Generation Function ----------------------
def generate_performance_report() -> str:
    """Generates a comprehensive performance report from the database in Arabic, including recent closed trades and USD profit/loss."""
    logger.info("ℹ️ [Report] إنشاء تقرير الأداء...")
    if not check_db_connection() or not conn or not cur:
        logger.error("❌ [Report] لا يمكن إنشاء التقرير، مشكلة في اتصال قاعدة البيانات.")
        return "❌ لا يمكن إنشاء التقرير، مشكلة في اتصال قاعدة البيانات."
    try:
        with conn.cursor() as report_cur:
            report_cur.execute("SELECT id, symbol, entry_price, entry_time FROM signals WHERE achieved_target = FALSE ORDER BY entry_time DESC;")
            open_signals = report_cur.fetchall()
            open_signals_count = len(open_signals)

            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed,
                    COUNT(*) FILTER (WHERE profit_percentage > 0) AS winning_signals,
                    COUNT(*) FILTER (WHERE profit_percentage <= 0) AS losing_signals,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct_sum,
                    COALESCE(AVG(profit_percentage) FILTER (WHERE profit_percentage > 0), 0) AS avg_win_pct,
                    COALESCE(AVG(profit_percentage) FILTER (WHERE profit_percentage <= 0), 0) AS avg_loss_pct,
                    COALESCE(SUM(entry_price * (1 + profit_percentage/100.0)), 0) AS total_exit_value
                FROM signals
                WHERE achieved_target = TRUE;
            """)
            closed_stats = report_cur.fetchone() or {}

            total_closed = closed_stats.get('total_closed', 0)
            winning_signals = closed_stats.get('winning_signals', 0)
            losing_signals = closed_stats.get('losing_signals', 0)
            total_profit_pct_sum = closed_stats.get('total_profit_pct_sum', 0.0)
            gross_profit_pct_sum = total_profit_pct_sum
            avg_win_pct = closed_stats.get('avg_win_pct', 0.0)
            avg_loss_pct = closed_stats.get('avg_loss_pct', 0.0)
            total_exit_value = closed_stats.get('total_exit_value', 0.0)

            gross_profit_usd = (total_profit_pct_sum / 100.0) * TRADE_VALUE
            total_fees_usd = (total_closed * TRADE_VALUE * BINANCE_FEE_RATE) + (total_exit_value * BINANCE_FEE_RATE)
            net_profit_usd = gross_profit_usd - total_fees_usd
            net_profit_pct = (net_profit_usd / (total_closed * TRADE_VALUE)) * 100 if total_closed * TRADE_VALUE > 0 else 0.0

            win_rate = (winning_signals / total_closed) * 100.0 if total_closed > 0 else 0.0
            profit_factor = float('inf') if abs(avg_loss_pct) < 1e-9 else (gross_profit_pct_sum / abs(avg_loss_pct * losing_signals)) if losing_signals > 0 else float('inf')


        report = (
            f"📊 *تقرير الأداء الشامل:*\n"
            f"_(افتراض حجم الصفقة: ${TRADE_VALUE:,.2f} ورسوم Binance: {BINANCE_FEE_RATE*100:.2f}% لكل صفقة)_ \n"
            f"——————————————\n"
            f"📈 الإشارات المفتوحة حالياً: *{open_signals_count}*\n"
        )

        if open_signals:
            report += "  • التفاصيل:\n"
            for signal in open_signals:
                safe_symbol = str(signal['symbol']).replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                entry_time_str = signal['entry_time'].strftime('%Y-%m-%d %H:%M') if signal['entry_time'] else 'N/A'
                report += f"    - `{safe_symbol}` (دخول: ${signal['entry_price']:.8g} | فتح: {entry_time_str})\n"
        else:
            report += "  • لا توجد إشارات مفتوحة حالياً.\n"

        report += (
            f"——————————————\n"
            f"📉 *إحصائيات الإشارات المغلقة (تم تحقيق الهدف فقط):*\n"
            f"  • إجمالي الإشارات المغلقة: *{total_closed}*\n"
            f"  ✅ إشارات رابحة: *{winning_signals}* ({win_rate:.2f}%)\n"
            f"  ❌ إشارات خاسرة: *{losing_signals}*\n"
            f"——————————————\n"
            f"💰 *الربحية الإجمالية (للصفقات التي حققت الهدف):*\n"
            f"  • إجمالي الربح الإجمالي: *{gross_profit_pct_sum:+.2f}%* (≈ *${gross_profit_usd:+.2f}*)\n"
            f"  • إجمالي الرسوم المدفوعة: *${total_fees_usd:,.2f}*\n"
            f"  • *الربح الصافي:* *{net_profit_pct:+.2f}%* (≈ *${net_profit_usd:+.2f}*)\n"
            f"  • متوسط الصفقة الرابحة: *{avg_win_pct:+.2f}%*\n"
            f"  • متوسط الصفقة الخاسرة: *{avg_loss_pct:+.2f}%*\n"
            f"  • عامل الربح: *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}*\n"
            f"——————————————\n"
        )

        report += "ℹ️ *ملاحظة: هذا التقرير يعرض فقط الصفقات التي حققت الهدف، حيث تم إزالة منطق وقف الخسارة.*"
        report += "\n——————————————\n"


        report += (
            f"🕰️ _التقرير محدث حتى: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )

        logger.info("✅ [Report] تم إنشاء تقرير الأداء بنجاح.")
        return report

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء: {db_err}")
        if conn: conn.rollback()
        return "❌ خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء."
    except Exception as e:
        logger.error(f"❌ [Report] خطأ غير متوقع أثناء إنشاء تقرير الأداء: {e}", exc_info=True)
        return "❌ حدث خطأ غير متوقع أثناء إنشاء تقرير الأداء."

# ---------------------- Flask Service (Optional for Webhook) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    main_bot_alive = main_bot_thread.is_alive() if 'main_bot_thread' in globals() and main_bot_thread else False
    status = "running" if ws_alive and tracker_alive and main_bot_alive else "partially running"
    return Response(f"📈 Crypto Signal Bot ({status}) - Last Check: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """Handles favicon request to avoid 404 errors in logs."""
    return Response(status=204)

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    """Handles incoming requests from Telegram (like button presses and commands)."""
    if not WEBHOOK_URL:
        logger.warning("⚠️ [Flask] تم استلام طلب webhook، ولكن WEBHOOK_URL غير مهيأ. تجاهل الطلب.")
        return "Webhook not configured", 200

    if not request.is_json:
        logger.warning("⚠️ [Flask] تم استلام طلب webhook غير JSON.")
        return "Invalid request format", 400

    try:
        data = request.get_json()
        logger.info(f"✅ [Flask] تم استلام بيانات webhook. حجم البيانات: {len(json.dumps(data))} بايت.")
        logger.debug(f"ℹ️ [Flask] بيانات webhook الكاملة: {json.dumps(data)}")

        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')

            logger.info(f"ℹ️ [Flask] تم استلام استعلام رد اتصال (Callback Query). ID: {callback_id}, البيانات: '{callback_data}'")

            if not message_info or not callback_data:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (ID: {callback_id}) يفتقد الرسالة أو البيانات. تجاهل.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200
            chat_id_callback = message_info.get('chat', {}).get('id')
            if not chat_id_callback:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (ID: {callback_id}) يفتقد معرف الدردشة. تجاهل.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200

            message_id = message_info['message_id']
            user_info = callback_query.get('from', {})
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] معالجة استعلام رد الاتصال: البيانات='{callback_data}', المستخدم={username}({user_id}), الدردشة={chat_id_callback}")

            try:
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                logger.debug(f"✅ [Flask] تم تأكيد استعلام رد الاتصال {callback_id}.")
            except Exception as ack_err:
                 logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال {callback_id}: {ack_err}")

            if callback_data == "get_report":
                logger.info(f"ℹ️ [Flask] تم استلام طلب 'get_report' من الدردشة {chat_id_callback}. جاري إنشاء التقرير...")
                report_content = generate_performance_report()
                logger.info(f"✅ [Flask] تم إنشاء التقرير. طول التقرير: {len(report_content)} حرف.")
                report_thread = Thread(target=lambda: send_telegram_message(chat_id_callback, report_content, parse_mode='Markdown'))
                report_thread.start()
                logger.info(f"✅ [Flask] تم بدء خيط إرسال التقرير للدردشة {chat_id_callback}.")
            else:
                logger.warning(f"⚠️ [Flask] تم استلام بيانات رد اتصال غير معالجة: '{callback_data}'")

        elif 'message' in data:
            message_data = data['message']
            chat_info = message_data.get('chat')
            user_info = message_data.get('from', {})
            text_msg = message_data.get('text', '').strip()

            if not chat_info or not text_msg:
                 logger.debug("ℹ️ [Flask] تم استلام رسالة بدون معلومات الدردشة أو النص.")
                 return "OK", 200

            chat_id_msg = chat_info['id']
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] تم استلام رسالة: النص='{text_msg}', المستخدم={username}({user_id}), الدردشة={chat_id_msg}")

            if text_msg.lower() == '/report':
                 report_thread = Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown'))
                 report_thread.start()
            elif text_msg.lower() == '/status':
                 status_thread = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread.start()
            elif text_msg.lower() == '/train_models': # أمر جديد لتدريب النماذج يدوياً
                 train_thread = Thread(target=lambda: run_training_for_all_symbols(chat_id_msg))
                 train_thread.start()

        else:
            logger.debug("ℹ️ [Flask] تم استلام بيانات webhook بدون 'callback_query' أو 'message'.")

        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] خطأ في معالجة webhook: {e}", exc_info=True)
         return "Internal Server Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Separate function to handle /status command to avoid blocking the Webhook."""
    logger.info(f"ℹ️ [Flask Status] معالجة أمر /status للدردشة {chat_id_msg}")
    status_msg = "⏳ جلب الحالة..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg)
    if not (msg_sent and msg_sent.get('ok')):
         logger.error(f"❌ [Flask Status] فشل إرسال رسالة الحالة الأولية إلى {chat_id_msg}")
         return
    message_id_to_edit = msg_sent['result']['message_id'] if msg_sent and msg_sent.get('result') else None

    if message_id_to_edit is None:
        logger.error(f"❌ [Flask Status] فشل الحصول على message_id لتحديث الحالة في الدردشة {chat_id_msg}")
        return

    try:
        open_count = 0
        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                open_count = (status_cur.fetchone() or {}).get('count', 0)

        ws_status = 'نشط ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'غير نشط ❌'
        tracker_status = 'نشط ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'غير نشط ❌'
        main_bot_alive = 'نشط ✅' if 'main_bot_thread' in globals() and main_bot_thread and main_bot_thread.is_alive() else 'غير نشط ❌'
        final_status_msg = (
            f"🤖 *حالة البوت:*\n"
            f"- تتبع الأسعار (WS): {ws_status}\n"
            f"- تتبع الإشارات: {tracker_status}\n"
            f"- حلقة البوت الرئيسية: {main_bot_alive}\n"
            f"- الإشارات النشطة: *{open_count}* / {MAX_OPEN_TRADES}\n"
            f"- وقت الخادم الحالي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        edit_payload = {
            'chat_id': chat_id_msg,
             'message_id': message_id_to_edit,
            'text': final_status_msg,
            'parse_mode': 'Markdown'
        }
        response = requests.post(edit_url, json=edit_payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Flask Status] تم تحديث الحالة للدردشة {chat_id_msg}")

    except Exception as status_err:
        logger.error(f"❌ [Flask Status] خطأ في جلب/تعديل تفاصيل الحالة للدردشة {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ حدث خطأ أثناء جلب تفاصيل الحالة.")


def run_flask() -> None:
    """Runs the Flask application to listen for the Webhook using a production server if available."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] استخدام خادم 'waitress'.")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
         logger.warning("⚠️ [Flask] 'waitress' غير مثبت. الرجوع إلى خادم تطوير Flask (لا يوصى به للإنتاج).")
         try:
             app.run(host=host, port=port)
         except Exception as flask_run_err:
              logger.critical(f"❌ [Flask] فشل بدء خادم التطوير: {flask_run_err}", exc_info=True)
    except Exception as serve_err:
         logger.critical(f"❌ [Flask] فشل بدء الخادم (waitress؟): {serve_err}", exc_info=True)

# ---------------------- Main Loop and Check Function ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لا توجد رموز صالحة تم تحميلها أو التحقق منها. لا يمكن المتابمة.")
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا صالحًا للمسح.")
    last_full_scan_time = time.time()

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] بدء دورة مسح السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] تخطي دورة المسح بسبب فشل اتصال قاعدة البيانات.")
                time.sleep(60)
                continue

            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء التحقق من عدد الإشارات المفتوحة: {db_err}. تخطي الدورة.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] الإشارات المفتوحة حالياً: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول إلى الحد الأقصى لعدد الإشارات المفتوحة. انتظار...")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60)
                continue

            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] تم الوصول إلى الحد الأقصى ({MAX_OPEN_TRADES}) أثناء المسح. إيقاف مسح الرموز لهذه الدورة.")
                      break

                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] مسح {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")

                 try:
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            continue

                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        continue

                    strategy = ScalpingTradingStrategy(symbol)
                    if strategy.ml_model is None:
                        logger.warning(f"⚠️ [Main] تخطي {symbol} لأن نموذج ML الخاص به لم يتم تحميله بنجاح. يرجى تدريب النموذج أولاً باستخدام أمر /train_models.")
                        continue

                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    if potential_signal:
                        logger.info(f"✨ [Main] تم العثور على إشارة محتملة لـ {symbol}! التحقق النهائي والإدراج...")
                        with conn.cursor() as final_check_cur:
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     time.sleep(2)
                                 else:
                                     logger.error(f"❌ [Main] فشل إدراج الإشارة لـ {symbol} في قاعدة البيانات.")
                             else:
                                 logger.warning(f"⚠️ [Main] تم الوصول إلى الحد الأقصى ({final_open_count}) قبل إدراج الإشارة لـ {symbol}. تم تجاهل الإشارة.")
                                 break

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء معالجة الرمز {symbol}: {db_loop_err}. الانتقال إلى التالي...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] خطأ عام في معالجة الرمز {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 time.sleep(0.1)

            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] انتهت دورة المسح. الإشارات التي تم إنشاؤها: {signals_generated_in_loop}. مدة المسح: {scan_duration:.2f} ثانية.")
            frame_minutes = get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME)
            wait_time = max(frame_minutes * 60, 120 - scan_duration)
            logger.info(f"⏳ [Main] انتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] تم طلب الإيقاف (KeyboardInterrupt). إيقاف التشغيل...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] خطأ فادح في قاعدة البيانات في الحلقة الرئيسية: {db_main_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60)
             try:
                 init_db()
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] فشل إعادة الاتصال بقاعدة البيانات: {recon_err}. خروج...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 120 ثانية قبل إعادة المحاولة...")
            time.sleep(120)

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] إغلاق الموارد...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ في إغلاق اتصال قاعدة البيانات: {close_err}")
    logger.info("✅ [Cleanup] اكتمل تنظيف الموارد.")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات التداول...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | وقت UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None
    main_bot_thread: Optional[Thread] = None

    try:
        # 1. تهيئة عميل Binance وقاعدة البيانات
        initialize_binance_client()
        init_db()

        # 2. بدء WebSocket Ticker
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر WebSocket.")
        logger.info("ℹ️ [Main] انتظار 5 ثوانٍ لتهيئة WebSocket...")
        time.sleep(5)
        if not ticker_data:
             logger.warning("⚠️ [Main] لم يتم استلام بيانات أولية من WebSocket بعد 5 ثوانٍ.")
        else:
             logger.info(f"✅ [Main] تم استلام بيانات أولية من WebSocket لـ {len(ticker_data)} رمزًا.")

        # 3. بدء Signal Tracker
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر الإشارة.")

        # 4. بدء حلقة البوت الرئيسية في خيط منفصل
        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        logger.info("✅ [Main] تم بدء حلقة البوت الرئيسية في خيط منفصل.")

        # 5. بدء خادم Flask (يعمل دائمًا، daemon=False للحفاظ على البرنامج الرئيسي نشطًا)
        flask_thread = Thread(target=run_flask, daemon=False, name="FlaskThread")
        flask_thread.start()
        logger.info("✅ [Main] تم بدء خادم Flask.")

        # انتظر حتى ينتهي خيط Flask (عادة لن ينتهي إلا إذا كان هناك خطأ)
        flask_thread.join()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل أو في الحلقة الرئيسية: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] يتم إيقاف تشغيل البرنامج...")
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف بوت إشارات التداول.")
        # os._exit(0)
