import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql, OperationalError, InterfaceError # لاستخدام استعلامات آمنة وأخطاء محددة
from psycopg2.extras import RealDictCursor # للحصول على النتائج كقواميس
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException # أخطاء Binance المحددة
from flask import Flask, request, Response
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union # لإضافة Type Hinting

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # إضافة اسم المسجل
    handlers=[
        logging.FileHandler('crypto_bot_bottom_fishing.log', encoding='utf-8'), # تغيير اسم ملف السجل
        logging.StreamHandler()
    ]
)
# استخدام اسم محدد للمسجل بدلاً من الجذر
logger = logging.getLogger('CryptoBot')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    # استخدام قيمة افتراضية None إذا لم يكن المتغير موجودًا
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1) # استخدام رمز خروج غير صفري للإشارة إلى خطأ

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Telegram Token: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT_ID}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'}")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0         # Default trade value in USDT
MAX_OPEN_TRADES: int = 4          # Maximum number of open trades simultaneously
SIGNAL_GENERATION_TIMEFRAME: str = '30m' # Timeframe for signal generation (كما طلب المستخدم، سنستخدمها للتوليد)
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 10 # Historical data lookback in days for signal generation (زيادة الأيام لالتقاط المزيد من القيعان المحتملة)
SIGNAL_TRACKING_TIMEFRAME: str = '1m' # Timeframe for signal tracking and stop loss updates (إطار زمني أصغر لتتبع أكثر آنية)
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 2   # Historical data lookback in days for signal tracking (أيام أقل للتتبع)

# =============================================================================
# --- Indicator Parameters ---
# Adjusted parameters for bottom fishing strategy
# =============================================================================
RSI_PERIOD: int = 14          # RSI Period
RSI_OVERSOLD: int = 35        # Increased Oversold threshold slightly to catch earlier signs
RSI_OVERBOUGHT: int = 65      # Decreased Overbought threshold
EMA_SHORT_PERIOD: int = 10      # Short EMA period (faster)
EMA_LONG_PERIOD: int = 20       # Long EMA period (faster)
VWMA_PERIOD: int = 20           # VWMA Period
SWING_ORDER: int = 5          # Order for swing point detection (for Elliott, etc.)
# ... (Other constants remain the same) ...
FIB_LEVELS_TO_CHECK: List[float] = [0.382, 0.5, 0.618] # قد تكون أقل أهمية لاستراتيجية القيعان البسيطة
FIB_TOLERANCE: float = 0.007
LOOKBACK_FOR_SWINGS: int = 100
ENTRY_ATR_PERIOD: int = 14     # ATR Period for entry/tracking
ENTRY_ATR_MULTIPLIER: float = 2.0 # Reduced ATR Multiplier for initial target/stop (أكثر تحفظًا في الهدف الأولي)
BOLLINGER_WINDOW: int = 20     # Bollinger Bands Window
BOLLINGER_STD_DEV: int = 2       # Bollinger Bands Standard Deviation
MACD_FAST: int = 12            # MACD Fast Period
MACD_SLOW: int = 26            # MACD Slow Period
MACD_SIGNAL: int = 9             # MACD Signal Line Period
ADX_PERIOD: int = 14            # ADX Period
SUPERTREND_PERIOD: int = 10     # SuperTrend Period (قد يكون أقل أهمية في القيعان)
SUPERTREND_MULTIPLIER: float = 2.5 # SuperTrend Multiplier (أقل شراسة)

# Trailing Stop Loss (Adjusted for tighter stops)
TRAILING_STOP_ACTIVATION_PROFIT_PCT: float = 0.01 # Profit percentage to activate trailing stop (1%)
TRAILING_STOP_ATR_MULTIPLIER: float = 1.5        # Reduced ATR Multiplier for tighter trailing stop
TRAILING_STOP_MOVE_INCREMENT_PCT: float = 0.002  # Price increase percentage to move trailing stop (0.2%) - Increased slightly for less frequent updates

# Additional Signal Conditions
MIN_PROFIT_MARGIN_PCT: float = 1.5 # Minimum required profit margin percentage (Reduced for potential faster trades)
MIN_VOLUME_15M_USDT: float = 200000.0 # Minimum liquidity in the last 15 minutes in USDT (Increased slightly)
MAX_TRADE_DURATION_HOURS: int = 72 # Maximum time to keep a trade open (72 hours = 3 days)
# =============================================================================
# --- End Indicator Parameters ---
# =============================================================================

# Global variables (will be initialized later)
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {} # Dictionary to store the latest closing prices for symbols

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping() # Check connection and keys validity
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance بنجاح. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except BinanceRequestException as req_err:
     logger.critical(f"❌ [Binance] خطأ طلب Binance (مشكلة شبكة أو طلب): {req_err}")
     exit(1)
except BinanceAPIException as api_err:
     logger.critical(f"❌ [Binance] خطأ API Binance (مفاتيح غير صالحة أو مشكلة خادم): {api_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}", exc_info=True)
    exit(1)

# ---------------------- Additional Indicator Functions (Keep existing) ----------------------
# Keep get_fear_greed_index, fetch_historical_data, calculate_ema,
# calculate_vwma, get_btc_trend_4h, calculate_rsi_indicator,
# calculate_atr_indicator, calculate_bollinger_bands, calculate_macd,
# calculate_adx, calculate_vwap, calculate_obv, calculate_supertrend,
# is_hammer, is_shooting_star, is_doji, compute_engulfing, detect_candlestick_patterns,
# detect_swings, detect_elliott_waves, fetch_recent_volume as they are.

# ... (Past in all the existing functions here, from get_fear_greed_index down to fetch_recent_volume) ...
# IMPORTANT: You need to copy and paste all the functions from the original code
# between the marker "---------------------- Additional Indicator Functions ----------------------"
# and "---------------------- Comprehensive Performance Report Generation Function ----------------------"
# and also the Candlestick pattern functions and Swing/Elliott/Volume functions.

# ---------------------- Database Connection Setup (Keep existing) ----------------------
# Keep init_db, check_db_connection, convert_np_values as they are.
# The signals table structure should be sufficient. We'll store the strategy name.

# ... (Past in all the existing DB functions here, from init_db down to convert_np_values) ...
# IMPORTANT: Copy and paste all the database related functions.


# ---------------------- Reading and Validating Symbols List (Keep existing) ----------------------
# Keep get_crypto_symbols as it is.

# ... (Past in get_crypto_symbols here) ...

# ---------------------- WebSocket Management for Ticker Prices (Keep existing) ----------------------
# Keep handle_ticker_message, run_ticker_socket_manager as they are. They are essential for real-time price data.

# ... (Past in handle_ticker_message and run_ticker_socket_manager here) ...


# ---------------------- Technical Indicator Functions (Keep existing) ----------------------
# Keep calculate_rsi_indicator, calculate_atr_indicator, calculate_bollinger_bands,
# calculate_macd, calculate_adx, calculate_vwap, calculate_obv, calculate_supertrend
# as they are. Ensure they handle NaNs appropriately (already done in the original code).

# ... (Past in all the existing indicator functions here) ...

# ---------------------- Candlestick Patterns (Keep existing) ----------------------
# Keep is_hammer, is_shooting_star, is_doji, compute_engulfing, detect_candlestick_patterns.
# Hammer and Bullish Engulfing are crucial for bottom identification.

# ... (Past in all the existing candlestick functions here) ...

# ---------------------- Other Helper Functions (Elliott, Swings, Volume) (Keep existing) ----------------------
# Keep detect_swings, detect_elliott_waves, fetch_recent_volume. Elliott wave detection is less crucial for a simple bottom strategy, but can remain. Fetching volume is important.

# ... (Past in all the existing helper functions here) ...


# ---------------------- Comprehensive Performance Report Generation Function (Keep existing) ----------------------
# Keep generate_performance_report as it is. It works with the existing DB structure.

# ... (Past in generate_performance_report here) ...


# ---------------------- Trading Strategy (Modified for Bottom Fishing) -------------------

class BottomFishingStrategy:
    """Encapsulates the trading strategy logic focused on capturing potential bottoms using a scoring system."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        # Required columns for indicator calculation (ensure all needed indicators are listed)
        self.required_cols_indicators = [
            'open', 'high', 'low', 'close', 'volume',
            'ema_10', 'ema_20', 'vwma', # Changed EMA periods
            'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            'vwap', 'obv', 'supertrend', 'supertrend_trend',
            'BullishCandleSignal', 'BearishCandleSignal'
        ]
        # Required columns for buy signal generation
        self.required_cols_buy_signal = [
            'close',
            'ema_10', 'ema_20', 'vwma',
            'rsi', 'atr',
            'macd', 'macd_signal', 'macd_hist',
            'supertrend_trend', 'adx', 'di_plus', 'di_minus', 'vwap', 'bb_lower', # Changed bb_upper to bb_lower
            'BullishCandleSignal', 'obv'
        ]

        # =====================================================================
        # --- Scoring System (Weights) for Optional Conditions ---
        # Adjusted weights and conditions for bottom fishing
        # =====================================================================
        self.condition_weights = {
            'rsi_bouncing_up': 2.5,       # RSI rising from oversold (High importance)
            'price_near_bb_lower': 2.0,   # Price touching or below lower Bollinger Band (High importance)
            'macd_bullish_momentum_shift': 2.0, # MACD showing upward momentum shift
            'price_crossing_vwma_up': 1.5, # Price crosses above VWMA
            'adx_low_and_di_cross': 1.0,  # Low ADX and DI+ crossing above DI-
            'price_crossing_ema10_up': 1.0, # Price crosses above faster EMA
            'obv_rising': 1.0,            # OBV is rising

            # Removed conditions that conflict with bottom fishing or breakout
            # 'ema_cross_bullish': 0,
            # 'supertrend_up': 0,
            # 'above_vwma': 0, # Now looking for cross up from below
            # 'macd_positive_or_cross': 0, # Looking for shift from below zero
            # 'adx_trending_bullish': 0, # Looking for low ADX turning
            # 'breakout_bb_upper': 0,
            # 'rsi_ok': 0, # Replaced by rsi_bouncing_up
            # 'not_bb_extreme': 0, # Replaced by price_near_bb_lower
            # 'rsi_filter_breakout': 0,
            # 'macd_filter_breakout': 0
        }
        # =====================================================================

        # =====================================================================
        # --- Mandatory Entry Conditions (All must be met for bottom fishing) ---
        # Focused on oversold state and bullish reversal candle
        # =====================================================================
        self.essential_conditions = [
            'rsi_oversold_or_bouncing', # RSI is oversold OR just bounced from oversold
            'bullish_reversal_candle', # Presence of a bullish reversal candle
            # Add others as deemed essential, e.g., price near BB lower
            'price_has_dropped_recently' # Add a simple check for recent price drop (needs implementation)
        ]
        # =====================================================================


        # Calculate total possible score for *optional* conditions
        self.total_possible_score = sum(self.condition_weights.values())

        # Required signal score threshold for *optional* conditions (as a percentage)
        # Adjust this threshold based on testing
        self.min_score_threshold_pct = 0.40 # Example: 40% of optional points (adjustable)
        self.min_signal_score = self.total_possible_score * self.min_score_threshold_pct


    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators for the strategy."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] حساب المؤشرات...")
        # Update minimum required rows based on the largest period of used indicators
        min_len_required = max(EMA_SHORT_PERIOD, EMA_LONG_PERIOD, VWMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD) + 5 # Add a small buffer

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] بيانات غير كافية ({len(df)} < {min_len_required}) لحساب المؤشرات.")
            return None

        try:
            df_calc = df.copy()
            # ATR is required for SuperTrend and Stop Loss/Target
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            # SuperTrend needs ATR calculated with its own period
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)

            # --- EMA Calculation (using adjusted periods) ---
            df_calc['ema_10'] = calculate_ema(df_calc['close'], EMA_SHORT_PERIOD) # Add EMA 10
            df_calc['ema_20'] = calculate_ema(df_calc['close'], EMA_LONG_PERIOD) # Add EMA 20
            # ----------------------

            # --- VWMA Calculation ---
            df_calc['vwma'] = calculate_vwma(df_calc, VWMA_PERIOD) # Calculate VWMA
            # ----------------------

            # Rest of the indicators
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_bollinger_bands(df_calc, BOLLINGER_WINDOW, BOLLINGER_STD_DEV)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL) # Ensure macd_hist is calculated here
            adx_df = calculate_adx(df_calc, ADX_PERIOD)
            df_calc = df_calc.join(adx_df)
            df_calc = calculate_vwap(df_calc) # Note: VWAP resets daily, VWMA is a rolling average
            df_calc = calculate_obv(df_calc)
            df_calc = detect_candlestick_patterns(df_calc)

            # Check for required columns after calculation
            missing_cols = [col for col in self.required_cols_indicators if col not in df_calc.columns]
            if missing_cols:
                 logger.error(f"❌ [Strategy {self.symbol}] أعمدة المؤشرات المطلوبة مفقودة بعد الحساب: {missing_cols}")
                 logger.debug(f"الأعمدة المتوفرة: {df_calc.columns.tolist()}")
                 return None

            # Handle NaNs after indicator calculation
            initial_len = len(df_calc)
            # Use required_cols_indicators which contains all calculated columns
            df_cleaned = df_calc.dropna(subset=self.required_cols_indicators).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] تم حذف {dropped_count} صف بسبب قيم NaN في المؤشرات.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ بعد إزالة قيم NaN للمؤشرات.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] تم حساب المؤشرات. آخر قيم - EMA10: {latest.get('ema_10', np.nan):.4f}, EMA20: {latest.get('ema_20', np.nan):.4f}, VWMA: {latest.get('vwma', np.nan):.4f}, RSI: {latest.get('rsi', np.nan):.1f}, MACD Hist: {latest.get('macd_hist', np.nan):.4f}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] خطأ: العمود المطلوب غير موجود أثناء حساب المؤشر: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ غير متوقع أثناء حساب المؤشرات: {e}", exc_info=True)
            return None


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal based on the processed DataFrame, mandatory bottom-fishing conditions,
        and a scoring system for optional conditions.
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] توليد إشارة شراء (صيد القيعان)...")

        # Check DataFrame and columns
        if df_processed is None or df_processed.empty or len(df_processed) < max(2, MACD_SLOW + 1): # Need at least 2 for diff, and enough for indicators
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ أو قصير جدًا، لا يمكن توليد الإشارة.")
            return None
        # Add required columns for signal if not already present
        required_cols_for_signal = list(set(self.required_cols_buy_signal))
        missing_cols = [col for col in required_cols_for_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame يفتقد الأعمدة المطلوبة للإشارة: {missing_cols}.")
            return None

        # Check Bitcoin trend (still a mandatory filter)
        btc_trend = get_btc_trend_4h()
        # Only allow signal if Bitcoin trend is bullish, neutral, or unknown (not bearish)
        if "هبوط" in btc_trend: # Downtrend
            logger.info(f"ℹ️ [Strategy {self.symbol}] التداول متوقف بسبب اتجاه البيتكوين الهبوطي ({btc_trend}).")
            return None
        # Do not reject if "N/A" or "استقرار" (Sideways) or "تذبذب" (Volatile)
        elif "N/A" in btc_trend:
             logger.warning(f"⚠️ [Strategy {self.symbol}] لا يمكن تحديد اتجاه البيتكوين، سيتم تجاهل هذا الشرط.")


        # Extract latest and previous candle data
        last_row = df_processed.iloc[-1]
        prev_row = df_processed.iloc[-2] if len(df_processed) >= 2 else pd.Series() # Handle case with only one row

        # Check for NaN in essential columns required for the signal (only last row needs full check)
        last_row_check = last_row[required_cols_for_signal]
        if last_row_check.isnull().any():
            nan_cols = last_row_check[last_row_check.isnull()].index.tolist()
            logger.warning(f"⚠️ [Strategy {self.symbol}] الصف الأخير يحتوي على قيم NaN في أعمدة الإشارة المطلوبة: {nan_cols}. لا يمكن توليد الإشارة.")
            return None
        # Check previous values needed for conditions
        if len(df_processed) < 2:
             logger.warning(f"⚠️ [Strategy {self.symbol}] بيانات غير كافية (أقل من شمعتين) للتحقق من الشروط التي تتطلب البيانات السابقة.")
             # Some conditions below might fail due to missing prev_row, they will contribute 0 points.

        # =====================================================================
        # --- Check Mandatory Bottom-Fishing Conditions First ---
        # If any mandatory condition fails, the signal is rejected immediately
        # =====================================================================
        essential_passed = True
        failed_essential_conditions = []
        signal_details = {} # To store details of checked conditions (mandatory and optional)

        # Mandatory Condition 1: RSI Oversold or Bouncing Up from Oversold
        # Check if RSI is currently oversold OR (was oversold and is now higher)
        is_oversold = last_row['rsi'] <= RSI_OVERSOLD
        bounced_from_oversold = (len(df_processed) >= 2 and
                                   pd.notna(prev_row.get('rsi')) and
                                   prev_row['rsi'] <= RSI_OVERSOLD and
                                   last_row['rsi'] > prev_row['rsi'])

        if not (is_oversold or bounced_from_oversold):
            essential_passed = False
            failed_essential_conditions.append('RSI Oversold or Bouncing')
            signal_details['RSI_Mandatory'] = f'فشل: RSI={last_row["rsi"]:.1f} (ليس في منطقة البيع المفرط أو لم يرتد)'
        else:
             signal_details['RSI_Mandatory'] = f'نجاح: RSI={last_row["rsi"]:.1f} (في منطقة البيع المفرط أو يرتد)'


        # Mandatory Condition 2: Bullish Reversal Candlestick Pattern
        if last_row['BullishCandleSignal'] != 1:
            essential_passed = False
            failed_essential_conditions.append('Bullish Reversal Candle')
            signal_details['Candle_Mandatory'] = 'فشل: لا يوجد نموذج شموع انعكاسي صعودي'
        else:
             signal_details['Candle_Mandatory'] = 'نجاح: يوجد نموذج شموع انعكاسي صعودي'

        # Mandatory Condition 3: Simple check for recent price drop (e.g., price is below EMA20 from a few bars ago)
        # This is a simplified way to check if the price was higher recently
        if len(df_processed) < EMA_LONG_PERIOD: # Need enough data for this check
             logger.warning(f"⚠️ [Strategy {self.symbol}] بيانات غير كافية للشمعات السابقة للتحقق من انخفاض السعر.")
             # This mandatory condition will be considered failed if we don't have enough history
             essential_passed = False
             failed_essential_conditions.append('Recent Price Drop (Insufficient Data)')
             signal_details['Recent_Drop_Mandatory'] = 'فشل: بيانات غير كافية للتحقق من انخفاض السعر الأخير'

        else:
            # Check if the current price is significantly lower than the price 'n' bars ago (e.g., EMA_LONG_PERIOD bars ago)
            price_n_bars_ago = df_processed['close'].iloc[-EMA_LONG_PERIOD]
            price_drop_threshold = price_n_bars_ago * 0.97 # Example: at least a 3% drop from n bars ago

            if not (last_row['close'] < price_drop_threshold):
                 essential_passed = False
                 failed_essential_conditions.append('Recent Price Drop')
                 signal_details['Recent_Drop_Mandatory'] = f'فشل: السعر الحالي ({last_row["close"]:.4f}) ليس أقل بكثير من سعر {EMA_LONG_PERIOD} شمعة سابقة ({price_n_bars_ago:.4f})'
            else:
                 signal_details['Recent_Drop_Mandatory'] = f'نجاح: السعر الحالي ({last_row["close"]:.4f}) أقل بشكل ملحوظ من سعر {EMA_LONG_PERIOD} شمعة سابقة ({price_n_bars_ago:.4f})'


        # If any mandatory condition failed, reject the signal immediately
        if not essential_passed:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] فشلت الشروط الإلزامية: {', '.join(failed_essential_conditions)}. تم رفض الإشارة.")
            signal_details['Mandatory_Status'] = 'فشل'
            signal_details['Failed_Mandatory'] = failed_essential_conditions
            return None
        else:
             signal_details['Mandatory_Status'] = 'نجاح'

        # =====================================================================
        # --- Calculate Score for Optional Conditions (if mandatory passed) ---
        # =====================================================================
        current_score = 0.0

        # Optional Condition 1: RSI bouncing up from oversold (already checked in mandatory, but add score if applicable)
        if bounced_from_oversold:
             current_score += self.condition_weights.get('rsi_bouncing_up', 0)
             signal_details['RSI_Bounce_Score'] = f'يرتد من البيع المفرط (+{self.condition_weights.get("rsi_bouncing_up", 0)})'
        else:
             signal_details['RSI_Bounce_Score'] = f'لا يرتد من البيع المفرط (0)'


        # Optional Condition 2: Price near or below lower Bollinger Band
        if pd.notna(last_row.get('bb_lower')) and last_row['close'] <= last_row['bb_lower'] * 1.005: # Within 0.5% of lower band or below
             current_score += self.condition_weights.get('price_near_bb_lower', 0)
             signal_details['BB_Lower_Score'] = f'السعر قريب أو تحت الحد السفلي لبولينجر (+{self.condition_weights.get("price_near_bb_lower", 0)})'
        else:
             signal_details['BB_Lower_Score'] = f'السعر ليس قريبا من الحد السفلي لبولينجر (0)'


        # Optional Condition 3: MACD bullish momentum shift (hist turning positive or bullish cross from below zero)
        if len(df_processed) >= 2 and pd.notna(prev_row.get('macd_hist')) and pd.notna(last_row.get('macd_hist')) and pd.notna(prev_row.get('macd')) and pd.notna(last_row.get('macd_signal')):
            macd_hist_turning_up = last_row['macd_hist'] > prev_row['macd_hist']
            macd_cross_from_below_zero = (last_row['macd'] > last_row['macd_signal'] and
                                            prev_row['macd'] <= prev_row['macd_signal'] and
                                            last_row['macd'] < 0) # Added condition that MACD is still below zero or just crossed

            if macd_hist_turning_up or macd_cross_from_below_zero:
                 current_score += self.condition_weights.get('macd_bullish_momentum_shift', 0)
                 detail_macd_score = f'MACD Hist يتحول للإيجابية' if macd_hist_turning_up else ''
                 detail_macd_score += ' و ' if detail_macd_score and macd_cross_from_below_zero else ''
                 detail_macd_score += f'تقاطع صعودي تحت الصفر' if macd_cross_from_below_zero else ''
                 signal_details['MACD_Score'] = f'تحول زخم MACD الصعودي (+{self.condition_weights.get("macd_bullish_momentum_shift", 0)}) ({detail_macd_score})'
            else:
                 signal_details['MACD_Score'] = f'لا يوجد تحول زخم MACD صعودي (0)'
        else:
             signal_details['MACD_Score'] = f'بيانات MACD غير كافية أو NaN (0)'


        # Optional Condition 4: Price crosses above VWMA
        if len(df_processed) >= 2 and pd.notna(last_row.get('close')) and pd.notna(last_row.get('vwma')) and pd.notna(prev_row.get('close')) and pd.notna(prev_row.get('vwma')):
            if last_row['close'] > last_row['vwma'] and prev_row['close'] <= prev_row['vwma']:
                 current_score += self.condition_weights.get('price_crossing_vwma_up', 0)
                 signal_details['VWMA_Cross_Score'] = f'السعر يتقاطع فوق VWMA (+{self.condition_weights.get("price_crossing_vwma_up", 0)})'
            else:
                 signal_details['VWMA_Cross_Score'] = f'السعر لم يتقاطع فوق VWMA (0)'
        else:
             signal_details['VWMA_Cross_Score'] = f'بيانات VWMA غير كافية أو NaN (0)'


        # Optional Condition 5: Low ADX and DI+ crossing above DI-
        if len(df_processed) >= 2 and pd.notna(last_row.get('adx')) and pd.notna(last_row.get('di_plus')) and pd.notna(last_row.get('di_minus')) and pd.notna(prev_row.get('di_plus')) and pd.notna(prev_row.get('di_minus')):
             if last_row['adx'] < 25 and last_row['di_plus'] > last_row['di_minus'] and prev_row['di_plus'] <= prev_row['di_minus']: # Slightly increased low ADX threshold
                 current_score += self.condition_weights.get('adx_low_and_di_cross', 0)
                 signal_details['ADX_DI_Score'] = f'ADX منخفض وتقاطع DI+ (+{self.condition_weights.get("adx_low_and_di_cross", 0)})'
             else:
                 signal_details['ADX_DI_Score'] = f'ADX ليس منخفضا أو لا يوجد تقاطع DI+ (0)'
        else:
             signal_details['ADX_DI_Score'] = f'بيانات ADX/DI غير كافية أو NaN (0)'


        # Optional Condition 6: Price crosses above EMA10
        if len(df_processed) >= 2 and pd.notna(last_row.get('close')) and pd.notna(last_row.get('ema_10')) and pd.notna(prev_row.get('close')) and pd.notna(prev_row.get('ema_10')):
            if last_row['close'] > last_row['ema_10'] and prev_row['close'] <= prev_row['ema_10']:
                 current_score += self.condition_weights.get('price_crossing_ema10_up', 0)
                 signal_details['EMA10_Cross_Score'] = f'السعر يتقاطع فوق EMA10 (+{self.condition_weights.get("price_crossing_ema10_up", 0)})'
            else:
                 signal_details['EMA10_Cross_Score'] = f'السعر لم يتقاطع فوق EMA10 (0)'
        else:
             signal_details['EMA10_Cross_Score'] = f'بيانات EMA10 غير كافية أو NaN (0)'

        # Optional Condition 7: OBV is rising
        if len(df_processed) >= 2 and pd.notna(last_row.get('obv')) and pd.notna(prev_row.get('obv')):
            if last_row['obv'] > prev_row['obv']:
                 current_score += self.condition_weights.get('obv_rising', 0)
                 signal_details['OBV_Score'] = f'OBV يرتفع (+{self.condition_weights.get("obv_rising", 0)})'
            else:
                 signal_details['OBV_Score'] = f'OBV لا يرتفع (0)'
        else:
             signal_details['OBV_Score'] = f'بيانات OBV غير كافية أو NaN (0)'

        # ------------------------------------------

        # Final buy decision based on the score of optional conditions
        if current_score < self.min_signal_score:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] نقاط الإشارة المطلوبة من الشروط الاختيارية لم يتم تحقيقها (النقاط: {current_score:.2f} / {self.total_possible_score:.2f}, الحد الأدنى: {self.min_signal_score:.2f}). تم رفض الإشارة.")
            signal_details['Optional_Score_Status'] = 'فشل'
            signal_details['Calculated_Score'] = float(f"{current_score:.2f}")
            signal_details['Min_Required_Score'] = float(f"{self.min_signal_score:.2f}")
            return None
        else:
             signal_details['Optional_Score_Status'] = 'نجاح'
             signal_details['Calculated_Score'] = float(f"{current_score:.2f}")
             signal_details['Min_Required_Score'] = float(f"{self.min_signal_score:.2f}")


        # Check trading volume (liquidity) - still a mandatory filter
        volume_recent = fetch_recent_volume(self.symbol)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى المطلوب ({MIN_VOLUME_15M_USDT:,.0f} USDT). تم رفض الإشارة.")
            signal_details['Liquidity_Status'] = 'فشل'
            signal_details['Volume_15m'] = volume_recent
            signal_details['Min_Volume_15m'] = MIN_VOLUME_15M_USDT
            return None
        else:
             signal_details['Liquidity_Status'] = 'نجاح'
             signal_details['Volume_15m'] = volume_recent
             signal_details['Min_Volume_15m'] = MIN_VOLUME_15M_USDT


        # Calculate initial target and stop loss based on ATR
        current_price = last_row['close']
        current_atr = last_row.get('atr')

        # Ensure ATR is not NaN before using it
        if pd.isna(current_atr) or current_atr <= 0:
             logger.warning(f"⚠️ [Strategy {self.symbol}] قيمة ATR غير صالحة ({current_atr}) لحساب الهدف ووقف الخسارة.")
             signal_details['ATR_Status'] = 'فشل'
             return None
        else:
             signal_details['ATR_Status'] = 'نجاح'
             signal_details['Current_ATR'] = float(f"{current_atr:.8g}")


        # These multipliers can be adjusted based on ADX or other factors for a more dynamic strategy if desired
        target_multiplier = ENTRY_ATR_MULTIPLIER
        stop_loss_multiplier = ENTRY_ATR_MULTIPLIER

        initial_target = current_price + (target_multiplier * current_atr)
        initial_stop_loss = current_price - (stop_loss_multiplier * current_atr)

        # Ensure stop loss is not zero or negative and is below the entry price
        if initial_stop_loss <= 0 or initial_stop_loss >= current_price:
            # Use a percentage as a minimum stop loss if the initial calculation is invalid
            min_sl_price_pct = current_price * (1 - 0.015) # Example: 1.5% below entry
            initial_stop_loss = max(min_sl_price_pct, current_price * 0.001) # Ensure it's not too close to zero
            logger.warning(f"⚠️ [Strategy {self.symbol}] وقف الخسارة المحسوب ({initial_stop_loss:.8g}) غير صالح أو أعلى من سعر الدخول. تم تعديله إلى {initial_stop_loss:.8f}")
            signal_details['Warning'] = f'تم تعديل وقف الخسارة الأولي (كان <= 0 أو >= الدخول، تم تعيينه إلى {initial_stop_loss:.8f})'
        else:
             # Ensure the initial stop loss is not too wide (optional)
             max_allowed_loss_pct = 0.10 # Example: Initial loss should not exceed 10%
             max_sl_price = current_price * (1 - max_allowed_loss_pct)
             if initial_stop_loss < max_sl_price:
                  logger.warning(f"⚠️ [Strategy {self.symbol}] وقف الخسارة المحسوب ({initial_stop_loss:.8g}) واسع جدًا. تم تعديله إلى {max_sl_price:.8f}")
                  initial_stop_loss = max_sl_price
                  signal_details['Warning'] = f'تم تعديل وقف الخسارة الأولي (كان واسعًا جدًا، تم تعيينه إلى {initial_stop_loss:.8f})' # Use the new value here


        # Check minimum profit margin (after calculating final target and stop loss) - still a mandatory filter
        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] هامش الربح ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%). تم رفض الإشارة.")
            signal_details['Profit_Margin_Status'] = 'فشل'
            signal_details['Profit_Margin_Pct'] = float(f"{profit_margin_pct:.2f}")
            signal_details['Min_Profit_Margin_Pct'] = MIN_PROFIT_MARGIN_PCT
            return None
        else:
             signal_details['Profit_Margin_Status'] = 'نجاح'
             signal_details['Profit_Margin_Pct'] = float(f"{profit_margin_pct:.2f}")
             signal_details['Min_Profit_Margin_Pct'] = MIN_PROFIT_MARGIN_PCT


        # Compile final signal data
        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'initial_stop_loss': float(f"{initial_stop_loss:.8g}"),
            'current_target': float(f"{initial_target:.8g}"),
            'current_stop_loss': float(f"{initial_stop_loss:.8g}"),
            'r2_score': float(f"{current_score:.2f}"), # Weighted score of optional conditions
            'strategy_name': 'Bottom_Fishing_Filtered', # اسم الاستراتيجية الجديد
            'signal_details': signal_details, # Now contains details of mandatory and optional conditions
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': float(f"{self.total_possible_score:.2f}") # Total points for optional conditions
        }

        logger.info(f"✅ [Strategy {self.symbol}] تم تأكيد إشارة الشراء (صيد القيعان). السعر: {current_price:.6f}, النقاط (اختيارية): {current_score:.2f}/{self.total_possible_score:.2f}, ATR: {current_atr:.6f}, السيولة: {volume_recent:,.0f}")
        return signal_output


# ---------------------- Telegram Functions (Adjusted message format) ----------------------
# Keep send_telegram_message as it is.

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends a new trading signal alert (Bottom Fishing) to Telegram in Arabic."""
    logger.debug(f"ℹ️ [Telegram Alert] تنسيق وإرسال تنبيه الإشارة: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['initial_stop_loss'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A').replace('_', ' ').title() # تنسيق اسم الاستراتيجية
        signal_score = signal_data.get('r2_score', 0.0) # Weighted score for optional conditions
        total_possible_score = signal_data.get('total_possible_score', 10.0) # Total points for optional conditions
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        signal_details = signal_data.get('signal_details', {}) # تفاصيل الشروط

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        loss_pct = ((stop_loss_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        profit_usdt = trade_value_signal * (profit_pct / 100)
        loss_usdt = abs(trade_value_signal * (loss_pct / 100))

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Escape special characters for Markdown
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        btc_trend = get_btc_trend_4h()

        # Build the message in Arabic with weighted score and condition details
        message = (
            f"💡 *إشارة تداول جديدة ({strategy_name})* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (صيد القيعان)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            # --- إضافة عرض النقاط ---
            f"📊 **قوة الإشارة (نقاط الشروط الاختيارية):** *{signal_score:.1f} / {total_possible_score:.1f}*\n"
            f"💧 **السيولة (15 دقيقة):** {volume_15m:,.0f} USDT\n"
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🎯 **الهدف الأولي:** `${target_price:,.8g}` ({profit_pct:+.2f}% / ≈ ${profit_usdt:+.2f})\n"
            f"🛑 **وقف الخسارة الأولي:** `${stop_loss_price:,.8g}` ({loss_pct:.2f}% / ≈ ${loss_usdt:.2f})\n"
            f"——————————————\n"
            f"✅ *الشروط الإلزامية المحققة:*\n"
            f"  - RSI: {signal_details.get('RSI_Mandatory', 'N/A')}\n"
            f"  - نموذج الشموع: {signal_details.get('Candle_Mandatory', 'N/A')}\n"
            f"  - انخفاض أخير في السعر: {signal_details.get('Recent_Drop_Mandatory', 'N/A')}\n"
            f"——————————————\n"
            f"⭐ *نقاط الشروط الاختيارية:*\n" # قسم جديد لتفاصيل النقاط الاختيارية
            f"  - ارتداد RSI: {signal_details.get('RSI_Bounce_Score', 'N/A')}\n"
            f"  - السعر قرب BB السفلي: {signal_details.get('BB_Lower_Score', 'N/A')}\n"
            f"  - تحول زخم MACD: {signal_details.get('MACD_Score', 'N/A')}\n"
            f"  - تقاطع السعر فوق VWMA: {signal_details.get('VWMA_Cross_Score', 'N/A')}\n"
            f"  - ADX منخفض وتقاطع DI: {signal_details.get('ADX_DI_Score', 'N/A')}\n"
            f"  - تقاطع السعر فوق EMA10: {signal_details.get('EMA10_Cross_Score', 'N/A')}\n"
            f"  - OBV يرتفع: {signal_details.get('OBV_Score', 'N/A')}\n"
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
        logger.error(f"❌ [Telegram Alert] بيانات الإشارة غير مكتملة للزوج {signal_data.get('symbol', 'N/A')}: مفتاح مفقود {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة للزوج {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

# Keep send_tracking_notification as it is. It seems general enough for different strategies.

# ... (Past in send_telegram_message and send_tracking_notification here) ...


# ---------------------- Database Functions (Insert and Update) (Keep existing) ----------------------
# Keep insert_signal_into_db as it is. It already stores strategy_name and r2_score.

# ... (Past in insert_signal_into_db here) ...


# ---------------------- Open Signal Tracking Function (Add max trade duration check) ----------------------
def track_signals() -> None:
    """Tracks open signals, checks targets and stop losses, applies trailing stop, and checks max trade duration."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة اتصال قاعدة البيانات.")
                time.sleep(15) # Wait a bit longer before retrying
                continue

            # Use a cursor with context manager to fetch open signals
            with conn.cursor() as track_cur: # Uses RealDictCursor
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_stop_loss, current_target, current_stop_loss,
                           is_trailing_active, last_trailing_update_price, sent_at
                    FROM signals
                    WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                # logger.debug("ℹ️ [Tracker] لا توجد إشارات مفتوحة لتتبعها.")
                time.sleep(10) # Wait less if no signals
                continue

            logger.debug(f"ℹ️ [Tracker] تتبع {len(open_signals)} إشارة مفتوحة...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False # To track if this signal was updated in the current cycle

                try:
                    # Extract and safely convert numeric data
                    entry_price = float(signal_row['entry_price'])
                    initial_stop_loss = float(signal_row['initial_stop_loss'])
                    current_target = float(signal_row['current_target'])
                    current_stop_loss = float(signal_row['current_stop_loss'])
                    is_trailing_active = signal_row['is_trailing_active']
                    last_update_px = signal_row['last_trailing_update_price']
                    last_trailing_update_price = float(last_update_px) if last_update_px is not None else None
                    sent_at = signal_row['sent_at'] # Get signal sent timestamp

                    # Get current price from WebSocket Ticker data
                    current_price = ticker_data.get(symbol)

                    if current_price is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): السعر الحالي غير متوفر في بيانات Ticker.")
                         continue # Skip this signal in this cycle

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f} SL={current_stop_loss:.4f} Trail={'On' if is_trailing_active else 'Off'}")

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id}

                    # --- Check for Max Trade Duration ---
                    if MAX_TRADE_DURATION_HOURS > 0:
                         trade_duration = datetime.now() - sent_at
                         if trade_duration > timedelta(hours=MAX_TRADE_DURATION_HOURS):
                              logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): تجاوز الحد الأقصى لمدة الصفقة ({MAX_TRADE_DURATION_HOURS} ساعات). إغلاق الصفقة.")
                              # Close the trade at the current price
                              closing_price = current_price
                              profit_pct = ((closing_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                              profitable_close = closing_price > entry_price

                              update_query = sql.SQL("UPDATE signals SET hit_stop_loss = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s, profitable_stop_loss = %s WHERE id = %s;")
                              update_params = (closing_price, profit_pct, profitable_close, signal_id) # Use hit_stop_loss flag for closure
                              log_message = f"⏳ [Tracker] {symbol}(ID:{signal_id}): تم الإغلاق بسبب تجاوز المدة القصوى ({trade_duration}). السعر: {closing_price:.8g} (النسبة: {profit_pct:.2f}%)."
                              notification_details.update({'type': 'stop_loss_hit', 'closing_price': closing_price, 'profit_pct': profit_pct, 'profitable_sl': profitable_close}) # Reuse stop_loss_hit type but include profitable status
                              update_executed = True
                              # If trade duration closes, skip other checks for this signal
                              if update_executed:
                                   try:
                                        with conn.cursor() as update_cur:
                                            update_cur.execute(update_query, update_params)
                                        conn.commit()
                                        if log_message: logger.info(log_message)
                                        if notification_details.get('type'):
                                           send_tracking_notification(notification_details)
                                   except psycopg2.Error as db_err:
                                       logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ قاعدة بيانات أثناء تحديث تجاوز المدة القصوى: {db_err}")
                                       if conn: conn.rollback()
                                   except Exception as exec_err:
                                       logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تنفيذ التحديث/الإشعار لتجاوز المدة: {exec_err}", exc_info=True)
                                       if conn: conn.rollback()
                                   continue # Move to the next signal after closing

                    # --- Check and Update Logic (Only if not closed by max duration) ---
                    if not update_executed:
                        # 1. Check for Target Hit
                        if current_price >= current_target:
                            profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                            update_query = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;")
                            update_params = (current_target, profit_pct, signal_id)
                            log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): تم الوصول إلى الهدف عند {current_target:.8g} (الربح: {profit_pct:+.2f}%)."
                            notification_details.update({'type': 'target_hit', 'closing_price': current_target, 'profit_pct': profit_pct})
                            update_executed = True

                        # 2. Check for Stop Loss Hit (Must be after Target check)
                        elif current_price <= current_stop_loss:
                            loss_pct = ((current_stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
                            profitable_sl = current_stop_loss > entry_price
                            sl_type_msg = "بربح ✅" if profitable_sl else "بخسارة ❌"
                            update_query = sql.SQL("UPDATE signals SET hit_stop_loss = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s, profitable_stop_loss = %s WHERE id = %s;")
                            update_params = (current_stop_loss, loss_pct, profitable_sl, signal_id)
                            log_message = f"🔻 [Tracker] {symbol}(ID:{signal_id}): تم ضرب وقف الخسارة ({sl_type_msg}) عند {current_stop_loss:.8g} (النسبة: {loss_pct:.2f}%)."
                            notification_details.update({'type': 'stop_loss_hit', 'closing_price': current_stop_loss, 'profit_pct': loss_pct, 'profitable_sl': profitable_sl}) # Pass the profitable_sl flag
                            update_executed = True

                        # 3. Check for Trailing Stop Activation or Update (Only if Target or SL not hit)
                        else:
                            activation_threshold_price = entry_price * (1 + TRAILING_STOP_ACTIVATION_PROFIT_PCT)
                            # a. Activate Trailing Stop
                            if not is_trailing_active and current_price >= activation_threshold_price:
                                logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر {current_price:.8g} وصل إلى عتبة تفعيل الوقف المتحرك ({activation_threshold_price:.8g}). جلب ATR...")
                                # Use the specified tracking timeframe
                                df_atr = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                                if df_atr is not None and not df_atr.empty:
                                    # Use the ATR period designated for entry/tracking
                                    df_atr = calculate_atr_indicator(df_atr, period=ENTRY_ATR_PERIOD)
                                    if not df_atr.empty and 'atr' in df_atr.columns and pd.notna(df_atr['atr'].iloc[-1]):
                                        current_atr_val = df_atr['atr'].iloc[-1]
                                        if current_atr_val > 0:
                                             # Calculate new stop loss based on current price and ATR
                                             new_stop_loss_calc = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val)
                                             # The new stop loss must be HIGHER than the current stop loss AND higher than the entry price (to lock in profit)
                                             new_stop_loss = max(new_stop_loss_calc, current_stop_loss, entry_price * (1 + 0.0001)) # Ensure at least a tiny profit

                                             if new_stop_loss > current_stop_loss: # Only if the new stop is actually higher than the *previous* stop
                                                update_query = sql.SQL("UPDATE signals SET is_trailing_active = TRUE, current_stop_loss = %s, last_trailing_update_price = %s WHERE id = %s;")
                                                update_params = (new_stop_loss, current_price, signal_id)
                                                log_message = f"⬆️✅ [Tracker] {symbol}(ID:{signal_id}): تم تفعيل الوقف المتحرك. السعر={current_price:.8g}, ATR={current_atr_val:.8g}. الوقف الجديد: {new_stop_loss:.8g}"
                                                notification_details.update({'type': 'trailing_activated', 'current_price': current_price, 'atr_value': current_atr_val, 'new_stop_loss': new_stop_loss, 'activation_profit_pct': TRAILING_STOP_ACTIVATION_PROFIT_PCT * 100})
                                                update_executed = True
                                             else:
                                                logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الوقف المتحرك المحسوب ({new_stop_loss:.8g}) ليس أعلى من الوقف الحالي ({current_stop_loss:.8g}). لن يتم التفعيل.")
                                    else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): قيمة ATR غير صالحة ({current_atr_val}) لتفعيل الوقف المتحرك.")
                                else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب ATR لتفعيل الوقف المتحرك.")
                            # b. Update Trailing Stop (Only if trailing is already active)
                            elif is_trailing_active and last_trailing_update_price is not None:
                                update_threshold_price = last_trailing_update_price * (1 + TRAILING_STOP_MOVE_INCREMENT_PCT)
                                if current_price >= update_threshold_price: # Check if price has increased enough since last update
                                    logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر {current_price:.8g} وصل إلى عتبة تحديث الوقف المتحرك ({update_threshold_price:.8g}). جلب ATR...")
                                    df_recent = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                                    if df_recent is not None and not df_recent.empty:
                                        df_recent = calculate_atr_indicator(df_recent, period=ENTRY_ATR_PERIOD)
                                        if not df_recent.empty and 'atr' in df_recent.columns and pd.notna(df_recent['atr'].iloc[-1]):
                                             current_atr_val_update = df_recent['atr'].iloc[-1]
                                             if current_atr_val_update > 0:
                                                 # Calculate the new potential stop loss
                                                 potential_new_stop_loss = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val_update)
                                                 # The new stop loss must be higher than the current stop loss
                                                 if potential_new_stop_loss > current_stop_loss:
                                                    new_stop_loss_update = potential_new_stop_loss
                                                    update_query = sql.SQL("UPDATE signals SET current_stop_loss = %s, last_trailing_update_price = %s WHERE id = %s;")
                                                    update_params = (new_stop_loss_update, current_price, signal_id)
                                                    log_message = f"➡️🔼 [Tracker] {symbol}(ID:{signal_id}): تم تحديث الوقف المتحرك. السعر={current_price:.8g}, ATR={current_atr_val_update:.8g}. السابق={current_stop_loss:.8g}, الجديد: {new_stop_loss_update:.8g}"
                                                    notification_details.update({'type': 'trailing_updated', 'current_price': current_price, 'atr_value': current_atr_val_update, 'old_stop_loss': current_stop_loss, 'new_stop_loss': new_stop_loss_update, 'trigger_price_increase_pct': TRAILING_STOP_MOVE_INCREMENT_PCT * 100})
                                                    update_executed = True
                                                 else:
                                                     logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الوقف المتحرك المحسوب ({potential_new_stop_loss:.8g}) ليس أعلى من الوقف الحالي ({current_stop_loss:.8g}). لن يتم التحديث.")
                                         else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): قيمة ATR غير صالحة ({current_atr_val_update}) للتحديث.")
                                    else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب ATR للتحديث.")
                                else:
                                     logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر ({current_price:.8g}) لم يصل إلى عتبة تحديث الوقف المتحرك ({update_threshold_price:.8g}) منذ آخر تحديث عند ({last_trailing_update_price:.8g}).")


                    # --- Execute Database Update and Send Notification ---
                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ قاعدة بيانات أثناء التحديث: {db_err}")
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
                logger.debug(f"ℹ️ [Tracker] حالة نهاية الدورة ({processed_in_cycle} معالجة): {'; '.join(active_signals_summary)}")

            time.sleep(3) # Wait between tracking cycles (real-time tracking needs short intervals)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] خطأ قاعدة بيانات في دورة التتبع الرئيسية: {db_cycle_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(30) # Wait longer on DB error
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارات: {cycle_err}", exc_info=True)
            time.sleep(30) # Wait longer on unexpected error


# ---------------------- Flask Service (Optional for Webhook) (Keep existing) ----------------------
# Keep app, home, favicon, webhook, handle_status_command, run_flask as they are.

# ... (Past in all the Flask related functions here) ...


# ---------------------- Main Loop and Check Function (Adjusted scan frequency) ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لم يتم تحميل أو التحقق من أي أزواج صالحة. لا يمكن المتابعة.")
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} زوجًا صالحًا للفحص.")
    # No need for last_full_scan_time if we use a fixed sleep time
    # last_full_scan_time = time.time()

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] بدء دورة فحص السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] تخطي دورة الفحص بسبب فشل اتصال قاعدة البيانات.")
                time.sleep(60)
                continue

            # 1. Check the current number of open signals
            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] خطأ قاعدة بيانات أثناء التحقق من عدد الإشارات المفتوحة: {db_err}. تخطي الدورة.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] الإشارات المفتوحة حالياً: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول إلى الحد الأقصى لعدد الإشارات المفتوحة. في انتظار...")
                time.sleep(30) # Wait 30 seconds before re-checking open count
                continue

            # 2. Iterate through the list of symbols and scan them
            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] تم الوصول إلى الحد الأقصى ({MAX_OPEN_TRADES}) أثناء الفحص. إيقاف فحص الأزواج لهذه الدورة.")
                      break

                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] فحص {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")

                 try:
                    # a. Check if there is already an open signal for this symbol
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE AND hit_stop_loss = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            logger.debug(f"ℹ️ [Main] توجد إشارة مفتوحة بالفعل للزوج {symbol}. تخطي.")
                            continue

                    # b. Fetch historical data (using SIGNAL_GENERATION_TIMEFRAME)
                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        logger.debug(f"ℹ️ [Main] لا تتوفر بيانات تاريخية كافية للزوج {symbol}.")
                        continue

                    # c. Apply the strategy and generate signal
                    # Use the new BottomFishingStrategy
                    strategy = BottomFishingStrategy(symbol)
                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        logger.debug(f"ℹ️ [Main] فشل حساب المؤشرات للزوج {symbol}.")
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    # d. Insert signal and send alert
                    if potential_signal:
                        logger.info(f"✨ [Main] تم العثور على إشارة محتملة للزوج {symbol}! (النقاط: {potential_signal.get('r2_score', 0):.2f}) التحقق النهائي والإدراج...")
                        # Re-check open count just before inserting to avoid exceeding the limit due to concurrent signals
                        with conn.cursor() as final_check_cur:
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     # Add a small delay after generating a signal to avoid rapid-fire signals
                                     time.sleep(5)
                                 else:
                                     logger.error(f"❌ [Main] فشل في إدراج إشارة الزوج {symbol} في قاعدة البيانات.")
                             else:
                                 logger.warning(f"⚠️ [Main] تم الوصول إلى الحد الأقصى ({final_open_count}) قبل إدراج إشارة الزوج {symbol}. تم تجاهل الإشارة.")
                                 # Break out of the symbol loop if the limit is reached
                                 break

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] خطأ قاعدة بيانات أثناء معالجة الزوج {symbol}: {db_loop_err}. الانتقال إلى الزوج التالي...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] خطأ عام أثناء معالجة الزوج {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 # Small delay between processing symbols to reduce load
                 time.sleep(0.1)

            # 3. Wait before starting the next cycle
            scan_duration = time.time() - scan_start_time
            # Adjust wait time to achieve a cycle of approximately 30 seconds
            # Ensure the wait time is not negative if scan_duration is long
            wait_time = max(0, 30 - scan_duration) # Target 30 seconds total cycle duration

            logger.info(f"🏁 [Main] انتهت دورة الفحص. الإشارات التي تم توليدها: {signals_generated_in_loop}. مدة الفحص: {scan_duration:.2f} ثانية.")
            logger.info(f"⏳ [Main] انتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] تم طلب الإيقاف (KeyboardInterrupt). إغلاق...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] خطأ قاعدة بيانات قاتل في الدورة الرئيسية: {db_main_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60) # Wait longer on fatal DB error
             try:
                 init_db()
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] فشل إعادة الاتصال بقاعدة البيانات: {recon_err}. الخروج...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الدورة الرئيسية: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 60 ثانية قبل إعادة المحاولة...") # Reduce wait time on general error
            time.sleep(60)

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] إغلاق الموارد...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ أثناء إغلاق اتصال قاعدة البيانات: {close_err}")
    logger.info("✅ [Cleanup] اكتمل تنظيف الموارد.")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل بوت إشارات التداول...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | وقت UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize Threads to be available as global variables
    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None

    try:
        # 1. Initialize the database first
        init_db()

        # 2. Start WebSocket Ticker
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء تشغيل مؤشر WebSocket.")
        logger.info("ℹ️ [Main] انتظار 5 ثوانٍ لتهيئة WebSocket...")
        time.sleep(5) # Give WebSocket a moment to connect and receive initial data
        if not ticker_data:
             logger.warning("⚠️ [Main] لم يتم تلقي بيانات أولية من WebSocket بعد 5 ثوانٍ.")
        else:
             logger.info(f"✅ [Main] تم تلقي بيانات أولية من WebSocket لـ {len(ticker_data)} زوجًا.")


        # 3. Start Signal Tracker
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء تشغيل متتبع الإشارات.")

        # 4. Start Flask Server (if Webhook configured)
        if WEBHOOK_URL:
            flask_thread = Thread(target=run_flask, daemon=True, name="FlaskThread")
            flask_thread.start()
            logger.info("✅ [Main] تم بدء تشغيل ثريد Flask Webhook.")
        else:
             logger.info("ℹ️ [Main] لم يتم تكوين Webhook URL، لن يتم بدء تشغيل خادم Flask.")

        # 5. Start the main loop
        main_loop()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل أو في الدورة الرئيسية: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] يتم إيقاف تشغيل البرنامج...")
        # send_telegram_message(CHAT_ID, "⚠️ Alert: Trading bot is shutting down now.") # Uncomment to send alert on shutdown
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف تشغيل بوت إشارات التداول.")
        os._exit(0)
