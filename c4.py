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

# \---------------------- إعداد التسجيل ----------------------

logging.basicConfig(
level=logging.INFO, \# يمكن تغيير هذا إلى logging.DEBUG للحصول على سجلات أكثر تفصيلاً
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \# إضافة اسم المسجل
handlers=[
logging.FileHandler('crypto\_bot\_elliott\_fib.log', encoding='utf-8'),
logging.StreamHandler()
]
)

# استخدام اسم محدد للمسجل بدلاً من الجذر

logger = logging.getLogger('CryptoBot')

# \---------------------- تحميل المتغيرات البيئية ----------------------

try:
API\_KEY: str = config('BINANCE\_API\_KEY')
API\_SECRET: str = config('BINANCE\_API\_SECRET')
TELEGRAM\_TOKEN: str = config('TELEGRAM\_BOT\_TOKEN')
CHAT\_ID: str = config('TELEGRAM\_CHAT\_ID')
DB\_URL: str = config('DATABASE\_URL')
\# استخدام قيمة افتراضية None إذا لم يكن المتغير موجودًا
WEBHOOK\_URL: Optional[str] = config('WEBHOOK\_URL', default=None)
except Exception as e:
logger.critical(f"❌ فشل تحميل متغيرات البيئة الأساسية: {e}")
exit(1) \# استخدام رمز خروج غير صفري للإشارة إلى خطأ

logger.info(f"Binance API Key: {'متوفر' if API\_KEY else 'غير متوفر'}")
logger.info(f"Telegram Token: {TELEGRAM\_TOKEN[:10]}...{'\*' \* (len(TELEGRAM\_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT\_ID}")
logger.info(f"Database URL: {'متوفر' if DB\_URL else 'غير متوفر'}")
logger.info(f"Webhook URL: {WEBHOOK\_URL if WEBHOOK\_URL else 'غير محدد'}")

# \---------------------- إعداد الثوابت والمتغيرات العامة (معدلة للفحص على إطار 15 دقيقة) ----------------------

TRADE\_VALUE: float = 10.0         \# Default trade value in USDT (Keep small for testing)
MAX\_OPEN\_TRADES: int = 5          \# Maximum number of open trades simultaneously (Increased slightly for scalping)
SIGNAL\_GENERATION\_TIMEFRAME: str = '15m' \# Timeframe for signal generation (Changed to 15m)
SIGNAL\_GENERATION\_LOOKBACK\_DAYS: int = 7 \# Increased historical data lookback for 15m timeframe
SIGNAL\_TRACKING\_TIMEFRAME: str = '15m' \# Timeframe for signal tracking and target updates (Changed to 15m)
SIGNAL\_TRACKING\_LOOKBACK\_DAYS: int = 3   \# Increased historical data lookback in days for signal tracking

# \--- New Constants for Multi-Timeframe Confirmation ---

CONFIRMATION\_TIMEFRAME: str = '30m' \# Larger timeframe for trend confirmation (Changed to 30m)
CONFIRMATION\_LOOKBACK\_DAYS: int = 14 \# Historical data lookback for confirmation timeframe (Increased for 30m)

# \--- Parameters for Improved Entry Point ---

ENTRY\_POINT\_EMA\_PROXIMITY\_PCT: float = 0.002 \# Price must be within this % of signal timeframe EMA\_SHORT (Increased tolerance slightly)
ENTRY\_POINT\_RECENT\_CANDLE\_LOOKBACK: int = 2 \# Look back this many candles on signal timeframe for bullish sign (Reduced lookback)

# \=============================================================================

# \--- Indicator Parameters (Adjusted for 15m Signal and 30m Confirmation) ---

# \=============================================================================

RSI\_PERIOD: int = 14 \# Standard RSI period
RSI\_OVERSOLD: int = 30
RSI\_OVERBOUGHT: int = 70
EMA\_SHORT\_PERIOD: int = 13 \# Adjusted for 15m
EMA\_LONG\_PERIOD: int = 34 \# Adjusted for 15m
VWMA\_PERIOD: int = 21 \# Adjusted for 15m
SWING\_ORDER: int = 3 \# Not used in current strategy logic
FIB\_LEVELS\_TO\_CHECK: List[float] = [0.382, 0.5, 0.618] \# Not used in current strategy logic
FIB\_TOLERANCE: float = 0.005 \# Not used in current strategy logic
LOOKBACK\_FOR\_SWINGS: int = 50 \# Not used in current strategy logic
ENTRY\_ATR\_PERIOD: int = 14 \# Adjusted for 15m
ENTRY\_ATR\_MULTIPLIER: float = 1.75 \# ATR Multiplier for initial target (Adjusted slightly)
BOLLINGER\_WINDOW: int = 20 \# Standard Bollinger period
BOLLINGER\_STD\_DEV: int = 2 \# Standard Bollinger std dev
MACD\_FAST: int = 12 \# Standard MACD fast period
MACD\_SLOW: int = 26 \# Standard MACD slow period
MACD\_SIGNAL: int = 9 \# Standard MACD signal period
ADX\_PERIOD: int = 14 \# Standard ADX period
SUPERTREND\_PERIOD: int = 10 \# Standard Supertrend period
SUPERTREND\_MULTIPLIER: float = 3.0 \# Adjusted Supertrend multiplier slightly

# \--- Parameters for Dynamic Target Update ---

DYNAMIC\_TARGET\_APPROACH\_PCT: float = 0.003 \# Percentage proximity to target to trigger re-evaluation (e.g., 0.3%) (Increased slightly)
DYNAMIC\_TARGET\_EXTENSION\_ATR\_MULTIPLIER: float = 1.0 \# ATR multiplier for extending the target (Increased)
MAX\_DYNAMIC\_TARGET\_UPDATES: int = 3 \# Maximum number of times a target can be dynamically updated for a single signal (Increased)
MIN\_ADX\_FOR\_DYNAMIC\_UPDATE: int = 25 \# Minimum ADX value to consider dynamic target update (Increased slightly)

MIN\_PROFIT\_MARGIN\_PCT: float = 1.5 \# Increased minimum profit margin
MIN\_VOLUME\_15M\_USDT: float = 500000.0 \# Increased minimum volume check (using 15m data now)

RECENT\_EMA\_CROSS\_LOOKBACK: int = 3 \# Adjusted for 15m
MIN\_ADX\_TREND\_STRENGTH: int = 25 \# Increased minimum ADX trend strength for essential condition
MACD\_HIST\_INCREASE\_CANDLES: int = 2 \# Reduced lookback for MACD Hist increase
OBV\_INCREASE\_CANDLES: int = 2 \# Reduced lookback for OBV increase

# \=============================================================================

# Global variables

conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker\_data: Dict[str, float] = {}

# \---------------------- Binance Client Setup ----------------------

try:
logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
client = Client(API\_KEY, API\_SECRET)
client.ping()
server\_time = client.get\_server\_time()
logger.info(f"✅ [Binance] تم تهيئة عميل Binance بنجاح. وقت الخادم: {datetime.fromtimestamp(server\_time['serverTime']/1000)}")
except (BinanceRequestException, BinanceAPIException) as binance\_err:
logger.critical(f"❌ [Binance] خطأ في Binance API/الطلب: {binance\_err}")
exit(1)
except Exception as e:
logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}", exc\_info=True)
exit(1)

# \---------------------- Additional Indicator Functions ----------------------

def get\_fear\_greed\_index() -\> str:
classification\_translation\_ar = {
"Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد",
"Greed": "جشع", "Extreme Greed": "جشع شديد",
}
url = "https://www.google.com/search?q=https://api.alternative.me/fng/"
logger.debug(f"ℹ️ [Indicators] جلب مؤشر الخوف والجشع من {url}...")
try:
response = requests.get(url, timeout=10)
response.raise\_for\_status()
data = response.json()
value = int(data["data"][0]["value"])
classification\_en = data["data"][0]["value\_classification"]
classification\_ar = classification\_translation\_ar.get(classification\_en, classification\_en)
logger.debug(f"✅ [Indicators] مؤشر الخوف والجشع: {value} ({classification\_ar})")
return f"{value} ({classification\_ar})"
except requests.exceptions.RequestException as e:
logger.error(f"❌ [Indicators] خطأ في الشبكة أثناء جلب مؤشر الخوف والجشع: {e}")
return "N/A (خطأ في الشبكة)"
except Exception as e:
logger.error(f"❌ [Indicators] خطأ أثناء جلب مؤشر الخوف والجشع: {e}", exc\_info=True)
return "N/A (خطأ)"

def fetch\_historical\_data(symbol: str, interval: str, days: int) -\> Optional[pd.DataFrame]:
"""يجلب البيانات التاريخية لزوج معين وإطار زمني."""
if not client:
logger.error(f"❌ [Data] عميل Binance غير مهيأ لجلب البيانات لـ {symbol}.")
return None
try:
start\_dt = datetime.utcnow() - timedelta(days=days + 1)
start\_str = start\_dt.strftime("%Y-%m-%d %H:%M:%S")
logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} منذ {start\_str} (حد أقصى 1000 شمعة)...")
klines = client.get\_historical\_klines(symbol, interval, start\_str, limit=1000)
if not klines:
logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) لـ {symbol}.")
return None
df = pd.DataFrame(klines, columns=[
'timestamp', 'open', 'high', 'low', 'close', 'volume',
'close\_time', 'quote\_volume', 'trades', 'taker\_buy\_base', 'taker\_buy\_quote', 'ignore'
])
numeric\_cols = ['open', 'high', 'low', 'close', 'volume']
for col in numeric\_cols:
df[col] = pd.to\_numeric(df[col], errors='coerce')
df['timestamp'] = pd.to\_datetime(df['timestamp'], unit='ms')
df.set\_index('timestamp', inplace=True)
df = df[numeric\_cols]
df.dropna(subset=numeric\_cols, inplace=True)
if df.empty:
logger.warning(f"⚠️ [Data] DataFrame لـ {symbol} فارغ بعد إزالة قيم NaN.")
return None
logger.debug(f"✅ [Data] تم جلب {len(df)} شمعة ({interval}) لـ {symbol}.")
return df
except (BinanceAPIException, BinanceRequestException) as binance\_err:
logger.error(f"❌ [Data] خطأ Binance أثناء جلب البيانات لـ {symbol}: {binance\_err}")
return None
except Exception as e:
logger.error(f"❌ [Data] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}", exc\_info=True)
return None

def calculate\_ema(series: pd.Series, span: int) -\> pd.Series:
"""يحسب المتوسط المتحرك الأسي (EMA)."""
if series is None or series.isnull().all() or len(series) \< span:
logger.debug(f"⚠️ [Indicators] بيانات غير كافية لحساب EMA span={span}.")
return pd.Series(index=series.index if series is not None else None, dtype=float)
ema = series.ewm(span=span, adjust=False).mean()
logger.debug(f"✅ [Indicators] تم حساب EMA span={span}.")
return ema

def calculate\_vwma(df: pd.DataFrame, period: int) -\> pd.Series:
"""يحسب المتوسط المتحرك المرجح بالحجم (VWMA)."""
df\_calc = df.copy()
if not all(col in df\_calc.columns for col in ['close', 'volume']) or df\_calc[['close', 'volume']].isnull().all().any() or len(df\_calc) \< period:
logger.debug(f"⚠️ [Indicators] بيانات غير كافية لحساب VWMA period={period}.")
return pd.Series(index=df\_calc.index, dtype=float)
df\_calc['price\_volume'] = df\_calc['close'] \* df\_calc['volume']
rolling\_price\_volume\_sum = df\_calc['price\_volume'].rolling(window=period, min\_periods=period).sum()
rolling\_volume\_sum = df\_calc['volume'].rolling(window=period, min\_periods=period).sum()
vwma = rolling\_price\_volume\_sum / rolling\_volume\_sum.replace(0, np.nan)
logger.debug(f"✅ [Indicators] تم حساب VWMA period={period}.")
return vwma

def get\_btc\_trend\_4h() -\> str:
"""يحسب اتجاه البيتكوين على إطار 4 ساعات."""
logger.debug("ℹ️ [Indicators] حساب اتجاه البيتكوين على إطار 4 ساعات...")
try:
df = fetch\_historical\_data("BTCUSDT", interval=Client.KLINE\_INTERVAL\_4HOUR, days=10)
if df is None or df.empty or len(df) \< 51: \# Ensure enough data for EMA50
logger.warning("⚠️ [Indicators] بيانات BTC/USDT 4H غير كافية لتحديد الاتجاه.")
return "N/A (بيانات غير كافية)"
df['close'] = pd.to\_numeric(df['close'], errors='coerce')
df.dropna(subset=['close'], inplace=True)
if len(df) \< 50:
logger.warning("⚠️ [Indicators] بيانات BTC/USDT 4H غير كافية بعد إزالة NaN.")
return "N/A (بيانات غير كافية)"
ema20 = calculate\_ema(df['close'], 20).iloc[-1]
ema50 = calculate\_ema(df['close'], 50).iloc[-1]
current\_close = df['close'].iloc[-1]
if pd.isna(ema20) or pd.isna(ema50) or pd.isna(current\_close):
logger.warning("⚠️ [Indicators] خطأ في حساب EMA20/EMA50 لـ BTC/USDT 4H.")
return "N/A (خطأ في الحساب)"
diff\_ema20\_pct = abs(current\_close - ema20) / current\_close if current\_close \> 0 else 0
if current\_close \> ema20 \> ema50: trend = "صعود 📈"
elif current\_close \< ema20 \< ema50: trend = "هبوط 📉"
elif diff\_ema20\_pct \< 0.005: trend = "استقرار 🔄" \# Sideways
else: trend = "تذبذب 🔀" \# Volatile
logger.debug(f"✅ [Indicators] اتجاه البيتكوين 4H: {trend}")
return trend
except Exception as e:
logger.error(f"❌ [Indicators] خطأ أثناء حساب اتجاه البيتكوين على إطار 4 ساعات: {e}", exc\_info=True)
return "N/A (خطأ)"

# \---------------------- Database Connection Setup ----------------------

def init\_db(retries: int = 5, delay: int = 5) -\> None:
"""تهيئة اتصال قاعدة البيانات وإنشاء الجداول إذا لم تكن موجودة."""
global conn, cur
logger.info("[DB] بدء تهيئة قاعدة البيانات...")
for attempt in range(retries):
try:
logger.info(f"[DB] محاولة الاتصال (المحاولة {attempt + 1}/{retries})..." )
conn = psycopg2.connect(DB\_URL, connect\_timeout=10, cursor\_factory=RealDictCursor)
conn.autocommit = False
cur = conn.cursor()
logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")