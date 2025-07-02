import os
import time
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from binance.client import Client
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Optional
from threading import Thread
from flask import Flask

# --- إعدادات أساسية ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ichimoku_calculator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IchimokuCalculatorService')

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
    exit(1)

# --- ثوابت وإعدادات ---
TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS: int = 180
RUN_INTERVAL_HOURS: int = 4  # تشغيل المهمة كل 4 ساعات
ICHIMOKU_TENKAN_PERIOD: int = 9
ICHIMOKU_KIJUN_PERIOD: int = 26
ICHIMOKU_SENKOU_B_PERIOD: int = 52
ICHIMOKU_CHIKOU_SHIFT: int = -26
ICHIMOKU_SENKOU_SHIFT: int = 26

# --- متغيرات عامة ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None

# --- دوال الاتصال والتهيئة ---
def init_db():
    """Initializes the database connection."""
    global conn
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        logger.info("✅ [DB] Database initialized successfully.")
    except Exception as e:
        logger.critical(f"❌ [DB] Database connection failed: {e}")
        # Don't exit, allow retries in the main loop
        conn = None

def get_binance_client():
    """Initializes the Binance client."""
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] Client initialized successfully.")
    except Exception as e:
        logger.critical(f"❌ [Binance] Client initialization failed: {e}")
        client = None

def create_ichimoku_table_if_not_exists():
    """Creates the ichimoku_features table if it doesn't exist."""
    if not conn:
        logger.warning("[DB] No database connection, skipping table creation check.")
        return
    query = """
    CREATE TABLE IF NOT EXISTS ichimoku_features (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        tenkan_sen FLOAT,
        kijun_sen FLOAT,
        senkou_span_a FLOAT,
        senkou_span_b FLOAT,
        chikou_span FLOAT,
        UNIQUE (symbol, timestamp, timeframe)
    );
    CREATE INDEX IF NOT EXISTS idx_ichimoku_symbol_timestamp 
    ON ichimoku_features (symbol, timestamp DESC);
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query)
        logger.info("✅ [DB] 'ichimoku_features' table checked/created successfully.")
    except Exception as e:
        logger.error(f"❌ [DB] Error creating 'ichimoku_features' table: {e}")

# --- دالة التحقق من الرموز (مطابقة لملف التدريب) ---
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """Reads a list of symbols from a file and validates them against Binance."""
    if not client:
        logger.error("❌ [Symbol Validation] Binance client not available.")
        return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"❌ Symbol list file not found at {file_path}")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted_symbols = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        
        validated_list = sorted(list(formatted_symbols.intersection(active_symbols)))
        logger.info(f"✅ Found {len(validated_list)} validated symbols to process.")
        return validated_list
    except Exception as e:
        logger.error(f"❌ [Symbol Validation] Error: {e}", exc_info=True)
        return []

# --- دوال حساب وحفظ إيشيموكو ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """Fetches historical kline data from Binance."""
    if not client: return None
    try:
        start_str = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close']]
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ Error fetching data for {symbol}: {e}")
        return None

def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all Ichimoku Cloud components."""
    high = df['high']
    low = df['low']
    close = df['close']

    df['tenkan_sen'] = (high.rolling(window=ICHIMOKU_TENKAN_PERIOD).max() + low.rolling(window=ICHIMOKU_TENKAN_PERIOD).min()) / 2
    df['kijun_sen'] = (high.rolling(window=ICHIMOKU_KIJUN_PERIOD).max() + low.rolling(window=ICHIMOKU_KIJUN_PERIOD).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(ICHIMOKU_SENKOU_SHIFT)
    df['senkou_span_b'] = ((high.rolling(window=ICHIMOKU_SENKOU_B_PERIOD).max() + low.rolling(window=ICHIMOKU_SENKOU_B_PERIOD).min()) / 2).shift(ICHIMOKU_SENKOU_SHIFT)
    df['chikou_span'] = close.shift(ICHIMOKU_CHIKOU_SHIFT)
    
    return df

def save_ichimoku_to_db(symbol: str, df_ichimoku: pd.DataFrame, timeframe: str):
    """Saves the calculated Ichimoku features to the database using ON CONFLICT."""
    if not conn or df_ichimoku.empty:
        return
    
    df_to_save = df_ichimoku[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']].copy()
    ichimoku_cols = df_to_save.columns.tolist()
    df_to_save.dropna(subset=ichimoku_cols, how='all', inplace=True)

    if df_to_save.empty:
        logger.warning(f"⚠️ No new Ichimoku data to save for {symbol} on {timeframe}.")
        return

    df_to_save.reset_index(inplace=True)
    tuples = [tuple(x) for x in df_to_save[['timestamp'] + ichimoku_cols].to_numpy()]
    
    # Add symbol and timeframe to each tuple for insertion
    data_to_insert = [(symbol, row[0], timeframe) + row[1:] for row in tuples]

    cols = ['symbol', 'timestamp', 'timeframe'] + ichimoku_cols
    update_cols = [f"{col} = EXCLUDED.{col}" for col in ichimoku_cols]
    
    query = f"""
        INSERT INTO ichimoku_features ({", ".join(cols)})
        VALUES %s
        ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
            {", ".join(update_cols)};
    """
    
    try:
        with conn.cursor() as cur:
            execute_values(cur, query, data_to_insert)
        logger.info(f"💾 Successfully saved/updated {len(data_to_insert)} Ichimoku records for {symbol} to DB.")
    except Exception as e:
        logger.error(f"❌ [DB] Error saving Ichimoku data for {symbol}: {e}")

# --- المهمة الرئيسية التي تعمل في الخلفية ---
def calculator_job():
    """Main function to run the calculation and saving pipeline periodically."""
    global conn, client  # <--- ✨ התיקון כאן (The fix is here)
    while True:
        logger.info("🚀 Starting new Ichimoku calculation cycle...")
        try:
            # Initialize connections at the start of each cycle
            init_db()
            get_binance_client()

            if not conn or not client:
                logger.error("Connections not established. Skipping this cycle.")
            else:
                create_ichimoku_table_if_not_exists()
                symbols_to_process = get_validated_symbols()
                
                if not symbols_to_process:
                    logger.warning("No symbols to process in this cycle.")
                else:
                    for symbol in symbols_to_process:
                        logger.info(f"\n--- ⏳ Processing {symbol} ---")
                        try:
                            df_ohlc = fetch_historical_data(symbol, TIMEFRAME, DATA_LOOKBACK_DAYS)
                            if df_ohlc is None or df_ohlc.empty:
                                logger.warning(f"Could not fetch data for {symbol}. Skipping.")
                                continue
                            
                            df_with_ichimoku = calculate_ichimoku(df_ohlc)
                            save_ichimoku_to_db(symbol, df_with_ichimoku, TIMEFRAME)
                            
                        except Exception as e:
                            logger.critical(f"❌ Critical error processing {symbol}: {e}", exc_info=True)
                        time.sleep(2) # Small delay between symbols to avoid rate limits

        except Exception as e:
            logger.critical(f"❌ An unexpected error occurred in the main job loop: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()
                logger.info("✅ Database connection closed for this cycle.")
                conn = None
            
        logger.info(f"✅ Cycle finished. Waiting for {RUN_INTERVAL_HOURS} hours before the next run.")
        time.sleep(RUN_INTERVAL_HOURS * 60 * 60)

# --- خادم الويب للفحص الصحي ---
app = Flask(__name__)
@app.route('/')
def health_check():
    """Health check endpoint for the hosting platform."""
    return "Ichimoku Calculator service is running.", 200

if __name__ == "__main__":
    # تشغيل مهمة الحساب في خيط منفصل
    calculator_thread = Thread(target=calculator_job)
    calculator_thread.daemon = True
    calculator_thread.start()
    
    # تشغيل خادم الويب في الخيط الرئيسي
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting health check server on port {port}")
    app.run(host='0.0.0.0', port=port)
