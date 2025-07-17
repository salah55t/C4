import os
import time
import json
import pickle
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta, timezone
from psycopg2.extras import RealDictCursor
from binance.client import Client
from decouple import config

# إعدادات الاتصال
API_KEY = config('BINANCE_API_KEY')
API_SECRET = config('BINANCE_API_SECRET')
DB_URL = config('DATABASE_URL')
MODEL_FOLDER = 'V8'
BASE_ML_MODEL_NAME = 'LightGBM_Scalping_V8_With_Momentum'
SIGNAL_GENERATION_TIMEFRAME = '15m'
BTC_SYMBOL = 'BTCUSDT'
ADX_PERIOD = 14; RSI_PERIOD = 14; ATR_PERIOD = 14
EMA_PERIODS = [21, 50, 200]
REL_VOL_PERIOD = 30; MOMENTUM_PERIOD = 12; EMA_SLOPE_PERIOD = 5

# تعريف الفلاتر كما هي في الكود الأساسي
FILTER_KEYS = [
    "adx", "rel_vol", "rsi", "roc", "slope", "min_rrr", "min_volatility_pct",
    "min_btc_correlation", "min_bid_ask_ratio", "relative_volume", "btc_correlation",
    f"roc_{MOMENTUM_PERIOD}", f"ema_slope_{EMA_SLOPE_PERIOD}", "atr"
]

def get_db():
    conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    conn.autocommit = False
    return conn

def fetch_historical_data(client, symbol, interval, days):
    limit = int((days * 24 * 60) / int(interval.replace('m', '')))
    klines = client.get_historical_klines(symbol, interval, limit=min(limit, 1000))
    if not klines:
        return None
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df.dropna()

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
    df_calc = df.copy()
    # EMAs
    for period in EMA_PERIODS:
        df_calc[f'ema_{period}'] = df_calc['close'].ewm(span=period, adjust=False).mean()
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    # ADX
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
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
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['btc_correlation'] = 0.0
    if btc_df is not None and not btc_df.empty:
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = df_calc['close'].pct_change().rolling(window=30).corr(merged_df['btc_returns'])
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['slope'] = df_calc['close'].diff()
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    return df_calc.astype('float32', errors='ignore')

def load_ml_model(symbol):
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)
    return model_bundle['model'], model_bundle['scaler'], model_bundle['feature_names']

def insert_signal(conn, symbol, ts, entry, atr, filter_values, ml_conf, target, stop_loss):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO signals_backtest (symbol, timestamp, entry_price, atr, filter_values, ml_confidence, target_price, stop_loss, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (symbol, ts, entry, atr, json.dumps(filter_values), ml_conf, target, stop_loss, 'open'))
        conn.commit()
        signal_id = cur.fetchone()['id']
    return signal_id

def update_signal_status(conn, signal_id, status, close_price, close_time):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE signals_backtest SET status=%s, close_price=%s, close_time=%s WHERE id=%s
        """, (status, close_price, close_time, signal_id))
        conn.commit()

def main():
    # Binance client
    client = Client(API_KEY, API_SECRET)
    conn = get_db()
    # تجهيز الجدول إذا لم يكن موجوداً
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS signals_backtest (
                id SERIAL PRIMARY KEY,
                symbol TEXT,
                timestamp TIMESTAMP,
                entry_price DOUBLE PRECISION,
                atr DOUBLE PRECISION,
                filter_values JSONB,
                ml_confidence DOUBLE PRECISION,
                target_price DOUBLE PRECISION,
                stop_loss DOUBLE PRECISION,
                status TEXT,
                close_price DOUBLE PRECISION,
                close_time TIMESTAMP
            );
        """)
        conn.commit()

    # جلب قائمة العملات المدعومة (ممكن تعديلها حسب المتوفر)
    symbols = []
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM signals;")
        symbols = [row['symbol'] for row in cur.fetchall()]
    if not symbols:
        symbols = ['BTCUSDT', 'ETHUSDT']

    # بيانات البيتكوين للارتباط
    btc_df = fetch_historical_data(client, BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, 3)
    if btc_df is not None:
        btc_df['btc_returns'] = btc_df['close'].pct_change()

    for symbol in symbols:
        print(f"Processing {symbol} ...")
        model, scaler, feats = load_ml_model(symbol)
        if not model: continue
        df = fetch_historical_data(client, symbol, SIGNAL_GENERATION_TIMEFRAME, 3)
        if df is None or df.empty: continue
        df_feat = calculate_features(df, btc_df)
        # بدء من الشمعة التي يكون فيها كل الميزات متوفرة
        for ts in df_feat.index:
            row = df_feat.loc[ts]
            # تجهيز الميزات بنفس ترتيب النموذج
            try:
                feats_row = row[feats].values.reshape(1, -1)
                feats_row = scaler.transform(feats_row)
                y_pred = model.predict(feats_row)[0]
                y_prob = model.predict_proba(feats_row)[0].max()
            except Exception as e:
                continue
            if y_pred == 1:
                entry = row['close']
                atr = row['atr']
                if not atr or atr <= 0: continue
                target = entry + (atr * 2.2)
                stop_loss = entry - (atr * 1.5)
                # حفظ كل قيم الفلاتر
                filter_values = {k: float(row.get(k, 0)) for k in FILTER_KEYS}
                signal_id = insert_signal(conn, symbol, ts, entry, atr, filter_values, float(y_prob), target, stop_loss)
                # تتبع التوصية لمدة 3 أيام أو حتى تحقق الهدف/وقف الخسارة
                close_status, close_price, close_time = None, None, None
                closing_window = df_feat.loc[ts:].iloc[1:]  # بعد التوصية مباشرة
                for ts2, row2 in closing_window.iterrows():
                    price = row2['close']
                    if price >= target:
                        close_status, close_price, close_time = 'target_hit', price, ts2
                        break
                    elif price <= stop_loss:
                        close_status, close_price, close_time = 'stop_loss_hit', price, ts2
                        break
                    # انتهاء 3 أيام؟ (تاريخ الشمعة الحالية - تاريخ التوصية)
                    if (ts2 - ts) > timedelta(days=3):
                        close_status, close_price, close_time = 'timeout', price, ts2
                        break
                if close_status:
                    update_signal_status(conn, signal_id, close_status, close_price, close_time)
                else:
                    # لم يحدث شيء خلال 3 أيام
                    update_signal_status(conn, signal_id, 'timeout', row2['close'], ts2)
    print("Backtest complete.")

if __name__ == "__main__":
    main()