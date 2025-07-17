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

# باقي الكود كما هو ...

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

    # جلب قائمة الرموز من ملفات النماذج في المجلد V8
    symbols = [filename.replace(f"{BASE_ML_MODEL_NAME}_", "").replace(".pkl", "") 
               for filename in os.listdir(MODEL_FOLDER) if filename.endswith(".pkl")]
    symbols = symbols[:20]  # تحديد أول 20 رمز فقط

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
        # باقي الكود كما هو ...
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