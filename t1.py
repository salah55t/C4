import os
import gc
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from psycopg2 import sql
import psycopg2
from decouple import config
import random
from typing import List, Dict, Any, Optional, Tuple
from binance.client import Client

# --- تهيئة الإعدادات ---
DB_URL = config('DATABASE_URL')
BATCH_SIZE = 10
RISK_PER_TRADE_PERCENT = 1.0
START_DATE = "2024-01-01"
END_DATE = "2024-06-01"
TIMEFRAME = '15m'
TEST_SYMBOLS = get_validated_symbols()  # دالة لتحميل الرموز الصالحة

# --- تهيئة قاعدة البيانات للاختبار الخلفي ---
def init_backtest_db():
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                entry_price DOUBLE PRECISION NOT NULL,
                target_price DOUBLE PRECISION NOT NULL,
                stop_loss DOUBLE PRECISION NOT NULL,
                closing_price DOUBLE PRECISION,
                profit_percentage DOUBLE PRECISION,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                filter_values JSONB NOT NULL,
                status TEXT CHECK(status IN ('win', 'loss', 'open'))
            );
        """)
        conn.commit()
    return conn

# --- دالة محاكاة التداول ---
def simulate_trade(
    df: pd.DataFrame, 
    entry_idx: int, 
    entry_price: float, 
    target_price: float, 
    stop_loss: float
) -> Tuple[float, float, str]:
    """
    محاكاة صفقة من نقطة الدخول حتى الخروج
    """
    for i in range(entry_idx + 1, len(df)):
        current_low = df.iloc[i]['low']
        current_high = df.iloc[i]['high']
        current_close = df.iloc[i]['close']
        
        # التحقق من ضرب وقف الخسارة
        if current_low <= stop_loss:
            return stop_loss, ((stop_loss / entry_price) - 1) * 100, 'loss'
        
        # التحقق من تحقيق الهدف
        if current_high >= target_price:
            return target_price, ((target_price / entry_price) - 1) * 100, 'win'
        
        # إغلاق عند نهاية البيانات
        if i == len(df) - 1:
            return current_close, ((current_close / entry_price) - 1) * 100, 'open'

    return entry_price, 0, 'open'

# --- دالة الاختبار الخلفي لرمز واحد ---
def backtest_symbol(symbol: str, client: Client) -> List[Dict[str, Any]]:
    results = []
    
    # جلب البيانات التاريخية
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=TIMEFRAME,
        start_str=START_DATE,
        end_str=END_DATE
    )
    
    if not klines:
        return results
        
    # تحويل إلى DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
        'taker_buy_quote', 'ignore'
    ])
    
    # تحويل الأنواع
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # حساب المؤشرات (يجب إضافة دوال حساب المؤشرات الفعلية)
    df = calculate_indicators(df)
    
    # المحاكاة
    for i in range(50, len(df) - 1):  # بدءاً من الشمعة الـ 50
        # تطبيق الاستراتيجية (يجب استبدال هذا بمنطق الإشارة الفعلي)
        signal = generate_signal(df.iloc[:i+1])
        
        if signal:
            # حساب قيم الفلاتر
            filter_values = {
                'adx': df.iloc[i]['adx'],
                'rsi': df.iloc[i]['rsi'],
                'atr': df.iloc[i]['atr'],
                'rel_vol': df.iloc[i]['relative_volume'],
                'btc_corr': df.iloc[i]['btc_correlation'],
                'roc': df.iloc[i][f'roc_{MOMENTUM_PERIOD}'],
                'ema_slope': df.iloc[i][f'ema_slope_{EMA_SLOPE_PERIOD}']
            }
            
            # محاكاة الصفقة
            entry_price = df.iloc[i]['close']
            target_price = entry_price * 1.02  # +2%
            stop_loss = entry_price * 0.98     # -2%
            
            # تشغيل المحاكاة
            closing_price, profit_pct, status = simulate_trade(
                df, i, entry_price, target_price, stop_loss
            )
            
            # تسجيل النتيجة
            trade_data = {
                'symbol': symbol,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'closing_price': closing_price,
                'profit_percentage': profit_pct,
                'entry_time': df.iloc[i]['timestamp'],
                'exit_time': df.iloc[i + 1]['timestamp'],  # تبسيط
                'filter_values': json.dumps(filter_values),
                'status': status
            }
            results.append(trade_data)
    
    return results

# --- دالة معالجة الدفعات ---
def run_backtest_in_batches():
    # تهيئة العميل والاتصال بقاعدة البيانات
    client = Client()
    db_conn = init_backtest_db()
    
    # تقسيم الرموز إلى دفعات
    batches = [TEST_SYMBOLS[i:i + BATCH_SIZE] 
               for i in range(0, len(TEST_SYMBOLS), BATCH_SIZE)]
    
    for batch_idx, batch in enumerate(batches):
        logger.info(f"🚀 بدء الدفعة {batch_idx + 1}/{len(batches)} - {len(batch)} رموز")
        batch_results = []
        
        for symbol in batch:
            try:
                symbol_results = backtest_symbol(symbol, client)
                batch_results.extend(symbol_results)
                logger.info(f"✅ تم معالجة {symbol} - {len(symbol_results)} صفقة")
            except Exception as e:
                logger.error(f"❌ فشل معالجة {symbol}: {str(e)}")
        
        # حفظ نتائج الدفعة في قاعدة البيانات
        save_batch_results(db_conn, batch_results)
        
        # تحرير الذاكرة
        del batch_results
        gc.collect()
        logger.info(f"♻️ تم تحرير ذاكرة الدفعة {batch_idx + 1}")

# --- دالة حفظ النتائج ---
def save_batch_results(conn, results: List[Dict[str, Any]]):
    if not results:
        return
        
    with conn.cursor() as cur:
        args = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
            r['symbol'],
            r['entry_price'],
            r['target_price'],
            r['stop_loss'],
            r['closing_price'],
            r['profit_percentage'],
            r['entry_time'],
            r['exit_time'],
            r['filter_values'],
            r['status']
        )).decode('utf-8') for r in results)
        
        cur.execute(
            sql.SQL("""
            INSERT INTO backtest_results (
                symbol, entry_price, target_price, stop_loss,
                closing_price, profit_percentage, entry_time,
                exit_time, filter_values, status
            ) VALUES {}
            """).format(sql.SQL(args))
        )
        conn.commit()
    logger.info(f"💾 تم حفظ {len(results)} صفقة في قاعدة البيانات")

if __name__ == "__main__":
    run_backtest_in_batches()