#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام التداول الآلي المحسن - الإصدار المطور
Enhanced Crypto Trading Bot with AI Strategy Optimization
تم تطويره بواسطة فريق التحليل الفني المتقدم

الميزات الجديدة:
- استراتيجية EMA Crossover المحسنة مع Stochastic Momentum
- تحسين معاملات الاستراتيجية بالذكاء الاصطناعي
- إدارة المخاطر المتقدمة مع ATR Trailing Stop
- واجهة ويب متطورة للمراقبة والتحكم
- دعم قاعدة البيانات PostgreSQL
"""

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import redis
import gc
import warnings
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor, Json
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Set, Tuple
from sklearn.preprocessing import StandardScaler
from collections import deque, Counter
import talib

# إعدادات تجاهل التحذيرات
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# إعداد نظام السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_crypto_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedCryptoBot')

# تحميل متغيرات البيئة
try:
    # إعدادات Binance API
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    
    # إعدادات قاعدة البيانات
    DATABASE_URL = config('DATABASE_URL')
    REDIS_URL = config('REDIS_URL', default='redis://localhost:6379/0')
    
    # إعدادات Telegram (اختيارية)
    TELEGRAM_BOT_TOKEN = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID = config('TELEGRAM_CHAT_ID', default='')
    
    # إعدادات Flask
    FLASK_HOST = config('FLASK_HOST', default='0.0.0.0')
    FLASK_PORT = int(config('FLASK_PORT', default='5000'))
    
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة: {e}")
    exit(1)

# === إعدادات النظام المحسن ===
# إعدادات التداول الأساسية
TRADING_ENABLED = False
RISK_PER_TRADE_PERCENT = 1.5
MAX_OPEN_TRADES = 3
MIN_PROFIT_PERCENT = 0.8
TRADING_FEE_PERCENT = 0.1

# إعدادات الاستراتيجية المحسنة
STRATEGY_NAME = 'EMA_Crossover_Enhanced_v2'
SIGNAL_TIMEFRAME = '15m'
LOOKBACK_DAYS = 90

# معاملات المؤشرات الفنية المحسنة
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
STOCH_PERIOD = 14
STOCH_K_PERIOD = 3
STOCH_D_PERIOD = 3
RSI_PERIOD = 14
ATR_PERIOD = 14

# إعدادات إدارة المخاطر المتقدمة
USE_ATR_TRAILING_STOP = True
ATR_MULTIPLIER = 2.5
CONFIDENCE_THRESHOLD = 0.65
VOLUME_CONFIRMATION_REQUIRED = True

# متغيرات الحالة العامة
conn = None
client = None
redis_client = None
exchange_info = {}
validated_symbols = []
trading_status_lock = Lock()
signal_cache = {}
notifications_cache = deque(maxlen=100)

class EnhancedStrategyOptimizer:
    """محرك تحسين الاستراتيجيات المتقدم"""
    
    def __init__(self):
        self.optimized_parameters = {
            'ema_fast': EMA_FAST_PERIOD,
            'ema_slow': EMA_SLOW_PERIOD,
            'stoch_period': STOCH_PERIOD,
            'rsi_period': RSI_PERIOD,
            'atr_period': ATR_PERIOD,
            'confidence_threshold': CONFIDENCE_THRESHOLD
        }
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب المؤشرات الفنية المحسنة"""
        try:
            # المتوسطات المتحركة الأسية
            df['ema_fast'] = talib.EMA(df['close'].values, timeperiod=self.optimized_parameters['ema_fast'])
            df['ema_slow'] = talib.EMA(df['close'].values, timeperiod=self.optimized_parameters['ema_slow'])
            
            # مؤشر الستوكاستيك
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                df['high'].values, df['low'].values, df['close'].values,
                fastk_period=self.optimized_parameters['stoch_period'],
                slowk_period=STOCH_K_PERIOD,
                slowd_period=STOCH_D_PERIOD
            )
            
            # مؤشر القوة النسبية
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.optimized_parameters['rsi_period'])
            
            # متوسط المدى الحقيقي
            df['atr'] = talib.ATR(
                df['high'].values, df['low'].values, df['close'].values,
                timeperiod=self.optimized_parameters['atr_period']
            )
            
            # مؤشر MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
            
            # خطوط البولينجر
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values)
            
            # حجم التداول النسبي
            df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشرات الفنية: {e}")
            return df
    
    def generate_enhanced_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """توليد إشارة تداول محسنة"""
        try:
            if len(df) < 50:
                return {'signal': 'HOLD', 'confidence': 0, 'reason': 'بيانات غير كافية'}
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signals = []
            confidence_factors = []
            
            # إشارة تقاطع المتوسطات المتحركة
            ema_signal = self._analyze_ema_crossover(latest, prev)
            signals.append(ema_signal['signal'])
            confidence_factors.append(ema_signal['confidence'])
            
            # إشارة الستوكاستيك
            stoch_signal = self._analyze_stochastic(latest, prev)
            signals.append(stoch_signal['signal'])
            confidence_factors.append(stoch_signal['confidence'])
            
            # إشارة RSI
            rsi_signal = self._analyze_rsi(latest)
            signals.append(rsi_signal['signal'])
            confidence_factors.append(rsi_signal['confidence'])
            
            # تأكيد الحجم
            volume_confirmation = self._analyze_volume(latest)
            
            # حساب الإشارة النهائية
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            
            if buy_signals >= 2 and volume_confirmation:
                final_signal = 'BUY'
                confidence = np.mean(confidence_factors) * 0.9 if volume_confirmation else np.mean(confidence_factors) * 0.7
            elif sell_signals >= 2 and volume_confirmation:
                final_signal = 'SELL'
                confidence = np.mean(confidence_factors) * 0.9 if volume_confirmation else np.mean(confidence_factors) * 0.7
            else:
                final_signal = 'HOLD'
                confidence = np.mean(confidence_factors) * 0.5
            
            # حساب مستويات الدخول والخروج
            entry_price = latest['close']
            atr_value = latest['atr']
            
            if final_signal == 'BUY':
                stop_loss = entry_price - (atr_value * ATR_MULTIPLIER)
                take_profit = entry_price + (atr_value * ATR_MULTIPLIER * 1.5)
            elif final_signal == 'SELL':
                stop_loss = entry_price + (atr_value * ATR_MULTIPLIER)
                take_profit = entry_price - (atr_value * ATR_MULTIPLIER * 1.5)
            else:
                stop_loss = take_profit = entry_price
            
            return {
                'signal': final_signal,
                'confidence': min(confidence, 0.95),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr_value,
                'volume_confirmation': volume_confirmation,
                'technical_details': {
                    'ema_fast': latest['ema_fast'],
                    'ema_slow': latest['ema_slow'],
                    'stoch_k': latest['stoch_k'],
                    'stoch_d': latest['stoch_d'],
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'volume_ratio': latest['volume_ratio']
                }
            }
            
        except Exception as e:
            logger.error(f"خطأ في توليد الإشارة: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'reason': f'خطأ: {str(e)}'}
    
    def _analyze_ema_crossover(self, latest: pd.Series, prev: pd.Series) -> Dict[str, Any]:
        """تحليل تقاطع المتوسطات المتحركة"""
        try:
            ema_fast_current = latest['ema_fast']
            ema_slow_current = latest['ema_slow']
            ema_fast_prev = prev['ema_fast']
            ema_slow_prev = prev['ema_slow']
            
            # تقاطع صاعد
            if ema_fast_prev <= ema_slow_prev and ema_fast_current > ema_slow_current:
                return {'signal': 'BUY', 'confidence': 0.8}
            # تقاطع هابط
            elif ema_fast_prev >= ema_slow_prev and ema_fast_current < ema_slow_current:
                return {'signal': 'SELL', 'confidence': 0.8}
            # اتجاه صاعد مستمر
            elif ema_fast_current > ema_slow_current:
                strength = min((ema_fast_current - ema_slow_current) / ema_slow_current * 100, 0.6)
                return {'signal': 'BUY', 'confidence': strength}
            # اتجاه هابط مستمر
            else:
                strength = min((ema_slow_current - ema_fast_current) / ema_slow_current * 100, 0.6)
                return {'signal': 'SELL', 'confidence': strength}
                
        except Exception:
            return {'signal': 'HOLD', 'confidence': 0}
    
    def _analyze_stochastic(self, latest: pd.Series, prev: pd.Series) -> Dict[str, Any]:
        """تحليل مؤشر الستوكاستيك"""
        try:
            stoch_k = latest['stoch_k']
            stoch_d = latest['stoch_d']
            
            # منطقة ذروة البيع
            if stoch_k < 20 and stoch_d < 20 and stoch_k > stoch_d:
                return {'signal': 'BUY', 'confidence': 0.7}
            # منطقة ذروة الشراء
            elif stoch_k > 80 and stoch_d > 80 and stoch_k < stoch_d:
                return {'signal': 'SELL', 'confidence': 0.7}
            # منطقة محايدة
            else:
                return {'signal': 'HOLD', 'confidence': 0.3}
                
        except Exception:
            return {'signal': 'HOLD', 'confidence': 0}
    
    def _analyze_rsi(self, latest: pd.Series) -> Dict[str, Any]:
        """تحليل مؤشر القوة النسبية"""
        try:
            rsi = latest['rsi']
            
            if rsi < 30:
                return {'signal': 'BUY', 'confidence': 0.6}
            elif rsi > 70:
                return {'signal': 'SELL', 'confidence': 0.6}
            else:
                return {'signal': 'HOLD', 'confidence': 0.2}
                
        except Exception:
            return {'signal': 'HOLD', 'confidence': 0}
    
    def _analyze_volume(self, latest: pd.Series) -> bool:
        """تحليل تأكيد الحجم"""
        try:
            return latest['volume_ratio'] > 1.2
        except Exception:
            return False

class DatabaseManager:
    """مدير قاعدة البيانات المحسن"""
    
    def __init__(self):
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """تهيئة قاعدة البيانات"""
        try:
            self.conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
            self.conn.autocommit = True
            
            with self.conn.cursor() as cur:
                # جدول الإشارات المحسن
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        strategy_name VARCHAR(50) NOT NULL,
                        signal_type VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(20,8) NOT NULL,
                        stop_loss DECIMAL(20,8),
                        take_profit DECIMAL(20,8),
                        confidence DECIMAL(5,4),
                        status VARCHAR(20) DEFAULT 'OPEN',
                        quantity DECIMAL(20,8),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        closed_at TIMESTAMP WITH TIME ZONE,
                        profit_loss DECIMAL(20,8),
                        technical_details JSONB,
                        notes TEXT
                    )
                """)
                
                # جدول معاملات الاستراتيجية
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_parameters (
                        id SERIAL PRIMARY KEY,
                        strategy_name VARCHAR(50) NOT NULL,
                        parameters JSONB NOT NULL,
                        performance_metrics JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # جدول الإشعارات
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY,
                        type VARCHAR(20) NOT NULL,
                        title VARCHAR(100) NOT NULL,
                        message TEXT NOT NULL,
                        is_read BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # إنشاء الفهارس
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON trading_signals(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_created_at ON trading_signals(created_at)")
                
            logger.info("✅ تم تهيئة قاعدة البيانات بنجاح")
            
        except Exception as e:
            logger.error(f"❌ خطأ في تهيئة قاعدة البيانات: {e}")
            raise
    
    def save_signal(self, signal_data: Dict[str, Any]) -> int:
        """حفظ إشارة تداول"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_signals 
                    (symbol, strategy_name, signal_type, entry_price, stop_loss, 
                     take_profit, confidence, quantity, technical_details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    signal_data['symbol'],
                    signal_data['strategy_name'],
                    signal_data['signal_type'],
                    signal_data['entry_price'],
                    signal_data['stop_loss'],
                    signal_data['take_profit'],
                    signal_data['confidence'],
                    signal_data.get('quantity', 0),
                    Json(signal_data.get('technical_details', {}))
                ))
                
                signal_id = cur.fetchone()['id']
                logger.info(f"✅ تم حفظ الإشارة رقم {signal_id}")
                return signal_id
                
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ الإشارة: {e}")
            return 0
    
    def get_open_signals(self) -> List[Dict[str, Any]]:
        """الحصول على الإشارات المفتوحة"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM trading_signals 
                    WHERE status = 'OPEN' 
                    ORDER BY created_at DESC
                """)
                return [dict(row) for row in cur.fetchall()]
                
        except Exception as e:
            logger.error(f"❌ خطأ في استرداد الإشارات: {e}")
            return []
    
    def add_notification(self, notification_type: str, title: str, message: str):
        """إضافة إشعار"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO notifications (type, title, message)
                    VALUES (%s, %s, %s)
                """, (notification_type, title, message))
                
            # إضافة للكاش أيضاً
            notifications_cache.appendleft({
                'type': notification_type,
                'title': title,
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ خطأ في إضافة الإشعار: {e}")

class TradingBot:
    """بوت التداول المحسن"""
    
    def __init__(self):
        self.client = None
        self.db = DatabaseManager()
        self.optimizer = EnhancedStrategyOptimizer()
        self.redis_client = None
        self.init_binance_client()
        self.init_redis()
        
    def init_binance_client(self):
        """تهيئة عميل Binance"""
        try:
            self.client = Client(API_KEY, API_SECRET)
            # اختبار الاتصال
            self.client.ping()
            logger.info("✅ تم الاتصال بـ Binance API بنجاح")
        except Exception as e:
            logger.error(f"❌ فشل الاتصال بـ Binance API: {e}")
            
    def init_redis(self):
        """تهيئة Redis"""
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ تم الاتصال بـ Redis بنجاح")
        except Exception as e:
            logger.warning(f"⚠️ فشل الاتصال بـ Redis: {e}")
    
    def get_historical_data(self, symbol: str, interval: str = '15m', days: int = 90) -> pd.DataFrame:
        """جلب البيانات التاريخية"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            klines = self.client.get_historical_klines(
                symbol, interval, start_time.strftime("%d %b %Y %H:%M:%S")
            )
            
            if not klines:
                return pd.DataFrame()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # تحويل أنواع البيانات
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """تحليل رمز معين"""
        try:
            # جلب البيانات التاريخية
            df = self.get_historical_data(symbol)
            if df.empty:
                return {'error': 'لا توجد بيانات كافية'}
            
            # حساب المؤشرات الفنية
            df = self.optimizer.calculate_technical_indicators(df)
            if df.empty:
                return {'error': 'فشل في حساب المؤشرات الفنية'}
            
            # توليد الإشارة
            signal_result = self.optimizer.generate_enhanced_signal(df)
            
            # إضافة معلومات إضافية
            signal_result['symbol'] = symbol
            signal_result['timestamp'] = datetime.now().isoformat()
            signal_result['strategy_name'] = STRATEGY_NAME
            
            return signal_result
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل {symbol}: {e}")
            return {'error': str(e)}
    
    def scan_symbols(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """مسح رموز متعددة"""
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol)
                if 'error' not in result and result['confidence'] >= CONFIDENCE_THRESHOLD:
                    results.append(result)
                    
                    # حفظ الإشارة في قاعدة البيانات
                    if result['signal'] in ['BUY', 'SELL']:
                        signal_id = self.db.save_signal({
                            'symbol': symbol,
                            'strategy_name': STRATEGY_NAME,
                            'signal_type': result['signal'],
                            'entry_price': result['entry_price'],
                            'stop_loss': result['stop_loss'],
                            'take_profit': result['take_profit'],
                            'confidence': result['confidence'],
                            'technical_details': result.get('technical_details', {})
                        })
                        
                        # إرسال إشعار
                        self.db.add_notification(
                            'SIGNAL',
                            f'إشارة {result["signal"]} جديدة',
                            f'تم اكتشاف إشارة {result["signal"]} لـ {symbol} بثقة {result["confidence"]:.2%}'
                        )
                
                # توقف قصير لتجنب تجاوز حدود API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ خطأ في مسح {symbol}: {e}")
                continue
        
        return results

# تهيئة Flask للواجهة الويب
app = Flask(__name__)
CORS(app)

# تهيئة بوت التداول
trading_bot = None

def init_trading_bot():
    """تهيئة بوت التداول"""
    global trading_bot
    try:
        trading_bot = TradingBot()
        logger.info("✅ تم تهيئة بوت التداول بنجاح")
    except Exception as e:
        logger.error(f"❌ فشل في تهيئة بوت التداول: {e}")

# === مسارات API ===

@app.route('/api/status', methods=['GET'])
def get_status():
    """الحصول على حالة النظام"""
    return jsonify({
        'status': 'active' if trading_bot else 'inactive',
        'trading_enabled': TRADING_ENABLED,
        'timestamp': datetime.now().isoformat(),
        'strategy': STRATEGY_NAME,
        'version': '2.0'
    })

@app.route('/api/analyze/<symbol>', methods=['GET'])
def analyze_symbol_api(symbol):
    """تحليل رمز معين"""
    if not trading_bot:
        return jsonify({'error': 'بوت التداول غير مهيأ'}), 500
    
    result = trading_bot.analyze_symbol(symbol.upper())
    return jsonify(result)

@app.route('/api/scan', methods=['POST'])
def scan_symbols_api():
    """مسح رموز متعددة"""
    if not trading_bot:
        return jsonify({'error': 'بوت التداول غير مهيأ'}), 500
    
    data = request.get_json()
    symbols = data.get('symbols', [])
    
    if not symbols:
        return jsonify({'error': 'لم يتم تحديد رموز للمسح'}), 400
    
    results = trading_bot.scan_symbols(symbols)
    return jsonify({'signals': results, 'count': len(results)})

@app.route('/api/signals', methods=['GET'])
def get_signals():
    """الحصول على الإشارات المفتوحة"""
    if not trading_bot:
        return jsonify({'error': 'بوت التداول غير مهيأ'}), 500
    
    signals = trading_bot.db.get_open_signals()
    return jsonify({'signals': signals})

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """الحصول على الإشعارات"""
    notifications = list(notifications_cache)
    return jsonify({'notifications': notifications})

@app.route('/api/toggle-trading', methods=['POST'])
def toggle_trading():
    """تبديل حالة التداول"""
    global TRADING_ENABLED
    TRADING_ENABLED = not TRADING_ENABLED
    
    status_msg = 'تم تفعيل التداول' if TRADING_ENABLED else 'تم إيقاف التداول'
    logger.info(status_msg)
    
    if trading_bot:
        trading_bot.db.add_notification('SYSTEM', 'تغيير حالة التداول', status_msg)
    
    return jsonify({
        'trading_enabled': TRADING_ENABLED,
        'message': status_msg
    })

# صفحة HTML بسيطة للمراقبة
@app.route('/')
def dashboard():
    """لوحة التحكم الرئيسية"""
    return render_template_string("""
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نظام التداول الآلي المحسن</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
               margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-card { background: #2d2d2d; padding: 20px; border-radius: 10px; 
                      margin-bottom: 20px; border-left: 4px solid #4CAF50; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #2d2d2d; padding: 20px; border-radius: 10px; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; 
                border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #45a049; }
        .signal-buy { border-left: 4px solid #4CAF50; }
        .signal-sell { border-left: 4px solid #f44336; }
        input { padding: 8px; margin: 5px; border: 1px solid #555; background: #333; color: #fff; }
        .notification { background: #333; padding: 10px; margin: 5px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 نظام التداول الآلي المحسن</h1>
            <p>الإصدار 2.0 - مع تحسينات الذكاء الاصطناعي</p>
        </div>
        
        <div class="status-card">
            <h3>حالة النظام</h3>
            <p id="system-status">جاري التحميل...</p>
            <button onclick="toggleTrading()">تبديل التداول</button>
            <button onclick="loadData()">تحديث البيانات</button>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>تحليل رمز</h3>
                <input type="text" id="symbol-input" placeholder="أدخل الرمز (مثل: BTCUSDT)" value="BTCUSDT">
                <button onclick="analyzeSymbol()">تحليل</button>
                <div id="analysis-result"></div>
            </div>
            
            <div class="card">
                <h3>الإشارات النشطة</h3>
                <div id="active-signals">جاري التحميل...</div>
            </div>
            
            <div class="card">
                <h3>الإشعارات الأخيرة</h3>
                <div id="notifications">جاري التحميل...</div>
            </div>
        </div>
    </div>

    <script>
        function loadData() {
            // تحميل حالة النظام
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('system-status').innerHTML = `
                        <strong>الحالة:</strong> ${data.status} | 
                        <strong>التداول:</strong> ${data.trading_enabled ? 'مفعل' : 'معطل'} | 
                        <strong>الاستراتيجية:</strong> ${data.strategy}
                    `;
                });
            
            // تحميل الإشارات
            fetch('/api/signals')
                .then(response => response.json())
                .then(data => {
                    const signalsDiv = document.getElementById('active-signals');
                    if (data.signals.length === 0) {
                        signalsDiv.innerHTML = '<p>لا توجد إشارات نشطة</p>';
                    } else {
                        signalsDiv.innerHTML = data.signals.map(signal => `
                            <div class="signal-${signal.signal_type.toLowerCase()}">
                                <strong>${signal.symbol}</strong> - ${signal.signal_type}<br>
                                <small>الثقة: ${(signal.confidence * 100).toFixed(1)}% | السعر: ${signal.entry_price}</small>
                            </div>
                        `).join('');
                    }
                });
            
            // تحميل الإشعارات
            fetch('/api/notifications')
                .then(response => response.json())
                .then(data => {
                    const notificationsDiv = document.getElementById('notifications');
                    if (data.notifications.length === 0) {
                        notificationsDiv.innerHTML = '<p>لا توجد إشعارات</p>';
                    } else {
                        notificationsDiv.innerHTML = data.notifications.slice(0, 5).map(notif => `
                            <div class="notification">
                                <strong>${notif.title}</strong><br>
                                <small>${notif.message}</small>
                            </div>
                        `).join('');
                    }
                });
        }
        
        function analyzeSymbol() {
            const symbol = document.getElementById('symbol-input').value.toUpperCase();
            if (!symbol) return;
            
            document.getElementById('analysis-result').innerHTML = 'جاري التحليل...';
            
            fetch(`/api/analyze/${symbol}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('analysis-result').innerHTML = `<p style="color: red;">خطأ: ${data.error}</p>`;
                    } else {
                        document.getElementById('analysis-result').innerHTML = `
                            <div class="signal-${data.signal.toLowerCase()}">
                                <h4>النتيجة: ${data.signal}</h4>
                                <p><strong>الثقة:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                                <p><strong>السعر:</strong> ${data.entry_price}</p>
                                <p><strong>وقف الخسارة:</strong> ${data.stop_loss}</p>
                                <p><strong>جني الأرباح:</strong> ${data.take_profit}</p>
                                <p><strong>تأكيد الحجم:</strong> ${data.volume_confirmation ? 'نعم' : 'لا'}</p>
                            </div>
                        `;
                    }
                });
        }
        
        function toggleTrading() {
            fetch('/api/toggle-trading', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    loadData();
                });
        }
        
        // تحديث البيانات كل 30 ثانية
        setInterval(loadData, 30000);
        
        // تحميل البيانات عند بدء الصفحة
        loadData();
    </script>
</body>
</html>
    """)

def main():
    """الدالة الرئيسية"""
    logger.info("🚀 بدء تشغيل نظام التداول الآلي المحسن...")
    
    # تهيئة بوت التداول
    init_trading_bot()
    
    # قائمة العملات للمسح (يمكن تخصيصها)
    default_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
        'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT'
    ]
    
    def continuous_scanning():
        """مسح مستمر للرموز"""
        while True:
            try:
                if trading_bot and TRADING_ENABLED:
                    logger.info("🔍 بدء مسح الرموز...")
                    results = trading_bot.scan_symbols(default_symbols)
                    
                    if results:
                        logger.info(f"✅ تم العثور على {len(results)} إشارة")
                    else:
                        logger.info("ℹ️ لم يتم العثور على إشارات جديدة")
                
                time.sleep(300)  # مسح كل 5 دقائق
                
            except Exception as e:
                logger.error(f"❌ خطأ في المسح المستمر: {e}")
                time.sleep(60)
    
    # تشغيل المسح في خيط منفصل
    scanning_thread = Thread(target=continuous_scanning, daemon=True)
    scanning_thread.start()
    
    # تشغيل Flask
    logger.info(f"🌐 تشغيل خادم الويب على http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()