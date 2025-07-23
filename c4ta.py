import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from binance.client import Client
from decouple import config
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# --- إعدادات اللوجر ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EmergencyBacktest')


# ==============================================================================
# --- CONFIGURATION - قم بتعديل هذه الإعدادات لاختبار سيناريوهات مختلفة ---
# ==============================================================================
CONFIG = {
    # كم يوماً في الماضي تريد اختباره؟ (فترة أطول تزيد من فرصة إيجاد انهيار)
    "BACKTEST_DAYS": 90,

    # درجة الطوارئ المطلوبة لتفعيل الإنذار (قيمة أقل = أكثر حساسية)
    "TRIGGER_SCORE_THRESHOLD": 60.0,

    # مضاعف حجم التداول الذي يعتبر طارئاً (مثال: 5.0 يعني أن الحجم الحالي 5 أضعاف المتوسط)
    "VOLUME_SPIKE_THRESHOLD": 5.0,

    # الأصول المراقبة، أوزانها في حساب الدرجة، وعتبة الهبوط المئوية لكل منها
    "ASSETS": {
        'BTCUSDT': {'weight': 0.4, 'threshold': -3.0},
        'ETHUSDT': {'weight': 0.3, 'threshold': -4.0},
        'BNBUSDT': {'weight': 0.2, 'threshold': -5.0},
        'SOLUSDT': {'weight': 0.1, 'threshold': -6.0}
    }
}
# ==============================================================================


# --- فئة كاشف الطوارئ المعدلة للاختبار الخلفي ---
class BacktestEmergencyDetector:
    """
    نسخة معدلة من كاشف الطوارئ مصممة للعمل مع بيانات تاريخية (DataFrame)
    بدلاً من استدعاءات API الحية.
    """
    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.emergency_assets = self.config["ASSETS"]
        self.volume_spike_threshold = self.config["VOLUME_SPIKE_THRESHOLD"]

    def get_15m_change(self, candle: pd.Series) -> float:
        """يحسب التغير من شمعة واحدة (DataFrame row)."""
        if pd.isna(candle['open']) or pd.isna(candle['close']) or candle['open'] == 0:
            return 0.0
        return ((candle['close'] - candle['open']) / candle['open']) * 100

    def get_volume_spike(self, current_candle: pd.Series, previous_candles: pd.DataFrame) -> float:
        """يحسب طفرة الحجم مقارنة بالشموع السابقة."""
        if previous_candles.empty or pd.isna(current_candle['volume']):
            return 1.0
        avg_vol = previous_candles['volume'].mean()
        return current_candle['volume'] / avg_vol if avg_vol > 0 else 1.0

    def calculate_emergency_score(self, data_slice: Dict[str, pd.Series], history_slice: Dict[str, pd.DataFrame]) -> Tuple[float, Dict]:
        """
        يحسب درجة الطوارئ بناءً على شريحة من البيانات التاريخية.
        """
        total_score = 0.0
        details = {}

        for symbol, config in self.emergency_assets.items():
            if symbol not in data_slice:
                continue

            current_candle = data_slice[symbol]
            previous_candles = history_slice.get(symbol, pd.DataFrame())

            # 1. حساب مساهمة هبوط السعر
            price_change = self.get_15m_change(current_candle)
            if price_change <= config['threshold']:
                asset_score = abs(price_change / config['threshold']) * config['weight'] * 100
                total_score += asset_score
                details[symbol] = {
                    'type': 'Price Drop',
                    'change_pct': round(price_change, 2),
                    'contribution': round(asset_score, 1)
                }

            # 2. حساب مساهمة طفرة الحجم
            volume_ratio = self.get_volume_spike(current_candle, previous_candles)
            if volume_ratio >= self.volume_spike_threshold:
                vol_score = min((volume_ratio / self.volume_spike_threshold) * 10, 20) * config['weight'] * 2
                total_score += vol_score
                details[f"{symbol}_volume"] = {
                    'type': 'Volume Spike',
                    'ratio': round(volume_ratio, 2),
                    'contribution': round(vol_score, 1)
                }
        
        return min(total_score, 100), details


class EmergencyBacktest:
    def __init__(self, client: Client, config_dict: Dict):
        self.client = client
        self.config = config_dict
        self.detector = BacktestEmergencyDetector(config_dict)
        
    def fetch_historical_data(self, symbol: str, interval: str, start_date_str: str) -> Optional[pd.DataFrame]:
        """يجلب البيانات التاريخية من تاريخ محدد."""
        logger.info(f"⏳ Fetching {interval} data for {symbol} from {start_date_str}...")
        try:
            klines = self.client.get_historical_klines(symbol, interval, start_date_str)
            if not klines:
                logger.warning(f"⚠️ No data returned for {symbol}.")
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df.set_index('timestamp', inplace=True)
            logger.info(f"✅ Fetched {len(df)} records for {symbol}.")
            return df.dropna(subset=numeric_cols)
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            return None
    
    def simulate_emergency_triggers(self) -> Dict:
        """يُحاكي تفعيلات الطوارئ خلال فترة زمنية محددة."""
        days = self.config["BACKTEST_DAYS"]
        results = {
            'backtest_period_days': days,
            'trigger_threshold': self.config["TRIGGER_SCORE_THRESHOLD"],
            'total_triggers': 0,
            'trigger_events': [],
            'max_score': 0.0,
            'daily_scores': []
        }
        
        start_date = datetime.now() - timedelta(days=days)
        start_date_str = start_date.strftime("%d %b, %Y")

        all_data = {}
        for symbol in self.detector.emergency_assets.keys():
            df = self.fetch_historical_data(symbol, '15m', start_date_str)
            if df is not None and not df.empty:
                all_data[symbol] = df
        
        if not all_data:
            logger.critical("No historical data fetched. Aborting backtest.")
            return results
        
        logger.info("Merging and aligning data...")
        merged_df = pd.concat(
            [df.add_prefix(f"{symbol}_") for symbol, df in all_data.items()],
            axis=1
        )
        merged_df.ffill(inplace=True)
        merged_df.dropna(inplace=True)

        logger.info(f"Simulation will run on {len(merged_df)} common timestamps.")

        for timestamp, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Simulating"):
            data_slice, history_slice = {}, {}
            
            for symbol in self.detector.emergency_assets.keys():
                prefix = f"{symbol}_"
                cols = [c for c in row.index if c.startswith(prefix)]
                if not cols: continue
                
                current_candle_data = row[cols]
                current_candle_data.index = [c.replace(prefix, '') for c in current_candle_data.index]
                data_slice[symbol] = pd.Series(current_candle_data)
                
                history_df = all_data[symbol]
                previous_candles = history_df[(history_df.index < timestamp) & (history_df.index >= timestamp - timedelta(hours=2))]
                history_slice[symbol] = previous_candles.tail(7)

            score, details = self.detector.calculate_emergency_score(data_slice, history_slice)
            
            results['daily_scores'].append({'date': timestamp.isoformat(), 'score': score})
            results['max_score'] = max(results['max_score'], score)

            if score >= self.config["TRIGGER_SCORE_THRESHOLD"]:
                results['total_triggers'] += 1
                trigger_event = {'date': timestamp.isoformat(), 'score': score, 'details': details}
                results['trigger_events'].append(trigger_event)
                logger.warning(f"🚨 Trigger! Date: {timestamp.isoformat()}, Score: {score:.1f}")

        return results
    
    def save_results(self, results: Dict, filename: str = 'emergency_backtest_results.json'):
        """يحفظ نتائج الاختبار في ملف JSON."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"✅ Backtest results saved to {filename}")

# --- نقطة انطلاق التشغيل ---
if __name__ == "__main__":
    try:
        api_key = config('BINANCE_API_KEY')
        api_secret = config('BINANCE_API_SECRET')
    except Exception as e:
        logger.critical("❌ لم يتم العثور على مفاتيح API في ملف .env. يرجى إضافتها.")
        exit(1)
        
    client = Client(api_key, api_secret)
    
    backtest = EmergencyBacktest(client, CONFIG)
    results = backtest.simulate_emergency_triggers()
    backtest.save_results(results)
    
    print("\n" + "="*50)
    print("📊 Backtest Results Summary 📊")
    print("="*50)
    print(f"  Backtest Period: {results['backtest_period_days']} days")
    print(f"  Trigger Threshold: {results['trigger_threshold']}")
    print(f"  Total Triggers: {results['total_triggers']}")
    print(f"  Max Score Reached: {results['max_score']:.2f}")
    print("\n--- Trigger Events ---")
    if results['trigger_events']:
        for event in results['trigger_events']:
            print(f"  - Date: {event['date']}, Score: {event['score']:.1f}")
            print(f"    Details: {json.dumps(event['details'], ensure_ascii=False)}")
    else:
        print("  No emergency events were triggered during the backtest period.")
    print("="*50)
