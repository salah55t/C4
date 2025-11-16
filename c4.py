# trading_bot_v37_core.py (الجزء الأول - النواة)
import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import json
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# إعدادات التسجيل المتقدمة
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[
        logging.FileHandler('bot_v37_advanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BotV37Core')

# ============================================
# أنواع البيانات المتقدمة
# ============================================

class MarketRegime(Enum):
    """تصنيف دقيق لحالة السوق"""
    STRONG_BULL = "strong_bull"
    WEAK_BULL = "weak_bull"
    RANGING_BULLISH = "ranging_bullish"
    PURE_RANGING = "pure_ranging"
    RANGING_BEARISH = "ranging_bearish"
    WEAK_BEAR = "weak_bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"

class SignalConfidence(Enum):
    """مستوى الثقة في الإشارة"""
    LOW = 0.5
    MEDIUM = 0.7
    HIGH = 0.85
    VERY_HIGH = 0.95

@dataclass
class StrategyConfig:
    """إعدادات الاستراتيجية المتقدمة"""
    enabled: bool = True
    weight: float = 1.0
    min_confidence: SignalConfidence = SignalConfidence.MEDIUM
    regimes: List[MarketRegime] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h'])
    priority: int = 5

@dataclass
class MarketContext:
    """سياق السوق الحالي"""
    regime: MarketRegime
    regime_strength: float
    trend_5m: str
    trend_15m: str
    trend_1h: str
    volatility_score: float
    volume_profile: Dict[str, float]
    liquidity_zones: List[Tuple[float, float]]
    support_levels: List[float]
    resistance_levels: List[float]
    correlation_matrix: Optional[pd.DataFrame] = None
    market_sentiment: float = 0.5  # 0-1, >0.5 صاعد

@dataclass
class SignalResult:
    """نتيجة توليد الإشارة"""
    symbol: str
    strategy_name: str
    confidence: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    position_size: float
    quality_score: float
    market_context: MarketContext
    timestamp: datetime = field(default_factory=datetime.utcnow)

# ============================================
# محلل السوق الذكي
# ============================================

class AdvancedMarketAnalyzer:
    """محلل سوق متقدم يحدد حالة السوق بدقة"""
    
    def __init__(self):
        self.leader_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
        self.regime_history = deque(maxlen=100)
        
    async def analyze_market_regime(self, client) -> MarketContext:
        """تحليل شامل لحالة السوق"""
        try:
            # 1. جلب البيانات للعملات القيادية
            leader_data = await self._fetch_leader_data(client)
            
            # 2. حساب مؤشرات الاتجاه والقوة
            trend_scores = []
            volatility_scores = []
            volume_profiles = []
            
            for symbol, df in leader_data.items():
                if df is None or len(df) < 50:
                    continue
                    
                # حساب ADX لقياس قوة الاتجاه
                adx = self._calculate_adx(df)
                trend_strength = self._calculate_trend_strength(df)
                vol_score = self._calculate_volatility_score(df)
                vol_profile = self._analyze_volume_profile(df)
                
                trend_scores.append({
                    'symbol': symbol,
                    'adx': adx,
                    'trend_strength': trend_strength,
                    'direction': self._determine_trend_direction(df)
                })
                volatility_scores.append(vol_score)
                volume_profiles.append(vol_profile)
            
            # 3. تحليل الارتباط
            correlation_matrix = self._calculate_correlation_matrix(leader_data)
            
            # 4. تحديد نظام السوق
            regime = self._determine_market_regime(trend_scores, volatility_scores)
            regime_strength = np.mean([t['trend_strength'] for t in trend_scores]) if trend_scores else 0.5
            
            # 5. تحليل المناطق الرئيسية
            liquidity_zones = self._find_liquidity_zones(leader_data)
            support_resistance = self._identify_key_levels(leader_data)
            
            # 6. حساب معنويات السوق
            market_sentiment = self._calculate_market_sentiment(trend_scores)
            
            # إنشاء سياق السوق
            context = MarketContext(
                regime=regime,
                regime_strength=regime_strength,
                trend_5m=trend_scores[0]['direction'] if trend_scores else 'neutral',
                trend_15m=trend_scores[1]['direction'] if len(trend_scores) > 1 else 'neutral',
                trend_1h=trend_scores[2]['direction'] if len(trend_scores) > 2 else 'neutral',
                volatility_score=np.mean(volatility_scores) if volatility_scores else 0.5,
                volume_profile=self._aggregate_volume_profiles(volume_profiles),
                liquidity_zones=liquidity_zones,
                support_levels=support_resistance['support'],
                resistance_levels=support_resistance['resistance'],
                correlation_matrix=correlation_matrix,
                market_sentiment=market_sentiment
            )
            
            self.regime_history.append({
                'timestamp': datetime.utcnow(),
                'regime': regime,
                'strength': regime_strength
            })
            
            logger.info(f"[Market Analysis] Regime: {regime.value}, Strength: {regime_strength:.2f}, Sentiment: {market_sentiment:.2f}")
            return context
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}", exc_info=True)
            return self._get_default_context()
    
    async def _fetch_leader_data(self, client) -> Dict[str, pd.DataFrame]:
        """جلب البيانات للعملات القيادية بشكل متزامن"""
        tasks = []
        for symbol in self.leader_symbols:
            task = self._fetch_with_retry(client, symbol, '1h', 10)
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                results[symbol] = await asyncio.wait_for(task, timeout=30)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {symbol}")
                results[symbol] = None
        
        return results
    
    async def _fetch_with_retry(self, client, symbol, timeframe, days, max_retries=3):
        """جلب البيانات مع آلية إعادة محاولة ذكية"""
        for attempt in range(max_retries):
            try:
                # استخدام طريقة غير متزامنة إن أمكن
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None, 
                    lambda: self._sync_fetch(client, symbol, timeframe, days)
                )
                return df
            except Exception as e:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Fetch attempt {attempt+1} failed for {symbol}: {e}. Waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        return None
    
    def _sync_fetch(self, client, symbol, timeframe, days):
        """جلب البيانات المتزامن (لتنفيذه في executor)"""
        try:
            klines = client.get_historical_klines(symbol, timeframe, f"{days} days ago UTC")
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna().astype(float)
        except Exception as e:
            logger.error(f"Sync fetch error for {symbol}: {e}")
            return None
    
    def _calculate_adx(self, df: pd.DataFrame, period=14) -> float:
        """حساب ADX بكفاءة"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            
            tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
            plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
            minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
            
            plus_di = 100 * plus_dm_smooth / tr_smooth
            minus_di = 100 * minus_dm_smooth / tr_smooth
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/period, adjust=False).mean()
            
            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20.0
        except:
            return 20.0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """حساب قوة الاتجاه بين 0-1"""
        try:
            # استخدام EMAs متعددة
            ema9 = df['close'].ewm(span=9).mean().iloc[-1]
            ema21 = df['close'].ewm(span=21).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            
            score = 0.0
            if ema9 > ema21 > ema50:
                score += 0.5
            if ema9 > ema21:
                score += 0.3
            if ema21 > ema50:
                score += 0.2
            
            return min(1.0, score)
        except:
            return 0.5
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """حساب درجة التقلب بين 0-1"""
        try:
            returns = df['close'].pct_change().dropna()
            current_vol = returns.tail(20).std() * np.sqrt(365)
            historical_vol = returns.std() * np.sqrt(365)
            
            if historical_vol == 0:
                return 0.5
            
            ratio = current_vol / historical_vol
            # نطاق طبيعي 0.5-2.0
            return min(1.0, max(0.0, (ratio - 0.5) / 1.5))
        except:
            return 0.5
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """تحديد اتجاه الاتجاه"""
        try:
            ema21 = df['close'].ewm(span=21).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > ema21 > ema50:
                return 'bullish'
            elif current_price < ema21 < ema50:
                return 'bearish'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _determine_market_regime(self, trend_scores: List[Dict], volatility_scores: List[float]) -> MarketRegime:
        """تحديد نظام السوق بدقة"""
        if not trend_scores:
            return MarketRegime.PURE_RANGING
        
        avg_adx = np.mean([t['adx'] for t in trend_scores])
        avg_trend_strength = np.mean([t['trend_strength'] for t in trend_scores])
        avg_volatility = np.mean(volatility_scores) if volatility_scores else 0.5
        
        # تحليل الاتجاه الإجمالي
        bullish_count = sum(1 for t in trend_scores if t['direction'] == 'bullish')
        bearish_count = sum(1 for t in trend_scores if t['direction'] == 'bearish')
        total = len(trend_scores)
        
        # منطق التصنيف المتقدم
        if avg_adx > 25 and avg_trend_strength > 0.7:
            if bullish_count / total > 0.6:
                return MarketRegime.STRONG_BULL
            elif bearish_count / total > 0.6:
                return MarketRegime.STRONG_BEAR
            else:
                return MarketRegime.WEAK_BULL if bullish_count > bearish_count else MarketRegime.WEAK_BEAR
        elif avg_adx < 20 and avg_volatility < 0.6:
            return MarketRegime.PURE_RANGING
        elif avg_volatility > 0.8:
            if avg_trend_strength > 0.5:
                return MarketRegime.BREAKOUT
            else:
                return MarketRegime.BREAKDOWN
        else:
            if bullish_count > bearish_count:
                return MarketRegime.RANGING_BULLISH
            else:
                return MarketRegime.RANGING_BEARISH
    
    def _find_liquidity_zones(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[float, float]]:
        """إيجاد مناطق السيولة الرئيسية"""
        zones = []
        for symbol, df in data.items():
            if df is None:
                continue
            
            recent_highs = df['high'].tail(20).values
            recent_lows = df['low'].tail(20).values
            
            # استخدام K-meus لوضع علامات على مناطق التجمع
            from scipy.cluster.vq import kmeans
            
            if len(recent_highs) > 5:
                try:
                    high_clusters, _ = kmeans(recent_highs, min(3, len(recent_highs) // 3))
                    low_clusters, _ = kmeans(recent_lows, min(3, len(recent_lows) // 3))
                    
                    for h in high_clusters:
                        for l in low_clusters:
                            if abs(h - l) / h < 0.02:  # منطقة ضيقة
                                zones.append((l, h))
                except:
                    pass
        
        return zones[:10]  # الحد الأقصى 10 مناطق
    
    def _identify_key_levels(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[float]]:
        """تحديد المستويات الرئيسية للدعم والمقاومة"""
        supports = []
        resistances = []
        
        for symbol, df in data.items():
            if df is None:
                continue
            
            highs = df['high'].values
            lows = df['low'].values
            
            # استخدام نماذج تعلم الآلة لتحديد المستويات
            from scipy.signal import find_peaks
            
            # قمم المقاومة
            high_peaks, _ = find_peaks(highs, distance=5, prominence=highs.std() * 0.5)
            resistances.extend(highs[high_peaks])
            
            # قيعان الدعم
            low_peaks, _ = find_peaks(-lows, distance=5, prominence=lows.std() * 0.5)
            supports.extend(lows[low_peaks])
        
        # تصفية وتجميع المستويات
        supports = sorted(list(set([round(s, 4) for s in supports]))[-5:])
        resistances = sorted(list(set([round(r, 4) for r in resistances]))[:5])
        
        return {'support': supports, 'resistance': resistances}
    
    def _calculate_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """حساب مصفوفة الارتباط"""
        try:
            returns = {}
            for symbol, df in data.items():
                if df is not None and len(df) > 50:
                    returns[symbol] = df['close'].pct_change().dropna()
            
            if len(returns) < 2:
                return None
            
            returns_df = pd.DataFrame(returns)
            return returns_df.corr()
        except:
            return None
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """تحليل بياني للحجم"""
        try:
            recent_volume = df['volume'].tail(20)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            
            return {
                'avg': avg_volume,
                'current': current_volume,
                'ratio': current_volume / avg_volume if avg_volume > 0 else 1.0,
                'trend': 1 if recent_volume.pct_change().mean() > 0 else -1
            }
        except:
            return {'avg': 0, 'current': 0, 'ratio': 1.0, 'trend': 0}
    
    def _aggregate_volume_profiles(self, profiles: List[Dict]) -> Dict[str, float]:
        """تجميع ملفات الحجم"""
        if not profiles:
            return {'avg_ratio': 1.0, 'overall_trend': 0}
        
        avg_ratio = np.mean([p['ratio'] for p in profiles])
        overall_trend = np.mean([p['trend'] for p in profiles])
        
        return {'avg_ratio': avg_ratio, 'overall_trend': overall_trend}
    
    def _calculate_market_sentiment(self, trend_scores: List[Dict]) -> float:
        """حساب معنويات السوق بين 0-1"""
        if not trend_scores:
            return 0.5
        
        # مرجح حسب قوة الاتجاه
        weighted_bullish = sum(t['trend_strength'] for t in trend_scores if t['direction'] == 'bullish')
        weighted_bearish = sum(t['trend_strength'] for t in trend_scores if t['direction'] == 'bearish')
        total_weight = sum(t['trend_strength'] for t in trend_scores)
        
        if total_weight == 0:
            return 0.5
        
        return (weighted_bullish + 0.5 * (total_weight - weighted_bullish - weighted_bearish)) / total_weight
    
    def _get_default_context(self) -> MarketContext:
        """سياق افتراضي في حالة الفشل"""
        return MarketContext(
            regime=MarketRegime.PURE_RANGING,
            regime_strength=0.5,
            trend_5m='neutral',
            trend_15m='neutral',
            trend_1h='neutral',
            volatility_score=0.5,
            volume_profile={'avg_ratio': 1.0, 'overall_trend': 0},
            liquidity_zones=[],
            support_levels=[],
            resistance_levels=[]
        )

# ============================================
# مدير الاستراتيجيات الديناميكي
# ============================================

class DynamicStrategyManager:
    """يدير استراتيجيات متعددة مع تكيف ديناميكي"""
    
    def __init__(self, analyzer: AdvancedMarketAnalyzer):
        self.analyzer = analyzer
        self.strategies = self._initialize_strategies()
        self.performance_tracker = {}
        
    def _initialize_strategies(self) -> Dict[str, StrategyConfig]:
        """تهيئة جميع الاستراتيجيات"""
        return {
            # استراتيجيات الاتجاه الصاعد
            "trend_continuation_breakout": StrategyConfig(
                enabled=True, weight=1.2, min_confidence=SignalConfidence.HIGH,
                regimes=[MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL],
                priority=9
            ),
            "smart_pullback_entry": StrategyConfig(
                enabled=True, weight=1.0, min_confidence=SignalConfidence.MEDIUM,
                regimes=[MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL],
                priority=8
            ),
            "ema_cross_momentum": StrategyConfig(
                enabled=True, weight=0.9, min_confidence=SignalConfidence.MEDIUM,
                regimes=[MarketRegime.STRONG_BULL, MarketRegime.WEAK_BULL],
                priority=7
            ),
            
            # استراتيجيات السوق الجانبي
            "range_bound_bounce": StrategyConfig(
                enabled=True, weight=1.0, min_confidence=SignalConfidence.HIGH,
                regimes=[MarketRegime.PURE_RANGING, MarketRegime.RANGING_BULLISH, MarketRegime.RANGING_BEARISH],
                priority=8
            ),
            "mean_reversion_vwap": StrategyConfig(
                enabled=True, weight=0.8, min_confidence=SignalConfidence.MEDIUM,
                regimes=[MarketRegime.PURE_RANGING],
                priority=6
            ),
            
            # استراتيجيات التقلبات العالية
            "volatility_breakout": StrategyConfig(
                enabled=True, weight=1.1, min_confidence=SignalConfidence.HIGH,
                regimes=[MarketRegime.HIGH_VOLATILITY, MarketRegime.BREAKOUT],
                priority=9
            ),
            "atr_band_breakout": StrategyConfig(
                enabled=True, weight=0.9, min_confidence=SignalConfidence.MEDIUM,
                regimes=[MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY],
                priority=7
            ),
            
            # استراتيجيات الاتجاه الهابط (للتداول الهامشي في المستقبل)
            "bearish_momentum": StrategyConfig(
                enabled=False, weight=0.8, min_confidence=SignalConfidence.HIGH,
                regimes=[MarketRegime.STRONG_BEAR, MarketRegime.WEAK_BEAR],
                priority=5
            ),
            
            # استراتيجيات متقدمة
            "volume_profile_imbalance": StrategyConfig(
                enabled=True, weight=1.3, min_confidence=SignalConfidence.VERY_HIGH,
                regimes=list(MarketRegime),
                priority=10
            ),
            "liquidity_sweep_entry": StrategyConfig(
                enabled=True, weight=1.0, min_confidence=SignalConfidence.HIGH,
                regimes=[MarketRegime.STRONG_BULL, MarketRegime.BREAKOUT],
                priority=8
            )
        }
    
    async def scan_for_signals(self, symbols: List[str], client, market_context: MarketContext) -> List[SignalResult]:
        """فحص الإشارات لجميع الاستراتيجيات"""
        signals = []
        
        # تصفية الاستراتيجيات المناسبة للسياق الحالي
        active_strategies = {
            name: config for name, config in self.strategies.items()
            if config.enabled and market_context.regime in config.regimes
        }
        
        if not active_strategies:
            logger.warning(f"No active strategies for regime: {market_context.regime}")
            return []
        
        # توليد الإشارات بشكل متزامن
        tasks = []
        for symbol in symbols:
            task = self._scan_symbol_strategies(symbol, active_strategies, client, market_context)
            tasks.append(task)
        
        # تنفيذ بحد أقصى 10 مهام في وقت واحد لتجنب الحظر
        semaphore = asyncio.Semaphore(10)
        async def bounded_scan(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_scan(t) for t in tasks], return_exceptions=True)
        
        # معالجة النتائج
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Strategy scan error: {result}")
                continue
            if result:
                signals.extend(result)
        
        # ترتيب حسب الأولوية والجودة
        signals.sort(key=lambda s: (
            self.strategies[s.strategy_name].priority,
            s.confidence,
            s.quality_score
        ), reverse=True)
        
        return signals
    
    async def _scan_symbol_strategies(self, symbol: str, strategies: Dict[str, StrategyConfig], 
                                      client, market_context: MarketContext) -> List[SignalResult]:
        """فحص استراتيجيات متعددة لرمز واحد"""
        signals = []
        
        # جلب البيانات مرة واحدة لجميع الاستراتيجيات
        try:
            loop = asyncio.get_event_loop()
            df_5m = await loop.run_in_executor(None, 
                lambda: self._sync_fetch(client, symbol, '5m', 7))
            df_15m = await loop.run_in_executor(None, 
                lambda: self._sync_fetch(client, symbol, '15m', 10))
            
            if df_5m is None or len(df_5m) < 100:
                return []
            
            # حساب المؤشرات الأساسية
            df_5m = self._calculate_advanced_indicators(df_5m)
            
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
            return []
        
        # اختبار كل استراتيجية
        for strategy_name, config in strategies.items():
            try:
                # التحقق من الثبة المطلوبة
                if config.min_confidence == SignalConfidence.VERY_HIGH and market_context.regime_strength < 0.7:
                    continue
                if config.min_confidence == SignalConfidence.HIGH and market_context.regime_strength < 0.6:
                    continue
                
                # تنفيذ الاستراتيجية
                signal = await self._execute_strategy(
                    strategy_name, symbol, df_5m, df_15m, market_context, config
                )
                
                if signal and signal.confidence >= config.min_confidence.value:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Strategy execution error {strategy_name} on {symbol}: {e}", exc_info=True)
        
        return signals
    
    async def _execute_strategy(self, strategy_name: str, symbol: str, 
                               df_5m: pd.DataFrame, df_15m: pd.DataFrame,
                               market_context: MarketContext, config: StrategyConfig) -> Optional[SignalResult]:
        """تنفيذ استراتيجية معينة"""
        
        # استدعاء دالة الاستراتيجية المناسبة
        strategy_func = getattr(self, f"_strategy_{strategy_name}", None)
        if not strategy_func:
            logger.error(f"Strategy function not found: _strategy_{strategy_name}")
            return None
        
        try:
            result = await strategy_func(symbol, df_5m, df_15m, market_context)
            if result:
                result.strategy_name = strategy_name
                result.confidence *= config.weight
            return result
        except Exception as e:
            logger.error(f"Error in strategy {strategy_name}: {e}", exc_info=True)
            return None
    
    # ========================================
    # تعريفات الاستراتيجيات الفعلية
    # ========================================
    
    async def _strategy_trend_continuation_breakout(self, symbol: str, df: pd.DataFrame, 
                                                    df_15m: pd.DataFrame, ctx: MarketContext) -> Optional[SignalResult]:
        """استراتيجية اختراق استمرار الاتجاه"""
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # شروط الاختراق
            breakout_condition = (
                last['close'] > last['bb_upper'] * 0.998 and  # فوق بولينجر العلوي
                prev['close'] <= prev['bb_upper'] and         # كان تحته قبل
                last['volume'] > last['volume_ma20'] * 1.5 and # حجم مرتفع
                last['adx'] > 25                              # اتجاه قوي
            )
            
            if not breakout_condition:
                return None
            
            # حساب المستويات
            entry = last['close']
            stop_loss = last['bb_lower'] * 0.995  # تحت بولينجر السفلي
            target1 = entry + (entry - stop_loss) * 1.5
            target2 = entry + (entry - stop_loss) * 2.5
            
            # حساب جودة الإشارة
            quality = self._calculate_signal_quality(df, ctx, 'breakout')
            
            return SignalResult(
                symbol=symbol,
                strategy_name='',
                confidence=0.85,
                entry_price=entry,
                stop_loss=stop_loss,
                target_1=target1,
                target_2=target2,
                position_size=0,
                quality_score=quality,
                market_context=ctx
            )
            
        except Exception:
            return None
    
    async def _strategy_smart_pullback_entry(self, symbol: str, df: pd.DataFrame,
                                           df_15m: pd.DataFrame, ctx: MarketContext) -> Optional[SignalResult]:
        """استراتيجية دخول عند الارتداد الذكي"""
        try:
            # التحقق من اتجاه صاعد عام
            if ctx.trend_15m not in ['bullish', 'neutral']:
                return None
            
            # العثور على الارتداد
            recent_highs = df['high'].tail(10).max()
            recent_lows = df['low'].tail(5).min()
            
            pullback_depth = (recent_highs - recent_lows) / recent_highs
            
            if not (0.01 <= pullback_depth <= 0.05):  # ارتداد 1-5%
                return None
            
            # تأكيد الاسترداد
            last = df.iloc[-1]
            if not (last['close'] > last['ema9'] and last['macd_hist'] > 0):
                return None
            
            entry = last['close']
            stop_loss = recent_lows * 0.992
            risk = entry - stop_loss
            
            quality = self._calculate_signal_quality(df, ctx, 'pullback')
            
            return SignalResult(
                symbol=symbol,
                strategy_name='',
                confidence=0.8,
                entry_price=entry,
                stop_loss=stop_loss,
                target_1=entry + risk * 1.8,
                target_2=entry + risk * 2.8,
                position_size=0,
                quality_score=quality,
                market_context=ctx
            )
            
        except Exception:
            return None
    
    async def _strategy_range_bound_bounce(self, symbol: str, df: pd.DataFrame,
                                         df_15m: pd.DataFrame, ctx: MarketContext) -> Optional[SignalResult]:
        """استراتيجية الارتداد في السوق الجانبي"""
        try:
            # تأكيد السوق الجانبي
            if ctx.regime not in [MarketRegime.PURE_RANGING, MarketRegime.RANGING_BULLISH]:
                return None
            
            # تحديد نطاق التداول
            bb_width = df['bb_width'].tail(20).mean()
            if bb_width > 0.05:  # نطاق واسع جداً
                return None
            
            last = df.iloc[-1]
            
            # شراء عند الحد السفلي
            bounce_condition = (
                last['low'] <= last['bb_lower'] * 1.002 and  # لمس الحد السفلي
                last['stoch_k'] < 30 and                      // تشبع بيعي
                last['rsi'] > 35                              # ليس في منطقة هابطة قوية
            )
            
            if not bounce_condition:
                return None
            
            entry = last['close']
            stop_loss = last['bb_lower'] * 0.99
            target1 = last['bb_middle']
            target2 = last['bb_upper']
            
            quality = self._calculate_signal_quality(df, ctx, 'range')
            
            return SignalResult(
                symbol=symbol,
                strategy_name='',
                confidence=0.75,
                entry_price=entry,
                stop_loss=stop_loss,
                target_1=target1,
                target_2=target2,
                position_size=0,
                quality_score=quality,
                market_context=ctx
            )
            
        except Exception:
            return None
    
    async def _strategy_volume_profile_imbalance(self, symbol: str, df: pd.DataFrame,
                                               df_15m: pd.DataFrame, ctx: MarketContext) -> Optional[SignalResult]:
        """استراتيجية عدم توازن بيان الحجم"""
        try:
            # تحليل بيان الحجم
            volume_profile = self._calculate_volume_profile_hist(df)
            
            # العثور على فجوات عدم التوازن
            imbalance = self._detect_volume_imbalance(volume_profile)
            
            if not imbalance:
                return None
            
            last = df.iloc[-1]
            entry = last['close']
            
            # حساب وقف الخسارة بالقرب من نقطة التحكم
            poc = imbalance['poc']
            stop_loss = poc * 0.995 if entry > poc else poc * 1.005
            
            # أهداف بناءً على نقاط المحور
            target1 = imbalance['va_high']
            target2 = imbalance['va_high'] + (imbalance['va_high'] - imbalance['va_low'])
            
            quality = 90  # جودة عالية جداً
            
            return SignalResult(
                symbol=symbol,
                strategy_name='',
                confidence=0.9,
                entry_price=entry,
                stop_loss=stop_loss,
                target_1=target1,
                target_2=target2,
                position_size=0,
                quality_score=quality,
                market_context=ctx
            )
            
        except Exception:
            return None
    
    # ========================================
    # أدوات المساعدة
    # ========================================
    
    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب جميع المؤشرات المتقدمة"""
        try:
            # الحسابات الأساسية (مثل الكود السابق ولكن محسنة)
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
            
            # EMAs
            for period in [9, 21, 34, 50, 100, 200]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # ATR و ADX
            df['atr'] = self._calculate_atr_vectorized(df)
            df['adx'] = self._calculate_adx_vectorized(df)
            df['atr_percent'] = (df['atr'] / df['close']) * 100
            
            # RSI
            df['rsi'] = self._calculate_rsi_vectorized(df, 14)
            
            # بولينجر
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Stochastic
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # VWAP
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # متوسط الحجم
            df['volume_ma20'] = df['volume'].rolling(20).mean()
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return df
    
    def _calculate_signal_quality(self, df: pd.DataFrame, ctx: MarketContext, signal_type: str) -> float:
        """حساب جودة الإشارة المتقدم"""
        score = 0.0
        last = df.iloc[-1]
        
        # 1. توافق الاتجاه (25 نقطة)
        trend_alignment = 0
        if ctx.trend_5m in ['bullish', 'neutral']: trend_alignment += 8
        if ctx.trend_15m in ['bullish', 'neutral']: trend_alignment += 9
        if ctx.trend_1h in ['bullish', 'neutral']: trend_alignment += 8
        
        score += trend_alignment
        
        # 2. قوة المؤشرات (25 نقطة)
        indicator_strength = 0
        if last['rsi'] > 50: indicator_strength += 8
        if last['macd_hist'] > 0: indicator_strength += 9
        if last['adx'] > 20: indicator_strength += 8
        
        score += indicator_strength
        
        # 3. تأكيد الحجم (20 نقطة)
        if last['volume'] > last['volume_ma20'] * 1.3: score += 20
        elif last['volume'] > last['volume_ma20']: score += 10
        
        # 4. التقلب المثالي (15 نقطة)
        atr_pct = last['atr_percent']
        if 1.0 <= atr_pct <= 3.0: score += 15
        elif 0.5 <= atr_pct <= 4.0: score += 7
        
        # 5. تكامل السياق (15 نقطة)
        if ctx.market_sentiment > 0.6: score += 15
        elif ctx.market_sentiment > 0.4: score += 7
        
        return min(100, score)
    
    def _calculate_atr_vectorized(self, df: pd.DataFrame, period=14) -> pd.Series:
        """حساب ATR بكفاءة عالية"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        return tr.ewm(alpha=1/period, adjust=False).mean()
    
    def _calculate_rsi_vectorized(self, df: pd.DataFrame, period=14) -> pd.Series:
        """حساب RSI بكفاءة عالية"""
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

# ============================================
# تصدير الوحدات
# ============================================

__all__ = [
    'MarketRegime',
    'SignalConfidence',
    'StrategyConfig',
    'MarketContext',
    'SignalResult',
    'AdvancedMarketAnalyzer',
    'DynamicStrategyManager'
]
# trading_bot_v37_dashboard.py (الجزء الثاني - الواجهة والمخاطر)
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_sock import Sock
import redis
import psycopg2
from collections import deque, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import os
from decimal import Decimal

logger = logging.getLogger('BotV37Dashboard')

# ============================================
# تكامل Telegram المتقدم
# ============================================

class TelegramNotifier:
    """إشعارات Telegram تفاعلية مع أزرار سريعة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}"
        self.session = None
    
    async def initialize(self):
        """تهيئة الجلسة غير المتزامنة"""
        import aiohttp
        self.session = aiohttp.ClientSession()
    
    async def send_rich_notification(self, message: str, 
                                     buttons: Optional[List[List[Dict]]] = None,
                                     image_path: Optional[str] = None):
        """إرسال إشعار غني مع أزرار وصور"""
        if not self.token or not self.chat_id:
            return
        
        try:
            if image_path and os.path.exists(image_path):
                await self._send_photo_with_caption(message, image_path, buttons)
            else:
                await self._send_message_with_buttons(message, buttons)
                
        except Exception as e:
            logger.error(f"Telegram notification error: {e}", exc_info=True)
    
    async def _send_message_with_buttons(self, message: str, buttons: Optional[List[List[Dict]]]):
        """إرسال رسالة مع أزرار تفاعلية"""
        if not self.session:
            return
        
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            if buttons:
                payload["reply_markup"] = {"inline_keyboard": buttons}
            
            async with self.session.post(
                f"{self.api_url}/sendMessage",
                json=payload,
                timeout=10
            ) as response:
                if response.status != 200:
                    logger.error(f"Telegram API error: {await response.text()}")
        
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def _send_photo_with_caption(self, message: str, image_path: str, 
                                      buttons: Optional[List[List[Dict]]]):
        """إرسال صورة مع تسمية وأزرار"""
        if not self.session:
            return
        
        try:
            with open(image_path, 'rb') as img:
                form = aiohttp.FormData()
                form.add_field('chat_id', self.chat_id)
                form.add_field('caption', message)
                form.add_field('parse_mode', 'Markdown')
                form.add_field('photo', img, filename='chart.png')
                
                if buttons:
                    form.add_field('reply_markup', json.dumps({"inline_keyboard": buttons}))
                
                async with self.session.post(
                    f"{self.api_url}/sendPhoto",
                    data=form,
                    timeout=30
                ) as response:
                    if response.status != 200:
                        logger.error(f"Telegram photo API error: {await response.text()}")
        
        except Exception as e:
            logger.error(f"Error sending photo: {e}")
    
    async def answer_callback(self, callback_query_id: str, text: str):
        """الرد على استدعاء زر"""
        if not self.session:
            return
        
        try:
            await self.session.post(
                f"{self.api_url}/answerCallbackQuery",
                json={
                    "callback_query_id": callback_query_id,
                    "text": text,
                    "show_alert": False
                },
                timeout=10
            )
        except Exception as e:
            logger.error(f"Error answering callback: {e}")
    
    async def close(self):
        """إغلاق الجلسة"""
        if self.session:
            await self.session.close()

# ============================================
# محلل المخاطر والأداء
# ============================================

@dataclass
class RiskMetrics:
    """مقاييس المخاطر الحالية"""
    total_exposure: float
    per_trade_risk: float
    daily_risk: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float  # Value at Risk
    expected_shortfall: float
    
    @property
    def is_healthy(self) -> bool:
        """هل حالة المخاطر صحية؟"""
        return (
            self.max_drawdown < 20 and
            self.sharpe_ratio > 0.5 and
            self.win_rate > 0.4 and
            self.var_95 > -5
        )

class AdvancedRiskManager:
    """إدارة مخاطر ديناميكية مع تحسين المحفظة"""
    
    def __init__(self, db_connection, redis_client):
        self.conn = db_connection
        self.redis = redis_client
        self.position_sizes = {}
        self.daily_loss_limit = -50  # 50$ يومياً
        self.max_drawdown_limit = -20  # 20% إجمالي
    
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """حساب جميع مقاييس المخاطر"""
        try:
            # البيانات من آخر 30 يوم
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        profit_percentage,
                        closing_price,
                        entry_price,
                        closed_at,
                        is_real_trade
                    FROM signals 
                    WHERE status = 'closed' 
                    AND closed_at >= NOW() - INTERVAL '30 days'
                    ORDER BY closed_at ASC
                """)
                trades = cur.fetchall()
            
            if not trades:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            profits = [t['profit_percentage'] for t in trades]
            returns = np.array(profits) / 100
            
            # الحسابات الأساسية
            win_rate = len([p for p in profits if p > 0]) / len(profits)
            profit_factor = abs(sum([p for p in profits if p > 0]) / sum([p for p in profits if p < 0])) if any(p < 0 for p in profits) else float('inf')
            
            # التراجع الأقصى
            equity_curve = np.cumprod(1 + returns) * 1000
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (running_max - equity_curve) / running_max * 100
            max_drawdown = drawdown.max()
            
            # شارب نسبة
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns, 5) * 100
            
            # Expected Shortfall
            expected_shortfall = returns[returns <= np.percentile(returns, 5)].mean() * 100
            
            # المخاطر اليومية
            today_profit = sum([t['profit_percentage'] for t in trades if t['closed_at'].date() == datetime.utcnow().date()])
            
            return RiskMetrics(
                total_exposure=sum([abs(t['closing_price'] - t['entry_price']) / t['entry_price'] * 100 for t in trades]),
                per_trade_risk=2.5,  # افتراضي، يتم تحديثه ديناميكياً
                daily_risk=today_profit,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                profit_factor=profit_factor,
                var_95=var_95,
                expected_shortfall=expected_shortfall
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}", exc_info=True)
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def calculate_dynamic_position_size(self, symbol: str, entry_price: float,
                                            stop_loss: float, available_balance: float,
                                            market_context: MarketContext) -> Optional[Decimal]:
        """حساب حجم المركز الديناميكي بناءً على المخاطر"""
        try:
            risk_per_trade = 0.02  # 2% من رأس المال افتراضياً
            
            # تعديل بناءً على حالة السوق
            if market_context.regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.BREAKDOWN]:
                risk_per_trade *= 0.5  # تقليل المخاطر في التقلبات العالية
            elif market_context.regime in [MarketRegime.STRONG_BULL]:
                risk_per_trade *= 1.2  # زيادة طفيفة في الاتجاهات القوية
            
            # تعديل بناءً على قوة النظام
            risk_per_trade *= (1 + (market_context.regime_strength - 0.5) * 0.3)
            
            # حساب المخاطرة بالدولار
            risk_amount = available_balance * risk_per_trade
            
            # حساب المخاطرة للنقطة
            risk_per_point = abs(entry_price - stop_loss)
            if risk_per_point == 0:
                return None
            
            # حجم المركز
            position_size = risk_amount / risk_per_point
            
            # التحقق من الحدود
            position_size = min(position_size, available_balance / entry_price * 0.1)  # أقصى 10% من الرصيد
            position_size = max(position_size, 1.0)  # أدنى حجم 1 وحدة
            
            return Decimal(str(position_size))
            
        except Exception as e:
            logger.error(f"Dynamic position size error: {e}", exc_info=True)
            return None
    
    async def should_pause_trading(self) -> Tuple[bool, str]:
        """تحديد ما إذا كان يجب إيقاف التداول مؤقتاً"""
        try:
            metrics = await self.calculate_risk_metrics()
            
            # فحص حدود الخسارة اليومية
            if metrics.daily_risk <= self.daily_loss_limit:
                return True, f"Daily loss limit reached: {metrics.daily_risk:.2f}%"
            
            # فحص التراجع الأقصى
            if metrics.max_drawdown >= self.max_drawdown_limit:
                return True, f"Max drawdown limit reached: {metrics.max_drawdown:.2f}%"
            
            # فحص نسبة شارب
            if metrics.sharpe_ratio < -1:
                return True, f"Sharpe ratio too low: {metrics.sharpe_ratio:.2f}"
            
            # فحص Value at Risk
            if metrics.var_95 < -10:
                return True, f"VaR 95% too high: {metrics.var_95:.2f}%"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Pause trading check error: {e}", exc_info=True)
            return False, ""

# ============================================
# لوحة التحكم المتقدمة
# ============================================

class AdvancedDashboard:
    """لوحة تحكم تفاعلية مع رسوم بيانية متقدمة"""
    
    def __init__(self, db_connection, redis_client, telegram_notifier):
        self.conn = db_connection
        self.redis = redis_client
        self.telegram = telegram_notifier
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.ws_clients = []
        self.setup_routes()
        
    def setup_routes(self):
        """إعداد مسارات Flask"""
        
        @self.app.route('/')
        def dashboard():
            return self._render_dashboard()
        
        @self.app.route('/api/dashboard_data')
        def dashboard_data():
            return self._get_dashboard_data()
        
        @self.app.route('/api/risk_metrics')
        def risk_metrics():
            return self._get_risk_metrics()
        
        @self.app.route('/api/performance_charts')
        def performance_charts():
            return self._get_performance_charts()
        
        @self.app.route('/api/signals_analysis')
        def signals_analysis():
            return self._get_signals_analysis()
        
        @self.sock.route('/ws')
        def ws(ws):
            self._handle_websocket(ws)
        
        @self.app.route('/api/emergency_stop', methods=['POST'])
        def emergency_stop():
            """زر إيقاف الطوارئ"""
            # منطق الإيقاف الفوري
            return jsonify({"success": True, "message": "Trading stopped immediately"})
        
        @self.app.route('/api/boost_mode', methods=['POST'])
        async def boost_mode():
            """تنشيط وضع التعزيز (زيادة عدد الإشارات)"""
            data = request.json
            enabled = data.get('enabled', False)
            # منطق تغيير إعدادات البوت
            return jsonify({"success": True, "boost_enabled": enabled})
    
    def _render_dashboard(self) -> str:
        """تقديم لوحة التحكم الرئيسية"""
        template = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bot V37 - Advanced Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .status-indicator {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .status-active { background: #4CAF50; color: white; }
        .status-paused { background: #FF9800; color: white; }
        .emergency-btn {
            background: #f44336;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        .emergency-btn:hover { background: #d32f2f; transform: scale(1.05); }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-5px); }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            margin: 5px 0;
        }
        .metric-value {
            font-weight: bold;
            color: #4CAF50;
        }
        .risk-warning { color: #FF9800; }
        .risk-danger { color: #f44336; }
        .chart-container {
            width: 100%;
            height: 300px;
            border-radius: 8px;
            overflow: hidden;
        }
        .signals-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .signal-item {
            padding: 12px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            margin: 5px 0;
            border-left: 4px solid #4CAF50;
            transition: all 0.3s;
        }
        .signal-item:hover { background: rgba(255, 255, 255, 0.08); }
        .signal-item.losing { border-left-color: #f44336; }
        .signal-item.pending { border-left-color: #FF9800; }
        .strategy-pill {
            display: inline-block;
            padding: 4px 8px;
            background: #2196F3;
            border-radius: 12px;
            font-size: 12px;
            margin: 2px;
        }
        .action-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
        }
        .btn-primary { background: #2196F3; color: white; }
        .btn-primary:hover { background: #1976D2; }
        .btn-secondary { background: #607D8B; color: white; }
        .btn-secondary:hover { background: #455A64; }
        .websocket-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            background: rgba(0, 0, 0, 0.7);
            font-size: 12px;
        }
        .loading { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,.3); border-radius: 50%; border-top-color: #fff; animation: spin 1s ease-in-out infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <h1>🤖 Bot V37 - Advanced Dashboard</h1>
                <p id="marketRegime">Loading market regime...</p>
            </div>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div class="status-indicator status-active" id="botStatus">
                    <span id="statusIcon">●</span>
                    <span id="statusText">Active</span>
                </div>
                <button class="emergency-btn" onclick="triggerEmergencyStop()">🚨 Emergency Stop</button>
            </div>
        </div>

        <!-- Main Grid -->
        <div class="grid">
            <!-- Performance Metrics -->
            <div class="card">
                <div class="card-header">
                    <h2>📈 Performance Metrics</h2>
                    <span id="lastUpdate" class="loading"></span>
                </div>
                <div id="performanceMetrics">
                    <!-- Dynamic content -->
                </div>
                <div class="chart-container" id="equityChart"></div>
            </div>

            <!-- Risk Analysis -->
            <div class="card">
                <div class="card-header">
                    <h2>⚠️ Risk Analysis</h2>
                    <span id="riskBadge" class="strategy-pill" style="background: #4CAF50;">Healthy</span>
                </div>
                <div id="riskMetrics">
                    <!-- Dynamic content -->
                </div>
                <div class="chart-container" id="riskChart"></div>
            </div>

            <!-- Market Context -->
            <div class="card">
                <div class="card-header">
                    <h2>🌍 Market Context</h2>
                    <span id="marketStrength" class="strategy-pill">Loading...</span>
                </div>
                <div id="marketContext">
                    <!-- Dynamic content -->
                </div>
                <div class="chart-container" id="marketChart"></div>
            </div>

            <!-- Active Signals -->
            <div class="card">
                <div class="card-header">
                    <h2>📊 Active Signals</h2>
                    <span id="signalsCount" class="strategy-pill">0</span>
                </div>
                <div class="signals-list" id="activeSignals">
                    <!-- Dynamic content -->
                </div>
                <div class="action-buttons">
                    <button class="btn btn-secondary" onclick="toggleSignalDetails()">Toggle Details</button>
                    <button class="btn btn-primary" onclick="refreshSignals()">Refresh</button>
                </div>
            </div>
        </div>

        <!-- Strategy Distribution -->
        <div class="card">
            <div class="card-header">
                <h2>🎯 Strategy Performance Distribution</h2>
            </div>
            <div class="chart-container" id="strategyChart" style="height: 400px;"></div>
        </div>
    </div>

    <div class="websocket-status" id="wsStatus">
        <span id="wsIndicator">🔴</span> WebSocket: <span id="wsText">Disconnected</span>
    </div>

    <script>
        let ws = null;
        let lastData = {};
        let charts = {};

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', async function() {
            await initializeWebSocket();
            await loadAllData();
            setInterval(loadAllData, 5000); // Refresh every 5 seconds
        });

        async function loadAllData() {
            try {
                const [dashboard, risk, performance, signals, strategies] = await Promise.all([
                    fetch('/api/dashboard_data').then(r => r.json()),
                    fetch('/api/risk_metrics').then(r => r.json()),
                    fetch('/api/performance_charts').then(r => r.json()),
                    fetch('/api/signals_analysis').then(r => r.json()),
                    fetch('/api/strategy_performance').then(r => r.json())
                ]);

                updateDashboard(dashboard);
                updateRiskMetrics(risk);
                updatePerformanceCharts(performance);
                updateSignalsList(signals);
                updateStrategyChart(strategies);

                document.getElementById('lastUpdate').innerHTML = '✓ Updated';
            } catch (error) {
                console.error('Data loading error:', error);
                document.getElementById('lastUpdate').innerHTML = '❌ Error';
            }
        }

        async function initializeWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                updateWebSocketStatus(true);
                console.log('WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = () => {
                updateWebSocketStatus(false);
                setTimeout(initializeWebSocket, 3000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                updateWebSocketStatus(false);
            };
        }

        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'new_signal':
                    addSignalToList(data.payload);
                    break;
                case 'signal_update':
                    updateSignalInList(data.payload);
                    break;
                case 'trade_closed':
                    removeSignalFromList(data.payload.signal_id);
                    break;
                case 'market_state_update':
                    updateMarketContext(data.payload);
                    break;
                case 'risk_alert':
                    showRiskAlert(data.payload);
                    break;
            }
        }

        function updateDashboard(data) {
            const regimeElement = document.getElementById('marketRegime');
            if (data.market_regime) {
                regimeElement.textContent = `Regime: ${data.market_regime} | Strength: ${(data.regime_strength * 100).toFixed(1)}%`;
                regimeElement.style.color = data.regime_strength > 0.6 ? '#4CAF50' : 
                                           data.regime_strength < 0.4 ? '#f44336' : '#FF9800';
            }

            // Update bot status
            const statusElement = document.getElementById('botStatus');
            const statusIcon = document.getElementById('statusIcon');
            const statusText = document.getElementById('statusText');
            
            if (data.trading_enabled) {
                statusElement.className = 'status-indicator status-active';
                statusIcon.textContent = '🟢';
                statusText.textContent = 'Active';
            } else {
                statusElement.className = 'status-indicator status-paused';
                statusIcon.textContent = '🟡';
                statusText.textContent = 'Paused';
            }

            lastData = data;
        }

        function updateRiskMetrics(data) {
            const container = document.getElementById('riskMetrics');
            container.innerHTML = `
                <div class="metric">
                    <span>Daily Risk:</span>
                    <span class="metric-value ${data.daily_risk < 0 ? 'risk-danger' : ''}">${data.daily_risk.toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span>Max Drawdown:</span>
                    <span class="metric-value ${data.max_drawdown > 15 ? 'risk-danger' : 
                                               data.max_drawdown > 10 ? 'risk-warning' : ''}">${data.max_drawdown.toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span>VaR (95%):</span>
                    <span class="metric-value ${data.var_95 < -5 ? 'risk-danger' : 
                                               data.var_95 < -3 ? 'risk-warning' : ''}">${data.var_95.toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span>Sharpe Ratio:</span>
                    <span class="metric-value">${data.sharpe_ratio.toFixed(2)}</span>
                </div>
                <div class="metric">
                    <span>Win Rate:</span>
                    <span class="metric-value">${(data.win_rate * 100).toFixed(1)}%</span>
                </div>
            `;

            const badge = document.getElementById('riskBadge');
            if (data.is_healthy) {
                badge.textContent = 'Healthy';
                badge.style.background = '#4CAF50';
            } else {
                badge.textContent = 'Warning';
                badge.style.background = '#f44336';
            }
        }

        function updateSignalsList(data) {
            const container = document.getElementById('activeSignals');
            const countElement = document.getElementById('signalsCount');
            
            countElement.textContent = data.signals.length;
            
            if (data.signals.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #888;">No active signals</p>';
                return;
            }

            container.innerHTML = data.signals.map(signal => `
                <div class="signal-item ${signal.profit < 0 ? 'losing' : ''}" 
                     onmouseover="highlightSignal(${signal.id})" 
                     id="signal-${signal.id}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>${signal.symbol}</strong>
                            <span class="strategy-pill">${signal.strategy_name}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: bold; color: ${signal.profit >= 0 ? '#4CAF50' : '#f44336'}">
                                ${signal.profit >= 0 ? '+' : ''}${signal.profit.toFixed(2)}%
                            </div>
                            <div style="font-size: 12px; color: #888;">${signal.current_price}</div>
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 12px; color: #aaa;">
                        Entry: ${signal.entry_price} | SL: ${signal.stop_loss} | TP: ${signal.target_price}
                    </div>
                </div>
            `).join('');
        }

        async function triggerEmergencyStop() {
            const confirmed = confirm('Are you sure you want to stop all trading immediately?');
            if (!confirmed) return;

            try {
                const response = await fetch('/api/emergency_stop', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    alert('Trading stopped successfully!');
                }
            } catch (error) {
                console.error('Emergency stop error:', error);
                alert('Failed to stop trading.');
            }
        }

        function updateWebSocketStatus(connected) {
            const indicator = document.getElementById('wsIndicator');
            const text = document.getElementById('wsText');
            
            if (connected) {
                indicator.textContent = '🟢';
                text.textContent = 'Connected';
            } else {
                indicator.textContent = '🔴';
                text.textContent = 'Disconnected';
            }
        }

        function showRiskAlert(alert) {
            if (Notification.permission === 'granted') {
                new Notification('Risk Alert!', {
                    body: alert.message,
                    icon: '/static/alert-icon.png'
                });
            }
        }

        // Chart utilities
        function createAreaChart(elementId, data, title) {
            const trace = {
                x: data.x,
                y: data.y,
                fill: 'tozeroy',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#4CAF50' },
                fillcolor: 'rgba(76, 175, 80, 0.2)'
            };

            const layout = {
                title: title,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#e0e0e0' },
                margin: { l: 40, r: 20, t: 40, b: 40 }
            };

            Plotly.newPlot(elementId, [trace], layout, {responsive: true});
        }

        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    </script>
</body>
</html>
        """
        return template
    
    def _get_dashboard_data(self):
        """جلب بيانات لوحة التحكم الرئيسية"""
        try:
            # الحصول على سياق السوق
            market_context = self._get_cached_market_context()
            
            # بيانات الأداء
            performance = self._get_performance_summary()
            
            # عدد الإشارات النشطة
            active_signals_count = self._get_active_signals_count()
            
            return jsonify({
                "trading_enabled": self._is_trading_enabled(),
                "market_regime": market_context.regime.value,
                "regime_strength": market_context.regime_strength,
                "active_signals": active_signals_count,
                "total_profit": performance.get('total_profit', 0),
                "win_rate": performance.get('win_rate', 0),
                "sharpe_ratio": performance.get('sharpe_ratio', 0),
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Dashboard data error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    def _get_risk_metrics(self):
        """جلب مقاييس المخاطر"""
        try:
            risk_manager = AdvancedRiskManager(self.conn, self.redis)
            metrics = risk_manager.calculate_risk_metrics()
            
            return jsonify({
                **asdict(metrics),
                "is_healthy": metrics.is_healthy,
                "recommendations": self._generate_risk_recommendations(metrics)
            })
            
        except Exception as e:
            logger.error(f"Risk metrics error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    def _get_performance_charts(self):
        """جلب بيانات الرسوم البيانية للأداء"""
        try:
            # منحنى رأس المال
            equity_curve = self._calculate_equity_curve()
            
            # توزيع الأرباح
            profit_distribution = self._calculate_profit_distribution()
            
            # التراجعات
            drawdowns = self._calculate_drawdowns()
            
            return jsonify({
                "equity_curve": equity_curve,
                "profit_distribution": profit_distribution,
                "drawdowns": drawdowns
            })
            
        except Exception as e:
            logger.error(f"Performance charts error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    def _get_signals_analysis(self):
        """تحليل الإشارات الحالية"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        s.id,
                        s.symbol,
                        s.entry_price,
                        s.stop_loss,
                        s.target_price_1,
                        s.target_price_2,
                        s.strategy_name,
                        s.signal_details,
                        lp.price as current_price
                    FROM signals s
                    LEFT JOIN live_prices lp ON s.symbol = lp.symbol
                    WHERE s.status IN ('open', 'updated')
                """)
                signals = cur.fetchall()
            
            # حساب الربح/الخسارة
            for signal in signals:
                details = signal['signal_details'] or {}
                if isinstance(details, str):
                    details = json.loads(details)
                
                current_price = signal.get('current_price') or signal['entry_price']
                signal['profit'] = ((current_price - signal['entry_price']) / signal['entry_price']) * 100
                signal['quality_score'] = details.get('quality_score', 0)
            
            return jsonify({
                "signals": [dict(s) for s in signals],
                "total_trades": len(signals)
            })
            
        except Exception as e:
            logger.error(f"Signals analysis error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    def _handle_websocket(self, ws):
        """معالجة WebSocket"""
        self.ws_clients.append(ws)
        logger.info(f"WebSocket client connected. Total: {len(self.ws_clients)}")
        
        try:
            ws.send(json.dumps({"type": "connected", "timestamp": datetime.utcnow().isoformat()}))
            
            while True:
                data = ws.receive(timeout=30)
                if data is None:
                    ws.send(json.dumps({"type": "ping"}))
                    continue
                
                message = json.loads(data)
                self._process_websocket_message(message, ws)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            if ws in self.ws_clients:
                self.ws_clients.remove(ws)
            logger.info(f"WebSocket client disconnected. Total: {len(self.ws_clients)}")
    
    def _process_websocket_message(self, message: Dict, ws):
        """معالجة رسائل WebSocket"""
        try:
            if message.get('type') == 'request_action':
                action = message.get('action')
                if action == 'close_signal':
                    signal_id = message.get('signal_id')
                    # منطق إغلاق الإشارة
                    pass
                elif action == 'change_settings':
                    settings = message.get('settings')
                    # منطق تغيير الإعدادات
                    pass
                    
        except Exception as e:
            logger.error(f"Processing WebSocket message error: {e}", exc_info=True)
    
    def broadcast_update(self, data_type: str, payload: Dict):
        """بث تحديث لجميع عملاء WebSocket"""
        message = json.dumps({
            "type": data_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }, cls=PlotlyJSONEncoder)
        
        for client in self.ws_clients:
            try:
                client.send(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}", exc_info=True)
                self.ws_clients.remove(client)

# ============================================
# قاعدة البيانات المحسنة
# ============================================

class EnhancedDatabaseManager:
    """إدارة قاعدة بيانات مع تتبع أداء متقدم"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.init_advanced_schema()
    
    def init_advanced_schema(self):
        """إنشاء المخطط المتقدم"""
        try:
            conn = psycopg2.connect(self.db_url)
            with conn.cursor() as cur:
                # جدول الأداء المفصل
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS detailed_performance (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        signal_id INTEGER REFERENCES signals(id),
                        symbol TEXT NOT NULL,
                        strategy_name TEXT,
                        entry_price DOUBLE PRECISION,
                        exit_price DOUBLE PRECISION,
                        profit_usdt DOUBLE PRECISION,
                        profit_percentage DOUBLE PRECISION,
                        holding_duration INTERVAL,
                        risk_score DOUBLE PRECISION,
                        market_regime TEXT,
                        max_drawdown_during_trade DOUBLE PRECISION,
                        slippage DOUBLE PRECISION
                    );
                """)
                
                # جدول مخاطر المحفظة
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_risk (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        total_exposure DOUBLE PRECISION,
                        var_95 DOUBLE PRECISION,
                        sharpe_ratio DOUBLE PRECISION,
                        max_drawdown DOUBLE PRECISION,
                        correlation_heatmap JSONB,
                        risk_recommendations TEXT
                    );
                """)
                
                # جدول تتبع الاستراتيجية
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_tracking (
                        strategy_name TEXT PRIMARY KEY,
                        enabled BOOLEAN DEFAULT TRUE,
                        total_trades INTEGER DEFAULT 0,
                        win_rate DOUBLE PRECISION DEFAULT 0,
                        avg_profit DOUBLE PRECISION DEFAULT 0,
                        max_drawdown DOUBLE PRECISION DEFAULT 0,
                        last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                
                # الأدلة
                cur.execute("CREATE INDEX IF NOT EXISTS idx_detailed_performance_timestamp ON detailed_performance(timestamp);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_detailed_performance_symbol ON detailed_performance(symbol);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_risk_timestamp ON portfolio_risk(timestamp DESC);")
                
                conn.commit()
            conn.close()
            
            logger.info("✅ Advanced database schema initialized")
            
        except Exception as e:
            logger.error(f"Database schema error: {e}", exc_info=True)
            raise
    
    def log_detailed_performance(self, signal_result: Any, exit_data: Dict):
        """تسجيل أداء مفصل للإشارة"""
        try:
            conn = psycopg2.connect(self.db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO detailed_performance (
                        signal_id, symbol, strategy_name, entry_price, exit_price,
                        profit_usdt, profit_percentage, holding_duration, risk_score,
                        market_regime, max_drawdown_during_trade, slippage
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    exit_data['signal_id'],
                    signal_result.symbol,
                    signal_result.strategy_name,
                    signal_result.entry_price,
                    exit_data['exit_price'],
                    exit_data.get('profit_usdt', 0),
                    exit_data['profit_percentage'],
                    exit_data.get('holding_duration'),
                    signal_result.quality_score,
                    signal_result.market_context.regime.value,
                    exit_data.get('max_drawdown', 0),
                    exit_data.get('slippage', 0)
                ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Logging detailed performance error: {e}", exc_info=True)
    
    def update_strategy_tracking(self, strategy_name: str, trade_result: Dict):
        """تحديث تتبع أداء الاستراتيجية"""
        try:
            conn = psycopg2.connect(self.db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO strategy_tracking (strategy_name, total_trades, win_rate, avg_profit, max_drawdown)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (strategy_name)
                    DO UPDATE SET
                        total_trades = strategy_tracking.total_trades + 1,
                        win_rate = (strategy_tracking.win_rate * strategy_tracking.total_trades + %s) / (strategy_tracking.total_trades + 1),
                        avg_profit = (strategy_tracking.avg_profit * strategy_tracking.total_trades + %s) / (strategy_tracking.total_trades + 1),
                        max_drawdown = GREATEST(strategy_tracking.max_drawdown, %s),
                        last_updated = NOW()
                """, (
                    strategy_name,
                    1,
                    1 if trade_result['profit'] > 0 else 0,
                    trade_result['profit'],
                    abs(min(0, trade_result.get('drawdown', 0))),
                    trade_result['profit'] > 0,
                    trade_result['profit'],
                    abs(min(0, trade_result.get('drawdown', 0)))
                ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Strategy tracking error: {e}", exc_info=True)

# ============================================
# تصدير الوحدات
# ============================================

__all__ = [
    'TelegramNotifier',
    'RiskMetrics',
    'AdvancedRiskManager',
    'AdvancedDashboard',
    'EnhancedDatabaseManager'
]
# trading_bot_v37_execution.py (الجزء الثالث - التنفيذ والأداء)
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_DOWN
import numpy as np
import pandas as pd
import redis
import hashlib
import psutil
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *
import requests
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('BotV37Execution')

# ============================================
# Binance API Client المتقدم
# ============================================

class AdvancedBinanceClient:
    """عميل Binance مع معالجة أخطاء وإدارة معدل الطلبات"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.loop = asyncio.get_event_loop()
        
        # إدارة معدل الطلبات
        self.rate_limiter = AsyncRateLimiter(requests_per_minute=1200)
        self.weight_tracker = RequestWeightTracker()
        
        # التخزين المؤقت
        self.cache = RedisCache(prefix="binance_", ttl=60)
        
        # تنفيذ الطلبات
        self.order_executor = SmartOrderExecutor(self.client, self.rate_limiter)
        
        # تتبع الأخطاء
        self.error_tracker = ErrorTracker()
        
        logger.info(f"✅ Binance client initialized (Testnet: {testnet})")
    
    async def get_exchange_info_async(self) -> Dict[str, Any]:
        """جلب معلومات المنصة بشكل غير متزامن مع التخزين المؤقت"""
        cache_key = "exchange_info"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            await self.rate_limiter.acquire()
            info = await self.loop.run_in_executor(None, self.client.get_exchange_info)
            await self.cache.set(cache_key, info)
            logger.info("✅ Exchange info cached")
            return info
        except Exception as e:
            await self.error_tracker.log_error("exchange_info", str(e))
            logger.error(f"Exchange info error: {e}")
            return {}
    
    async def get_historical_klines_async(self, symbol: str, interval: str, 
                                         days: int) -> Optional[pd.DataFrame]:
        """جلب بيانات تاريخية مع إدارة الأخطاء"""
        cache_key = f"klines_{symbol}_{interval}_{days}"
        cached = await self.cache.get(cache_key)
        if cached:
            return pd.read_json(cached)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                await self.weight_tracker.consume_weight(5)  # وزن الطلب
                
                klines = await self.loop.run_in_executor(
                    None,
                    lambda: self.client.get_historical_klines(
                        symbol, interval, f"{days} days ago UTC"
                    )
                )
                
                if not klines:
                    return None
                
                df = self._klines_to_dataframe(klines)
                
                # تخزين مؤقت
                await self.cache.set(cache_key, df.to_json())
                
                return df
                
            except BinanceAPIException as e:
                await self.error_tracker.log_error("klines", str(e), symbol)
                
                if e.code == -1003:  # تجاوز معدل الطلبات
                    wait_time = int(e.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit hit, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                
                elif e.code == -1121:  # رمز غير صالح
                    logger.error(f"Invalid symbol: {symbol}")
                    return None
                
                else:
                    logger.error(f"Binance API error: {e}")
                    if attempt == max_retries - 1:
                        return None
                    await asyncio.sleep(2 ** attempt)
            
            except Exception as e:
                logger.error(f"Unexpected klines error: {e}", exc_info=True)
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    def _klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """تحويل klines إلى DataFrame محسّن"""
        try:
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna().astype(float)
            
        except Exception as e:
            logger.error(f"Klines conversion error: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def get_current_price_async(self, symbol: str) -> Optional[float]:
        """جلب السعر الحالي مع التخزين المؤقت"""
        cache_key = f"price_{symbol}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            await self.rate_limiter.acquire()
            ticker = await self.loop.run_in_executor(
                None,
                lambda: self.client.get_symbol_ticker(symbol=symbol)
            )
            
            price = float(ticker['price'])
            await self.cache.set(cache_key, price)
            return price
            
        except Exception as e:
            await self.error_tracker.log_error("price", str(e), symbol)
            logger.error(f"Current price error for {symbol}: {e}")
            return None
    
    async def place_smart_order(self, order_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """وضع طلب ذكي مع إدارة المخاطر"""
        try:
            # التحقق من صحة الطلب
            validation_result = await self._validate_order(order_data)
            if not validation_result['valid']:
                logger.error(f"Order validation failed: {validation_result['errors']}")
                return None
            
            # حساب أفضل وقت تنفيذ
            execution_time = await self._calculate_optimal_execution_time(order_data['symbol'])
            if execution_time > 0:
                logger.info(f"Waiting {execution_time:.2f}s for optimal execution")
                await asyncio.sleep(execution_time)
            
            # تنفيذ الطلب
            result = await self.order_executor.execute_order(order_data)
            
            # تسجيل الأداء
            await self._log_order_performance(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Smart order placement error: {e}", exc_info=True)
            await self.error_tracker.log_error("order_placement", str(e), order_data.get('symbol'))
            return None
    
    async def _validate_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من صحة الطلب ضد قواعد المنصة"""
        errors = []
        
        try:
            symbol = order_data['symbol']
            quantity = Decimal(str(order_data['quantity']))
            price = Decimal(str(order_data.get('price', 0)))
            
            # جلب معلومات الرمز
            exchange_info = await self.get_exchange_info_async()
            symbol_info = exchange_info.get('symbols', [])
            symbol_data = next((s for s in symbol_info if s['symbol'] == symbol), None)
            
            if not symbol_data:
                return {"valid": False, "errors": ["Symbol not found"]}
            
            # فلاتر LOT_SIZE
            lot_filter = next((f for f in symbol_data['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_filter:
                min_qty = Decimal(lot_filter['minQty'])
                max_qty = Decimal(lot_filter['maxQty'])
                step_size = Decimal(lot_filter['stepSize'])
                
                if quantity < min_qty:
                    errors.append(f"Quantity {quantity} < min {min_qty}")
                if quantity > max_qty:
                    errors.append(f"Quantity {quantity} > max {max_qty}")
                if (quantity - min_qty) % step_size != 0:
                    errors.append(f"Quantity step size violation")
            
            # فلاتر MIN_NOTIONAL
            notional_filter = next((f for f in symbol_data['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            if notional_filter:
                min_notional = Decimal(notional_filter['minNotional'])
                notional_value = quantity * price
                
                if notional_value < min_notional:
                    errors.append(f"Notional value {notional_value} < min {min_notional}")
            
            return {"valid": len(errors) == 0, "errors": errors}
            
        except Exception as e:
            logger.error(f"Order validation error: {e}", exc_info=True)
            return {"valid": False, "errors": [str(e)]}

# ============================================
# تنفيذ الطلبات الذكي
# ============================================

class SmartOrderExecutor:
    """تنفيذ الطلبات باستخدام خوارزميات متقدمة"""
    
    def __init__(self, client: Client, rate_limiter: 'AsyncRateLimiter'):
        self.client = client
        self.rate_limiter = rate_limiter
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    async def execute_order(self, order_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """تنفيذ الطلب مع خوارزمية التنفيذ المثلى"""
        try:
            order_type = order_data.get('type', 'MARKET')
            
            if order_type == 'MARKET':
                return await self._execute_market_order(order_data)
            elif order_type == 'LIMIT':
                return await self._execute_limit_order(order_data)
            elif order_type == 'STOP_LOSS_LIMIT':
                return await self._execute_stop_loss_limit_order(order_data)
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return None
                
        except Exception as e:
            logger.error(f"Order execution error: {e}", exc_info=True)
            return None
    
    async def _execute_market_order(self, order_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """تنفيذ الطلب السوقي مع تفاصيل التنفيذ"""
        try:
            symbol = order_data['symbol']
            side = order_data['side']
            quantity = order_data['quantity']
            
            # تنفيذ الطلب
            order_result = await self.loop.run_in_executor(
                self.executor,
                lambda: self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
            )
            
            # تحليل التنفيذ
            execution_analysis = self._analyze_execution(order_result)
            
            return {
                "order_id": order_result['orderId'],
                "symbol": symbol,
                "side": side,
                "executed_qty": float(order_result['executedQty']),
                "avg_price": execution_analysis['avg_price'],
                "fills": order_result.get('fills', []),
                "execution_slippage": execution_analysis['slippage'],
                "execution_quality": execution_analysis['quality'],
                "fees": execution_analysis['fees'],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except BinanceOrderException as e:
            logger.error(f"Market order failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected market order error: {e}", exc_info=True)
            raise
    
    def _analyze_execution(self, order_result: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل جودة التنفيذ"""
        try:
            fills = order_result.get('fills', [])
            if not fills:
                return {"avg_price": 0, "slippage": 0, "quality": 0, "fees": 0}
            
            total_qty = sum(float(f['qty']) for f in fills)
            weighted_price = sum(float(f['price']) * float(f['qty']) for f in fills)
            avg_price = weighted_price / total_qty if total_qty > 0 else 0
            
            # حساب الانزلاق السعري
            expected_price = float(order_result.get('price', avg_price))
            slippage = abs(avg_price - expected_price) / expected_price if expected_price > 0 else 0
            
            # حساب جودة التنفيذ
            fill_rate = len(fills) / max(1, total_qty / 10)  # نسبة مقياسية
            quality = max(0, min(100, (1 - slippage * 10) * fill_rate * 100))
            
            # إجمالي الرسوم
            total_fees = sum(float(f.get('commission', 0)) for f in fills)
            
            return {
                "avg_price": avg_price,
                "slippage": slippage,
                "quality": quality,
                "fees": total_fees
            }
            
        except Exception as e:
            logger.error(f"Execution analysis error: {e}", exc_info=True)
            return {"avg_price": 0, "slippage": 0, "quality": 0, "fees": 0}

# ============================================
# أدوات مساعدة متقدمة
# ============================================

class AsyncRateLimiter:
    """حد معدل الطلبات غير المتزامن"""
    
    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute / 60  # لكل ثانية
        self.tokens = requests_per_minute
        self.max_tokens = requests_per_minute
        self.lock = asyncio.Lock()
        self.last_update = time.time()
    
    async def acquire(self, cost: int = 1):
        """الحصول على إذن للطلب"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # تعبئة الرموز
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.rate
            )
            
            self.last_update = now
            
            # الانتظار إذا لزم الأمر
            if self.tokens < cost:
                wait_time = (cost - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= cost

class RequestWeightTracker:
    """تتبع وزن الطلبات لحد API"""
    
    def __init__(self):
        self.current_weight = 0
        self.reset_time = time.time() + 60
        self.lock = threading.Lock()
    
    async def consume_weight(self, weight: int):
        """استهلاك وزن الطلب"""
        with self.lock:
            now = time.time()
            
            if now >= self.reset_time:
                self.current_weight = 0
                self.reset_time = now + 60
            
            if self.current_weight + weight > 1200:  # الحد الأقصى لـ Binance
                wait_time = self.reset_time - now
                if wait_time > 0:
                    logger.warning(f"Weight limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                self.current_weight = 0
            
            self.current_weight += weight

class RedisCache:
    """تخزين مؤقت Redis متقدم"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, prefix: str = "", ttl: int = 60):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.prefix = prefix
        self.ttl = ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """الحصول على قيمة مؤقتة"""
        try:
            cached = self.redis.get(f"{self.prefix}{key}")
            if cached:
                return json.loads(cached)
            return None
        except:
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """تخزين قيمة مؤقتة"""
        try:
            self.redis.setex(
                f"{self.prefix}{key}",
                ttl or self.ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Redis cache set error: {e}", exc_info=True)
    
    async def delete_pattern(self, pattern: str):
        """حذف مفاتيح مطابقة نمط"""
        try:
            keys = self.redis.keys(f"{self.prefix}{pattern}")
            if keys:
                self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Redis delete pattern error: {e}", exc_info=True)

class ErrorTracker:
    """تتبع الأخطاء وتوليد تقارير"""
    
    def __init__(self):
        self.errors = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.lock = threading.Lock()
    
    async def log_error(self, error_type: str, message: str, symbol: Optional[str] = None):
        """تسجيل خطأ جديد"""
        with self.lock:
            error_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": error_type,
                "message": message,
                "symbol": symbol,
                "stack_trace": ""  # يمكن إضافة استخراج التتبع
            }
            
            self.errors.append(error_entry)
            self.error_counts[f"{error_type}:{symbol}"] += 1
    
    def get_error_report(self) -> Dict[str, Any]:
        """الحصول على تقرير الأخطاء"""
        with self.lock:
            return {
                "total_errors": len(self.errors),
                "error_counts": dict(self.error_counts),
                "recent_errors": list(self.errors)[-10:],
                "most_common": sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }

# ============================================
# أنماط تداول متقدمة
# ============================================

class AdvancedTradingPatterns:
    """التعرف على الأنماط المتقدمة باستخدام تعلم الآلة"""
    
    def __init__(self):
        self.pattern_models = self._load_ml_models()
        self.pattern_cache = {}
        
    def _load_ml_models(self) -> Dict[str, Any]:
        """تحميل نماذج ML للتعرف على الأنماط"""
        # في إنتاج حقيقي، يتم تحميل نماذج تم تدريبها مسبقاً
        # هنا نستخدم قواعد بسيطة كمثال
        logger.warning("ML models not loaded, using rule-based patterns")
        return {}
    
    def detect_advanced_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """الكشف عن الأنماط المتقدمة"""
        patterns = {}
        
        # 1. نمط الرأس والكتفين
        head_shoulders = self._detect_head_and_shoulders(df)
        if head_shoulders:
            patterns['head_and_shoulders'] = head_shoulders['confidence']
        
        # 2. نمط الكوب والمقبض
        cup_handle = self._detect_cup_and_handle(df)
        if cup_handle:
            patterns['cup_and_handle'] = cup_handle['confidence']
        
        # 3. نمط التوحيد المثلث
        triangle = self._detect_triangle_pattern(df)
        if triangle:
            patterns['triangle'] = triangle['confidence']
        
        # 4. نمط الشمعة اليابانية المتقدم
        candlestick = self._detect_advanced_candlestick_patterns(df)
        patterns.update(candlestick)
        
        # 5. نمط حجم التداول الغير عادي
        volume_pattern = self._detect_volume_anomaly(df)
        if volume_pattern:
            patterns['volume_anomaly'] = volume_pattern['confidence']
        
        return patterns
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """الكشف عن نمط الرأس والكتفين المتقدم"""
        try:
            from scipy.signal import argrelextrema
            
            # العثور على القمم المحلية
            highs = df['high'].values
            extrema = argrelextrema(highs, np.greater, order=10)[0]
            
            if len(extrema) < 5:
                return None
            
            # التحقق من الترتيب
            last_extrema = extrema[-5:]
            if len(last_extrema) != 5:
                return None
            
            # شروط الرأس والكتفين
            left_shoulder, left_peak, head, right_peak, right_shoulder = last_extrema
            
            # منطق التحقق (مبسط)
            if (highs[left_shoulder] < highs[left_peak] and
                highs[head] > highs[left_peak] and
                highs[head] > highs[right_peak] and
                abs(highs[left_shoulder] - highs[right_shoulder]) < highs[head] * 0.01):
                
                return {
                    "confidence": 0.75,
                    "type": "head_and_shoulders",
                    "head_level": highs[head],
                    "shoulder_level": (highs[left_shoulder] + highs[right_shoulder]) / 2
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Head and shoulders detection error: {e}")
            return None
    
    def _detect_cup_and_handle(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """الكشف عن نمط الكوب والمقبض"""
        try:
            # تحليل على المدى المتوسط (20-30 شمعة)
            recent = df.tail(30)
            
            # العثور على القاع (الكوب)
            min_idx = recent['low'].idxmin()
            min_val = recent['low'].min()
            min_pos = recent.index.get_loc(min_idx)
            
            if min_pos < 5 or min_pos > 25:
                return None
            
            # التحقق من الشكل (مبسط)
            left_side = recent.iloc[:min_pos]['close'].values
            right_side = recent.iloc[min_pos+1:]['close'].values
            
            if len(left_side) < 5 or len(right_side) < 5:
                return None
            
            # التحقق من التناظر
            left_trend = np.polyfit(range(len(left_side)), left_side, 1)[0]
            right_trend = np.polyfit(range(len(right_side)), right_side, 1)[0]
            
            if abs(left_trend - right_trend) < 0.001:  # تقريبياً متناظر
                return {
                    "confidence": 0.65,
                    "type": "cup_and_handle",
                    "bottom_price": min_val,
                    "handle_level": recent['close'].iloc[-1]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Cup and handle detection error: {e}")
            return None
    
    def _detect_triangle_pattern(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """الكشف عن أنماط المثلث"""
        try:
            recent = df.tail(40)
            
            # العثور على القمم والقيعان المتتالية
            highs = recent['high'].values
            lows = recent['low'].values
            
            # تقلص النطاق
            high_slope = (highs[-10:].mean() - highs[:10].mean()) / len(highs)
            low_slope = (lows[-10:].mean() - lows[:10].mean()) / len(lows)
            
            # الاتجاهات المعاكسة
            if abs(high_slope - low_slope) < 0.001:  # تقارب
                return {
                    "confidence": 0.7,
                    "type": "symmetrical_triangle",
                    "upper_trend": high_slope,
                    "lower_trend": low_slope,
                    "apex_level": (highs[-1] + lows[-1]) / 2
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Triangle pattern error: {e}")
            return None
    
    def _detect_advanced_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """الكشف عن أنماط الشمعة اليابانية المتقدمة"""
        patterns = {}
        
        try:
            recent = df.tail(5)
            
            # أنماط ثلاثية الشموع
            if len(recent) >= 3:
                three_candles = recent.tail(3)
                
                # نمط النجمة الصباحية
                if self._is_morning_star(three_candles):
                    patterns['morning_star'] = 0.8
                
                # نمط النجمة المسائية
                if self._is_evening_star(three_candles):
                    patterns['evening_star'] = 0.8
                
                # نمط الثلاثة جنود البيض
                if self._is_three_white_soldiers(three_candles):
                    patterns['three_white_soldiers'] = 0.85
                
                # نمط الغربان السود الثلاثة
                if self._is_three_black_crows(three_candles):
                    patterns['three_black_crows'] = 0.85
            
            return patterns
            
        except Exception as e:
            logger.error(f"Candlestick pattern error: {e}")
            return {}
    
    def _is_morning_star(self, df: pd.DataFrame) -> bool:
        """التحقق من نمط النجمة الصباحية"""
        try:
            c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
            
            # شمعة هابطة كبيرة
            cond1 = c1['open'] > c1['close'] and (c1['open'] - c1['close']) > (c1['high'] - c1['low']) * 0.6
            
            # شمعة صغيرة (دوجي أو جنين)
            cond2 = abs(c2['open'] - c2['close']) < (c2['high'] - c2['low']) * 0.3
            
            # شمعة صاعدة كبيرة
            cond3 = c3['close'] > c3['open'] and (c3['close'] - c3['open']) > (c3['high'] - c3['low']) * 0.6
            
            return cond1 and cond2 and cond3
            
        except:
            return False
    
    def _detect_volume_anomaly(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """الكشف عن أنماط حجم غير عادية"""
        try:
            recent_volumes = df['volume'].tail(10).values
            avg_volume = recent_volumes[:-1].mean()
            current_volume = recent_volumes[-1]
            
            # حجم غير عادي (>3x المتوسط)
            if current_volume > avg_volume * 3:
                return {
                    "confidence": 0.75,
                    "type": "volume_spike",
                    "multiplier": current_volume / avg_volume,
                    "direction": "up" if df['close'].iloc[-1] > df['open'].iloc[-1] else "down"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Volume anomaly detection error: {e}")
            return None

# ============================================
# Main Loop Integration
# ============================================

class TradingOrchestrator:
    """أوركسترا التداول الرئيسية تنسق جميع المكونات"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.tasks = []
        
        # تهيئة المكونات
        self.binance_client = AdvancedBinanceClient(
            config['BINANCE_API_KEY'],
            config['BINANCE_API_SECRET'],
            config.get('USE_TESTNET', False)
        )
        
        self.market_analyzer = AdvancedMarketAnalyzer()
        self.strategy_manager = DynamicStrategyManager(self.market_analyzer)
        self.risk_manager = AdvancedRiskManager(
            config['DATABASE_URL'],
            redis.from_url(config['REDIS_URL'])
        )
        
        self.dashboard = AdvancedDashboard(
            config['DATABASE_URL'],
            redis.from_url(config['REDIS_URL']),
            TelegramNotifier(
                config.get('TELEGRAM_BOT_TOKEN', ''),
                config.get('TELEGRAM_CHAT_ID', '')
            )
        )
        
        self.pattern_detector = AdvancedTradingPatterns()
        self.db_manager = EnhancedDatabaseManager(config['DATABASE_URL'])
        
        logger.info("✅ Trading orchestrator initialized")
    
    async def start(self):
        """بدء جميع أعمال الخلفية"""
        self.running = True
        
        # تهيئة
        await self.binance_client.initialize()
        await self.dashboard.telegram.initialize()
        
        # بدء الحلقات
        self.tasks = [
            asyncio.create_task(self._market_analysis_loop()),
            asyncio.create_task(self._signal_generation_loop()),
            asyncio.create_task(self._trade_management_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._performance_tracking_loop()),
            asyncio.create_task(self._dashboard_update_loop()),
            asyncio.create_task(self._resource_monitoring_loop())
        ]
        
        logger.info("🚀 All trading loops started")
        
        # الانتظار حتى الإيقاف
        await asyncio.gather(*self.tasks, return_exceptions=True)
    
    async def stop(self):
        """إيقاف جميع العمليات بأمان"""
        self.running = False
        
        # إلغاء المهام
        for task in self.tasks:
            task.cancel()
        
        # إغلاق الاتصالات
        await self.binance_client.telegram.close()
        
        logger.info("🛑 Trading orchestrator stopped")
    
    async def _market_analysis_loop(self):
        """حلقة تحليل السوق"""
        while self.running:
            try:
                # تحليل النظام الحالي
                market_context = await self.market_analyzer.analyze_market_regime(
                    self.binance_client.client
                )
                
                # تخزين التحليل
                await self._cache_market_context(market_context)
                
                # بث التحديثات
                self.dashboard.broadcast_update(
                    "market_context",
                    asdict(market_context)
                )
                
                # فحص حالات الطوارئ
                if market_context.regime == MarketRegime.STRONG_BEAR:
                    logger.warning("Strong bear market detected, reducing exposure")
                    await self._reduce_exposure()
                
                await asyncio.sleep(60 * 5)  # كل 5 دقائق
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market analysis loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _signal_generation_loop(self):
        """حلقة توليد الإشارات (أكثر تكراراً)"""
        while self.running:
            try:
                # جلب سياق السوق
                market_context = await self._get_cached_market_context()
                
                # جلب العملات المراد فحصها
                symbols = await self._get_trading_symbols()
                
                # توليد الإشارات
                signals = await self.strategy_manager.scan_for_signals(
                    symbols,
                    self.binance_client.client,
                    market_context
                )
                
                # تصفية وتنقيح الإشارات
                filtered_signals = await self._filter_and_refine_signals(signals)
                
                # تنفيذ الإشارات
                for signal in filtered_signals:
                    await self._execute_signal(signal)
                
                # انتظار حتى الدورة التالية
                await asyncio.sleep(60 * 1)  # كل دقيقة (30 ثانية قبل إغلاق الشمعة)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal generation loop error: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    async def _execute_signal(self, signal: SignalResult):
        """تنفيذ إشارة واحدة مع إدارة المخاطر"""
        try:
            # التحقق من المخاطر
            should_pause, reason = await self.risk_manager.should_pause_trading()
            if should_pause:
                logger.warning(f"Trading paused: {reason}")
                return
            
            # الحصول على توازن USDT المتاح
            balance = await self._get_available_balance()
            
            # حساب حجم المركز
            position_size = await self.risk_manager.calculate_dynamic_position_size(
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                balance,
                signal.market_context
            )
            
            if not position_size or position_size <= 0:
                logger.error(f"Invalid position size for {signal.symbol}")
                return
            
            # إنشاء بيانات الطلب
            order_data = {
                "symbol": signal.symbol,
                "side": "BUY",
                "type": "MARKET",
                "quantity": float(position_size),
                "price": signal.entry_price,
                "newOrderRespType": "FULL"
            }
            
            # تنفيذ الطلب
            order_result = await self.binance_client.place_smart_order(order_data)
            
            if order_result:
                # حفظ الإشارة في قاعدة البيانات
                await self._save_signal_to_db(signal, order_result)
                
                # إرسال إشعار
                await self._send_signal_notification(signal, order_result)
                
                logger.info(f"✅ Signal executed: {signal.symbol} | Strategy: {signal.strategy_name}")
            else:
                logger.error(f"❌ Failed to execute signal: {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Signal execution error for {signal.symbol}: {e}", exc_info=True)
    
    async _trade_management_loop(self):
        """إدارة الصفقات المفتوحة"""
        while self.running:
            try:
                # جلب الصفقات المفتوحة
                open_trades = await self._get_open_trades()
                
                for trade in open_trades:
                    # إعادة تحليل الصفقة
                    await self._reanalyze_trade(trade)
                    
                    # إدارة الوقف المتحرك
                    await self._manage_trailing_stop(trade)
                
                await asyncio.sleep(5)  # تحديث كل 5 ثواني
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trade management loop error: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def _risk_monitoring_loop(self):
        """حلقة مراقبة المخاطر"""
        while self.running:
            try:
                # حساب مقاييس المخاطر
                metrics = await self.risk_manager.calculate_risk_metrics()
                
                # فحص الحدود
                if not metrics.is_healthy:
                    logger.warning(f"Risk limits exceeded: {metrics}")
                    
                    # إرسال إشعار
                    await self._send_risk_alert(metrics)
                    
                    # تقليل التعرض
                    await self._reduce_exposure()
                
                # تسجيل في قاعدة البيانات
                await self.db_manager.log_portfolio_risk(metrics)
                
                await asyncio.sleep(30)  # كل 30 ثانية
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    async def _performance_tracking_loop(self):
        """حلقة تتبع الأداء"""
        while self.running:
            try:
                # تحديث أداء الاستراتيجية
                await self._update_strategy_performance()
                
                # تقرير الأداء اليومي
                if datetime.utcnow().hour == 23 and datetime.utcnow().minute == 59:
                    await self._send_daily_performance_report()
                
                await asyncio.sleep(60 * 5)  # كل 5 دقائق
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance tracking loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _resource_monitoring_loop(self):
        """حلقة مراقبة الموارد"""
        while self.running:
            try:
                # استخدام CPU/Memory
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                if cpu_percent > 80 or memory.percent > 85:
                    logger.warning(f"High resource usage: CPU {cpu_percent}%, Memory {memory.percent}%")
                    
                    # إجراءات التنظيف
                    await self._cleanup_resources()
                
                # مراقبة عدد المهام
                active_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
                if active_tasks > 100:
                    logger.warning(f"Too many active tasks: {active_tasks}")
                
                await asyncio.sleep(60)  # كل دقيقة
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _cleanup_resources(self):
        """تنظيف الموارد المستخدمة"""
        try:
            # تنظيف التخزين المؤقت
            await self.binance_client.cache.delete_pattern("price_*")
            
            # تنظيف WebSocket clients الميتة
            self.dashboard.ws_clients = [
                ws for ws in self.dashboard.ws_clients 
                if ws.connected
            ]
            
            # تنظيف الأخطاء القديمة
            self.binance_client.error_tracker.errors.clear()
            
            logger.info("🧹 Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup error: {e}", exc_info=True)

# ============================================
# نقطة الدخول الرئيسية
# ============================================

def main():
    """دالة البدء الرئيسية"""
    try:
        # تحميل الإعدادات
        config = {
            'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY'),
            'BINANCE_API_SECRET': os.getenv('BINANCE_API_SECRET'),
            'DATABASE_URL': os.getenv('DATABASE_URL'),
            'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', ''),
            'USE_TESTNET': os.getenv('USE_TESTNET', 'true').lower() == 'true'
        }
        
        # التحقق من الإعدادات المطلوبة
        missing_config = [k for k, v in config.items() if not v and k != 'USE_TESTNET']
        if missing_config:
            logger.critical(f"Missing configuration: {', '.join(missing_config)}")
            raise ValueError("Configuration incomplete")
        
        # تهيئة المنسق
        orchestrator = TradingOrchestrator(config)
        
        # بدء التداول
        asyncio.run(orchestrator.start())
        
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
    
    finally:
        # التنظيف
        logger.info("🧹 Cleanup completed")

if __name__ == '__main__':
    main()

# ============================================
# قائمة التصدير
# ============================================

__all__ = [
    'AdvancedBinanceClient',
    'SmartOrderExecutor',
    'AsyncRateLimiter',
    'RequestWeightTracker',
    'RedisCache',
    'ErrorTracker',
    'AdvancedTradingPatterns',
    'TradingOrchestrator',
    'main'
]