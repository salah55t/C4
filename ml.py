import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import warnings
import gc

# --- NEW IMPORTS FOR TTM ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread

# ---------------------- تجاهل التحذيرات المستقبلية من Pandas ----------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_ttm.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_TTM_MemFix')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'TTM_Scalping_V2_MemFix'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 120
BTC_SYMBOL = 'BTCUSDT'

# --- TTM Model & Training Parameters (MEMORY FIXES APPLIED) ---
SEQUENCE_LENGTH: int = 24 # MODIFIED: Reduced from 32 to 24
N_FEATURES: int = 0 
D_MODEL: int = 48 # MODIFIED: Reduced from 64 to 48
N_BLOCKS: int = 3 # MODIFIED: Reduced from 4 to 3
LEARNING_RATE: float = 0.001
N_EPOCHS: int = 20
BATCH_SIZE: int = 32 # MODIFIED: Reduced from 64 to 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Indicator & Feature Parameters ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30

# Triple-Barrier Method Parameters
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None


# -------------------------------------------------
# TTM Model Definition
# -------------------------------------------------
class TimeMixerBlock(nn.Module):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.time_mixer = nn.Sequential(
            nn.Linear(sequence_length, sequence_length),
            nn.GELU(),
        )
    def forward(self, x): # x shape: (batch, seq_len, d_model)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2) # (batch, d_model, seq_len)
        x = self.time_mixer(x)
        x = x.transpose(1, 2) # (batch, seq_len, d_model)
        return x + residual

class ChannelMixerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
    def forward(self, x): # x shape: (batch, seq_len, d_model)
        residual = x
        x = self.norm(x)
        x = self.channel_mixer(x)
        return x + residual

class TTM(nn.Module):
    def __init__(self, n_features, sequence_length, d_model, n_blocks, n_classes=3):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.mixer_layers = nn.ModuleList([
            nn.Sequential(
                TimeMixerBlock(sequence_length, d_model),
                ChannelMixerBlock(d_model)
            ) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(d_model * sequence_length, n_classes)

    def forward(self, x): # x shape: (batch, seq_len, n_features)
        x = self.embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.head(x)
        return x

# --- دوال الاتصال والتحقق ---
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
            """)
        conn.commit()
        logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}"); exit(1)

def keep_db_alive():
    if not conn: return
    try:
        with conn.cursor() as cur: cur.execute("SELECT 1;")
        logger.debug("[DB Keep-Alive] Ping successful.")
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.error(f"❌ [DB Keep-Alive] انقطع اتصال قاعدة البيانات: {e}. محاولة إعادة الاتصال...")
        if conn: conn.close()
        init_db()
    except Exception as e:
        logger.error(f"❌ [DB Keep-Alive] خطأ غير متوقع أثناء فحص الاتصال: {e}")
        if conn: conn.rollback()

def get_trained_symbols_from_db() -> set:
    if not conn: return set()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT model_name FROM ml_models WHERE model_name LIKE %s;", (f"{BASE_ML_MODEL_NAME}_%",))
            trained_models = cur.fetchall()
            prefix_to_remove = f"{BASE_ML_MODEL_NAME}_"
            trained_symbols = {row['model_name'].replace(prefix_to_remove, '') for row in trained_models if row['model_name'].startswith(prefix_to_remove)}
            logger.info(f"✅ [DB Check] تم العثور على {len(trained_symbols)} نموذج مدرب مسبقاً في قاعدة البيانات.")
            return trained_symbols
    except Exception as e:
        logger.error(f"❌ [DB Check] لا يمكن جلب الرموز المدربة من قاعدة البيانات: {e}")
        if conn: conn.rollback()
        return set()

def get_binance_client():
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل تهيئة عميل Binance: {e}"); exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in symbols}
        info = client.get_exchange_info()
        active = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول.")
        return validated
    except FileNotFoundError:
        logger.error(f"❌ [Validation] ملف قائمة العملات '{filename}' غير موجود.")
        return []
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ في التحقق من الرموز: {e}"); return []

# --- دالة تحسين استهلاك الذاكرة ---
def optimize_memory_usage(df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    end_mem = df.memory_usage().sum() / 1024**2
    if start_mem > end_mem:
         logger.info(f"🧠 [{log_prefix}] Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction).")
    return df

# --- دوال جلب ومعالجة البيانات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base']
        df = df[required_cols]
        
        numeric_cols = {
            'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 
            'volume': 'float', 'quote_volume': 'float', 'taker_buy_base': 'float'
        }
        df = df.astype(numeric_cols)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.dropna(inplace=True)
        
        return optimize_memory_usage(df, log_prefix=f"Fetch {symbol} {interval}")
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol} على إطار {interval}: {e}"); return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change()

# --- دوال حساب الميزات (Feature Calculation Functions) ---
def calculate_advanced_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1e-9)
    return df

def add_all_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    # ---------- Bullish ----------
    bullish_div = ((df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2)) &
                   (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'].shift(1) > df['rsi'].shift(2)))
    df['rsi_divergence'] = bullish_div.astype(int)
    ema3 = df['close'].ewm(span=3).mean()
    ema8 = df['close'].ewm(span=8).mean()
    ema21 = df['close'].ewm(span=21).mean()
    df['ema_3_8_21_cross'] = ((ema3 > ema8) & (ema8 > ema21) & (ema3.shift(1) <= ema8.shift(1))).astype(int)
    df['macd_zero_break'] = ((df['macd'] > 0) & (df['macd'].shift(1) <= 0)).astype(int)
    stoch_pop = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & (df['stoch_k'] < 20)
    df['stoch_oversold_pop'] = stoch_pop.astype(int)
    vol_ma = df['volume'].rolling(20).mean()
    df['vol_bb_break'] = ((df['close'] > df['bb_upper']) & (df['volume'] > 2*vol_ma)).astype(int)
    ema50 = df['close'].ewm(span=50).mean()
    ema200 = df['close'].ewm(span=200).mean()
    df['golden_cross'] = ((ema50 > ema200) & (ema50.shift(50) <= ema200.shift(50))).astype(int)
    support = df['low'].rolling(20).min()
    df['support_reclaim'] = ((df['close'] > support) & (df['close'].shift(1) <= support.shift(1))).astype(int)
    df['morning_star'] = 0
    atr_trail = df['close'] - 3*df['atr']
    df['atr_trail_flip'] = ((df['close'] > atr_trail) & (df['close'].shift(1) <= atr_trail.shift(1))).astype(int)
    df['fib_bounce'] = 0
    # ---------- Bearish ----------
    bear_div = ((df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2)) &
                (df['rsi'] < df['rsi'].shift(1)) & (df['rsi'].shift(1) < df['rsi'].shift(2)))
    df['bear_rsi_div'] = bear_div.astype(int)
    df['death_cross'] = ((ema50 < ema200) & (ema50.shift(50) >= ema200.shift(50))).astype(int)
    df['macd_zero_down'] = ((df['macd'] < 0) & (df['macd'].shift(1) >= 0)).astype(int)
    df['bear_engulf'] = 0
    df['break_support_vol'] = ((df['close'] < support) & (df['volume'] > 1.5*vol_ma)).astype(int)
    kelt_lower = ema21 - 1.5*df['atr']
    df['keltner_lower_break'] = ((df['close'] < kelt_lower) & (df['close'].shift(1) >= kelt_lower.shift(1))).astype(int)
    df['evening_star'] = 0
    # ---------- Neutral / Range ----------
    bb_width = df['bb_upper'] - df['bb_lower']
    df['bb_squeeze'] = (bb_width < 2*df['atr']).astype(int)
    rsi_50_rej = ((df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)) | ((df['rsi'] > 50) & (df['rsi'].shift(1) <= 50))
    df['rsi_50_reject'] = rsi_50_rej.astype(int)
    df['adx_no_trend'] = (df['adx'] < 20).astype(int)
    return df

def calculate_all_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("ℹ️ [Features] Calculating all features for TTM...")
    df_calc = df.copy()
    # --- Standard Features ---
    df_calc['atr'] = (df_calc['high'] - df_calc['low']).rolling(window=ATR_PERIOD).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    asset_returns = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = asset_returns.rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    
    # --- Advanced and Strategy Features ---
    df_calc = calculate_advanced_momentum_features(df_calc)
    df_calc = add_all_strategy_features(df_calc)
    
    logger.info("✅ [Features] All features calculated successfully.")
    return optimize_memory_usage(df_calc, log_prefix="All Features")

# --- دوال إعداد البيانات والتدريب (Data Prep & Training Functions) ---

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index)
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0: continue
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        for j in range(1, MAX_HOLD_PERIOD + 1):
            if i + j >= len(prices): break
            if prices.iloc[i + j] >= upper_barrier:
                labels.iloc[i] = 1; break
            if prices.iloc[i + j] <= lower_barrier:
                labels.iloc[i] = -1; break
    return labels

def create_sequences(X, y, sequence_length):
    xs, ys = [], []
    for i in range(len(X) - sequence_length):
        xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(xs), np.array(ys)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    global N_FEATURES
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    
    df_featured = calculate_all_features(df_15m, btc_df)
    
    # MTF Features
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    mtf_features = df_4h[['rsi_4h']]
    df_featured = df_featured.join(mtf_features).fillna(method='ffill')
    
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    # Remap target for PyTorch CrossEntropyLoss: {-1, 0, 1} -> {0, 1, 2}
    df_featured['target'] = df_featured['target'].replace({-1: 0, 0: 1, 1: 2})
    
    feature_columns = [col for col in df_featured.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'target']]
    N_FEATURES = len(feature_columns)

    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(subset=feature_columns, inplace=True)

    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes after cleaning. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    return X, y, feature_columns

def train_ttm_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Dict], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info(f"🧠 [TTM Train] Starting TTM model training...")

    # Use TimeSeriesSplit for a more robust train/test split
    tscv = TimeSeriesSplit(n_splits=5)
    train_index, test_index = list(tscv.split(X))[-1] # Use the last split for final train/test

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, SEQUENCE_LENGTH)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        logger.warning("⚠️ Not enough data to create sequences. Skipping training.")
        return None, None, None

    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TTM(n_features=N_FEATURES, sequence_length=SEQUENCE_LENGTH, d_model=D_MODEL, n_blocks=N_BLOCKS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"🔥 Training on {DEVICE} for {N_EPOCHS} epochs...")
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_train_loss:.4f}")

    # Final evaluation on test set
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(y_batch.cpu().numpy())
    
    if not all_true:
        logger.warning("⚠️ No predictions were made in the test set.")
        return None, None, None

    final_report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
    final_metrics = {
        'accuracy': accuracy_score(all_true, all_preds),
        'precision_class_2': final_report.get('2', {}).get('precision', 0), # Class 2 is BUY
        'recall_class_2': final_report.get('2', {}).get('recall', 0),
        'f1_score_class_2': final_report.get('2', {}).get('f1-score', 0),
        'precision_class_0': final_report.get('0', {}).get('precision', 0), # Class 0 is SELL
        'num_samples_trained': len(X_train_seq),
    }
    
    metrics_log_str = f"Accuracy: {final_metrics['accuracy']:.4f}, P(BUY): {final_metrics['precision_class_2']:.4f}, R(BUY): {final_metrics['recall_class_2']:.4f}"
    logger.info(f"📊 [TTM Train] Final Performance: {metrics_log_str}")

    return model.state_dict(), scaler, final_metrics

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        model_binary = pickle.dumps(model_bundle)
        metrics_json = json.dumps(metrics)
        with conn.cursor() as db_cur:
            db_cur.execute("""
                INSERT INTO ml_models (model_name, model_data, trained_at, metrics) 
                VALUES (%s, %s, NOW(), %s) ON CONFLICT (model_name) DO UPDATE SET 
                model_data = EXCLUDED.model_data, trained_at = NOW(), metrics = EXCLUDED.metrics;
            """, (model_name, model_binary, metrics_json))
        conn.commit()
        logger.info(f"✅ [DB Save] Model bundle '{model_name}' saved successfully.")
    except Exception as e:
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}"); conn.rollback()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def run_training_job():
    logger.info(f"🚀 Starting TTM model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    fetch_and_cache_btc_data()
    
    all_valid_symbols = get_validated_symbols(filename='crypto_list.txt')
    if not all_valid_symbols:
        logger.critical("❌ [Main] لم يتم العثور على رموز صالحة. سيتم الخروج."); return
    
    trained_symbols = get_trained_symbols_from_db()
    symbols_to_train = [s for s in all_valid_symbols if s not in trained_symbols]
    
    if not symbols_to_train:
        logger.info("✅ [Main] جميع الرموز مدربة بالفعل ومحدثة.");
        if conn: conn.close()
        return

    logger.info(f"ℹ️ [Main] Total: {len(all_valid_symbols)}. Trained: {len(trained_symbols)}. To Train: {len(symbols_to_train)}.")
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} new symbols.")
    
    successful_models, failed_models = 0, 0
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] بدء تدريب النموذج لـ {symbol} ---")
        try:
            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            
            if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty:
                logger.warning(f"⚠️ [Main] لا توجد بيانات كافية لـ {symbol}, سيتم التجاوز."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_15m, df_4h, btc_data_cache, symbol)
            del df_15m, df_4h; gc.collect()

            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            training_result = train_ttm_model(X, y)
            if not all(training_result) or training_result[0] is None:
                 logger.warning(f"⚠️ [Main] فشل تدريب النموذج لـ {symbol}."); failed_models += 1
                 del X, y, prepared_data; gc.collect()
                 continue
            
            model_state_dict, final_scaler, model_metrics = training_result
            
            if model_state_dict and final_scaler and model_metrics.get('precision_class_2', 0) > 0.35:
                model_bundle = {
                    'model_state_dict': model_state_dict, 
                    'scaler': final_scaler, 
                    'feature_names': feature_names,
                    'sequence_length': SEQUENCE_LENGTH,
                    'n_features': N_FEATURES,
                    'd_model': D_MODEL,
                    'n_blocks': N_BLOCKS
                }
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                logger.warning(f"⚠️ [Main] النموذج الخاص بـ {symbol} غير مفيد (Precision < 0.35). سيتم تجاهله."); failed_models += 1
            
            del X, y, prepared_data, training_result, model_state_dict, final_scaler, model_metrics; gc.collect()

        except Exception as e:
            logger.critical(f"❌ [Main] حدث خطأ فادح للرمز {symbol}: {e}", exc_info=True); failed_models += 1
            gc.collect()

        keep_db_alive()
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} new models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Processed this run: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: conn.close()
    logger.info("👋 [Main] انتهت مهمة تدريب النماذج.")

app = Flask(__name__)

@app.route('/')
def health_check():
    return "TTM ML Trainer (MemFix) service is running and healthy.", 200

if __name__ == "__main__":
    logger.info(f"PyTorch device set to: {DEVICE}")
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10002))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
