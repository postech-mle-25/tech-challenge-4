# src/data/data_loader.py
import os
import io
import time
import math
import random
import datetime as dt
from typing import Optional, List

import numpy as np
import pandas as pd

# fontes
import yfinance as yf

try:
    import requests
except Exception:
    requests = None

try:
    from pandas_datareader.stooq import StooqDailyReader
except Exception:
    StooqDailyReader = None

try:
    from alpha_vantage.timeseries import TimeSeries
except Exception:
    TimeSeries = None


CACHE_DIR = "data/cache"
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
    return df.dropna().sort_index()


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[REQUIRED_COLS].dropna().sort_index()


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol.upper()}.csv")
    df.to_csv(path, index=True)


def _load_cache(symbol: str) -> Optional[pd.DataFrame]:
    path = os.path.join(CACHE_DIR, f"{symbol.upper()}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0)
            df = _ensure_datetime_index(df)
            return _normalize_cols(df)
        except Exception:
            return None
    return None


def _cut_period(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if start_date:
        df = df[df.index.date >= pd.to_datetime(start_date).date()]
    if end_date:
        df = df[df.index.date <= pd.to_datetime(end_date).date()]
    return df


def _synthetic_series(start_date: str, end_date: str, n_days_min: int = 260, seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if not end_date:
        end_date = dt.date.today().isoformat()
    if not start_date:
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=365 * 3)).date().isoformat()

    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    if len(dates) < n_days_min:
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=n_days_min * 2)).date().isoformat()
        dates = pd.bdate_range(start=start_date, end=end_date, freq="B")

    S0, mu, sigma = 100.0, 0.08, 0.25
    dt_daily = 1.0 / 252.0
    prices = [S0]
    for _ in range(1, len(dates)):
        z = np.random.normal()
        S_prev = prices[-1]
        S_new = S_prev * math.exp((mu - 0.5 * sigma * sigma) * dt_daily + sigma * math.sqrt(dt_daily) * z)
        prices.append(S_new)

    close = np.array(prices)
    high = close * (1 + np.random.uniform(0.0, 0.02, size=len(close)))
    low = close * (1 - np.random.uniform(0.0, 0.02, size=len(close)))
    open_ = close * (1 + np.random.uniform(-0.01, 0.01, size=len(close)))
    volume = np.random.randint(1_000_000, 5_000_000, size=len(close))

    df = pd.DataFrame(
        {"Open": open_, "High": np.maximum.reduce([open_, high, close]),
         "Low": np.minimum.reduce([open_, low, close]), "Close": close, "Volume": volume},
        index=dates,
    )
    return _normalize_cols(df)


def _try_yfinance(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Tenta baixar do Yahoo; silencioso e curto. Retorna None em qualquer erro."""
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            repair=True,
            ignore_tz=True,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return _normalize_cols(_ensure_datetime_index(df))
    except Exception:
        return None
    return None


def _try_stooq_csv(symbol: str) -> Optional[pd.DataFrame]:
    if requests is None:
        return None
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        return _normalize_cols(df)
    except Exception:
        return None


def _try_stooq_reader(symbol: str) -> Optional[pd.DataFrame]:
    if StooqDailyReader is None:
        return None
    try:
        reader = StooqDailyReader(symbol.lower())
        df = reader.read()
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
        return _normalize_cols(df)
    except Exception:
        return None


def _alpha_symbol_variants(symbol: str) -> List[str]:
    variants = [symbol]
    if symbol.endswith(".SA"):
        base = symbol.replace(".SA", "")
        variants += [f"BVMF:{base}", f"SAO:{base}", base]
    return list(dict.fromkeys(variants))  # unique order


def _try_alpha_vantage(symbol: str) -> Optional[pd.DataFrame]:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key or TimeSeries is None:
        return None
    ts = TimeSeries(key=api_key, output_format="pandas")
    for s in _alpha_symbol_variants(symbol):
        try:
            data, _ = ts.get_daily_adjusted(symbol=s, outputsize="full")
            if data is None or data.empty:
                continue
            data.index = pd.to_datetime(data.index, errors="coerce", utc=False)
            data = data.sort_index()
            rename_map = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "6. volume": "Volume",
            }
            if not set(rename_map.keys()).issubset(set(data.columns)):
                continue
            df = data.rename(columns=rename_map)[list(rename_map.values())].copy()
            df = _normalize_cols(df)
            if not df.empty:
                return df
        except Exception:
            continue
    return None


class StockDataLoader:
    """
    Loader robusto com múltiplas fontes e fallbacks.
    Atributos:
      - source: 'stooq_csv' | 'stooq_reader' | 'alpha_vantage' | 'yfinance' | 'cache' | 'synthetic'
    """

    def __init__(self, symbol: str, start_date: str = None, end_date: str = None, allow_synthetic: bool = True):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.allow_synthetic = allow_synthetic
        self.source: Optional[str] = None

    def get_source(self) -> Optional[str]:
        return self.source

    def fetch_data(self) -> pd.DataFrame:
        if not self.end_date:
            self.end_date = dt.date.today().isoformat()
        if not self.start_date:
            self.start_date = (dt.date.today() - dt.timedelta(days=365 * 3)).isoformat()

        # nunca deixe df virar None; mantenha um DataFrame vazio como padrão
        df: pd.DataFrame = pd.DataFrame()
        disable_yf = os.getenv("DISABLE_YFINANCE", "0").lower() in ("1", "true", "yes")

        # 1) stooq csv
        if df.empty:
            tmp = _try_stooq_csv(self.symbol)
            if tmp is not None and not tmp.empty:
                df = tmp
                self.source = "stooq_csv"

        # 2) stooq reader
        if df.empty:
            tmp = _try_stooq_reader(self.symbol)
            if tmp is not None and not tmp.empty:
                df = tmp
                self.source = "stooq_reader"

        # 3) alpha vantage
        if df.empty:
            tmp = _try_alpha_vantage(self.symbol)
            if tmp is not None and not tmp.empty:
                df = tmp
                self.source = "alpha_vantage"

        # 4) cache
        if df.empty:
            cache = _load_cache(self.symbol)
            if cache is not None and not cache.empty:
                df = cache.copy()
                self.source = "cache"

        # 5) yfinance (se permitido)
        if df.empty and not disable_yf:
            tmp = _try_yfinance(self.symbol, self.start_date, self.end_date)
            if tmp is not None and not tmp.empty:
                df = tmp
                self.source = "yfinance"

        # 6) synthetic
        if df.empty and self.allow_synthetic:
            df = _synthetic_series(self.start_date, self.end_date)
            self.source = "synthetic"

        if df.empty:
            raise RuntimeError(
                f"Falha ao baixar dados para {self.symbol}. "
                f"Tente outro ticker (ex.: MSFT, ITUB4.SA) ou verifique sua rede."
            )

        # cacheia somente fontes reais
        if self.source in {"stooq_csv", "stooq_reader", "alpha_vantage", "yfinance"}:
            try:
                _save_cache(self.symbol, df)
            except Exception:
                pass

        df = _cut_period(df, self.start_date, self.end_date)
        if df.empty:
            raise RuntimeError(
                f"Sem dados no período solicitado para {self.symbol} "
                f"({self.start_date} → {self.end_date})."
            )
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not set(REQUIRED_COLS).issubset(out.columns):
            return out

        out["SMA_10"] = out["Close"].rolling(window=10, min_periods=10).mean()
        out["SMA_20"] = out["Close"].rolling(window=20, min_periods=20).mean()
        out["EMA_10"] = out["Close"].ewm(span=10, adjust=False).mean()
        out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()

        delta = out["Close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / (loss.replace(0, np.nan))
        out["RSI_14"] = 100 - (100 / (1 + rs))
        out["RSI_14"] = out["RSI_14"].bfill()

        ema12 = out["Close"].ewm(span=12, adjust=False).mean()
        ema26 = out["Close"].ewm(span=26, adjust=False).mean()
        out["MACD"] = ema12 - ema26
        out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

        ma20 = out["Close"].rolling(window=20, min_periods=20).mean()
        sd20 = out["Close"].rolling(window=20, min_periods=20).std()
        out["BB_upper"] = ma20 + 2 * sd20
        out["BB_lower"] = ma20 - 2 * sd20

        return out.dropna()
