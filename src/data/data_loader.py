import os
import datetime as dt
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from pandas_datareader.stooq import StooqDailyReader
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = "data/cache"
REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


class StockDataLoader:
    def __init__(self, symbol: str, start_date: str = None, end_date: str = None):
        self.symbol = symbol.upper()
        self.end_date = end_date if end_date else dt.date.today().isoformat()
        self.start_date = start_date if start_date else (dt.date.today() - dt.timedelta(days = 365 * 2)).isoformat()
        self.source = None

    def get_source(self):
        return self.source

    def fetch_data(self) -> pd.DataFrame:
        """
        Tries to fetch data in the order: YFinance -> Alpha Vantage -> Brapi.
        """
        df = pd.DataFrame()

        # 1. Try YFinance (Primary)
        if df.empty:
            try:
                df = self._fetch_yfinance()
                if not df.empty:
                    self.source = "yfinance"
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("YFinance failed for %s: %s", self.symbol, e)

        # 2. Try Stooq (Fallback 1)
        if df.empty:
            try:
                df = self._fetch_stooq()
                if not df.empty:
                    self.source = "stooq"
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Stooq failed for %s: %s", self.symbol, e)

        # 3. Try Brapi (Fallback 2)
        if df.empty:
            try:
                token = os.getenv("BRAPI_TOKEN")
                if token:
                    df = self._fetch_brapi(token)
                    if not df.empty:
                        self.source = "brapi"
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Brapi failed for %s: %s", self.symbol, e)

        #4. Try cache (Fallback 3)

        if df.empty:
            cache = self._load_cache()
            if cache is not None and not cache.empty:
                df = cache.copy()
                self.source = "cache"

        if df.empty:
            raise RuntimeError(f"Could not fetch data for {self.symbol} from any source.")

        return self._normalize(df)

    def _fetch_yfinance(self) -> pd.DataFrame:
        """Fetches from Yahoo Finance"""
        ticker = self.symbol

        df = yf.download(ticker, start = self.start_date, end = self.end_date, progress = False, auto_adjust = True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df


    def _fetch_stooq(self) -> pd.DataFrame:

        reader = StooqDailyReader(self.symbol.lower())
        df = reader.read()
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index, errors = "coerce", utc = False)
        return df


    def _fetch_brapi(self, token: str) -> pd.DataFrame:
        """Fetches from Brapi.dev with dynamic range selection."""
        start_dt = pd.to_datetime(self.start_date).date()
        days_diff = (dt.date.today() - start_dt).days

        symbol = self.symbol.split('.')[0] if '.' in self.symbol else self.symbol


        # Brapi valid ranges: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        if days_diff <= 365:
            r_param = "1y"
        elif days_diff <= 365 * 2:
            r_param = "2y"
        elif days_diff <= 365 * 5:
            r_param = "5y"
        elif days_diff <= 365 * 10:
            r_param = "10y"
        else:
            r_param = "max"

        url = f"https://brapi.dev/api/quote/{symbol}"
        params = {"token": token, "range": r_param, "interval": "1d"}

        r = requests.get(url, params = params, timeout = 15)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        history = results[0].get("historicalDataPrice", [])
        df = pd.DataFrame(history)

        if "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"], unit="s")
            df.set_index("Date", inplace=True)

        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df.rename(columns=rename_map, inplace=True)
        return df

    def _load_cache(self) -> Optional[pd.DataFrame]:
        path = os.path.join(CACHE_DIR, f"{self.symbol.upper()}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, index_col = 0)
                df = self._ensure_datetime_index(df)
                return df

            except Exception:
                return None
        return None

    def has_local_cache(self) -> bool:
        """Returns True if there's a usable cache file for this symbol."""
        df = self._load_cache()
        return df is not None and not df.empty

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures standard columns and datetime index."""
        # Convert index to datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors = "coerce")

        df = df.sort_index()

        # Ensure all required columns exist and are numeric
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors = "coerce")

        # Filter date range
        mask = (df.index >= pd.to_datetime(self.start_date)) & (df.index <= pd.to_datetime(self.end_date))
        df = df.loc[mask]

        return df[REQUIRED_COLS].dropna()

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors = "coerce", utc = False)
        return df.dropna().sort_index()


if __name__ == "__main__":
    # Simple smoke test for development. Run `python -m src.data.data_loader`.
    loader = StockDataLoader("MSFT")
    try:
        data = loader.fetch_data()
        data = loader.add_technical_indicators(data)
        import logging
        logging.getLogger(__name__).info("Success! Source: %s", loader.source)
        print(data.tail())
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("Data loader test failed: %s", e)