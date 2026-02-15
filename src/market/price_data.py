"""
Market Data Provider
---------------------
Fetches and caches historical stock prices from Yahoo Finance.
Provides filing-date-aligned return calculations for backtesting.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Fetches and caches stock price data for backtest evaluation."""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Settings.MARKET_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}_prices.csv"

    def _load_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load price data — from memory cache, disk cache, or yfinance."""
        ticker = ticker.upper()

        if ticker in self._price_cache:
            return self._price_cache[ticker]

        cache_file = self._cache_path(ticker)
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    self._price_cache[ticker] = df
                    logger.debug(f"Loaded {ticker} prices from cache ({len(df)} rows)")
                    return df
            except Exception as e:
                logger.warning(f"Corrupt cache for {ticker}, re-fetching: {e}")

        return self._fetch_and_cache(ticker)

    def _fetch_and_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance and save to disk."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — run: pip install yfinance")
            return None

        ticker = ticker.upper()
        logger.info(f"Fetching price data for {ticker} from Yahoo Finance...")

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="max", auto_adjust=True)

            if df.empty:
                logger.error(f"No price data returned for {ticker}")
                return None

            # Keep only what we need — Close price
            df = df[["Close"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()

            # Save to cache
            df.to_csv(self._cache_path(ticker))
            self._price_cache[ticker] = df
            logger.info(f"Cached {ticker}: {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {ticker} from Yahoo Finance: {e}")
            return None

    def get_price_at_date(self, ticker: str, date: str) -> Optional[float]:
        """
        Get closing price at or near a given date.

        If the date falls on a weekend/holiday, returns the next available trading day.
        """
        df = self._load_prices(ticker)
        if df is None:
            return None

        target = pd.Timestamp(date)

        # Find the nearest trading day on or after the target date
        mask = df.index >= target
        if not mask.any():
            # Date is after all available data
            logger.warning(f"No price data for {ticker} on or after {date}")
            return None

        nearest_idx = df.index[mask][0]
        price = float(df.loc[nearest_idx, "Close"])

        gap_days = (nearest_idx - target).days
        if gap_days > 5:
            logger.warning(f"{ticker} price lookup: {date} → {nearest_idx.date()} ({gap_days} day gap)")

        return price

    def get_returns(self, ticker: str, filing_date: str, windows: List[int] = None) -> Dict[int, Optional[float]]:
        """
        Calculate actual returns over multiple time windows after a filing date.

        Args:
            ticker: Stock ticker.
            filing_date: Date of the 10-K filing (YYYY-MM-DD).
            windows: List of days after filing to measure (default from Settings).

        Returns:
            Dict mapping window (days) to return percentage, or None if unavailable.
        """
        windows = windows or Settings.MARKET_RETURN_WINDOWS
        df = self._load_prices(ticker)
        if df is None:
            return {w: None for w in windows}

        base_price = self.get_price_at_date(ticker, filing_date)
        if base_price is None or base_price == 0:
            return {w: None for w in windows}

        base_date = pd.Timestamp(filing_date)
        results = {}

        for window in windows:
            target_date = base_date + timedelta(days=window)
            future_price = self.get_price_at_date(ticker, str(target_date.date()))
            if future_price is not None:
                results[window] = (future_price - base_price) / base_price
            else:
                results[window] = None

        return results

    def get_filing_outcome(
        self, ticker: str, filing_date: str, window: int = 90
    ) -> Dict:
        """
        Classify the market outcome after a filing.

        Returns:
            Dict with return_pct, direction ("up"/"down"/"flat"), and prices.
        """
        returns = self.get_returns(ticker, filing_date, windows=[window])
        ret = returns.get(window)

        if ret is None:
            return {"return_pct": None, "direction": "unknown", "window_days": window}

        if ret > Settings.MARKET_FLAT_THRESHOLD:
            direction = "up"
        elif ret < -Settings.MARKET_FLAT_THRESHOLD:
            direction = "down"
        else:
            direction = "flat"

        return {
            "return_pct": round(ret * 100, 2),
            "direction": direction,
            "window_days": window,
            "filing_date": filing_date,
            "ticker": ticker.upper(),
        }

    def refresh_cache(self, ticker: str) -> bool:
        """Force re-fetch price data for a ticker."""
        ticker = ticker.upper()
        self._price_cache.pop(ticker, None)
        cache_file = self._cache_path(ticker)
        if cache_file.exists():
            cache_file.unlink()
        df = self._fetch_and_cache(ticker)
        return df is not None

    def get_cached_tickers(self) -> List[str]:
        """List all tickers with cached price data."""
        return [f.stem.replace("_prices", "") for f in self.cache_dir.glob("*_prices.csv")]
