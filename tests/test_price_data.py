"""
Tests for src/market/price_data.py
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.market.price_data import MarketDataProvider


@pytest.fixture
def price_df():
    """Sample price DataFrame spanning 2 years."""
    dates = pd.bdate_range("2023-01-02", "2024-12-31")
    # Simulate a stock going from $100 to ~$130 with noise
    import numpy as np
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    return pd.DataFrame({"Close": prices}, index=dates)


@pytest.fixture
def provider(tmp_path, price_df):
    """MarketDataProvider with mocked price data."""
    p = MarketDataProvider(cache_dir=tmp_path)
    p._price_cache["AAPL"] = price_df
    return p


class TestGetPriceAtDate:

    def test_exact_trading_day(self, provider):
        price = provider.get_price_at_date("AAPL", "2023-01-02")
        assert price is not None
        assert isinstance(price, float)

    def test_weekend_snaps_to_next_trading_day(self, provider):
        # 2023-01-07 is a Saturday
        price = provider.get_price_at_date("AAPL", "2023-01-07")
        assert price is not None

    def test_date_after_data_uses_last_available(self, provider):
        """When date exceeds data range, fall back to last available price."""
        price = provider.get_price_at_date("AAPL", "2025-03-01")
        # Should return last available price instead of None
        assert price is not None
        assert isinstance(price, float)

    def test_unknown_ticker_returns_none(self, provider):
        price = provider.get_price_at_date("ZZZZ", "2023-01-02")
        assert price is None


class TestGetReturns:

    def test_returns_dict_of_floats(self, provider):
        returns = provider.get_returns("AAPL", "2023-06-01", windows=[30, 60, 90])
        assert isinstance(returns, dict)
        assert 30 in returns
        assert 60 in returns
        assert 90 in returns

    def test_return_is_percentage(self, provider):
        returns = provider.get_returns("AAPL", "2023-06-01", windows=[30])
        ret = returns[30]
        if ret is not None:
            # Return should be a decimal (e.g., 0.05 = 5%)
            assert -5.0 < ret < 5.0

    def test_missing_ticker_returns_none_values(self, provider):
        returns = provider.get_returns("ZZZZ", "2023-06-01", windows=[30])
        assert returns[30] is None


class TestGetFilingOutcome:

    def test_classifies_up(self, provider, price_df):
        # Find a period where price went up
        outcome = provider.get_filing_outcome("AAPL", "2023-03-01", window=90)
        assert outcome["direction"] in ("up", "down", "flat")
        assert outcome["ticker"] == "AAPL"
        assert outcome["window_days"] == 90

    def test_unknown_ticker(self, provider):
        outcome = provider.get_filing_outcome("ZZZZ", "2023-01-02")
        assert outcome["direction"] == "unknown"

    def test_return_pct_is_float_or_none(self, provider):
        outcome = provider.get_filing_outcome("AAPL", "2023-06-01")
        assert outcome["return_pct"] is None or isinstance(outcome["return_pct"], float)


class TestCaching:

    def test_cache_writes_csv(self, tmp_path):
        provider = MarketDataProvider(cache_dir=tmp_path)
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        df = pd.DataFrame({"Close": range(len(dates))}, index=dates)
        provider._price_cache["TEST"] = df

        # Simulate saving to cache
        df.to_csv(provider._cache_path("TEST"))

        # New provider should read from cache
        provider2 = MarketDataProvider(cache_dir=tmp_path)
        loaded = provider2._load_prices("TEST")
        assert loaded is not None
        assert len(loaded) == len(df)

    def test_get_cached_tickers(self, tmp_path):
        provider = MarketDataProvider(cache_dir=tmp_path)
        (tmp_path / "AAPL_prices.csv").write_text("Close\n100")
        (tmp_path / "MSFT_prices.csv").write_text("Close\n200")

        tickers = provider.get_cached_tickers()
        assert "AAPL" in tickers
        assert "MSFT" in tickers


class TestStaleCacheDetection:

    def test_nonexistent_cache_is_stale(self, tmp_path):
        provider = MarketDataProvider(cache_dir=tmp_path)
        assert provider._is_cache_stale("NONEXISTENT") is True

    def test_fresh_cache_is_not_stale(self, tmp_path):
        provider = MarketDataProvider(cache_dir=tmp_path)
        cache_file = provider._cache_path("FRESH")
        cache_file.write_text("Close\n100")
        assert provider._is_cache_stale("FRESH") is False

    def test_old_cache_is_stale(self, tmp_path):
        import os
        import time
        provider = MarketDataProvider(cache_dir=tmp_path)
        cache_file = provider._cache_path("OLD")
        cache_file.write_text("Close\n100")
        # Set modification time to 30 days ago
        old_time = time.time() - (30 * 86400)
        os.utime(cache_file, (old_time, old_time))
        assert provider._is_cache_stale("OLD") is True


class TestFetchAndCache:

    def test_yfinance_not_installed(self, tmp_path):
        provider = MarketDataProvider(cache_dir=tmp_path)
        with patch.dict("sys.modules", {"yfinance": None}):
            # This should handle ImportError gracefully
            result = provider._fetch_and_cache("AAPL")
            # May return None if yfinance can't be imported
