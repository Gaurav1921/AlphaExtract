"""
Tests for Portfolio Signal Aggregation
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.analysis.portfolio import (
    aggregate_portfolio,
    equal_weight_portfolio,
    sector_portfolio,
    multi_portfolio_comparison,
    save_portfolio,
    HoldingSignal,
    PortfolioSignal,
    _get_ticker_sector,
)


@pytest.fixture
def mock_sentiment_dir(tmp_path):
    """Create a temp directory with sentiment and ensemble files."""
    sentiment_dir = tmp_path / "sentiment"
    sentiment_dir.mkdir()

    # AAPL sentiment
    (sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text(
        json.dumps({
            "overall": {"compound": 0.35, "signal": "BUY"},
            "sections": {"item_7": {"scores": {"compound": 0.4}}},
        }),
        encoding="utf-8",
    )

    # AAPL ensemble
    (sentiment_dir / "AAPL_2024-01-15_ensemble.json").write_text(
        json.dumps({
            "ensemble": {"score": 0.28, "signal": "BUY"},
        }),
        encoding="utf-8",
    )

    # MSFT sentiment
    (sentiment_dir / "MSFT_2024-07-30_sentiment.json").write_text(
        json.dumps({
            "overall": {"compound": 0.20, "signal": "BUY"},
            "sections": {"item_7": {"scores": {"compound": 0.25}}},
        }),
        encoding="utf-8",
    )

    # GOOGL sentiment
    (sentiment_dir / "GOOGL_2024-02-01_sentiment.json").write_text(
        json.dumps({
            "overall": {"compound": -0.10, "signal": "HOLD"},
            "sections": {"item_7": {"scores": {"compound": -0.05}}},
        }),
        encoding="utf-8",
    )

    return sentiment_dir


class TestHoldingSignal:
    def test_creates_holding(self):
        h = HoldingSignal(ticker="AAPL", weight=0.5, sentiment_score=0.3)
        assert h.ticker == "AAPL"
        assert h.weight == 0.5

    def test_to_dict(self):
        h = HoldingSignal(ticker="MSFT", weight=0.25, sector="Technology")
        d = h.to_dict()
        assert d["ticker"] == "MSFT"
        assert d["sector"] == "Technology"


class TestGetTickerSector:
    def test_finds_sector(self):
        assert _get_ticker_sector("AAPL") == "Technology"
        assert _get_ticker_sector("JPM") == "Financials"
        assert _get_ticker_sector("XOM") == "Energy"

    def test_unknown_ticker(self):
        assert _get_ticker_sector("ZZZZ") is None


class TestAggregatePortfolio:
    def test_basic_aggregation(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {"Technology": ["AAPL", "MSFT", "GOOGL"]}
            mock_settings.generate_signal = lambda score: "BUY" if score >= 0.2 else ("HOLD" if score >= -0.2 else "SELL")

            result = aggregate_portfolio(
                {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2},
                name="Test Portfolio",
            )

        assert result.name == "Test Portfolio"
        assert result.holdings_count == 3
        assert result.holdings_with_data == 3
        assert result.coverage_pct == 100.0

    def test_normalizes_weights(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {"Technology": ["AAPL", "MSFT"]}
            mock_settings.generate_signal = lambda score: "HOLD"

            result = aggregate_portfolio({"AAPL": 2, "MSFT": 8})

        # Weights should be normalized to sum to 1.0
        weights = [h["weight"] for h in result.holdings]
        assert abs(sum(weights) - 1.0) < 0.01

    def test_handles_missing_data(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {}
            mock_settings.generate_signal = lambda score: "HOLD"

            result = aggregate_portfolio({"AAPL": 0.5, "UNKNOWN": 0.5})

        assert result.holdings_count == 2
        assert result.holdings_with_data == 1
        assert result.coverage_pct == 50.0

    def test_sector_breakdown(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {
                "Technology": ["AAPL", "MSFT", "GOOGL"],
            }
            mock_settings.generate_signal = lambda score: "BUY" if score > 0 else "HOLD"

            result = aggregate_portfolio({"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2})

        assert "Technology" in result.sector_breakdown
        assert result.sector_breakdown["Technology"]["count"] == 3

    def test_risk_concentration(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {"Technology": ["AAPL", "MSFT"]}
            mock_settings.generate_signal = lambda score: "HOLD"

            result = aggregate_portfolio({"AAPL": 0.8, "MSFT": 0.2})

        assert result.risk_concentration["max_single_holding"]["ticker"] == "AAPL"
        assert result.risk_concentration["herfindahl_index"] > 0

    def test_signal_distribution(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {}
            mock_settings.generate_signal = lambda score: "BUY" if score > 0 else "HOLD"

            result = aggregate_portfolio({"AAPL": 0.5, "MSFT": 0.5})

        assert isinstance(result.signal_distribution, dict)
        assert sum(result.signal_distribution.values()) == 2


class TestEqualWeightPortfolio:
    def test_equal_weights(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {}
            mock_settings.generate_signal = lambda score: "HOLD"

            result = equal_weight_portfolio(["AAPL", "MSFT"])

        weights = [h["weight"] for h in result.holdings]
        assert all(abs(w - 0.5) < 0.01 for w in weights)


class TestSectorPortfolio:
    def test_sector_portfolio(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {"Technology": ["AAPL", "MSFT", "GOOGL"]}
            mock_settings.generate_signal = lambda score: "HOLD"

            result = sector_portfolio("Technology")

        assert result.holdings_count == 3
        assert "Technology" in result.name

    def test_unknown_sector(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {}
            mock_settings.generate_signal = lambda score: "HOLD"

            result = sector_portfolio("Nonexistent")

        assert result.holdings_count == 0


class TestMultiPortfolioComparison:
    def test_compare_two_portfolios(self, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {}
            mock_settings.generate_signal = lambda score: "BUY" if score > 0 else "HOLD"

            result = multi_portfolio_comparison({
                "Tech Heavy": {"AAPL": 0.7, "MSFT": 0.3},
                "Balanced": {"AAPL": 0.34, "MSFT": 0.33, "GOOGL": 0.33},
            })

        assert "Tech Heavy" in result["portfolios"]
        assert "Balanced" in result["portfolios"]
        assert result["best"] is not None
        assert len(result["ranking"]) == 2


class TestSavePortfolio:
    def test_saves_to_json(self, tmp_path, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {}
            mock_settings.generate_signal = lambda score: "HOLD"

            result = equal_weight_portfolio(["AAPL", "MSFT"], name="Test")

        path = save_portfolio(result, output_dir=tmp_path)
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == "Test"
        assert data["holdings_count"] == 2


class TestPortfolioSignal:
    def test_to_dict(self):
        ps = PortfolioSignal(
            name="Test",
            holdings_count=2,
            holdings_with_data=2,
            coverage_pct=100.0,
            weighted_sentiment=0.25,
            weighted_ensemble=None,
            portfolio_signal="BUY",
            sector_breakdown={},
            signal_distribution={"BUY": 2},
            risk_concentration={"max_single_holding": {}, "max_single_sector": {}, "herfindahl_index": 0.5},
            holdings=[],
        )
        d = ps.to_dict()
        assert d["name"] == "Test"
        assert d["portfolio_signal"] == "BUY"
