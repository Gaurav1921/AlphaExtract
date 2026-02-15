"""
Tests for src/backtesting/backtester.py and src/backtesting/metrics.py
"""

import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.backtesting import metrics as m
from src.backtesting.backtester import Backtester, BacktestResult


# --- Sample result data for metrics testing ---

@pytest.fixture
def sample_results():
    """Realistic backtest results for metrics testing."""
    return [
        {"signal": "STRONG_BUY", "score": 0.6, "actual_return_pct": 15.2, "actual_direction": "up", "ticker": "AAPL", "filing_date": "2023-01-20"},
        {"signal": "BUY", "score": 0.3, "actual_return_pct": 8.1, "actual_direction": "up", "ticker": "AAPL", "filing_date": "2022-01-18"},
        {"signal": "BUY", "score": 0.25, "actual_return_pct": -5.3, "actual_direction": "down", "ticker": "MSFT", "filing_date": "2023-07-30"},
        {"signal": "HOLD", "score": 0.05, "actual_return_pct": 1.2, "actual_direction": "flat", "ticker": "MSFT", "filing_date": "2022-07-28"},
        {"signal": "SELL", "score": -0.3, "actual_return_pct": -12.0, "actual_direction": "down", "ticker": "TSLA", "filing_date": "2023-01-30"},
        {"signal": "STRONG_SELL", "score": -0.55, "actual_return_pct": 22.0, "actual_direction": "up", "ticker": "TSLA", "filing_date": "2022-01-28"},
        {"signal": "HOLD", "score": 0.0, "actual_return_pct": 3.5, "actual_direction": "up", "ticker": "GOOGL", "filing_date": "2023-02-01"},
    ]


class TestHitRate:

    def test_basic(self, sample_results):
        rate = m.hit_rate(sample_results)
        # STRONG_BUY+up=hit, BUY+up=hit, BUY+down=miss, HOLD+flat=hit, SELL+down=hit, STRONG_SELL+up=miss, HOLD+up=miss
        # 4 hits out of 7 scorable
        assert abs(rate - 4/7) < 0.01

    def test_empty(self):
        assert m.hit_rate([]) == 0.0

    def test_all_correct(self):
        results = [
            {"signal": "BUY", "actual_direction": "up"},
            {"signal": "SELL", "actual_direction": "down"},
        ]
        assert m.hit_rate(results) == 1.0


class TestDirectionalAccuracy:

    def test_ignores_hold_and_flat(self, sample_results):
        acc = m.directional_accuracy(sample_results)
        # Only non-HOLD, non-flat: STRONG_BUY+up(hit), BUY+up(hit), BUY+down(miss), SELL+down(hit), STRONG_SELL+up(miss)
        # 3 hits out of 5
        assert abs(acc - 3/5) < 0.01

    def test_empty(self):
        assert m.directional_accuracy([]) == 0.0


class TestPrecisionBySignal:

    def test_structure(self, sample_results):
        precision = m.precision_by_signal(sample_results)
        assert isinstance(precision, dict)
        for signal, stats in precision.items():
            assert "correct" in stats
            assert "total" in stats
            assert "precision" in stats

    def test_strong_buy_precision(self, sample_results):
        precision = m.precision_by_signal(sample_results)
        assert precision.get("STRONG_BUY", {}).get("precision") == 1.0  # 1/1


class TestAvgReturnBySignal:

    def test_structure(self, sample_results):
        avg = m.avg_return_by_signal(sample_results)
        assert isinstance(avg, dict)
        for signal, stats in avg.items():
            assert "avg_return_pct" in stats
            assert "count" in stats

    def test_buy_returns(self, sample_results):
        avg = m.avg_return_by_signal(sample_results)
        buy_stats = avg.get("BUY", {})
        assert buy_stats["count"] == 2
        expected_avg = (8.1 + (-5.3)) / 2
        assert abs(buy_stats["avg_return_pct"] - expected_avg) < 0.1


class TestSharpeRatio:

    def test_returns_float(self, sample_results):
        sharpe = m.sharpe_ratio(sample_results)
        assert sharpe is not None
        assert isinstance(sharpe, float)

    def test_too_few_results(self):
        assert m.sharpe_ratio([]) is None
        assert m.sharpe_ratio([{"actual_return_pct": 5}]) is None


class TestConfusionMatrix:

    def test_structure(self, sample_results):
        cm = m.confusion_matrix(sample_results)
        assert "bullish" in cm
        assert "neutral" in cm
        assert "bearish" in cm
        for row in cm.values():
            assert "up" in row
            assert "flat" in row
            assert "down" in row

    def test_values(self, sample_results):
        cm = m.confusion_matrix(sample_results)
        # STRONG_BUY+up -> bullish+up
        assert cm["bullish"]["up"] >= 1
        # SELL+down -> bearish+down
        assert cm["bearish"]["down"] >= 1


class TestCompareModels:

    def test_comparison_structure(self, sample_results):
        comparison = m.compare_models(sample_results, sample_results, "A", "B")
        assert "A" in comparison
        assert "B" in comparison
        assert "hit_rate" in comparison["A"]
        assert "directional_accuracy" in comparison["B"]


class TestBacktestResult:

    def test_to_dict(self, sample_results):
        result = BacktestResult(sample_results, {"mode": "finbert", "tickers": ["AAPL"]})
        d = result.to_dict()
        assert "summary" in d
        assert "config" in d
        assert "results" in d
        assert d["summary"]["total_signals"] == 7

    def test_properties(self, sample_results):
        result = BacktestResult(sample_results, {})
        assert 0 <= result.hit_rate <= 1
        assert 0 <= result.directional_accuracy <= 1
        assert isinstance(result.precision_by_signal, dict)


class TestBacktester:

    def test_backtest_finbert(self, tmp_data_dir, sample_sentiment_result):
        """Test FinBERT backtest with mocked market data."""
        # Create sentiment file
        sentiment_dir = tmp_data_dir / "sentiment"
        (sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text(
            json.dumps(sample_sentiment_result)
        )

        # Mock market data
        mock_market = MagicMock()
        mock_market.get_filing_outcome.return_value = {
            "return_pct": 10.5,
            "direction": "up",
            "window_days": 90,
        }

        with patch("src.backtesting.backtester.Settings") as mock_s:
            mock_s.SENTIMENT_DIR = sentiment_dir
            mock_s.MARKET_FLAT_THRESHOLD = 0.02

            backtester = Backtester(return_window=90, market_provider=mock_market)
            result = backtester.run(["AAPL"], mode="finbert")

        assert len(result.results) == 1
        assert result.results[0]["ticker"] == "AAPL"
        assert result.results[0]["actual_return_pct"] == 10.5

    def test_save_results(self, tmp_path, sample_results):
        result = BacktestResult(sample_results, {"mode": "test"})

        with patch("src.backtesting.backtester.Settings") as mock_s:
            mock_s.BACKTEST_DIR = tmp_path
            backtester = Backtester()
            path = backtester.save_results(result, name="test")

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["config"]["mode"] == "test"
