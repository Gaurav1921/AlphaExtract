"""
Tests for Multi-Quarter Comparison Engine
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.analysis.comparison import (
    load_all_filings,
    compare_quarters,
    compare_tickers,
    section_deep_dive,
)


@pytest.fixture
def sentiment_data_2023():
    return {
        "ticker": "TEST",
        "filing_date": "2023-02-15",
        "sections": {
            "item_1a": {
                "scores": {"compound": -0.15, "positive": 0.2, "negative": 0.35, "neutral": 0.45},
                "signal": "HOLD",
                "word_count": 5000,
            },
            "item_7": {
                "scores": {"compound": 0.25, "positive": 0.45, "negative": 0.2, "neutral": 0.35},
                "signal": "BUY",
                "word_count": 12000,
            },
            "item_8": {
                "scores": {"compound": 0.10, "positive": 0.35, "negative": 0.25, "neutral": 0.40},
                "signal": "HOLD",
                "word_count": 3000,
            },
        },
        "overall": {"compound": 0.15, "signal": "HOLD"},
    }


@pytest.fixture
def sentiment_data_2024():
    return {
        "ticker": "TEST",
        "filing_date": "2024-02-20",
        "sections": {
            "item_1a": {
                "scores": {"compound": -0.05, "positive": 0.25, "negative": 0.30, "neutral": 0.45},
                "signal": "HOLD",
                "word_count": 5500,
            },
            "item_7": {
                "scores": {"compound": 0.35, "positive": 0.50, "negative": 0.15, "neutral": 0.35},
                "signal": "BUY",
                "word_count": 13000,
            },
            "item_8": {
                "scores": {"compound": 0.20, "positive": 0.40, "negative": 0.20, "neutral": 0.40},
                "signal": "BUY",
                "word_count": 3200,
            },
        },
        "overall": {"compound": 0.28, "signal": "BUY"},
    }


@pytest.fixture
def ensemble_data_2024():
    return {
        "ticker": "TEST",
        "filing_date": "2024-02-20",
        "signals": {
            "finbert": {"score": 0.28, "signal": "BUY", "weight": 0.60},
            "keywords": {"score": -0.05, "signal": "HOLD", "weight": 0.40},
            "llm": {"score": 0.0, "signal": "N/A", "weight": 0.00},
        },
        "ensemble": {"score": 0.148, "signal": "HOLD", "components_agree": False},
    }


@pytest.fixture
def mock_sentiment_dir(tmp_path, sentiment_data_2023, sentiment_data_2024, ensemble_data_2024):
    """Create a temporary sentiment directory with test files."""
    sentiment_dir = tmp_path / "sentiment"
    sentiment_dir.mkdir()

    # Write sentiment files
    (sentiment_dir / "TEST_2023-02-15_sentiment.json").write_text(
        json.dumps(sentiment_data_2023), encoding="utf-8"
    )
    (sentiment_dir / "TEST_2024-02-20_sentiment.json").write_text(
        json.dumps(sentiment_data_2024), encoding="utf-8"
    )
    (sentiment_dir / "TEST_2024-02-20_ensemble.json").write_text(
        json.dumps(ensemble_data_2024), encoding="utf-8"
    )

    return sentiment_dir


class TestLoadAllFilings:
    def test_loads_filings_chronologically(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            filings = load_all_filings("TEST")

        assert len(filings) == 2
        assert filings[0]["filing_date"] == "2023-02-15"
        assert filings[1]["filing_date"] == "2024-02-20"
        assert filings[0]["year"] == "2023"
        assert filings[1]["year"] == "2024"

    def test_loads_ensemble_when_available(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            filings = load_all_filings("TEST")

        assert filings[0]["ensemble"] is None  # No ensemble for 2023
        assert filings[1]["ensemble"] is not None  # Has ensemble for 2024
        assert filings[1]["ensemble"]["ensemble"]["score"] == 0.148

    def test_empty_for_unknown_ticker(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            filings = load_all_filings("UNKNOWN")

        assert filings == []

    def test_case_insensitive(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            filings = load_all_filings("test")

        assert len(filings) == 2


class TestCompareQuarters:
    def test_full_comparison(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = compare_quarters("TEST")

        assert result["ticker"] == "TEST"
        assert result["filings_available"] == 2
        assert result["trend"] in ("improving", "stable", "deteriorating")
        assert len(result["yearly_data"]) == 2
        assert len(result["deltas"]) == 1

    def test_delta_calculation(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = compare_quarters("TEST")

        delta = result["deltas"][0]
        assert delta["from_year"] == "2023"
        assert delta["to_year"] == "2024"
        assert delta["overall_delta"] == pytest.approx(0.13, abs=0.01)

    def test_error_with_insufficient_data(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = compare_quarters("UNKNOWN")

        assert "error" in result

    def test_summary_statistics(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = compare_quarters("TEST")

        summary = result["summary"]
        assert "avg_score" in summary
        assert "min_score" in summary
        assert "max_score" in summary
        assert "score_range" in summary
        assert summary["avg_score"] == pytest.approx(0.215, abs=0.01)

    def test_section_data_included(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = compare_quarters("TEST")

        yearly = result["yearly_data"][0]
        assert "item_1a" in yearly["sections"]
        assert "item_7" in yearly["sections"]
        assert "item_8" in yearly["sections"]
        assert "compound" in yearly["sections"]["item_7"]


class TestCompareTickers:
    def test_compare_multiple_tickers(self, mock_sentiment_dir):
        # Add a second ticker
        (mock_sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text(
            json.dumps({
                "sections": {
                    "item_1a": {"scores": {"compound": -0.1}},
                    "item_7": {"scores": {"compound": 0.4}},
                    "item_8": {"scores": {"compound": 0.2}},
                },
                "overall": {"compound": 0.3, "signal": "BUY"},
            }),
            encoding="utf-8",
        )

        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = compare_tickers(["TEST", "AAPL"])

        assert "TEST" in result["tickers"]
        assert "AAPL" in result["tickers"]
        assert result["tickers"]["AAPL"]["overall_score"] == 0.3

    def test_missing_ticker_returns_error(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = compare_tickers(["TEST", "MISSING"])

        assert "error" in result["tickers"]["MISSING"]


class TestSectionDeepDive:
    def test_deep_dive_returns_data(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = section_deep_dive("TEST", "item_7")

        assert result["ticker"] == "TEST"
        assert result["section"] == "item_7"
        assert len(result["data"]) == 2
        assert result["data"][0]["compound"] == 0.25
        assert result["data"][1]["compound"] == 0.35

    def test_word_count_change(self, mock_sentiment_dir):
        with patch("src.analysis.comparison.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            result = section_deep_dive("TEST", "item_7")

        # 12000 -> 13000 = +8.3%
        assert result["word_count_change_pct"] == pytest.approx(8.3, abs=0.1)
