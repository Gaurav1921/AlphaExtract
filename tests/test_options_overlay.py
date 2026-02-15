"""
Tests for Options Sentiment Overlay
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.market.options_overlay import (
    OptionsOverlay,
    OptionsSnapshot,
    CompositeSignal,
)


@pytest.fixture
def mock_sentiment_dir(tmp_path):
    """Create temp directory with sentiment/ensemble data."""
    sentiment_dir = tmp_path / "sentiment"
    sentiment_dir.mkdir()

    (sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text(
        json.dumps({
            "overall": {"compound": 0.35, "signal": "BUY"},
        }),
        encoding="utf-8",
    )

    (sentiment_dir / "AAPL_2024-01-15_ensemble.json").write_text(
        json.dumps({
            "ensemble": {"score": 0.28, "signal": "BUY"},
        }),
        encoding="utf-8",
    )

    return sentiment_dir


@pytest.fixture
def overlay():
    return OptionsOverlay(filing_weight=0.70, options_weight=0.30)


class TestOptionsSnapshot:
    def test_creates_snapshot(self):
        snap = OptionsSnapshot(
            ticker="AAPL",
            snapshot_date="2024-11-15",
            put_call_ratio=0.85,
            total_call_oi=500000,
            total_put_oi=425000,
        )
        assert snap.ticker == "AAPL"
        assert snap.put_call_ratio == 0.85

    def test_to_dict(self):
        snap = OptionsSnapshot(ticker="MSFT", snapshot_date="2024-11-15")
        d = snap.to_dict()
        assert d["ticker"] == "MSFT"
        assert "options_score" in d


class TestCompositeSignal:
    def test_creates_composite(self):
        sig = CompositeSignal(
            ticker="AAPL",
            filing_sentiment=0.35,
            filing_signal="BUY",
            ensemble_score=0.28,
            ensemble_signal="BUY",
            options_score=-0.15,
            options_signal="HOLD",
            composite_score=0.15,
            composite_signal="HOLD",
            filing_weight=0.70,
            options_weight=0.30,
            agreement=False,
        )
        assert sig.ticker == "AAPL"
        assert sig.agreement is False

    def test_to_dict(self):
        sig = CompositeSignal(
            ticker="AAPL",
            filing_sentiment=0.3,
            filing_signal="BUY",
            ensemble_score=None,
            ensemble_signal=None,
            options_score=0.1,
            options_signal="HOLD",
            composite_score=0.24,
            composite_signal="BUY",
            filing_weight=0.7,
            options_weight=0.3,
            agreement=True,
        )
        d = sig.to_dict()
        assert d["composite_score"] == 0.24


class TestScoreOptions:
    def test_bearish_high_pc_ratio(self):
        score, signal = OptionsOverlay._score_options(1.6, 1.6, 0.15)
        assert score < 0

    def test_bullish_low_pc_ratio(self):
        score, signal = OptionsOverlay._score_options(0.4, 0.5, -0.15)
        assert score > 0

    def test_neutral_balanced(self):
        score, signal = OptionsOverlay._score_options(0.8, 0.8, 0.0)
        assert -0.3 < score < 0.3

    def test_no_data(self):
        score, signal = OptionsOverlay._score_options(None, None, None)
        assert score == 0.0
        assert signal == "N/A"

    def test_partial_data_pc_only(self):
        score, signal = OptionsOverlay._score_options(1.2, None, None)
        assert score < 0  # High P/C = bearish

    def test_partial_data_skew_only(self):
        score, signal = OptionsOverlay._score_options(None, None, 0.15)
        assert score < 0  # High skew = bearish

    def test_score_clamped(self):
        score, signal = OptionsOverlay._score_options(0.1, 0.1, -0.5)
        assert -1.0 <= score <= 1.0


class TestCompositeSignalComputation:
    def test_composite_with_filing_data(self, mock_sentiment_dir):
        overlay = OptionsOverlay(filing_weight=0.70, options_weight=0.30)

        mock_snapshot = OptionsSnapshot(
            ticker="AAPL",
            snapshot_date="2024-11-15",
            put_call_ratio=0.8,
            options_score=-0.1,
            options_signal="HOLD",
        )

        with patch("src.market.options_overlay.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.generate_signal = lambda score: "BUY" if score >= 0.2 else ("HOLD" if score >= -0.2 else "SELL")

            result = overlay.composite_signal("AAPL", options=mock_snapshot)

        assert result.ticker == "AAPL"
        assert result.filing_sentiment == 0.35
        assert result.ensemble_score == 0.28
        assert result.options_score == -0.1
        # Composite: 0.28 * 0.70 + (-0.1) * 0.30 = 0.196 - 0.03 = 0.166
        assert -1.0 <= result.composite_score <= 1.0

    def test_composite_without_options(self, mock_sentiment_dir):
        overlay = OptionsOverlay()

        with patch("src.market.options_overlay.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.generate_signal = lambda score: "BUY" if score >= 0.2 else "HOLD"

            # Patch fetch_options to return None (no yfinance)
            with patch.object(overlay, "fetch_options", return_value=None):
                result = overlay.composite_signal("AAPL")

        # Without options, should use filing score only
        assert result.ticker == "AAPL"
        assert result.options_signal == "N/A"

    def test_composite_no_filing_data(self, tmp_path):
        empty_dir = tmp_path / "sentiment"
        empty_dir.mkdir()
        overlay = OptionsOverlay()

        mock_snapshot = OptionsSnapshot(
            ticker="ZZZZ",
            snapshot_date="2024-11-15",
            options_score=0.2,
            options_signal="BUY",
        )

        with patch("src.market.options_overlay.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = empty_dir
            mock_settings.generate_signal = lambda score: "BUY" if score >= 0.2 else "HOLD"

            result = overlay.composite_signal("ZZZZ", options=mock_snapshot)

        assert result.filing_sentiment is None

    def test_agreement_detection(self, mock_sentiment_dir):
        overlay = OptionsOverlay()

        # Options bullish -> agrees with filing
        bullish_options = OptionsSnapshot(
            ticker="AAPL",
            snapshot_date="2024-11-15",
            options_score=0.3,
            options_signal="BUY",
        )

        with patch("src.market.options_overlay.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.generate_signal = lambda score: "BUY" if score >= 0.2 else "HOLD"

            result = overlay.composite_signal("AAPL", options=bullish_options)

        assert result.agreement is True


class TestBatchComposite:
    def test_batch_returns_list(self, mock_sentiment_dir):
        overlay = OptionsOverlay()

        with patch("src.market.options_overlay.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.generate_signal = lambda score: "HOLD"

            with patch.object(overlay, "fetch_options", return_value=None):
                results = overlay.batch_composite(["AAPL"])

        assert len(results) == 1
        assert results[0].ticker == "AAPL"


class TestSaveSnapshot:
    def test_saves_snapshot(self, tmp_path):
        overlay = OptionsOverlay()
        snap = OptionsSnapshot(
            ticker="AAPL",
            snapshot_date="2024-11-15",
            put_call_ratio=0.85,
            options_score=-0.1,
        )

        path = overlay.save_snapshot(snap, output_dir=tmp_path)
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["ticker"] == "AAPL"
        assert data["put_call_ratio"] == 0.85


class TestSaveComposite:
    def test_saves_composite(self, tmp_path):
        overlay = OptionsOverlay()
        sig = CompositeSignal(
            ticker="AAPL",
            filing_sentiment=0.35,
            filing_signal="BUY",
            ensemble_score=0.28,
            ensemble_signal="BUY",
            options_score=-0.1,
            options_signal="HOLD",
            composite_score=0.15,
            composite_signal="HOLD",
            filing_weight=0.70,
            options_weight=0.30,
            agreement=False,
        )

        path = overlay.save_composite(sig, output_dir=tmp_path)
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["composite_score"] == 0.15
