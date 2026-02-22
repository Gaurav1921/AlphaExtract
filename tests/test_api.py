"""
Tests for FastAPI REST API
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_sentiment_dir(tmp_path):
    """Create temp directory with sentiment/ensemble data."""
    sentiment_dir = tmp_path / "sentiment"
    sentiment_dir.mkdir()

    (sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text(
        json.dumps({
            "overall": {"compound": 0.35, "signal": "BUY"},
            "sections": {"item_7": {"scores": {"compound": 0.4}}},
        }),
        encoding="utf-8",
    )

    (sentiment_dir / "AAPL_2024-01-15_ensemble.json").write_text(
        json.dumps({
            "ensemble": {"score": 0.28, "signal": "BUY", "components_agree": True},
            "signals": {"finbert": {"score": 0.35}, "keywords": {"score": 0.2}},
        }),
        encoding="utf-8",
    )

    return sentiment_dir


class TestHealthEndpoint:
    def test_health_check(self, client, mock_sentiment_dir):
        with patch("src.api.app.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.VERSION = "1.0.0"

            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert data["tickers_available"] >= 0


class TestTickersEndpoint:
    def test_list_tickers(self, client):
        response = client.get("/tickers")
        assert response.status_code == 200
        data = response.json()
        assert "tickers" in data
        assert "sectors" in data
        assert data["total"] > 0


class TestSentimentEndpoint:
    def test_get_sentiment(self, client, mock_sentiment_dir):
        with patch("src.api.app.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir

            response = client.get("/sentiment/AAPL")

        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["overall_score"] == 0.35
        assert data["overall_signal"] == "BUY"

    def test_sentiment_not_found(self, client, tmp_path):
        empty_dir = tmp_path / "sentiment"
        empty_dir.mkdir()

        with patch("src.api.app.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = empty_dir

            response = client.get("/sentiment/ZZZZ")

        assert response.status_code == 404

    def test_sentiment_case_insensitive(self, client, mock_sentiment_dir):
        with patch("src.api.app.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir

            response = client.get("/sentiment/aapl")

        assert response.status_code == 200
        assert response.json()["ticker"] == "AAPL"


class TestEnsembleEndpoint:
    def test_get_ensemble(self, client, mock_sentiment_dir):
        with patch("src.api.app.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir

            response = client.get("/ensemble/AAPL")

        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["ensemble_score"] == 0.28
        assert data["ensemble_signal"] == "BUY"

    def test_ensemble_not_found(self, client, tmp_path):
        empty_dir = tmp_path / "sentiment"
        empty_dir.mkdir()

        with patch("src.api.app.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = empty_dir

            response = client.get("/ensemble/ZZZZ")

        assert response.status_code == 404


class TestPortfolioEndpoint:
    def test_portfolio_computation(self, client, mock_sentiment_dir):
        with patch("src.analysis.portfolio.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir
            mock_settings.SECTOR_TICKERS = {"Technology": ["AAPL", "MSFT"]}
            mock_settings.generate_signal = lambda score: "BUY" if score > 0 else "HOLD"

            response = client.post("/portfolio", json={
                "holdings": {"AAPL": 0.6, "MSFT": 0.4},
                "name": "Test Portfolio",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Portfolio"
        assert data["holdings_count"] == 2

    def test_portfolio_empty_holdings(self, client):
        response = client.post("/portfolio", json={
            "holdings": {},
            "name": "Empty",
        })
        assert response.status_code == 400


class TestSignalsEndpoint:
    def test_get_all_signals(self, client, mock_sentiment_dir):
        with patch("src.api.app.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = mock_sentiment_dir

            response = client.get("/signals/AAPL")

        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "sentiment" in data
        assert "ensemble" in data

    def test_signals_not_found(self, client, tmp_path):
        empty_dir = tmp_path / "sentiment"
        empty_dir.mkdir()

        with patch("src.api.app.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = empty_dir

            response = client.get("/signals/ZZZZ")

        assert response.status_code == 404
