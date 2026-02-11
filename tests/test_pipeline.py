"""
Tests for src/pipeline/automated.py
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline.automated import (
    PipelineResult,
    FilingStatus,
    get_filing_status,
    get_available_years,
    get_sentiment_for_date,
    get_all_sentiment_data,
)


class TestPipelineResult:
    """Test the PipelineResult dataclass."""

    def test_success_result(self):
        result = PipelineResult(success=True, message="Done")
        assert result.success is True
        assert result.message == "Done"
        assert result.data == {}
        assert result.errors == []

    def test_failure_result(self):
        result = PipelineResult(
            success=False, message="Failed", errors=["Error 1", "Error 2"]
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_with_data(self):
        result = PipelineResult(
            success=True, message="Ok", data={"downloaded": 5}
        )
        assert result.data["downloaded"] == 5


class TestFilingStatus:
    """Test the FilingStatus dataclass."""

    def test_default_values(self):
        status = FilingStatus(filing_date="2024-01-15", year="2024")
        assert status.downloaded is False
        assert status.parsed is False
        assert status.sections_extracted is False
        assert status.sentiment_analyzed is False

    def test_is_complete_true(self):
        status = FilingStatus(
            filing_date="2024-01-15",
            year="2024",
            downloaded=True,
            parsed=True,
            sections_extracted=True,
            sentiment_analyzed=True,
        )
        assert status.is_complete is True

    def test_is_complete_false(self):
        status = FilingStatus(
            filing_date="2024-01-15", year="2024", downloaded=True, parsed=True
        )
        assert status.is_complete is False

    def test_to_dict(self):
        status = FilingStatus(
            filing_date="2024-01-15", year="2024", downloaded=True
        )
        d = status.to_dict()
        assert d["filing_date"] == "2024-01-15"
        assert d["year"] == "2024"
        assert d["downloaded"] is True
        assert d["parsed"] is False
        assert d["complete"] is False


class TestGetAvailableYears:
    """Test get_available_years function."""

    def test_returns_years(self, tmp_data_dir):
        sentiment_dir = tmp_data_dir / "sentiment"
        (sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text("{}")
        (sentiment_dir / "AAPL_2023-01-20_sentiment.json").write_text("{}")
        (sentiment_dir / "AAPL_2022-01-18_sentiment.json").write_text("{}")

        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = sentiment_dir
            result = get_available_years("AAPL")

        assert result == ["2024", "2023", "2022"]

    def test_no_files_returns_empty(self, tmp_data_dir):
        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = tmp_data_dir / "sentiment"
            result = get_available_years("AAPL")

        assert result == []

    def test_case_insensitive(self, tmp_data_dir):
        sentiment_dir = tmp_data_dir / "sentiment"
        (sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text("{}")

        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = sentiment_dir
            result = get_available_years("aapl")

        assert "2024" in result


class TestGetSentimentForDate:
    """Test get_sentiment_for_date function."""

    def test_loads_existing_file(self, tmp_data_dir, sample_sentiment_result):
        sentiment_dir = tmp_data_dir / "sentiment"
        filepath = sentiment_dir / "AAPL_2024-01-15_sentiment.json"
        filepath.write_text(json.dumps(sample_sentiment_result))

        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = sentiment_dir
            result = get_sentiment_for_date("AAPL", "2024-01-15")

        assert result is not None
        assert result["ticker"] == "AAPL"

    def test_returns_none_for_missing(self, tmp_data_dir):
        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = tmp_data_dir / "sentiment"
            result = get_sentiment_for_date("AAPL", "2099-01-01")

        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_data_dir):
        sentiment_dir = tmp_data_dir / "sentiment"
        filepath = sentiment_dir / "AAPL_2024-01-15_sentiment.json"
        filepath.write_text("not valid json{{{")

        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = sentiment_dir
            result = get_sentiment_for_date("AAPL", "2024-01-15")

        assert result is None


class TestGetAllSentimentData:
    """Test get_all_sentiment_data function."""

    def test_returns_sorted_data(self, tmp_data_dir, sample_sentiment_result, sample_sentiment_previous):
        sentiment_dir = tmp_data_dir / "sentiment"
        (sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text(
            json.dumps(sample_sentiment_result)
        )
        (sentiment_dir / "AAPL_2023-01-20_sentiment.json").write_text(
            json.dumps(sample_sentiment_previous)
        )

        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = sentiment_dir
            result = get_all_sentiment_data("AAPL")

        assert len(result) == 2
        # Should be sorted ascending
        assert result[0][0] < result[1][0]

    def test_empty_ticker(self, tmp_data_dir):
        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.SENTIMENT_DIR = tmp_data_dir / "sentiment"
            result = get_all_sentiment_data("NONEXISTENT")

        assert result == []


class TestGetFilingStatus:
    """Test get_filing_status function."""

    def test_detects_downloaded_files(self, tmp_data_dir):
        raw_dir = tmp_data_dir / "raw"
        (raw_dir / "AAPL_10K_2024-01-15.html").write_text("<html>test</html>")

        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.RAW_DIR = raw_dir
            mock_settings.PROCESSED_DIR = tmp_data_dir / "processed"
            mock_settings.SECTIONS_DIR = tmp_data_dir / "sections"
            mock_settings.SENTIMENT_DIR = tmp_data_dir / "sentiment"
            result = get_filing_status("AAPL")

        assert "2024-01-15" in result
        assert result["2024-01-15"].downloaded is True
        assert result["2024-01-15"].parsed is False

    def test_detects_fully_processed(self, tmp_data_dir):
        raw_dir = tmp_data_dir / "raw"
        processed_dir = tmp_data_dir / "processed"
        sections_dir = tmp_data_dir / "sections"
        sentiment_dir = tmp_data_dir / "sentiment"

        (raw_dir / "AAPL_10K_2024-01-15.html").write_text("<html>")
        (processed_dir / "AAPL_2024-01-15.md").write_text("# Parsed")
        (sections_dir / "AAPL_2024-01-15_item_1a.txt").write_text("Risk")
        (sentiment_dir / "AAPL_2024-01-15_sentiment.json").write_text("{}")

        with patch("src.pipeline.automated.Settings") as mock_settings:
            mock_settings.RAW_DIR = raw_dir
            mock_settings.PROCESSED_DIR = processed_dir
            mock_settings.SECTIONS_DIR = sections_dir
            mock_settings.SENTIMENT_DIR = sentiment_dir
            result = get_filing_status("AAPL")

        status = result["2024-01-15"]
        assert status.downloaded is True
        assert status.parsed is True
        assert status.sections_extracted is True
        assert status.sentiment_analyzed is True
        assert status.is_complete is True
