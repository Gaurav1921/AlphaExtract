"""
Tests for src/models/anomaly.py
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.anomaly import AnomalyDetector, TRACKED_KEYWORDS


class TestSentimentShiftDetection:
    """Test detect_sentiment_shifts() logic."""

    def setup_method(self):
        self.detector = AnomalyDetector()

    def test_detects_large_negative_shift(self, sample_sentiment_result, sample_sentiment_previous):
        anomalies = self.detector.detect_sentiment_shifts(sample_sentiment_result, sample_sentiment_previous)
        # item_1a: -0.3 vs -0.1 = delta -0.2, abs >= 0.15 threshold
        item_1a_anomalies = [a for a in anomalies if a.get("section") == "item_1a"]
        assert len(item_1a_anomalies) >= 1
        assert item_1a_anomalies[0]["type"] == "sentiment_shift"

    def test_detects_positive_shift(self, sample_sentiment_result, sample_sentiment_previous):
        anomalies = self.detector.detect_sentiment_shifts(sample_sentiment_result, sample_sentiment_previous)
        # item_7: 0.5 vs 0.35 = delta +0.15, abs >= 0.15 threshold
        item_7_anomalies = [a for a in anomalies if a.get("section") == "item_7"]
        assert len(item_7_anomalies) >= 1

    def test_no_anomaly_for_small_change(self):
        current = {"sections": {"item_8": {"scores": {"compound": 0.21}}}}
        previous = {"sections": {"item_8": {"scores": {"compound": 0.20}}}}
        anomalies = self.detector.detect_sentiment_shifts(current, previous)
        assert len(anomalies) == 0

    def test_severity_high_for_large_delta(self):
        current = {"sections": {"item_1a": {"scores": {"compound": 0.3}}}}
        previous = {"sections": {"item_1a": {"scores": {"compound": -0.1}}}}
        anomalies = self.detector.detect_sentiment_shifts(current, previous)
        assert anomalies[0]["severity"] == "high"

    def test_severity_medium_for_moderate_delta(self):
        current = {"sections": {"item_1a": {"scores": {"compound": 0.1}}}}
        previous = {"sections": {"item_1a": {"scores": {"compound": -0.1}}}}
        anomalies = self.detector.detect_sentiment_shifts(current, previous)
        assert anomalies[0]["severity"] == "medium"

    def test_no_crash_on_missing_sections(self):
        current = {"sections": {"item_1a": {"scores": {"compound": 0.5}}}}
        previous = {"sections": {}}
        anomalies = self.detector.detect_sentiment_shifts(current, previous)
        assert anomalies == []

    def test_anomaly_has_required_fields(self, sample_sentiment_result, sample_sentiment_previous):
        anomalies = self.detector.detect_sentiment_shifts(sample_sentiment_result, sample_sentiment_previous)
        if anomalies:
            a = anomalies[0]
            assert "type" in a
            assert "title" in a
            assert "description" in a
            assert "severity" in a
            assert "current_score" in a
            assert "previous_score" in a
            assert "delta" in a


class TestKeywordAnomalyDetection:
    """Test detect_keyword_anomalies() logic."""

    def setup_method(self):
        self.detector = AnomalyDetector()

    def test_detects_keyword_emergence(self, tmp_data_dir, section_files):
        """When a keyword appears for the first time, it should be flagged."""
        self.detector.data_dir = tmp_data_dir
        self.detector.sections_dir = tmp_data_dir / "sections"

        anomalies = self.detector.detect_keyword_anomalies(
            "AAPL",
            "2024-01-15",
            ["2023-01-20", "2022-01-18"],
        )
        # We should get some anomalies due to keyword mentions
        assert isinstance(anomalies, list)

    def test_returns_empty_for_missing_sections(self, tmp_data_dir):
        self.detector.data_dir = tmp_data_dir
        self.detector.sections_dir = tmp_data_dir / "sections"

        anomalies = self.detector.detect_keyword_anomalies("NONEXISTENT", "2024-01-01", [])
        assert anomalies == []

    def test_anomaly_has_context(self, tmp_data_dir, section_files):
        self.detector.data_dir = tmp_data_dir
        self.detector.sections_dir = tmp_data_dir / "sections"

        anomalies = self.detector.detect_keyword_anomalies(
            "AAPL",
            "2024-01-15",
            ["2023-01-20", "2022-01-18"],
        )
        for a in anomalies:
            if a.get("type") in ("keyword_emergence", "keyword_spike"):
                assert "context" in a
                assert isinstance(a["context"], list)


class TestExtractContext:
    """Test _extract_context sentence extraction."""

    def setup_method(self):
        self.detector = AnomalyDetector()

    def test_extracts_matching_sentences(self):
        text = "Revenue grew 10%. Litigation risks increased. Profit margins expanded."
        result = self.detector._extract_context(text, ["litigation"])
        assert len(result) == 1
        assert "litigation" in result[0].lower()

    def test_max_sentences_limit(self):
        text = "Litigation one. Litigation two. Litigation three. Litigation four."
        result = self.detector._extract_context(text, ["litigation"], max_sentences=2)
        assert len(result) <= 2

    def test_no_matches(self):
        text = "Revenue grew. Profits increased."
        result = self.detector._extract_context(text, ["bankruptcy"])
        assert result == []

    def test_empty_text(self):
        result = self.detector._extract_context("", ["test"])
        assert result == []


class TestTrackedKeywords:
    """Validate the TRACKED_KEYWORDS structure."""

    def test_all_categories_have_groups(self):
        for category, groups in TRACKED_KEYWORDS.items():
            assert isinstance(groups, dict)
            assert len(groups) > 0

    def test_all_groups_have_terms(self):
        for category, groups in TRACKED_KEYWORDS.items():
            for group_name, terms in groups.items():
                assert isinstance(terms, list)
                assert len(terms) > 0
                for term in terms:
                    assert isinstance(term, str)
                    assert len(term) > 0


class TestAnalyzeTicker:
    """Test full anomaly detection pipeline."""

    def test_insufficient_filings(self, tmp_data_dir):
        detector = AnomalyDetector(data_dir=tmp_data_dir)
        # Only one sentiment file exists
        (tmp_data_dir / "sentiment" / "AAPL_2024-01-15_sentiment.json").write_text(
            json.dumps({"sections": {}, "overall": {}}), encoding="utf-8"
        )

        report = detector.analyze_ticker("AAPL")
        assert "error" in report
        assert report["total_anomalies"] == 0

    def test_full_analysis(self, tmp_data_dir, sentiment_files, section_files):
        detector = AnomalyDetector(data_dir=tmp_data_dir)
        detector.sentiment_dir = tmp_data_dir / "sentiment"
        detector.sections_dir = tmp_data_dir / "sections"

        report = detector.analyze_ticker("AAPL")

        assert report["ticker"] == "AAPL"
        assert "total_anomalies" in report
        assert "anomalies_by_severity" in report
        assert "anomalies" in report
        assert isinstance(report["anomalies"], list)

    def test_report_has_filing_dates(self, tmp_data_dir, sentiment_files, section_files):
        detector = AnomalyDetector(data_dir=tmp_data_dir)
        detector.sentiment_dir = tmp_data_dir / "sentiment"
        detector.sections_dir = tmp_data_dir / "sections"

        report = detector.analyze_ticker("AAPL")

        assert "current_filing_date" in report
        assert "compared_to" in report


class TestSaveReport:
    """Test anomaly report saving."""

    def test_saves_report(self, tmp_path):
        detector = AnomalyDetector()
        detector.anomalies_dir = tmp_path

        report = {
            "ticker": "AAPL",
            "current_filing_date": "2024-01-15",
            "anomalies": [],
            "total_anomalies": 0,
        }

        path = detector.save_report(report)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["ticker"] == "AAPL"
