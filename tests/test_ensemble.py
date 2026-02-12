"""
Tests for src/models/ensemble.py
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.ensemble import (
    KeywordScorer,
    LLMScorer,
    EnsembleScorer,
    load_section_texts,
    load_historical_texts,
)


class TestKeywordScorer:

    def setup_method(self):
        self.scorer = KeywordScorer()

    def test_score_returns_dict(self):
        texts = {"item_1a": "Some risk factors text about litigation and cybersecurity threats."}
        result = self.scorer.score(texts)
        assert "score" in result
        assert -1.0 <= result["score"] <= 1.0

    def test_bearish_keywords_produce_negative_score(self):
        # Text heavy on bearish keywords
        texts = {
            "item_1a": (
                "litigation lawsuit legal proceedings class action arbitration "
                "regulatory action SEC investigation compliance violation "
                "liquidity risk going concern debt restructuring "
            ) * 5
        }
        result = self.scorer.score(texts)
        assert result["score"] < 0

    def test_bullish_keywords_produce_positive_score(self):
        texts = {
            "item_7": (
                "acquisition merger joint venture artificial intelligence "
                "machine learning generative AI "
            ) * 10
        }
        result = self.scorer.score(texts)
        assert result["score"] > 0

    def test_empty_text_returns_zero(self):
        result = self.scorer.score({"item_1a": ""})
        assert result["score"] == 0.0

    def test_delta_scoring_with_history(self):
        current = {"item_1a": "litigation lawsuit " * 20}
        # History with no litigation mentions
        history = [
            {"item_1a": "Business operations were normal."},
            {"item_1a": "Standard risk factors apply."},
            {"item_1a": "No significant changes."},
        ]
        result = self.scorer.score(current, historical_texts=history)
        assert result["method"] == "delta"
        assert result["score"] < 0  # New negative keywords = bearish

    def test_absolute_scoring_without_history(self):
        texts = {"item_1a": "Some text."}
        result = self.scorer.score(texts)
        assert result["method"] == "absolute"

    def test_top_risks_populated(self):
        texts = {"item_1a": "litigation lawsuit cybersecurity data breach supply chain disruption " * 5}
        result = self.scorer.score(texts)
        assert isinstance(result.get("top_risks"), list)


class TestLLMScorer:

    def test_unavailable_returns_zero(self):
        scorer = LLMScorer()
        with patch.object(type(scorer), "client", new_callable=lambda: property(lambda self: None)):
            result = scorer.score("AAPL", {"item_7": "text"})
            assert result["score"] == 0.0
            assert result["available"] is False

    def test_parse_bullish_response(self):
        scorer = LLMScorer()
        response = json.dumps({
            "outlook": "bullish",
            "confidence": "high",
            "key_factors": ["revenue growth", "strong margins"],
            "one_line_summary": "Strong execution",
        })
        result = scorer._parse_response(response)
        assert result["score"] == 1.0
        assert result["outlook"] == "bullish"
        assert result["confidence"] == "high"

    def test_parse_bearish_low_confidence(self):
        scorer = LLMScorer()
        response = json.dumps({
            "outlook": "bearish",
            "confidence": "low",
            "key_factors": ["revenue concerns"],
            "one_line_summary": "Weak outlook",
        })
        result = scorer._parse_response(response)
        assert result["score"] == -0.3
        assert result["outlook"] == "bearish"

    def test_parse_neutral(self):
        scorer = LLMScorer()
        response = json.dumps({
            "outlook": "neutral",
            "confidence": "medium",
            "key_factors": [],
            "one_line_summary": "Mixed signals",
        })
        result = scorer._parse_response(response)
        assert result["score"] == 0.0

    def test_parse_invalid_json(self):
        scorer = LLMScorer()
        result = scorer._parse_response("not valid json at all")
        assert result["score"] == 0.0
        assert result["outlook"] == "parse_error"

    def test_parse_markdown_wrapped_json(self):
        scorer = LLMScorer()
        response = '```json\n{"outlook": "bullish", "confidence": "medium", "key_factors": [], "one_line_summary": "ok"}\n```'
        result = scorer._parse_response(response)
        assert result["outlook"] == "bullish"
        assert result["score"] == 0.65


class TestEnsembleScorer:

    def test_score_with_sentiment_only(self):
        scorer = EnsembleScorer()
        sentiment = {
            "overall": {"compound": 0.3, "signal": "BUY"},
            "sections": {"item_7": {"scores": {"compound": 0.3}}},
        }
        result = scorer.score_filing("AAPL", "2024-01-15", sentiment_data=sentiment)

        assert result["ticker"] == "AAPL"
        assert result["filing_date"] == "2024-01-15"
        assert "ensemble" in result
        assert "signals" in result
        assert -1.0 <= result["ensemble"]["score"] <= 1.0

    def test_weight_redistribution_without_llm(self):
        scorer = EnsembleScorer(weight_finbert=0.4, weight_keywords=0.25, weight_llm=0.35)
        weights = scorer._redistribute_weights(exclude_llm=True)

        assert weights["llm"] == 0.0
        assert abs(weights["finbert"] + weights["keywords"] - 1.0) < 0.001

    def test_all_signals_present(self):
        scorer = EnsembleScorer()
        sentiment = {"overall": {"compound": 0.5, "signal": "STRONG_BUY"}, "sections": {}}
        result = scorer.score_filing("AAPL", "2024-01-15", sentiment_data=sentiment)

        assert "finbert" in result["signals"]
        assert "keywords" in result["signals"]
        assert "llm" in result["signals"]

    def test_components_agree_flag(self):
        scorer = EnsembleScorer()
        sentiment = {"overall": {"compound": 0.5, "signal": "STRONG_BUY"}, "sections": {}}
        # Without section texts, keyword and LLM will be 0 — so not all agree with positive finbert
        result = scorer.score_filing("AAPL", "2024-01-15", sentiment_data=sentiment)
        assert "components_agree" in result["ensemble"]

    def test_save_result(self, tmp_path):
        scorer = EnsembleScorer()
        result = {
            "ticker": "AAPL",
            "filing_date": "2024-01-15",
            "ensemble": {"score": 0.3, "signal": "BUY"},
            "signals": {},
        }
        path = scorer.save_result(result, output_dir=tmp_path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["ticker"] == "AAPL"


class TestLoadHelpers:

    def test_load_section_texts(self, tmp_data_dir, section_files):
        with patch("src.models.ensemble.Settings") as mock_s:
            mock_s.SECTIONS_DIR = tmp_data_dir / "sections"
            texts = load_section_texts("AAPL", "2024-01-15")
        assert isinstance(texts, dict)
        assert len(texts) > 0

    def test_load_historical_texts(self, tmp_data_dir, section_files):
        with patch("src.models.ensemble.Settings") as mock_s:
            mock_s.SECTIONS_DIR = tmp_data_dir / "sections"
            history = load_historical_texts("AAPL", exclude_date="2024-01-15")
        assert isinstance(history, list)
        # Should have data for 2023 and 2022, but not 2024
        assert len(history) >= 1

    def test_load_missing_ticker(self, tmp_data_dir):
        with patch("src.models.ensemble.Settings") as mock_s:
            mock_s.SECTIONS_DIR = tmp_data_dir / "sections"
            texts = load_section_texts("ZZZZ", "2024-01-15")
        assert texts == {}
