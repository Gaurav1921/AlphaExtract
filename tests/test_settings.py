"""
Tests for src/config/settings.py
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Settings, setup_logging


class TestSignalGeneration:
    """Test generate_signal() trading signal logic."""

    def test_strong_buy(self):
        assert Settings.generate_signal(0.5) == "STRONG_BUY"
        assert Settings.generate_signal(0.8) == "STRONG_BUY"
        assert Settings.generate_signal(1.0) == "STRONG_BUY"

    def test_buy(self):
        assert Settings.generate_signal(0.2) == "BUY"
        assert Settings.generate_signal(0.35) == "BUY"
        assert Settings.generate_signal(0.49) == "BUY"

    def test_hold(self):
        assert Settings.generate_signal(0.0) == "HOLD"
        assert Settings.generate_signal(0.19) == "HOLD"
        assert Settings.generate_signal(-0.19) == "HOLD"

    def test_sell(self):
        # SELL: SIGNAL_STRONG_SELL <= compound < SIGNAL_SELL
        assert Settings.generate_signal(-0.21) == "SELL"
        assert Settings.generate_signal(-0.35) == "SELL"
        assert Settings.generate_signal(-0.49) == "SELL"

    def test_strong_sell(self):
        # STRONG_SELL: compound < SIGNAL_STRONG_SELL
        assert Settings.generate_signal(-0.51) == "STRONG_SELL"
        assert Settings.generate_signal(-0.8) == "STRONG_SELL"
        assert Settings.generate_signal(-1.0) == "STRONG_SELL"

    def test_boundary_values(self):
        """Boundaries are inclusive (>=), so exact threshold maps to the band above."""
        assert Settings.generate_signal(0.5) == "STRONG_BUY"
        assert Settings.generate_signal(0.2) == "BUY"
        assert Settings.generate_signal(-0.2) == "HOLD"   # -0.2 >= -0.2 -> HOLD
        assert Settings.generate_signal(-0.5) == "SELL"    # -0.5 >= -0.5 -> SELL


class TestValidation:
    """Test Settings.validate() configuration checks."""

    def test_validate_returns_list(self):
        result = Settings.validate()
        assert isinstance(result, list)

    def test_validate_warns_on_default_email(self):
        with patch.object(Settings, "SEC_USER_EMAIL", "student@example.com"):
            warnings = Settings.validate()
            assert any("SEC_EMAIL" in w for w in warnings)

    def test_validate_warns_on_missing_llm_keys(self):
        with patch.object(Settings, "GEMINI_API_KEY", None), \
             patch.object(Settings, "GROQ_API_KEY", None):
            warnings = Settings.validate()
            assert any("LLM" in w for w in warnings)

    def test_validate_warns_on_missing_supabase(self):
        with patch.object(Settings, "SUPABASE_URL", None):
            warnings = Settings.validate()
            assert any("SUPABASE" in w for w in warnings)


class TestInitDirs:
    """Test directory initialization."""

    def test_init_dirs_creates_directories(self, tmp_path):
        with patch.object(Settings, "DATA_DIR", tmp_path / "data"), \
             patch.object(Settings, "RAW_DIR", tmp_path / "data" / "raw"), \
             patch.object(Settings, "PROCESSED_DIR", tmp_path / "data" / "processed"), \
             patch.object(Settings, "SECTIONS_DIR", tmp_path / "data" / "sections"), \
             patch.object(Settings, "SENTIMENT_DIR", tmp_path / "data" / "sentiment"), \
             patch.object(Settings, "ANOMALIES_DIR", tmp_path / "data" / "anomalies"), \
             patch.object(Settings, "LOG_DIR", tmp_path / "logs"):
            Settings.init_dirs()

            assert (tmp_path / "data" / "raw").exists()
            assert (tmp_path / "data" / "processed").exists()
            assert (tmp_path / "data" / "sections").exists()
            assert (tmp_path / "data" / "sentiment").exists()
            assert (tmp_path / "data" / "anomalies").exists()
            assert (tmp_path / "logs").exists()


class TestSettingsConstants:
    """Sanity checks on configuration constants."""

    def test_section_weights_sum_to_one(self):
        total = sum(Settings.SECTION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_signal_thresholds_ordered(self):
        assert Settings.SIGNAL_STRONG_BUY > Settings.SIGNAL_BUY
        assert Settings.SIGNAL_BUY > Settings.SIGNAL_SELL
        assert Settings.SIGNAL_SELL > Settings.SIGNAL_STRONG_SELL

    def test_chunk_size_greater_than_overlap(self):
        assert Settings.CHUNK_SIZE > Settings.CHUNK_OVERLAP

    def test_chunk_min_less_than_max(self):
        assert Settings.CHUNK_MIN_SIZE < Settings.CHUNK_SIZE

    def test_embedding_dim_positive(self):
        assert Settings.EMBEDDING_DIM > 0

    def test_rate_limit_positive(self):
        assert Settings.SEC_RATE_LIMIT_DELAY > 0


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_does_not_raise(self, tmp_path):
        with patch.object(Settings, "LOG_DIR", tmp_path):
            setup_logging("DEBUG")

    def test_setup_logging_with_invalid_level_falls_back(self, tmp_path):
        with patch.object(Settings, "LOG_DIR", tmp_path):
            setup_logging("INVALID_LEVEL")
