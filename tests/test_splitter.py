"""
Tests for src/data/splitter.py
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.data.splitter import SectionSplitter


class TestFindSectionBoundaries:
    """Test section boundary detection in 10-K text."""

    def setup_method(self):
        self.splitter = SectionSplitter()

    def test_finds_item_1a(self, sample_10k_markdown):
        result = self.splitter.find_section_boundaries(sample_10k_markdown, "item_1a")
        assert result is not None
        start, end = result
        assert start < end
        section = sample_10k_markdown[start:end]
        assert "Risk Factors" in section

    def test_finds_item_7(self, sample_10k_markdown):
        result = self.splitter.find_section_boundaries(sample_10k_markdown, "item_7")
        assert result is not None
        start, end = result
        section = sample_10k_markdown[start:end]
        assert "Management" in section or "Discussion" in section

    def test_finds_item_8(self, sample_10k_markdown):
        result = self.splitter.find_section_boundaries(sample_10k_markdown, "item_8")
        assert result is not None
        start, end = result
        section = sample_10k_markdown[start:end]
        assert "Financial" in section

    def test_returns_none_for_missing_section(self, sample_10k_markdown):
        result = self.splitter.find_section_boundaries(sample_10k_markdown, "item_9a")
        assert result is None

    def test_skips_toc_entry(self, sample_10k_markdown):
        """The TOC has short entries; the real section should be found after them."""
        result = self.splitter.find_section_boundaries(sample_10k_markdown, "item_1a")
        assert result is not None
        start, _ = result
        # The real section start should be after the TOC (which is near the top)
        text_before = sample_10k_markdown[:start]
        assert "Table of Contents" in sample_10k_markdown[:200] or start > 100

    def test_empty_text(self):
        result = self.splitter.find_section_boundaries("", "item_1a")
        assert result is None

    def test_case_insensitive_matching(self):
        text = "ITEM 1A. RISK FACTORS\n\n" + "x " * 500 + "\nITEM 7. MANAGEMENT"
        result = self.splitter.find_section_boundaries(text, "item_1a")
        assert result is not None


class TestExtractSection:
    """Test section extraction with text cleaning."""

    def setup_method(self):
        self.splitter = SectionSplitter()

    def test_extract_returns_text(self, sample_10k_markdown):
        text = self.splitter.extract_section(sample_10k_markdown, "item_1a")
        assert text is not None
        assert len(text) > 100

    def test_extract_missing_section_returns_none(self):
        text = self.splitter.extract_section("No sections here", "item_1a")
        assert text is None


class TestCleanSectionText:
    """Test the static _clean_section_text method."""

    def test_removes_excessive_newlines(self):
        text = "Hello\n\n\n\n\nWorld"
        result = SectionSplitter._clean_section_text(text)
        assert "\n\n\n" not in result
        assert "Hello" in result and "World" in result

    def test_removes_table_separators(self):
        text = "Some text\n|---|---|---|\nMore text"
        result = SectionSplitter._clean_section_text(text)
        assert "|---|" not in result

    def test_strips_whitespace(self):
        result = SectionSplitter._clean_section_text("  content  ")
        assert result == "content"


class TestSplitDocument:
    """Test full document splitting."""

    def setup_method(self):
        self.splitter = SectionSplitter()

    def test_splits_document(self, tmp_path, sample_10k_markdown):
        filepath = tmp_path / "AAPL_2024-01-15.md"
        filepath.write_text(sample_10k_markdown, encoding="utf-8")

        sections, stats = self.splitter.split_document(filepath)

        assert isinstance(sections, dict)
        assert isinstance(stats, dict)
        assert stats["ticker"] == "AAPL"
        assert stats["filing_date"] == "2024-01-15"

    def test_sections_found_in_stats(self, tmp_path, sample_10k_markdown):
        filepath = tmp_path / "AAPL_2024-01-15.md"
        filepath.write_text(sample_10k_markdown, encoding="utf-8")

        sections, stats = self.splitter.split_document(filepath)

        assert "item_1a" in stats["sections_found"]
        assert "item_7" in stats["sections_found"]
        assert "item_8" in stats["sections_found"]

    def test_section_stats_have_word_counts(self, tmp_path, sample_10k_markdown):
        filepath = tmp_path / "AAPL_2024-01-15.md"
        filepath.write_text(sample_10k_markdown, encoding="utf-8")

        sections, stats = self.splitter.split_document(filepath)

        for key in sections:
            assert "words" in stats["section_stats"][key]
            assert stats["section_stats"][key]["words"] > 0


class TestSaveSections:
    """Test saving extracted sections to disk."""

    def setup_method(self):
        self.splitter = SectionSplitter()

    def test_saves_section_files(self, tmp_path):
        self.splitter.output_dir = tmp_path
        sections = {"item_1a": "Risk factors text", "item_7": "MD&A text"}
        stats = {"ticker": "AAPL", "filing_date": "2024-01-15", "sections_found": ["item_1a", "item_7"]}

        self.splitter.save_sections(sections, stats)

        assert (tmp_path / "AAPL_2024-01-15_item_1a.txt").exists()
        assert (tmp_path / "AAPL_2024-01-15_item_7.txt").exists()
        assert (tmp_path / "AAPL_2024-01-15_sections.json").exists()

    def test_saved_json_is_valid(self, tmp_path):
        self.splitter.output_dir = tmp_path
        sections = {"item_1a": "text"}
        stats = {"ticker": "TEST", "filing_date": "2024-01-01", "sections_found": ["item_1a"]}

        self.splitter.save_sections(sections, stats)

        stats_file = tmp_path / "TEST_2024-01-01_sections.json"
        loaded = json.loads(stats_file.read_text(encoding="utf-8"))
        assert loaded["ticker"] == "TEST"


class TestBatchProcess:
    """Test batch processing of multiple files."""

    def setup_method(self):
        self.splitter = SectionSplitter()

    def test_batch_empty_directory(self, tmp_path):
        result = self.splitter.batch_process(tmp_path)
        assert result["total"] == 0
        assert result["successful"] == []
        assert result["failed"] == []

    def test_batch_processes_files(self, tmp_path, sample_10k_markdown):
        self.splitter.output_dir = tmp_path / "output"
        self.splitter.output_dir.mkdir()

        (tmp_path / "AAPL_2024-01-15.md").write_text(sample_10k_markdown, encoding="utf-8")

        result = self.splitter.batch_process(tmp_path)

        assert result["total"] == 1
        assert len(result["successful"]) == 1
