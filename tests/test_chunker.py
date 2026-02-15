"""
Tests for src/data/chunker.py
"""

import pytest
from pathlib import Path

from src.data.chunker import DocumentChunker


class TestWordCount:
    """Test static word counting."""

    def test_simple_text(self):
        assert DocumentChunker._word_count("hello world") == 2

    def test_empty_string(self):
        assert DocumentChunker._word_count("") == 0

    def test_multiline(self):
        assert DocumentChunker._word_count("one two\nthree four") == 4

    def test_extra_whitespace(self):
        assert DocumentChunker._word_count("  one   two  ") == 2


class TestSplitIntoParagraphs:
    """Test paragraph splitting."""

    def test_double_newline_split(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = DocumentChunker._split_into_paragraphs(text)
        assert len(result) == 3

    def test_strips_empty_paragraphs(self):
        text = "One.\n\n\n\n\nTwo."
        result = DocumentChunker._split_into_paragraphs(text)
        assert len(result) == 2

    def test_single_paragraph(self):
        text = "Just one paragraph here."
        result = DocumentChunker._split_into_paragraphs(text)
        assert len(result) == 1

    def test_empty_string(self):
        result = DocumentChunker._split_into_paragraphs("")
        assert len(result) == 0


class TestChunkText:
    """Test semantic text chunking."""

    def test_short_text_single_chunk(self):
        chunker = DocumentChunker(chunk_size=500, overlap=50, min_chunk_size=10)
        text = "Short paragraph. " * 20
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        chunker = DocumentChunker(chunk_size=100, overlap=10, min_chunk_size=20)
        paragraphs = [f"Paragraph {i}. " + "word " * 60 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_min_chunk_size_filter(self):
        chunker = DocumentChunker(chunk_size=100, overlap=10, min_chunk_size=50)
        text = "Small. " * 5  # ~5 words, below min
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 0

    def test_chunks_have_overlap(self):
        chunker = DocumentChunker(chunk_size=50, overlap=10, min_chunk_size=5)
        paragraphs = [f"Para {i}. " + "word " * 30 for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.chunk_text(text)

        if len(chunks) >= 2:
            # Last paragraph of chunk[0] should appear in chunk[1]
            chunk0_words = chunks[0].split()[-10:]
            chunk1_words = chunks[1].split()[:20]
            # At least some overlap should exist
            overlap = set(chunk0_words) & set(chunk1_words)
            assert len(overlap) > 0

    def test_empty_text(self):
        chunker = DocumentChunker(chunk_size=100, overlap=10, min_chunk_size=5)
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0

    def test_preserves_all_content(self):
        """No words should be lost during chunking (except possible min-size filter)."""
        chunker = DocumentChunker(chunk_size=100, overlap=0, min_chunk_size=1)
        paragraphs = [f"Para {i}. " + "word " * 50 for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.chunk_text(text)
        all_chunk_text = " ".join(chunks)

        for i in range(5):
            assert f"Para {i}" in all_chunk_text


class TestChunkFile:
    """Test file-based chunking with metadata."""

    def test_chunk_file_returns_documents(self, tmp_path):
        filepath = tmp_path / "AAPL_2024-01-15_item_1a.txt"
        text = "\n\n".join(["Paragraph content. " * 30 for _ in range(10)])
        filepath.write_text(text, encoding="utf-8")

        chunker = DocumentChunker(chunk_size=100, overlap=10, min_chunk_size=20)
        docs = chunker.chunk_file(filepath)

        assert len(docs) > 0
        for doc in docs:
            assert doc["ticker"] == "AAPL"
            assert doc["filing_date"] == "2024-01-15"
            assert doc["section"] == "item_1a"
            assert "text" in doc
            assert "word_count" in doc
            assert doc["word_count"] > 0

    def test_chunk_ids_are_sequential(self, tmp_path):
        filepath = tmp_path / "MSFT_2023-06-30_item_7.txt"
        text = "\n\n".join(["Content. " * 60 for _ in range(10)])
        filepath.write_text(text, encoding="utf-8")

        chunker = DocumentChunker(chunk_size=100, overlap=10, min_chunk_size=10)
        docs = chunker.chunk_file(filepath)

        for i, doc in enumerate(docs):
            assert doc["chunk_id"] == i

    def test_metadata_from_filename(self, tmp_path):
        filepath = tmp_path / "TSLA_2024-03-01_item_8.txt"
        filepath.write_text("Some financial data. " * 50, encoding="utf-8")

        chunker = DocumentChunker(chunk_size=500, overlap=10, min_chunk_size=10)
        docs = chunker.chunk_file(filepath)

        assert docs[0]["ticker"] == "TSLA"
        assert docs[0]["filing_date"] == "2024-03-01"
        assert docs[0]["section"] == "item_8"


class TestChunkDirectory:
    """Test directory-level chunking."""

    def test_chunk_empty_dir(self, tmp_path):
        chunker = DocumentChunker()
        docs, summary = chunker.chunk_directory(tmp_path)
        assert len(docs) == 0
        assert summary["total_files"] == 0
        assert summary["total_chunks"] == 0
        assert summary["avg_words_per_chunk"] == 0

    def test_chunk_directory_with_files(self, tmp_path):
        for name in ["AAPL_2024-01-15_item_1a.txt", "AAPL_2024-01-15_item_7.txt"]:
            (tmp_path / name).write_text("Content paragraph. " * 100, encoding="utf-8")

        chunker = DocumentChunker(chunk_size=100, overlap=10, min_chunk_size=10)
        docs, summary = chunker.chunk_directory(tmp_path)

        assert summary["total_files"] == 2
        assert summary["total_chunks"] > 0
        assert "AAPL" in summary["chunks_by_ticker"]
