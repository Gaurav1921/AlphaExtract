"""
Tests for Local RAG Engine
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from src.rag.local_rag import LocalRAG


@pytest.fixture
def sections_dir(tmp_path):
    """Create a temp sections directory with sample files."""
    sections = tmp_path / "sections"
    sections.mkdir()

    # Create some test section files
    (sections / "AAPL_2024-01-15_item_7.txt").write_text(
        "Apple reported strong revenue growth of 8% year over year. "
        "iPhone sales were particularly strong in emerging markets. "
        "Services revenue reached a new all-time high.\n\n"
        "The company invested heavily in artificial intelligence and "
        "machine learning capabilities. Research and development "
        "spending increased by 15% compared to the prior year.\n\n"
        "Supply chain improvements led to better inventory management "
        "and reduced costs. Gross margins expanded by 120 basis points.",
        encoding="utf-8",
    )

    (sections / "AAPL_2024-01-15_item_1a.txt").write_text(
        "The company faces significant competition in the smartphone "
        "market from Samsung, Google, and Chinese manufacturers. "
        "Market share in China has declined over the past two years.\n\n"
        "Regulatory risks remain elevated as antitrust investigations "
        "continue in the European Union. The Digital Markets Act may "
        "require changes to the App Store business model.\n\n"
        "Foreign currency fluctuations continue to impact reported "
        "revenue. A stronger US dollar negatively affects international sales.",
        encoding="utf-8",
    )

    (sections / "MSFT_2024-07-30_item_7.txt").write_text(
        "Microsoft Azure revenue grew 29% in constant currency terms. "
        "Cloud infrastructure demand continues to accelerate driven "
        "by enterprise adoption of AI workloads.\n\n"
        "LinkedIn and Dynamics 365 both showed strong momentum with "
        "double-digit revenue growth. The integration of Copilot "
        "across Microsoft 365 is driving new enterprise subscriptions.",
        encoding="utf-8",
    )

    return sections


@pytest.fixture
def rag(sections_dir):
    """Create a LocalRAG instance with test directory."""
    return LocalRAG(sections_dir=sections_dir)


class TestLocalRAGInit:
    def test_creates_instance(self, sections_dir):
        rag = LocalRAG(sections_dir=sections_dir)
        assert not rag.is_indexed
        assert rag.document_count == 0
        assert len(rag.conversation) == 0

    def test_default_max_history(self, sections_dir):
        rag = LocalRAG(sections_dir=sections_dir)
        assert rag.max_history == 5


class TestDocumentLoading:
    def test_loads_documents(self, rag):
        docs = rag._load_documents()
        assert len(docs) > 0

    def test_loads_for_specific_ticker(self, rag):
        docs = rag._load_documents(ticker="AAPL")
        for doc in docs:
            assert doc["ticker"] == "AAPL"

    def test_document_metadata(self, rag):
        docs = rag._load_documents(ticker="AAPL")
        for doc in docs:
            assert "text" in doc
            assert "ticker" in doc
            assert "filing_date" in doc
            assert "section" in doc
            assert "chunk_id" in doc

    def test_filters_short_paragraphs(self, rag):
        docs = rag._load_documents()
        for doc in docs:
            assert len(doc["text"].split()) >= 20


class TestIndexing:
    def test_index_without_embeddings(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        assert rag.is_indexed
        assert rag.document_count > 0
        assert not rag.has_embeddings

    def test_index_for_ticker(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index(ticker="MSFT")
        for doc in rag._documents:
            assert doc["ticker"] == "MSFT"


class TestKeywordSearch:
    def test_finds_relevant_docs(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        results = rag._keyword_search("revenue growth", k=3)
        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_filters_by_ticker(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        results = rag._keyword_search("revenue", k=5, ticker="AAPL")
        for r in results:
            assert r["ticker"] == "AAPL"

    def test_empty_results_for_no_match(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        results = rag._keyword_search("quantum computing nanotechnology", k=5)
        assert len(results) == 0


class TestRetrieve:
    def test_retrieve_auto_indexes(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            results = rag.retrieve("Apple iPhone sales")
        assert rag.is_indexed

    def test_retrieve_returns_results(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            results = rag.retrieve("revenue growth")
        assert len(results) > 0


class TestQuery:
    def test_query_without_llm(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            result = rag.query("What are Apple's risks?", ticker="AAPL")

        assert "answer" in result
        assert "sources" in result
        assert len(result["answer"]) > 0

    def test_query_updates_conversation(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.query("Test question", ticker="AAPL")

        assert len(rag.conversation) == 2  # user + assistant
        assert rag.conversation[0]["role"] == "user"
        assert rag.conversation[1]["role"] == "assistant"

    def test_clear_history(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.query("Test", ticker="AAPL")
        rag.clear_history()
        assert len(rag.conversation) == 0


class TestStatus:
    def test_status_returns_dict(self, rag):
        status = rag.status()
        assert "indexed" in status
        assert "documents" in status
        assert "has_embeddings" in status
        assert "conversation_length" in status

    def test_status_after_index(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        status = rag.status()
        assert status["indexed"] is True
        assert status["documents"] > 0


class TestFormatExcerpts:
    def test_formats_docs_as_text(self):
        docs = [
            {"ticker": "AAPL", "filing_date": "2024-01-15", "section": "item_7",
             "text": "Revenue grew 8% year over year.", "score": 0.8},
        ]
        result = LocalRAG._format_excerpts(docs)
        assert "AAPL" in result
        assert "2024-01-15" in result
        assert "Revenue grew" in result

    def test_handles_empty_docs(self):
        result = LocalRAG._format_excerpts([])
        assert "No relevant information" in result
