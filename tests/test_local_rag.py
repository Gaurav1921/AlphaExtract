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

    def test_documents_have_file_path(self, rag):
        """DocumentChunker provides file_path metadata."""
        docs = rag._load_documents()
        for doc in docs:
            assert "file_path" in doc
            assert doc["file_path"]  # not empty

    def test_documents_have_word_count(self, rag):
        """DocumentChunker provides word_count metadata."""
        docs = rag._load_documents()
        for doc in docs:
            assert "word_count" in doc
            assert doc["word_count"] > 0


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

    def test_bm25_index_built(self, rag):
        """BM25 index should be built during indexing."""
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        assert len(rag._idf) > 0
        assert rag._avg_doc_len > 0
        assert len(rag._doc_term_freqs) == rag.document_count


class TestBM25Search:
    def test_finds_relevant_docs(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        results = rag._bm25_search("revenue growth", k=3)
        assert len(results) > 0
        assert all("score" in r for r in results)
        assert all(r["retrieval"] == "bm25" for r in results)

    def test_filters_by_ticker(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        results = rag._bm25_search("revenue", k=5, ticker="AAPL")
        for r in results:
            assert r["ticker"] == "AAPL"

    def test_scores_are_positive(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        results = rag._bm25_search("competition smartphone market", k=5)
        for r in results:
            assert r["score"] > 0

    def test_results_sorted_by_score(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        results = rag._bm25_search("revenue growth", k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_for_nonsense_query(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        results = rag._bm25_search("xyzzyplugh42 foobarbaz99", k=5)
        assert len(results) == 0


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


class TestRRFFusion:
    def test_combines_two_lists(self):
        list_a = [
            {"text": "doc A1", "ticker": "AAPL", "filing_date": "2024-01-15",
             "section": "item_7", "chunk_id": 0, "score": 1.0},
            {"text": "doc A2", "ticker": "MSFT", "filing_date": "2024-07-30",
             "section": "item_7", "chunk_id": 0, "score": 0.8},
        ]
        list_b = [
            {"text": "doc B1", "ticker": "MSFT", "filing_date": "2024-07-30",
             "section": "item_7", "chunk_id": 0, "score": 5.0},
            {"text": "doc B2", "ticker": "AAPL", "filing_date": "2024-01-15",
             "section": "item_1a", "chunk_id": 0, "score": 3.0},
        ]
        results = LocalRAG._reciprocal_rank_fusion(list_a, list_b, final_k=5)
        assert len(results) >= 2
        assert all("rrf_score" in r for r in results)

    def test_rrf_scores_decrease(self):
        list_a = [
            {"text": f"doc {i}", "ticker": "T", "filing_date": "2024",
             "section": f"s{i}", "chunk_id": 0, "score": 1.0}
            for i in range(5)
        ]
        results = LocalRAG._reciprocal_rank_fusion(list_a, final_k=5)
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_shared_docs_rank_higher(self):
        """Documents appearing in both lists should get higher RRF scores."""
        shared = {"text": "shared", "ticker": "A", "filing_date": "2024",
                  "section": "s1", "chunk_id": 0, "score": 0.5}
        unique = {"text": "unique", "ticker": "B", "filing_date": "2024",
                  "section": "s2", "chunk_id": 0, "score": 1.0}
        list_a = [shared.copy()]
        list_b = [shared.copy(), unique.copy()]
        results = LocalRAG._reciprocal_rank_fusion(list_a, list_b, final_k=5)
        # shared doc appears in both lists → higher RRF score
        assert results[0]["ticker"] == "A"


class TestRetrieve:
    def test_retrieve_auto_indexes(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            results = rag.retrieve("Apple iPhone sales")
        assert rag.is_indexed

    def test_retrieve_returns_results(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            results = rag.retrieve("revenue growth")
        assert len(results) > 0

    def test_retrieve_uses_bm25_without_embeddings(self, rag):
        """Without embedding model, BM25 should be the primary retrieval method."""
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
            results = rag.retrieve("revenue growth", k=3)
        assert len(results) > 0


class TestReranking:
    def test_rerank_with_no_reranker(self, rag):
        """Without reranker, should return docs unchanged."""
        docs = [
            {"text": "doc1", "score": 0.5},
            {"text": "doc2", "score": 0.8},
        ]
        with patch.object(type(rag), "reranker", new_callable=lambda: property(lambda self: None)):
            result = rag._rerank("query", docs, k=2)
        assert len(result) == 2

    def test_rerank_with_mock_reranker(self, rag):
        """With reranker, docs should be reordered by rerank scores."""
        docs = [
            {"text": "low relevance doc", "score": 0.9},
            {"text": "high relevance doc", "score": 0.1},
        ]
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.2, 0.9]  # second doc scores higher
        rag._reranker_model = mock_reranker
        result = rag._rerank("test query", docs, k=2)
        assert result[0]["text"] == "high relevance doc"
        assert result[0]["rerank_score"] == 0.9


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

    def test_query_sources_have_file_paths(self, rag):
        """Sources should include file_path for traceability."""
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            result = rag.query("revenue growth", ticker="AAPL")
        for source in result["sources"]:
            assert "file_path" in source
            assert "chunk_id" in source

    def test_query_reports_retrieval_method(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            result = rag.query("revenue", ticker="AAPL")
        assert "retrieval_method" in result
        assert len(result["retrieval_method"]) > 0


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

    def test_status_includes_bm25(self, rag):
        with patch.object(type(rag), "embedding_model", new_callable=lambda: property(lambda self: None)):
            rag.index()
        status = rag.status()
        assert "has_bm25" in status
        assert status["has_bm25"] is True

    def test_status_includes_reranker(self, rag):
        status = rag.status()
        assert "has_reranker" in status


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

    def test_includes_source_numbers(self):
        docs = [
            {"ticker": "AAPL", "filing_date": "2024-01-15", "section": "item_7",
             "text": "Revenue data.", "score": 0.8, "file_path": "/path/file.txt"},
            {"ticker": "MSFT", "filing_date": "2024-07-30", "section": "item_7",
             "text": "Cloud data.", "score": 0.6, "file_path": "/path/file2.txt"},
        ]
        result = LocalRAG._format_excerpts(docs)
        assert "[Source 1]" in result
        assert "[Source 2]" in result

    def test_includes_file_path(self):
        docs = [
            {"ticker": "AAPL", "filing_date": "2024-01-15", "section": "item_7",
             "text": "Some text.", "score": 0.8, "file_path": "/data/AAPL_file.txt"},
        ]
        result = LocalRAG._format_excerpts(docs)
        assert "/data/AAPL_file.txt" in result


class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = LocalRAG._tokenize("Hello World! This is a Test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_numbers_preserved(self):
        tokens = LocalRAG._tokenize("Revenue grew 8% to $45.2 billion")
        assert "8" in tokens
        assert "45" in tokens
        assert "revenue" in tokens

    def test_empty_string(self):
        tokens = LocalRAG._tokenize("")
        assert tokens == []
