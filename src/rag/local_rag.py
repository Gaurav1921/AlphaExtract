"""
Local RAG Engine
-----------------
Lightweight RAG with hybrid retrieval (BM25 + vector search), cross-encoder
reranking, and grounded answer generation with source citations.

Works entirely offline without Docker or OpenSearch.
Uses in-memory BM25 + optional sentence-transformers vector search + optional
cross-encoder reranking. Falls back gracefully when models are unavailable.
"""

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config.settings import Settings
from src.data.chunker import DocumentChunker

logger = logging.getLogger(__name__)


class LocalRAG:
    """
    Self-contained RAG engine with hybrid retrieval and reranking.

    No external services required. Supports:
    - Layout-aware chunking via DocumentChunker (respects paragraph boundaries)
    - Hybrid retrieval: BM25 + vector cosine similarity with RRF fusion
    - Cross-encoder reranking (optional, lazy-loaded)
    - Grounded answer generation with source citations
    - Conversation history for follow-up questions
    """

    def __init__(self, sections_dir: Path = None):
        self.sections_dir = sections_dir or Settings.SECTIONS_DIR
        self._embedding_model = None
        self._reranker_model = None
        self._llm_client = None
        self._llm_provider = None
        self._llm_model = None
        self._documents: List[Dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._indexed = False
        self.conversation: List[Dict] = []
        self.max_history = 5
        self._chunker = DocumentChunker(
            chunk_size=Settings.CHUNK_SIZE,
            overlap=Settings.CHUNK_OVERLAP,
            min_chunk_size=20,  # low threshold so short documents are still indexed
        )
        # BM25 precomputed data
        self._idf: Dict[str, float] = {}
        self._doc_term_freqs: List[Dict[str, int]] = []
        self._avg_doc_len: float = 0.0

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(Settings.EMBEDDING_MODEL)
                logger.info(f"Loaded embedding model: {Settings.EMBEDDING_MODEL}")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
        return self._embedding_model

    @property
    def reranker(self):
        """Lazy-load cross-encoder reranker for second-stage scoring."""
        if self._reranker_model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker_model = CrossEncoder(Settings.RERANKER_MODEL)
                logger.info(f"Loaded reranker: {Settings.RERANKER_MODEL}")
            except Exception as e:
                logger.warning(f"Could not load reranker: {e}")
        return self._reranker_model

    @property
    def llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        self._init_llm()
        return self._llm_client

    def _init_llm(self):
        """Initialize LLM client (Groq > Gemini > Ollama)."""
        if Settings.GROQ_API_KEY:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI(
                    api_key=Settings.GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                )
                self._llm_provider = "groq"
                self._llm_model = Settings.GROQ_MODEL
                logger.info(f"LLM: Groq ({self._llm_model})")
                return
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")

        if Settings.GEMINI_API_KEY:
            try:
                from google import genai
                self._llm_client = genai.Client(api_key=Settings.GEMINI_API_KEY)
                self._llm_provider = "gemini"
                self._llm_model = Settings.GEMINI_MODEL
                logger.info(f"LLM: Gemini ({self._llm_model})")
                return
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        try:
            import urllib.request
            urllib.request.urlopen(f"{Settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
            from openai import OpenAI
            self._llm_client = OpenAI(
                api_key="ollama",
                base_url=f"{Settings.OLLAMA_BASE_URL}/v1",
            )
            self._llm_provider = "ollama"
            self._llm_model = Settings.OLLAMA_MODEL
            logger.info(f"LLM: Ollama ({self._llm_model})")
        except Exception:
            logger.info("No LLM provider available — using retrieval-only mode")

    # ------------------------------------------------------------------
    # Document loading & indexing
    # ------------------------------------------------------------------

    def _load_documents(self, ticker: str = None) -> List[Dict]:
        """Load section text files using layout-aware DocumentChunker."""
        pattern = f"{ticker.upper()}_*_item_*.txt" if ticker else "*_item_*.txt"
        section_files = sorted(self.sections_dir.glob(pattern))

        documents = []
        for filepath in section_files:
            try:
                chunks = self._chunker.chunk_file(filepath)
                documents.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to chunk {filepath.name}: {e}")
                continue

        logger.info(f"Loaded {len(documents)} chunks from {len(section_files)} files")
        return documents

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Whitespace + lowercase tokenizer for BM25."""
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def _build_bm25_index(self):
        """Precompute IDF and per-document term frequencies for BM25 scoring."""
        if not self._documents:
            return

        num_docs = len(self._documents)
        doc_freq: Dict[str, int] = defaultdict(int)
        self._doc_term_freqs = []

        for doc in self._documents:
            tokens = self._tokenize(doc["text"])
            tf = Counter(tokens)
            self._doc_term_freqs.append(tf)
            for term in set(tokens):
                doc_freq[term] += 1

        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {}
        for term, df in doc_freq.items():
            self._idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

        doc_lengths = [sum(tf.values()) for tf in self._doc_term_freqs]
        self._avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0

        logger.info(f"BM25 index: {len(self._idf)} terms, avg doc len {self._avg_doc_len:.0f}")

    def index(self, ticker: str = None):
        """Build in-memory search index for a ticker (or all tickers)."""
        self._documents = self._load_documents(ticker)

        if not self._documents:
            logger.warning("No documents to index")
            return

        # BM25 index (always available)
        self._build_bm25_index()

        # Vector index (only when embedding model is available)
        if self.embedding_model is not None:
            texts = [d["text"][:1000] for d in self._documents]
            self._embeddings = self.embedding_model.encode(
                texts, show_progress_bar=False, batch_size=64
            )
            logger.info(f"Indexed {len(self._documents)} chunks with embeddings + BM25")
        else:
            self._embeddings = None
            logger.info(f"Indexed {len(self._documents)} chunks (BM25-only mode)")

        self._indexed = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, k: int = 5, ticker: str = None) -> List[Dict]:
        """Retrieve top-k documents using cosine similarity."""
        if self._embeddings is None or self.embedding_model is None:
            return []

        query_embedding = self.embedding_model.encode(query)
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(self._embeddings, query_embedding) / norms

        if ticker:
            ticker = ticker.upper()
            mask = np.array([d["ticker"] == ticker for d in self._documents])
            similarities = np.where(mask, similarities, -1)

        top_indices = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_indices:
            if similarities[idx] <= 0:
                continue
            doc = self._documents[idx].copy()
            doc["score"] = float(similarities[idx])
            doc["retrieval"] = "vector"
            results.append(doc)

        return results

    def _bm25_search(self, query: str, k: int = 5, ticker: str = None) -> List[Dict]:
        """BM25 scoring with IDF weighting."""
        if not self._idf:
            return self._keyword_search(query, k, ticker)

        query_tokens = self._tokenize(query)
        k1, b = 1.5, 0.75  # standard BM25 parameters

        scored = []
        for i, doc in enumerate(self._documents):
            if ticker and doc["ticker"] != ticker.upper():
                continue

            tf = self._doc_term_freqs[i]
            doc_len = sum(tf.values())
            bm25_score = 0.0

            for term in query_tokens:
                if term not in tf:
                    continue
                term_tf = tf[term]
                idf = self._idf.get(term, 0.0)
                numerator = term_tf * (k1 + 1)
                denominator = term_tf + k1 * (1 - b + b * doc_len / self._avg_doc_len)
                bm25_score += idf * (numerator / denominator)

            if bm25_score > 0:
                result = doc.copy()
                result["score"] = bm25_score
                result["retrieval"] = "bm25"
                scored.append(result)

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    def _keyword_search(self, query: str, k: int = 5, ticker: str = None) -> List[Dict]:
        """Fallback: simple keyword-based retrieval using term frequency."""
        query_terms = set(query.lower().split())
        scored = []

        for doc in self._documents:
            if ticker and doc["ticker"] != ticker.upper():
                continue

            text_lower = doc["text"].lower()
            matches = sum(1 for term in query_terms if term in text_lower)
            if matches > 0:
                scored.append({**doc, "score": matches / len(query_terms), "retrieval": "keyword"})

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: List[Dict], k_rrf: int = 60, final_k: int = 5
    ) -> List[Dict]:
        """Combine multiple ranked result lists using Reciprocal Rank Fusion (RRF).

        RRF assigns each result a score of 1/(k + rank) for each list it appears in,
        then sums across lists. This is robust to score scale differences between
        retrieval methods.
        """
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Dict] = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                # Use a stable ID from document metadata
                doc_id = f"{doc.get('ticker', '')}_{doc.get('filing_date', '')}_{doc.get('section', '')}_{doc.get('chunk_id', 0)}"
                scores[doc_id] += 1.0 / (k_rrf + rank + 1)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:final_k]:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = round(score, 6)
            results.append(doc)

        return results

    def _rerank(self, query: str, docs: List[Dict], k: int = 5) -> List[Dict]:
        """Rerank documents using cross-encoder model for better precision."""
        if not docs or self.reranker is None:
            return docs[:k]

        pairs = [(query, doc["text"][:512]) for doc in docs]
        try:
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                docs[i]["rerank_score"] = float(score)
            docs.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
            logger.debug(f"Reranked {len(docs)} documents")
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")

        return docs[:k]

    def retrieve(self, query: str, k: int = 5, ticker: str = None) -> List[Dict]:
        """Retrieve relevant documents using hybrid search + reranking.

        Pipeline:
          1. BM25 search (always available)
          2. Vector search (when embedding model loaded)
          3. Reciprocal Rank Fusion to merge both result lists
          4. Cross-encoder reranking (when reranker model loaded)
        """
        if not self._indexed:
            self.index(ticker)

        fetch_k = k * 3  # over-fetch candidates for reranking

        # Stage 1: Dual retrieval
        vector_results = self._vector_search(query, k=fetch_k, ticker=ticker)
        bm25_results = self._bm25_search(query, k=fetch_k, ticker=ticker)

        # Stage 2: Fusion
        if vector_results and bm25_results:
            candidates = self._reciprocal_rank_fusion(
                vector_results, bm25_results, final_k=fetch_k
            )
        elif vector_results:
            candidates = vector_results[:fetch_k]
        elif bm25_results:
            candidates = bm25_results[:fetch_k]
        else:
            # Ultimate fallback
            return self._keyword_search(query, k=k, ticker=ticker)

        # Stage 3: Cross-encoder reranking
        results = self._rerank(query, candidates, k=k)

        return results

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def _generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using LLM with source citations, or fallback to excerpts."""
        if not context_docs:
            return "No relevant information found in the indexed filings."

        # Build numbered source context with clear citations
        source_blocks = []
        for i, doc in enumerate(context_docs[:5], 1):
            source_label = f"[Source {i}: {doc['ticker']} {doc['filing_date']} {doc['section']}]"
            file_ref = doc.get("file_path", "")
            text = doc["text"][:800]
            source_blocks.append(f"{source_label}\nFile: {file_ref}\n{text}")

        context = "\n\n---\n\n".join(source_blocks)

        # Build conversation context
        conv_context = ""
        if self.conversation:
            recent = self.conversation[-self.max_history * 2:]
            conv_parts = [
                f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content'][:200]}"
                for h in recent
            ]
            conv_context = "Prior conversation:\n" + "\n".join(conv_parts) + "\n\n"

        prompt = (
            "You are a financial analyst AI assistant. Answer the question using ONLY "
            "the provided 10-K filing excerpts.\n\n"
            "## STRICT RULES:\n"
            "1. ONLY state facts found in the excerpts below — never invent data\n"
            "2. Cite sources using [Source N] notation for every factual claim\n"
            "3. If the excerpts don't contain sufficient information, explicitly state: "
            "\"The available filings do not contain enough information to fully answer this.\"\n"
            "4. Include specific numbers, percentages, and metrics when available\n"
            "5. Compare across filings/dates when the question asks about trends\n\n"
            f"{conv_context}"
            f"## 10-K Filing Excerpts:\n\n{context}\n\n"
            f"## Question: {query}\n\n"
            "Answer (cite sources with [Source N]):"
        )

        if self.llm_client is None:
            return self._format_excerpts(context_docs)

        try:
            if self._llm_provider == "gemini":
                response = self._llm_client.models.generate_content(
                    model=self._llm_model, contents=prompt
                )
                return response.text.strip()
            else:
                response = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1000,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return self._format_excerpts(context_docs)

    @staticmethod
    def _format_excerpts(docs: List[Dict]) -> str:
        """Format a non-LLM answer from retrieved documents with source references."""
        if not docs:
            return "No relevant information found."
        parts = ["Based on the 10-K filings:\n"]
        for i, doc in enumerate(docs[:3], 1):
            file_ref = doc.get("file_path", "")
            file_note = f"\n_File: {file_ref}_" if file_ref else ""
            parts.append(
                f"**[Source {i}] {doc['ticker']} ({doc['filing_date']}) — {doc['section']}:**"
                f"{file_note}\n"
                f"{doc['text'][:500]}\n"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        ticker: str = None,
        k: int = 5,
        verbose: bool = False,
    ) -> Dict:
        """
        Full RAG query: retrieve -> rerank -> generate -> return answer with sources.

        Args:
            user_query: Natural language question.
            ticker: Optional ticker filter.
            k: Number of documents to retrieve.
            verbose: Log retrieval details.

        Returns:
            Dict with answer, sources, retrieval_method, and metadata.
        """
        # Handle follow-up queries
        follow_up_words = {"it", "that", "this", "they", "what about", "how about", "also"}
        query_lower = user_query.lower().strip()
        is_follow_up = any(query_lower.startswith(w) for w in follow_up_words)

        if is_follow_up and self.conversation:
            last_user = next(
                (h["content"] for h in reversed(self.conversation) if h["role"] == "user"),
                None,
            )
            expanded_query = f"{last_user} {user_query}" if last_user else user_query
        else:
            expanded_query = user_query

        # Retrieve with hybrid search + reranking
        docs = self.retrieve(expanded_query, k=k, ticker=ticker)
        if verbose:
            logger.info(f"Retrieved {len(docs)} documents for: {expanded_query[:80]}")
            for d in docs:
                logger.info(
                    f"  [{d.get('retrieval', 'hybrid')}] "
                    f"{d['ticker']} {d['section']} "
                    f"score={d.get('rrf_score', d.get('score', 0)):.4f}"
                )

        # Generate answer
        answer = self._generate_answer(user_query, docs)

        # Update conversation
        self.conversation.append({"role": "user", "content": user_query})
        self.conversation.append({"role": "assistant", "content": answer})

        # Determine retrieval method used
        methods = set()
        for d in docs:
            if "rrf_score" in d:
                methods.add("hybrid")
            elif d.get("retrieval"):
                methods.add(d["retrieval"])
        retrieval_method = "+".join(sorted(methods)) if methods else "keyword"
        if any("rerank_score" in d for d in docs):
            retrieval_method += "+reranked"

        return {
            "answer": answer,
            "sources": [
                {
                    "ticker": d["ticker"],
                    "filing_date": d["filing_date"],
                    "section": d["section"],
                    "score": round(
                        d.get("rrf_score", d.get("rerank_score", d.get("score", 0))), 4
                    ),
                    "file_path": d.get("file_path", ""),
                    "chunk_id": d.get("chunk_id", 0),
                }
                for d in docs
            ],
            "query_expanded": expanded_query != user_query,
            "retrieval_method": retrieval_method,
        }

    def clear_history(self):
        """Clear conversation history."""
        self.conversation.clear()

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def has_embeddings(self) -> bool:
        return self._embeddings is not None

    @property
    def llm_available(self) -> bool:
        return self.llm_client is not None

    def status(self) -> Dict:
        """Return current RAG engine status."""
        return {
            "indexed": self._indexed,
            "documents": len(self._documents),
            "has_embeddings": self._embeddings is not None,
            "has_bm25": bool(self._idf),
            "has_reranker": self._reranker_model is not None,
            "embedding_model": Settings.EMBEDDING_MODEL if self.embedding_model else None,
            "llm_provider": self._llm_provider,
            "llm_model": self._llm_model,
            "conversation_length": len(self.conversation),
        }
