"""
Local RAG Engine
-----------------
Lightweight RAG that works entirely offline without Docker or OpenSearch.
Uses in-memory vector search with sentence-transformers + numpy cosine similarity.
Falls back to keyword-based retrieval when no embedding model is available.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class LocalRAG:
    """
    Self-contained RAG engine that loads section files directly from disk.

    No external services required. Supports:
    - In-memory vector search (sentence-transformers)
    - Keyword-based fallback retrieval
    - LLM answer generation (Groq / Gemini / Ollama)
    - Conversation history for follow-up questions
    """

    def __init__(self, sections_dir: Path = None):
        self.sections_dir = sections_dir or Settings.SECTIONS_DIR
        self._embedding_model = None
        self._llm_client = None
        self._llm_provider = None
        self._llm_model = None
        self._documents: List[Dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._indexed = False
        self.conversation: List[Dict] = []
        self.max_history = 5

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
    def llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        self._init_llm()
        return self._llm_client

    def _init_llm(self):
        """Initialize LLM client (Groq > Gemini > Ollama)."""
        # Try Groq first (generous free tier)
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

        # Try Gemini
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

        # Try Ollama (local)
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
        """Load section text files into memory as documents."""
        pattern = f"{ticker.upper()}_*_item_*.txt" if ticker else "*_item_*.txt"
        section_files = sorted(self.sections_dir.glob(pattern))

        documents = []
        for filepath in section_files:
            parts = filepath.stem.split("_")
            if len(parts) < 3:
                continue

            doc_ticker = parts[0]
            filing_date = parts[1]

            # Reconstruct section key
            section_parts = []
            capture = False
            for part in parts:
                if part == "item":
                    capture = True
                if capture:
                    section_parts.append(part)
            section_key = "_".join(section_parts) if section_parts else "unknown"

            try:
                text = filepath.read_text(encoding="utf-8")
            except OSError:
                continue

            # Split into paragraphs for finer-grained retrieval
            paragraphs = re.split(r"\n\s*\n", text)
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para.split()) < 20:
                    continue
                documents.append({
                    "text": para,
                    "ticker": doc_ticker,
                    "filing_date": filing_date,
                    "section": section_key,
                    "chunk_id": i,
                })

        logger.info(f"Loaded {len(documents)} text chunks from {len(section_files)} files")
        return documents

    def index(self, ticker: str = None):
        """Build in-memory search index for a ticker (or all tickers)."""
        self._documents = self._load_documents(ticker)

        if not self._documents:
            logger.warning("No documents to index")
            return

        if self.embedding_model is not None:
            texts = [d["text"][:1000] for d in self._documents]  # Truncate for embedding
            self._embeddings = self.embedding_model.encode(
                texts, show_progress_bar=False, batch_size=64
            )
            logger.info(f"Indexed {len(self._documents)} chunks with embeddings")
        else:
            self._embeddings = None
            logger.info(f"Indexed {len(self._documents)} chunks (keyword-only mode)")

        self._indexed = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, k: int = 5, ticker: str = None) -> List[Dict]:
        """Retrieve top-k documents using cosine similarity."""
        if self._embeddings is None or self.embedding_model is None:
            return []

        query_embedding = self.embedding_model.encode(query)
        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(self._embeddings, query_embedding) / norms

        # Filter by ticker if specified
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
            results.append(doc)

        return results

    def _keyword_search(self, query: str, k: int = 5, ticker: str = None) -> List[Dict]:
        """Fallback: keyword-based retrieval using term frequency."""
        query_terms = set(query.lower().split())
        scored = []

        for doc in self._documents:
            if ticker and doc["ticker"] != ticker.upper():
                continue

            text_lower = doc["text"].lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in text_lower)
            if matches > 0:
                scored.append({**doc, "score": matches / len(query_terms)})

        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    def retrieve(self, query: str, k: int = 5, ticker: str = None) -> List[Dict]:
        """Retrieve relevant documents using best available method."""
        if not self._indexed:
            self.index(ticker)

        # Try vector search first
        results = self._vector_search(query, k=k, ticker=ticker)
        if results:
            return results

        # Fallback to keyword search
        return self._keyword_search(query, k=k, ticker=ticker)

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def _generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using LLM or fallback to excerpts."""
        if not context_docs:
            return "No relevant information found in the indexed filings."

        context = "\n\n---\n\n".join(
            f"[{doc['ticker']} {doc['filing_date']} {doc['section']}]\n{doc['text'][:600]}"
            for doc in context_docs[:5]
        )

        # Build conversation context
        conv_context = ""
        if self.conversation:
            recent = self.conversation[-self.max_history * 2:]
            conv_parts = []
            for h in recent:
                role = "User" if h["role"] == "user" else "Assistant"
                conv_parts.append(f"{role}: {h['content'][:200]}")
            conv_context = "\n".join(conv_parts)

        prompt = (
            "You are a financial analyst AI. Answer the question using ONLY the provided "
            "10-K filing excerpts. Be specific and cite which section/year.\n\n"
            f"{'Conversation context:\\n' + conv_context + '\\n\\n' if conv_context else ''}"
            f"10-K Filing Excerpts:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
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
                # OpenAI-compatible (Groq, Ollama)
                response = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return self._format_excerpts(context_docs)

    @staticmethod
    def _format_excerpts(docs: List[Dict]) -> str:
        """Format a non-LLM answer from retrieved documents."""
        if not docs:
            return "No relevant information found."
        parts = ["Based on the 10-K filings:\n"]
        for doc in docs[:3]:
            parts.append(
                f"**{doc['ticker']} ({doc['filing_date']}) - {doc['section']}:**\n"
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
        Full RAG query: retrieve -> generate -> return answer with sources.

        Args:
            user_query: Natural language question.
            ticker: Optional ticker filter.
            k: Number of documents to retrieve.
            verbose: Log retrieval details.

        Returns:
            Dict with answer, sources, and metadata.
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
            if last_user:
                expanded_query = f"{last_user} {user_query}"
            else:
                expanded_query = user_query
        else:
            expanded_query = user_query

        # Retrieve
        docs = self.retrieve(expanded_query, k=k, ticker=ticker)
        if verbose:
            logger.info(f"Retrieved {len(docs)} documents for: {expanded_query[:80]}")

        # Generate answer
        answer = self._generate_answer(user_query, docs)

        # Update conversation
        self.conversation.append({"role": "user", "content": user_query})
        self.conversation.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": [
                {
                    "ticker": d["ticker"],
                    "filing_date": d["filing_date"],
                    "section": d["section"],
                    "score": round(d.get("score", 0), 4),
                }
                for d in docs
            ],
            "query_expanded": expanded_query != user_query,
            "retrieval_method": "vector" if self._embeddings is not None else "keyword",
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
            "embedding_model": Settings.EMBEDDING_MODEL if self.embedding_model else None,
            "llm_provider": self._llm_provider,
            "llm_model": self._llm_model,
            "conversation_length": len(self.conversation),
        }
