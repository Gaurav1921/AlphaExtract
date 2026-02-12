"""
Enhanced RAG Engine
--------------------
Multi-turn conversational RAG with query expansion, re-ranking, and Gemini LLM.
"""

import re
import logging
from typing import Dict, List, Optional

from src.config.settings import Settings

logger = logging.getLogger(__name__)


FINANCIAL_SYNONYMS = {
    "revenue": ["revenue", "sales", "top line", "turnover"],
    "profit": ["profit", "net income", "earnings", "bottom line"],
    "risk": ["risk factors", "risk", "uncertainty", "exposure"],
    "ai": ["artificial intelligence", "machine learning", "AI", "deep learning"],
    "debt": ["debt", "liabilities", "borrowings", "leverage"],
    "competition": ["competition", "competitors", "competitive landscape", "market share"],
}


class ConversationHistory:
    """Manages multi-turn conversation context."""

    FOLLOW_UP_INDICATORS = {"it", "that", "this", "they", "what about", "how about", "also", "and"}

    def __init__(self, max_history: int = None):
        self.max_history = max_history or Settings.MAX_CONVERSATION_HISTORY
        self.history: List[Dict] = []

    def add(self, role: str, content: str, sources: List[str] = None):
        self.history.append({"role": role, "content": content, "sources": sources or []})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

    def is_follow_up(self, query: str) -> bool:
        """Detect if query references previous conversation."""
        query_lower = query.lower().strip()
        return any(query_lower.startswith(kw) for kw in self.FOLLOW_UP_INDICATORS)

    def expand_follow_up(self, query: str) -> str:
        """Expand a follow-up query using conversation context."""
        if not self.history:
            return query
        last_user = next((h["content"] for h in reversed(self.history) if h["role"] == "user"), None)
        if last_user:
            return f"{last_user} {query}"
        return query

    def get_context_string(self) -> str:
        """Format recent history for LLM context."""
        recent = self.history[-self.max_history * 2 :]
        parts = []
        for h in recent:
            role = "User" if h["role"] == "user" else "Assistant"
            content = h["content"][:200]
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


class QueryExpander:
    """Expands queries with financial domain synonyms."""

    @staticmethod
    def expand(query: str) -> List[str]:
        """Generate query variations using financial synonyms."""
        variations = [query]
        query_lower = query.lower()

        for key, synonyms in FINANCIAL_SYNONYMS.items():
            for syn in synonyms:
                if syn.lower() in query_lower:
                    for replacement in synonyms:
                        if replacement.lower() != syn.lower():
                            new_query = re.sub(re.escape(syn), replacement, query, flags=re.IGNORECASE)
                            if new_query not in variations:
                                variations.append(new_query)
                    break

        return variations[:3]  # Limit to 3 variations


class EnhancedRAG:
    """Production-grade RAG with conversation history, query expansion, re-ranking."""

    def __init__(self):
        self._embedding_model = None
        self._reranker = None
        self._llm = None
        self._os_client = None
        self.conversation = ConversationHistory()
        self.expander = QueryExpander()

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {Settings.EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(Settings.EMBEDDING_MODEL)
        return self._embedding_model

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker: {Settings.RERANKER_MODEL}")
            self._reranker = CrossEncoder(Settings.RERANKER_MODEL)
        return self._reranker

    @property
    def llm(self):
        if self._llm is None and Settings.GEMINI_API_KEY:
            from google import genai
            self._llm = genai.Client(api_key=Settings.GEMINI_API_KEY)
            logger.info(f"LLM client initialized: {Settings.GEMINI_MODEL}")
        return self._llm

    @property
    def os_client(self):
        if self._os_client is None:
            from src.search.opensearch_client import OpenSearchClient
            self._os_client = OpenSearchClient()
        return self._os_client

    def _retrieve(self, query: str, ticker: str = None, k: int = None) -> List[Dict]:
        """Retrieve documents using query expansion and multi-query fusion."""
        k = k or Settings.TOP_K
        variations = self.expander.expand(query)

        all_results = {}
        for variation in variations:
            embedding = self.embedding_model.encode(variation).tolist()
            results = self.os_client.knn_search(embedding, k=k * 2, ticker=ticker)
            for doc in results:
                key = f"{doc['ticker']}_{doc['filing_date']}_{doc['chunk_id']}"
                if key not in all_results or doc["score"] > all_results[key]["score"]:
                    all_results[key] = doc

        return list(all_results.values())

    def _rerank(self, query: str, documents: List[Dict], k: int = None) -> List[Dict]:
        """Re-rank documents using cross-encoder."""
        k = k or Settings.TOP_K
        if not documents:
            return []

        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(key=lambda d: d["rerank_score"], reverse=True)
        return documents[:k]

    def _generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using Gemini LLM."""
        if not self.llm:
            return self._format_fallback(context_docs)

        context = "\n\n---\n\n".join(
            f"[{doc['ticker']} {doc['filing_date']} {doc['section']}]\n{doc['text']}"
            for doc in context_docs
        )

        conv_context = self.conversation.get_context_string()

        prompt = (
            "You are a financial analyst AI. Answer the user's question using ONLY the provided "
            "10-K filing excerpts. Be specific and cite which section/year the information comes from.\n\n"
            f"{'Conversation context:' + chr(10) + conv_context + chr(10) + chr(10) if conv_context else ''}"
            f"10-K Filing Excerpts:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            response = self.llm.models.generate_content(
                model=Settings.GEMINI_MODEL, contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._format_fallback(context_docs)

    @staticmethod
    def _format_fallback(docs: List[Dict]) -> str:
        """Format a non-LLM answer from retrieved documents."""
        if not docs:
            return "No relevant information found in the indexed filings."
        parts = ["Based on the 10-K filings:\n"]
        for doc in docs[:3]:
            parts.append(f"**{doc['ticker']} ({doc['filing_date']}) - {doc['section']}:**\n{doc['text'][:500]}\n")
        return "\n".join(parts)

    def query(self, user_query: str, ticker: str = None, k: int = None, verbose: bool = False) -> Dict:
        """Full RAG query pipeline: expand -> retrieve -> rerank -> generate."""
        k = k or Settings.TOP_K

        # Handle follow-up queries
        if self.conversation.is_follow_up(user_query):
            expanded = self.conversation.expand_follow_up(user_query)
            logger.debug(f"Follow-up expanded: {expanded}")
        else:
            expanded = user_query

        # Retrieve
        candidates = self._retrieve(expanded, ticker=ticker, k=k)
        if verbose:
            logger.info(f"Retrieved {len(candidates)} candidates")

        # Re-rank
        ranked = self._rerank(expanded, candidates, k=k)
        if verbose:
            logger.info(f"Re-ranked to top {len(ranked)}")

        # Generate
        answer = self._generate_answer(user_query, ranked)

        # Update conversation
        self.conversation.add("user", user_query)
        self.conversation.add("assistant", answer, sources=[d.get("file_path", "") for d in ranked])

        return {
            "answer": answer,
            "sources": [
                {"ticker": d["ticker"], "filing_date": d["filing_date"], "section": d["section"], "score": d.get("rerank_score", d.get("score", 0))}
                for d in ranked
            ],
            "query_expanded": expanded != user_query,
        }
