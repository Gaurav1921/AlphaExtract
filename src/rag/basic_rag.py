"""
Basic RAG Engine
-----------------
Simple retrieve-and-generate pipeline for 10-K filing Q&A.
"""

import logging
from typing import Dict, List

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class BasicRAG:
    """Simple RAG: embed query -> KNN search -> generate answer."""

    def __init__(self):
        self._embedding_model = None
        self._llm = None
        self._os_client = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(Settings.EMBEDDING_MODEL)
        return self._embedding_model

    @property
    def llm(self):
        if self._llm is None and Settings.GEMINI_API_KEY:
            import google.generativeai as genai
            genai.configure(api_key=Settings.GEMINI_API_KEY)
            self._llm = genai.GenerativeModel(Settings.GEMINI_MODEL)
        return self._llm

    @property
    def os_client(self):
        if self._os_client is None:
            from src.search.opensearch_client import OpenSearchClient
            self._os_client = OpenSearchClient()
        return self._os_client

    def query(self, user_query: str, ticker: str = None, k: int = None) -> Dict:
        """Simple RAG query: retrieve -> generate."""
        k = k or Settings.TOP_K

        embedding = self.embedding_model.encode(user_query).tolist()
        results = self.os_client.knn_search(embedding, k=k, ticker=ticker)

        if not results:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n---\n\n".join(
            f"[{doc['ticker']} {doc['filing_date']} {doc['section']}]\n{doc['text']}"
            for doc in results
        )

        if self.llm:
            prompt = (
                "You are a financial analyst. Answer using ONLY the provided 10-K excerpts.\n\n"
                f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
            )
            try:
                response = self.llm.generate_content(prompt)
                answer = response.text.strip()
            except Exception as e:
                logger.error(f"LLM failed: {e}")
                answer = self._fallback_answer(results)
        else:
            answer = self._fallback_answer(results)

        return {
            "answer": answer,
            "sources": [
                {"ticker": d["ticker"], "filing_date": d["filing_date"], "section": d["section"], "score": d["score"]}
                for d in results
            ],
        }

    @staticmethod
    def _fallback_answer(docs: List[Dict]) -> str:
        parts = ["Based on the filings:\n"]
        for doc in docs[:3]:
            parts.append(f"**{doc['ticker']} ({doc['filing_date']}):** {doc['text'][:400]}\n")
        return "\n".join(parts)
