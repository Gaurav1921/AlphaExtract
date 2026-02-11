"""
Embedding Pipeline
-------------------
Generates vector embeddings for document chunks and indexes them in OpenSearch.
"""

import logging
from pathlib import Path
from typing import Dict, List

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Generate embeddings and index into OpenSearch."""

    def __init__(self, model_name: str = None, batch_size: int = None):
        self.model_name = model_name or Settings.EMBEDDING_MODEL
        self.batch_size = batch_size or Settings.EMBEDDING_BATCH_SIZE
        self._model = None
        self._os_client = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def os_client(self):
        if self._os_client is None:
            from src.search.opensearch_client import OpenSearchClient
            self._os_client = OpenSearchClient()
        return self._os_client

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """Add embeddings to document dicts."""
        texts = [doc["text"] for doc in documents]
        embeddings = self.generate_embeddings(texts)

        for doc, emb in zip(documents, embeddings):
            doc["embedding"] = emb

        return documents

    def index_documents(self, documents: List[Dict]) -> Dict:
        """Embed and index documents into OpenSearch."""
        if not documents:
            return {"indexed": 0, "failed": 0}

        embedded_docs = self.embed_documents(documents)
        return self.os_client.bulk_index(embedded_docs)

    def index_from_sections(self, ticker: str = None) -> Dict:
        """Generate chunks, embed, and index from section files."""
        from src.data.chunker import DocumentChunker

        chunker = DocumentChunker()
        sections_dir = Settings.SECTIONS_DIR

        if ticker:
            files = sorted(sections_dir.glob(f"{ticker.upper()}_*_item_*.txt"))
        else:
            files = sorted(sections_dir.glob("*_item_*.txt"))

        if not files:
            logger.warning("No section files found to index")
            return {"indexed": 0, "failed": 0}

        all_docs: List[Dict] = []
        for filepath in files:
            try:
                all_docs.extend(chunker.chunk_file(filepath))
            except Exception as e:
                logger.error(f"Failed to chunk {filepath.name}: {e}")

        logger.info(f"Embedding and indexing {len(all_docs)} chunks from {len(files)} files")
        return self.index_documents(all_docs)
