"""
OpenSearch Client
------------------
Manages connection, indexing, and KNN search against OpenSearch.
"""

import logging
from typing import Dict, List

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """Client for OpenSearch vector database operations."""

    def __init__(self, host: str = None, port: int = None, index_name: str = None):
        self.host = host or Settings.OPENSEARCH_HOST
        self.port = port or Settings.OPENSEARCH_PORT
        self.index_name = index_name or Settings.OPENSEARCH_INDEX
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self):
        """Establish connection to OpenSearch."""
        from opensearchpy import OpenSearch
        try:
            client = OpenSearch(
                hosts=[{"host": self.host, "port": self.port}],
                http_compress=True,
                use_ssl=Settings.OPENSEARCH_USE_SSL,
                verify_certs=False,
                ssl_show_warn=False,
                timeout=Settings.OPENSEARCH_BULK_TIMEOUT,
            )
            info = client.info()
            logger.info(f"Connected to OpenSearch {info['version']['number']} at {self.host}:{self.port}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch at {self.host}:{self.port}: {e}")
            raise

    def create_index(self, recreate: bool = False):
        """Create the vector index with KNN mapping."""
        if self.client.indices.exists(self.index_name):
            if recreate:
                logger.info(f"Deleting existing index: {self.index_name}")
                self.client.indices.delete(self.index_name)
            else:
                logger.info(f"Index {self.index_name} already exists")
                return

        body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": Settings.HNSW_EF_SEARCH,
                },
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": Settings.EMBEDDING_DIM,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": Settings.HNSW_EF_CONSTRUCTION,
                                "m": Settings.HNSW_M,
                            },
                        },
                    },
                    "text": {"type": "text"},
                    "ticker": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "filing_date": {"type": "keyword"},
                    "chunk_id": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "file_path": {"type": "keyword"},
                },
            },
        }

        self.client.indices.create(self.index_name, body=body)
        logger.info(f"Created index: {self.index_name}")

    def bulk_index(self, documents: List[Dict]) -> Dict:
        """Index documents in bulk."""
        from opensearchpy import helpers

        if not documents:
            return {"indexed": 0, "failed": 0}

        self.create_index()

        actions = []
        for doc in documents:
            actions.append({"_index": self.index_name, "_source": doc})

        try:
            success, errors = helpers.bulk(
                self.client,
                actions,
                max_retries=2,
                request_timeout=Settings.OPENSEARCH_BULK_TIMEOUT,
            )
            failed = len(errors) if isinstance(errors, list) else errors
            logger.info(f"Bulk index: {success} indexed, {failed} failed")
            self.client.indices.refresh(self.index_name)
            return {"indexed": success, "failed": failed}
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return {"indexed": 0, "failed": len(documents), "error": str(e)}

    def knn_search(
        self,
        query_vector: List[float],
        k: int = None,
        ticker: str = None,
        section: str = None,
    ) -> List[Dict]:
        """Search for similar documents using KNN."""
        k = k or Settings.TOP_K

        query = {
            "size": k,
            "query": {"knn": {"embedding": {"vector": query_vector, "k": k}}},
        }

        # Apply filters
        filters = []
        if ticker:
            filters.append({"term": {"ticker": ticker.upper()}})
        if section:
            filters.append({"term": {"section": section}})

        if filters:
            query["query"] = {
                "bool": {
                    "must": [{"knn": {"embedding": {"vector": query_vector, "k": k}}}],
                    "filter": filters,
                }
            }

        try:
            response = self.client.search(index=self.index_name, body=query)
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "text": source.get("text", ""),
                    "ticker": source.get("ticker", ""),
                    "section": source.get("section", ""),
                    "filing_date": source.get("filing_date", ""),
                    "chunk_id": source.get("chunk_id", 0),
                    "score": hit.get("_score", 0),
                    "file_path": source.get("file_path", ""),
                })
            return results
        except Exception as e:
            logger.error(f"KNN search failed: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get index statistics."""
        try:
            if not self.client.indices.exists(self.index_name):
                return {"exists": False}
            stats = self.client.indices.stats(self.index_name)
            count = self.client.count(index=self.index_name)
            return {
                "exists": True,
                "document_count": count["count"],
                "store_size": stats["_all"]["primaries"]["store"]["size_in_bytes"],
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"exists": False, "error": str(e)}

    def delete_by_ticker(self, ticker: str) -> Dict:
        """Delete all documents for a ticker."""
        try:
            result = self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"ticker": ticker.upper()}}},
            )
            deleted = result.get("deleted", 0)
            logger.info(f"Deleted {deleted} documents for {ticker}")
            return {"deleted": deleted}
        except Exception as e:
            logger.error(f"Failed to delete documents for {ticker}: {e}")
            return {"deleted": 0, "error": str(e)}
