"""
OpenSearch Setup for AlphaExtract
----------------------------------
Initializes OpenSearch index with K-NN vector search capabilities.

Index structure:
- Vector embeddings (384 dimensions from sentence-transformers)
- Metadata: ticker, section, filing_date, chunk_id
- Full text for keyword search fallback
"""

from opensearchpy import OpenSearch, helpers
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenSearchManager:
    """
    Manages OpenSearch connection and index operations.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "alphaextract-docs"
    ):
        """
        Initialize OpenSearch connection.
        
        Args:
            host: OpenSearch host
            port: OpenSearch port
            index_name: Name of the index
        """
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        
        self.index_name = index_name
        
        # Test connection
        try:
            info = self.client.info()
            logger.info(f"Connected to OpenSearch {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to OpenSearch: {e}")
            raise
    
    def create_index(self, embedding_dim: int = 384, force: bool = False):
        """
        Create index with K-NN vector search enabled.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            force: If True, delete existing index first
        """
        # Delete existing index if force=True
        if force and self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info(f"Deleted existing index: {self.index_name}")
        
        # Check if index already exists
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index {self.index_name} already exists")
            return
        
        # Index settings and mappings
        index_body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True,  # Enable K-NN
                    "knn.algo_param.ef_search": 100  # Search accuracy
                }
            },
            "mappings": {
                "properties": {
                    # Vector embedding for semantic search
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    
                    # Text content (for keyword fallback)
                    "text": {
                        "type": "text",
                        "analyzer": "english"
                    },
                    
                    # Metadata fields
                    "ticker": {
                        "type": "keyword"
                    },
                    "section": {
                        "type": "keyword"  # item_1a, item_7, item_8
                    },
                    "filing_date": {
                        "type": "date",
                        "format": "yyyy-MM-dd"
                    },
                    "chunk_id": {
                        "type": "integer"
                    },
                    "word_count": {
                        "type": "integer"
                    },
                    "file_path": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        self.client.indices.create(index=self.index_name, body=index_body)
        logger.info(f"✓ Created index: {self.index_name}")
        logger.info(f"  - K-NN enabled with {embedding_dim}D vectors")
        logger.info(f"  - HNSW algorithm (cosine similarity)")
    
    def index_document(self, doc: Dict):
        """
        Index a single document.
        
        Args:
            doc: Document with embedding, text, and metadata
        """
        self.client.index(
            index=self.index_name,
            body=doc,
            refresh=True
        )
    
    def bulk_index(self, docs: List[Dict]):
        """
        Bulk index multiple documents.
        
        Args:
            docs: List of documents
        """
        actions = [
            {
                "_index": self.index_name,
                "_source": doc
            }
            for doc in docs
        ]
        
        success, failed = helpers.bulk(
            self.client,
            actions,
            raise_on_error=False,
            request_timeout=60
        )
        
        # failed is a list of error details, count it
        failed_count = len(failed) if isinstance(failed, list) else failed
        
        logger.info(f"Indexed {success} documents ({failed_count} failed)")
        return success, failed_count
    
    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filters: Dict = None
    ) -> List[Dict]:
        """
        K-NN vector search with optional metadata filtering.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            filters: Metadata filters (e.g., {"ticker": "AAPL"})
            
        Returns:
            List of search results with scores
        """
        # Build query
        query = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": k
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # Add filters if provided
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                filter_clauses.append({"term": {field: value}})
            
            query["query"]["bool"]["filter"] = filter_clauses
        
        # Execute search
        response = self.client.search(
            index=self.index_name,
            body=query
        )
        
        # Parse results
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'score': hit['_score'],
                'text': hit['_source']['text'],
                'ticker': hit['_source']['ticker'],
                'section': hit['_source']['section'],
                'filing_date': hit['_source']['filing_date'],
                'chunk_id': hit['_source']['chunk_id'],
                'file_path': hit['_source']['file_path']
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        if not self.client.indices.exists(index=self.index_name):
            return {"exists": False}
        
        stats = self.client.indices.stats(index=self.index_name)
        count = self.client.count(index=self.index_name)
        
        return {
            "exists": True,
            "document_count": count['count'],
            "size_bytes": stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
            "size_mb": round(stats['indices'][self.index_name]['total']['store']['size_in_bytes'] / 1024 / 1024, 2)
        }
    
    def delete_by_query(self, filters: Dict):
        """
        Delete documents matching filters.
        
        Args:
            filters: Metadata filters (e.g., {"ticker": "AAPL"})
        """
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {field: value}}
                        for field, value in filters.items()
                    ]
                }
            }
        }
        
        response = self.client.delete_by_query(
            index=self.index_name,
            body=query
        )
        
        logger.info(f"Deleted {response['deleted']} documents")
        return response['deleted']


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("OPENSEARCH SETUP TEST")
    print("=" * 80)
    
    # Initialize manager
    print("\n[1] Connecting to OpenSearch...")
    try:
        manager = OpenSearchManager()
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure OpenSearch is running:")
        print("  docker-compose up -d")
        exit(1)
    
    # Create index
    print("\n[2] Creating index...")
    manager.create_index(force=True)
    
    # Test document
    print("\n[3] Indexing test document...")
    test_doc = {
        "embedding": [0.1] * 384,  # Dummy embedding
        "text": "Tesla faces battery supply chain risks.",
        "ticker": "TSLA",
        "section": "item_1a",
        "filing_date": "2025-01-30",
        "chunk_id": 1,
        "word_count": 6,
        "file_path": "data/sections/TSLA_2025-01-30_item_1a.txt"
    }
    
    manager.index_document(test_doc)
    print("✓ Document indexed")
    
    # Get stats
    print("\n[4] Index statistics:")
    stats = manager.get_stats()
    print(f"  Documents: {stats['document_count']}")
    print(f"  Size: {stats['size_mb']} MB")
    
    # Test search
    print("\n[5] Testing K-NN search...")
    results = manager.search(
        query_vector=[0.1] * 384,
        k=1
    )
    
    if results:
        print(f"✓ Found {len(results)} results")
        print(f"  Top result: {results[0]['text'][:50]}...")
    
    print("\n" + "=" * 80)
    print("OpenSearch setup complete! ✓")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run the document chunker")
    print("2. Generate embeddings")
    print("3. Bulk upload to OpenSearch")