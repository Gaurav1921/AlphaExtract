"""
Embedding Generator & Ingestion Pipeline
-----------------------------------------
Generates vector embeddings and uploads to OpenSearch.

Model: all-MiniLM-L6-v2
- 384 dimensions
- 22M parameters
- Fast inference (1000 docs/sec on CPU)
- Good quality for semantic search
"""

from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict
import json
import logging
from tqdm import tqdm
import numpy as np

from src.search.opensearch_client import OpenSearchManager
from src.data.chunker import DocumentChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    End-to-end pipeline: Chunk → Embed → Index
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        index_name: str = "alphaextract-docs"
    ):
        """
        Initialize embedding model and OpenSearch connection.
        
        Args:
            model_name: SentenceTransformer model name
            opensearch_host: OpenSearch host
            opensearch_port: OpenSearch port
            index_name: Index name
        """
        logger.info("Initializing embedding pipeline...")
        
        # Load embedding model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✓ Model loaded ({self.embedding_dim}D embeddings)")
        
        # Initialize OpenSearch
        self.opensearch = OpenSearchManager(
            host=opensearch_host,
            port=opensearch_port,
            index_name=index_name
        )
        
        # Initialize chunker
        self.chunker = DocumentChunker()
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            
        Returns:
            Array of embeddings (shape: [len(texts), embedding_dim])
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def prepare_documents(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add embeddings to chunk documents.
        
        Args:
            chunks: List of chunk documents (without embeddings)
            
        Returns:
            List of documents with embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to documents
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = chunk.copy()
            doc['embedding'] = embedding.tolist()
            documents.append(doc)
        
        logger.info(f"✓ Generated {len(documents)} embeddings")
        return documents
    
    def index_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Bulk index documents to OpenSearch.
        
        Args:
            documents: List of documents with embeddings
            batch_size: Batch size for bulk indexing
        """
        logger.info(f"Indexing {len(documents)} documents to OpenSearch...")
        
        total_success = 0
        total_failed = 0
        
        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing"):
            batch = documents[i:i + batch_size]
            success, failed = self.opensearch.bulk_index(batch)
            total_success += success
            total_failed += failed
        
        logger.info(f"✓ Indexed {total_success} documents ({total_failed} failed)")
    
    def process_directory(
        self,
        sections_dir: Path,
        recreate_index: bool = False
    ):
        """
        Complete pipeline: Chunk → Embed → Index
        
        Args:
            sections_dir: Directory with section .txt files
            recreate_index: If True, delete and recreate index
        """
        logger.info(f"\n{'='*80}")
        logger.info("EMBEDDING PIPELINE START")
        logger.info(f"{'='*80}")
        
        # Step 1: Create/verify index
        logger.info("\n[1] Setting up OpenSearch index...")
        self.opensearch.create_index(
            embedding_dim=self.embedding_dim,
            force=recreate_index
        )
        
        # Step 2: Chunk documents
        logger.info("\n[2] Chunking documents...")
        chunks, summary = self.chunker.chunk_directory(sections_dir)
        
        # Step 3: Generate embeddings
        logger.info("\n[3] Generating embeddings...")
        documents = self.prepare_documents(chunks)
        
        # Step 4: Index to OpenSearch
        logger.info("\n[4] Indexing to OpenSearch...")
        self.index_documents(documents)
        
        # Step 5: Verify
        logger.info("\n[5] Verifying index...")
        stats = self.opensearch.get_stats()
        
        logger.info(f"\n{'='*80}")
        logger.info("PIPELINE COMPLETE ✓")
        logger.info(f"{'='*80}")
        logger.info(f"  Documents indexed: {stats['document_count']}")
        logger.info(f"  Index size: {stats['size_mb']} MB")
        logger.info(f"  Embedding dimension: {self.embedding_dim}D")
        
        return stats
    
    def index_single_company(
        self,
        ticker: str,
        sections_dir: Path = None
    ):
        """
        Index documents for a single company.
        
        Args:
            ticker: Stock ticker
            sections_dir: Directory with section files
        """
        if sections_dir is None:
            sections_dir = Path("data/sections")
        
        # Find files for this ticker
        section_files = list(sections_dir.glob(f"{ticker}_*_item_*.txt"))
        
        if not section_files:
            logger.error(f"No files found for ticker: {ticker}")
            return
        
        logger.info(f"Found {len(section_files)} files for {ticker}")
        
        # Chunk files
        chunks = []
        for filepath in section_files:
            file_chunks = self.chunker.chunk_file(filepath)
            chunks.extend(file_chunks)
        
        # Generate embeddings and index
        documents = self.prepare_documents(chunks)
        self.index_documents(documents)
        
        logger.info(f"✓ Indexed {len(documents)} chunks for {ticker}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("EMBEDDING PIPELINE TEST")
    print("=" * 80)
    
    # Initialize pipeline
    print("\n[1] Initializing pipeline...")
    pipeline = EmbeddingPipeline()
    
    # Process all documents
    sections_dir = Path("data/sections")
    
    if not sections_dir.exists():
        print(f"\n❌ Directory not found: {sections_dir}")
        exit(1)
    
    # Ask user if they want to recreate index
    print("\n[2] Index setup:")
    stats = pipeline.opensearch.get_stats()
    
    if stats['exists']:
        print(f"  Existing index found:")
        print(f"    Documents: {stats['document_count']}")
        print(f"    Size: {stats['size_mb']} MB")
        
        response = input("\n  Recreate index? (y/n): ")
        recreate = response.lower() == 'y'
    else:
        print("  No existing index found")
        recreate = True
    
    # Run pipeline
    print("\n[3] Running pipeline...")
    final_stats = pipeline.process_directory(
        sections_dir=sections_dir,
        recreate_index=recreate
    )
    
    print("\n" + "=" * 80)
    print("Ready for RAG queries!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Test semantic search")
    print("2. Integrate with Gemini")
    print("3. Build query interface")