"""
Document Chunker for RAG
-------------------------
Intelligently splits 10-K sections into semantic chunks.

Strategy:
- Target: 500 words per chunk
- Overlap: 50 words (preserves context)
- Preserve: Paragraph boundaries
- Add: Metadata (ticker, section, chunk_id)
"""

import re
from pathlib import Path
from typing import List, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Semantic document chunking for RAG.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,  # words
        overlap: int = 50,      # words
        min_chunk_size: int = 100  # words
    ):
        """
        Initialize chunker with size parameters.
        
        Args:
            chunk_size: Target words per chunk
            overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def word_count(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into semantic units.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        # Split into paragraphs first
        paragraphs = self.split_into_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para_words = self.word_count(para)
            
            # If adding this paragraph exceeds chunk size
            if current_word_count + para_words > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))
                
                # Start new chunk with overlap
                # Keep last paragraph for context
                if len(current_chunk) > 0:
                    overlap_text = current_chunk[-1]
                    overlap_words = self.word_count(overlap_text)
                    
                    if overlap_words <= self.overlap:
                        current_chunk = [overlap_text]
                        current_word_count = overlap_words
                    else:
                        # Split last paragraph for overlap
                        words = overlap_text.split()
                        overlap_text = ' '.join(words[-self.overlap:])
                        current_chunk = [overlap_text]
                        current_word_count = self.overlap
                else:
                    current_chunk = []
                    current_word_count = 0
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_word_count += para_words
        
        # Add final chunk
        if current_chunk:
            final_chunk = '\n\n'.join(current_chunk)
            if self.word_count(final_chunk) >= self.min_chunk_size:
                chunks.append(final_chunk)
        
        return chunks
    
    def chunk_file(self, filepath: Path) -> List[Dict]:
        """
        Chunk a section file into documents.
        
        Args:
            filepath: Path to section .txt file
            
        Returns:
            List of chunk documents with metadata
        """
        # Parse filename: TICKER_DATE_SECTION.txt
        parts = filepath.stem.split('_')
        
        if len(parts) >= 3:
            ticker = parts[0]
            filing_date = parts[1]
            section = parts[2]
        else:
            logger.warning(f"Unexpected filename format: {filepath.name}")
            ticker = "UNKNOWN"
            filing_date = "UNKNOWN"
            section = "UNKNOWN"
        
        # Read text
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                'text': chunk,
                'ticker': ticker,
                'section': section,
                'filing_date': filing_date,
                'chunk_id': i,
                'word_count': self.word_count(chunk),
                'file_path': str(filepath)
            }
            documents.append(doc)
        
        logger.info(
            f"Chunked {filepath.name}: "
            f"{len(chunks)} chunks, "
            f"avg {sum(d['word_count'] for d in documents) / len(documents):.0f} words/chunk"
        )
        
        return documents
    
    def chunk_directory(self, directory: Path) -> List[Dict]:
        """
        Chunk all section files in a directory.
        
        Args:
            directory: Directory with section files
            
        Returns:
            List of all chunk documents
        """
        section_files = list(directory.glob("*_item_*.txt"))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"CHUNKING {len(section_files)} FILES")
        logger.info(f"{'='*80}")
        
        all_documents = []
        
        for filepath in section_files:
            try:
                docs = self.chunk_file(filepath)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to chunk {filepath.name}: {e}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"CHUNKING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"  Total chunks: {len(all_documents)}")
        logger.info(f"  Avg words/chunk: {sum(d['word_count'] for d in all_documents) / len(all_documents):.0f}")
        
        # Save summary
        summary = {
            'total_files': len(section_files),
            'total_chunks': len(all_documents),
            'avg_words_per_chunk': sum(d['word_count'] for d in all_documents) / len(all_documents),
            'chunks_by_ticker': {},
            'chunks_by_section': {}
        }
        
        # Group by ticker
        for doc in all_documents:
            ticker = doc['ticker']
            section = doc['section']
            
            summary['chunks_by_ticker'][ticker] = summary['chunks_by_ticker'].get(ticker, 0) + 1
            summary['chunks_by_section'][section] = summary['chunks_by_section'].get(section, 0) + 1
        
        return all_documents, summary


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DOCUMENT CHUNKER TEST")
    print("=" * 80)
    
    # Initialize chunker
    chunker = DocumentChunker(
        chunk_size=500,
        overlap=50,
        min_chunk_size=100
    )
    
    # Test on sections directory
    sections_dir = Path("data/sections")
    
    if not sections_dir.exists():
        print(f"\n❌ Directory not found: {sections_dir}")
        print("Run split_sections.py first!")
        exit(1)
    
    print(f"\n[1] Chunking documents from {sections_dir}...")
    documents, summary = chunker.chunk_directory(sections_dir)
    
    print(f"\n[2] Summary:")
    print(f"  Total chunks: {summary['total_chunks']}")
    print(f"  Avg words/chunk: {summary['avg_words_per_chunk']:.0f}")
    
    print(f"\n[3] Chunks by company:")
    for ticker, count in sorted(summary['chunks_by_ticker'].items()):
        print(f"  {ticker}: {count} chunks")
    
    print(f"\n[4] Chunks by section:")
    for section, count in sorted(summary['chunks_by_section'].items()):
        print(f"  {section}: {count} chunks")
    
    print(f"\n[5] Sample chunk:")
    print("-" * 80)
    sample = documents[0]
    print(f"Ticker: {sample['ticker']}")
    print(f"Section: {sample['section']}")
    print(f"Chunk ID: {sample['chunk_id']}")
    print(f"Words: {sample['word_count']}")
    print(f"\nText preview:")
    print(sample['text'][:300] + "...")
    
    print("\n" + "=" * 80)
    print("Next: Generate embeddings for these chunks")
    print("=" * 80)