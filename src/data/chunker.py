"""
Document Chunker for RAG
-------------------------
Splits 10-K sections into semantic chunks for vector indexing.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Semantic document chunking for RAG."""

    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        min_chunk_size: int = None,
    ):
        self.chunk_size = chunk_size or Settings.CHUNK_SIZE
        self.overlap = overlap or Settings.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or Settings.CHUNK_MIN_SIZE

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _split_into_paragraphs(text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into semantic units preserving paragraph boundaries."""
        paragraphs = self._split_into_paragraphs(text)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_wc = 0

        for para in paragraphs:
            para_wc = self._word_count(para)

            if current_wc + para_wc > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))

                # Overlap: keep tail of previous chunk
                overlap_text = current_chunk[-1]
                overlap_wc = self._word_count(overlap_text)

                if overlap_wc <= self.overlap:
                    current_chunk = [overlap_text]
                    current_wc = overlap_wc
                else:
                    words = overlap_text.split()
                    current_chunk = [" ".join(words[-self.overlap :])]
                    current_wc = self.overlap

            current_chunk.append(para)
            current_wc += para_wc

        if current_chunk:
            final = "\n\n".join(current_chunk)
            if self._word_count(final) >= self.min_chunk_size:
                chunks.append(final)

        return chunks

    def chunk_file(self, filepath: Path) -> List[Dict]:
        """Chunk a section file into documents with metadata."""
        parts = filepath.stem.split("_")
        ticker = parts[0] if parts else "UNKNOWN"

        # Find filing_date (YYYY-MM-DD) and section from filename parts
        filing_date = "UNKNOWN"
        section = "UNKNOWN"
        for i, part in enumerate(parts[1:], start=1):
            if len(part) >= 8 and "-" in part:
                filing_date = part
            elif part.startswith("item"):
                # Remaining parts form the section key: item_1a, item_7, etc.
                section = "_".join(parts[i:])
                break

        text = filepath.read_text(encoding="utf-8")
        chunks = self.chunk_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "text": chunk,
                    "ticker": ticker,
                    "section": section,
                    "filing_date": filing_date,
                    "chunk_id": i,
                    "word_count": self._word_count(chunk),
                    "file_path": str(filepath),
                }
            )

        if documents:
            avg_wc = sum(d["word_count"] for d in documents) / len(documents)
            logger.info(f"Chunked {filepath.name}: {len(chunks)} chunks, avg {avg_wc:.0f} words")

        return documents

    def chunk_directory(self, directory: Path) -> Tuple[List[Dict], Dict]:
        """Chunk all section files in a directory."""
        section_files = sorted(directory.glob("*_item_*.txt"))
        logger.info(f"Chunking {len(section_files)} files from {directory}")

        all_documents: List[Dict] = []
        for filepath in section_files:
            try:
                all_documents.extend(self.chunk_file(filepath))
            except Exception as e:
                logger.error(f"Failed to chunk {filepath.name}: {e}")

        summary = {
            "total_files": len(section_files),
            "total_chunks": len(all_documents),
            "avg_words_per_chunk": (
                sum(d["word_count"] for d in all_documents) / len(all_documents) if all_documents else 0
            ),
            "chunks_by_ticker": {},
            "chunks_by_section": {},
        }
        for doc in all_documents:
            summary["chunks_by_ticker"][doc["ticker"]] = summary["chunks_by_ticker"].get(doc["ticker"], 0) + 1
            summary["chunks_by_section"][doc["section"]] = summary["chunks_by_section"].get(doc["section"], 0) + 1

        logger.info(f"Chunking complete: {summary['total_chunks']} total chunks")
        return all_documents, summary
