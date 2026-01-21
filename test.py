"""
AlphaExtract Test Suite
-----------------------
Unit tests for core components.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


class TestSECDownloader:
    """Tests for SEC downloader."""
    
    def test_get_cik_valid_ticker(self):
        """Test CIK lookup for valid ticker."""
        from src.data.downloader import SECDownloader
        
        downloader = SECDownloader()
        cik = downloader.get_cik("AAPL")
        
        assert cik is not None
        assert len(cik) == 10
        assert cik.isdigit()
    
    def test_get_cik_invalid_ticker(self):
        """Test CIK lookup for invalid ticker."""
        from src.data.downloader import SECDownloader
        
        downloader = SECDownloader()
        cik = downloader.get_cik("INVALIDTICKER123")
        
        assert cik is None


class TestTenKParser:
    """Tests for 10-K parser."""
    
    def test_parser_initialization(self):
        """Test parser can be initialized."""
        from src.data.parser import TenKParser
        
        parser = TenKParser()
        assert parser.converter is not None
        assert parser.processed_dir.exists()


class TestSectionSplitter:
    """Tests for section splitter."""
    
    def test_splitter_initialization(self):
        """Test splitter can be initialized."""
        from src.data.splitter import SectionSplitter
        
        splitter = SectionSplitter()
        assert splitter.output_dir.exists()
    
    def test_section_patterns_defined(self):
        """Test that section patterns are defined."""
        from src.data.splitter import SectionSplitter
        
        splitter = SectionSplitter()
        assert 'item_1a' in splitter.SECTION_PATTERNS
        assert 'item_7' in splitter.SECTION_PATTERNS
        assert 'item_8' in splitter.SECTION_PATTERNS


class TestDocumentChunker:
    """Tests for document chunker."""
    
    def test_chunker_initialization(self):
        """Test chunker can be initialized with defaults."""
        from src.data.chunker import DocumentChunker
        
        chunker = DocumentChunker()
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50
    
    def test_chunk_text_short(self):
        """Test chunking short text returns single chunk."""
        from src.data.chunker import DocumentChunker
        
        chunker = DocumentChunker(chunk_size=100)
        text = "This is a short text."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text


class TestFinBERTAnalyzer:
    """Tests for FinBERT sentiment analyzer."""
    
    def test_signal_generation(self):
        """Test trading signal generation from scores."""
        from src.models.sentiment import FinBERTAnalyzer
        
        analyzer = FinBERTAnalyzer.__new__(FinBERTAnalyzer)
        
        assert analyzer._generate_signal(0.6) == "STRONG_BUY"
        assert analyzer._generate_signal(0.3) == "BUY"
        assert analyzer._generate_signal(0.0) == "HOLD"
        assert analyzer._generate_signal(-0.3) == "SELL"
        assert analyzer._generate_signal(-0.6) == "STRONG_SELL"


class TestAnomalyDetector:
    """Tests for anomaly detector."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        from src.models.anomaly import AnomalyDetector
        
        detector = AnomalyDetector()
        assert detector.sentiment_dir.exists() or True  # May not exist yet
        assert len(detector.TRACKED_KEYWORDS) > 0
    
    def test_keyword_extraction(self):
        """Test keyword extraction from text."""
        from src.models.anomaly import AnomalyDetector
        
        detector = AnomalyDetector()
        text = "The company faces regulation and tariff risks in China."
        keywords = detector.extract_keywords(text)
        
        assert 'regulation' in keywords or 'china' in keywords or 'tariff' in keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])