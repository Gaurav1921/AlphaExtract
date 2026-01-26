"""
FinBERT Sentiment Analyzer - UPDATED
-------------------------------------
LOCATION: src/models/sentiment.py

Analyzes sentiment of extracted 10-K sections using FinBERT,
a BERT model fine-tuned on financial text.

CHANGES in v0.5.1:
- Consistent key format: item_1a, item_7, item_8 (not "1a", "7", "8")
- Better error handling
- Cleaner output format

Key features:
- Handles long documents (chunking for BERT's 512 token limit)
- Returns sentiment scores: negative, neutral, positive
- Aggregates scores across chunks
- Generates trading signals based on sentiment
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinBERTAnalyzer:
    """
    Sentiment analysis using FinBERT for financial documents.
    
    Model: ProsusAI/finbert
    Output: [negative, neutral, positive] probabilities
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """Initialize FinBERT model and tokenizer."""
        logger.info(f"Loading FinBERT model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on device: {self.device}")
        
        self.output_dir = Path("data/sentiment")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def chunk_text(self, text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """Split text into chunks that fit BERT's token limit."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_length:
            return [text]
        
        chunks = []
        stride = max_length - overlap
        
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + max_length]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            if i + max_length >= len(tokens):
                break
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def analyze_chunk(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text chunk."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)[0]
        
        return {
            'negative': probs[0].item(),
            'neutral': probs[1].item(),
            'positive': probs[2].item()
        }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of potentially long text."""
        if not text or len(text.strip()) < 50:
            return {
                'negative': 0.33,
                'neutral': 0.34,
                'positive': 0.33,
                'compound': 0.0,
                'num_chunks': 0
            }
        
        chunks = self.chunk_text(text)
        chunk_scores = []
        
        for chunk in chunks:
            scores = self.analyze_chunk(chunk)
            chunk_scores.append(scores)
        
        # Aggregate scores (weighted by chunk length)
        chunk_lengths = [len(chunk.split()) for chunk in chunks]
        total_length = sum(chunk_lengths)
        
        if total_length == 0:
            return {
                'negative': 0.33,
                'neutral': 0.34,
                'positive': 0.33,
                'compound': 0.0,
                'num_chunks': 0
            }
        
        agg_negative = sum(
            s['negative'] * l / total_length 
            for s, l in zip(chunk_scores, chunk_lengths)
        )
        agg_neutral = sum(
            s['neutral'] * l / total_length 
            for s, l in zip(chunk_scores, chunk_lengths)
        )
        agg_positive = sum(
            s['positive'] * l / total_length 
            for s, l in zip(chunk_scores, chunk_lengths)
        )
        
        compound = agg_positive - agg_negative
        
        return {
            'negative': float(agg_negative),
            'neutral': float(agg_neutral),
            'positive': float(agg_positive),
            'compound': float(compound),
            'num_chunks': len(chunks)
        }
    
    def _generate_signal(self, compound: float) -> str:
        """Generate trading signal from compound score."""
        if compound > 0.5:
            return "STRONG_BUY"
        elif compound > 0.2:
            return "BUY"
        elif compound > -0.2:
            return "HOLD"
        elif compound > -0.5:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _determine_label(self, scores: Dict) -> str:
        """Determine sentiment label from scores."""
        if scores['positive'] > scores['negative'] and scores['positive'] > scores['neutral']:
            return 'positive'
        elif scores['negative'] > scores['positive'] and scores['negative'] > scores['neutral']:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_section_file(self, filepath: Path) -> Dict:
        """
        Analyze sentiment of a section file.
        
        Args:
            filepath: Path to section text file
            
        Returns:
            Dict with scores, signal, and metadata
        """
        logger.info(f"Analyzing: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        scores = self.analyze_text(text)
        label = self._determine_label(scores)
        signal = self._generate_signal(scores['compound'])
        
        result = {
            'file': filepath.name,
            'word_count': len(text.split()),
            'scores': scores,
            'label': label,
            'signal': signal
        }
        
        logger.info(f"  {label.upper()} ({scores['compound']:.3f}) → {signal}")
        return result
    
    def _normalize_section_key(self, filename: str) -> Optional[str]:
        """
        Extract and normalize section key from filename.
        
        Examples:
            'AAPL_2025-10-31_item_1a.txt' → 'item_1a'
            'AAPL_2025-10-31_item_7.txt' → 'item_7'
            'AAPL_2024-11-01_1a.txt' → 'item_1a'  (legacy format)
        
        Returns:
            Normalized key like 'item_1a', 'item_7', 'item_8' or None
        """
        stem = filename.replace('.txt', '')
        parts = stem.split('_')
        
        # Get the last part (section identifier)
        if len(parts) < 3:
            return None
        
        section_part = parts[-1].lower()
        
        # Already normalized
        if section_part.startswith('item_'):
            return section_part
        
        # Handle 'item' as second-to-last part: ticker_date_item_1a
        if len(parts) >= 4 and parts[-2].lower() == 'item':
            return f"item_{section_part}"
        
        # Legacy format: ticker_date_1a → item_1a
        key_map = {
            '1a': 'item_1a',
            '1b': 'item_1b',
            '1': 'item_1',
            '7': 'item_7',
            '7a': 'item_7a',
            '8': 'item_8',
            '9': 'item_9',
        }
        
        return key_map.get(section_part, f"item_{section_part}")
    
    def analyze_company(self, ticker: str, filing_date: str = None) -> Dict:
        """
        Analyze all sections for a company.
        
        Args:
            ticker: Stock ticker
            filing_date: Optional specific filing date (YYYY-MM-DD)
            
        Returns:
            Dict with section and overall sentiment
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING: {ticker}")
        logger.info(f"{'='*80}")
        
        sections_dir = Path("data/sections")
        
        # Find section files
        if filing_date:
            pattern = f"{ticker}_{filing_date}_*.txt"
        else:
            pattern = f"{ticker}_*_item_*.txt"
        
        section_files = list(sections_dir.glob(pattern))
        
        # Also check for legacy format without 'item_' prefix
        if not section_files:
            pattern = f"{ticker}_*.txt"
            all_files = list(sections_dir.glob(pattern))
            # Filter to only section files (not metadata)
            section_files = [f for f in all_files if any(
                s in f.name.lower() for s in ['1a', '1b', 'item_7', 'item_8', '_7.', '_8.']
            )]
        
        if not section_files:
            logger.warning(f"No section files found for {ticker}")
            return {
                'ticker': ticker,
                'error': 'No section files found',
                'sections': {},
                'overall': {'compound': 0, 'signal': 'N/A'}
            }
        
        # Determine filing date from files if not provided
        if not filing_date:
            # Extract date from first file: TICKER_YYYY-MM-DD_section.txt
            sample_file = section_files[0]
            parts = sample_file.stem.split('_')
            if len(parts) >= 2:
                filing_date = parts[1]
        
        results = {
            'ticker': ticker,
            'filing_date': filing_date,
            'analysis_date': datetime.now().isoformat(),
            'sections': {},
            'overall': {}
        }
        
        # Analyze each section
        section_scores = []
        
        for filepath in section_files:
            # Get normalized section key (item_1a, item_7, item_8)
            section_key = self._normalize_section_key(filepath.name)
            
            if not section_key:
                logger.warning(f"Could not determine section for: {filepath.name}")
                continue
            
            # Skip if not a main section we care about
            if section_key not in ['item_1a', 'item_7', 'item_8']:
                continue
            
            section_result = self.analyze_section_file(filepath)
            
            # Store with CONSISTENT key format: item_1a, item_7, item_8
            results['sections'][section_key] = section_result
            section_scores.append({
                'key': section_key,
                'compound': section_result['scores']['compound'],
                'word_count': section_result['word_count']
            })
        
        # Calculate overall sentiment (weighted by section importance)
        if section_scores:
            # Weights: MD&A most important, then Risk Factors, then Financials
            weights = {
                'item_1a': 0.25,  # Risk Factors
                'item_7': 0.60,   # MD&A - most important
                'item_8': 0.15    # Financial Statements
            }
            
            weighted_compound = 0
            total_weight = 0
            
            for score in section_scores:
                key = score['key']
                weight = weights.get(key, 0.1)
                weighted_compound += score['compound'] * weight
                total_weight += weight
            
            overall_compound = weighted_compound / total_weight if total_weight > 0 else 0
            overall_signal = self._generate_signal(overall_compound)
            
            results['overall'] = {
                'compound': overall_compound,
                'signal': overall_signal
            }
            
            logger.info(f"\n{'='*80}")
            logger.info(f"OVERALL: {overall_signal} ({overall_compound:.3f})")
            logger.info(f"{'='*80}")
        else:
            results['overall'] = {
                'compound': 0,
                'signal': 'N/A'
            }
        
        return results
    
    def save_results(self, results: Dict, ticker: str = None, filing_date: str = None):
        """Save analysis results to JSON."""
        ticker = ticker or results.get('ticker', 'UNKNOWN')
        filing_date = filing_date or results.get('filing_date', 'unknown')
        
        output_path = self.output_dir / f"{ticker}_{filing_date}_sentiment.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved: {output_path}")
        return output_path
    
    def batch_analyze(self, tickers: List[str] = None) -> Dict:
        """Analyze sentiment for multiple companies."""
        sections_dir = Path("data/sections")
        
        if tickers is None:
            # Find all unique tickers from section files
            section_files = list(sections_dir.glob("*_item_*.txt"))
            tickers = list(set(f.stem.split('_')[0] for f in section_files))
        
        summary = {'total': len(tickers), 'results': {}}
        
        for ticker in sorted(tickers):
            try:
                results = self.analyze_company(ticker)
                self.save_results(results, ticker, results.get('filing_date'))
                
                if 'overall' in results and results['overall'].get('signal') != 'N/A':
                    summary['results'][ticker] = {
                        'signal': results['overall']['signal'],
                        'compound': results['overall']['compound']
                    }
            except Exception as e:
                logger.error(f"Failed {ticker}: {e}")
                summary['results'][ticker] = {'error': str(e)}
        
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY")
        logger.info(f"{'='*80}")
        
        for ticker, result in summary['results'].items():
            if 'signal' in result:
                logger.info(f"{ticker:6s}: {result['signal']:12s} ({result['compound']:+.3f})")
        
        return summary


# Convenience alias
SentimentAnalyzer = FinBERTAnalyzer


if __name__ == "__main__":
    print("=" * 80)
    print("FINBERT SENTIMENT ANALYZER")
    print("=" * 80)
    
    analyzer = FinBERTAnalyzer()
    
    print("\n[TEST] Analyzing Apple...")
    results = analyzer.analyze_company("AAPL")
    analyzer.save_results(results)
    
    print("\n[TEST] Results structure:")
    print(f"  Ticker: {results.get('ticker')}")
    print(f"  Sections: {list(results.get('sections', {}).keys())}")
    print(f"  Overall: {results.get('overall')}")
    
    response = input("\nAnalyze all companies? (y/n): ")
    if response.lower() == 'y':
        analyzer.batch_analyze()