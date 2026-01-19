"""
FinBERT Sentiment Analyzer
--------------------------
Analyzes sentiment of extracted 10-K sections using FinBERT,
a BERT model fine-tuned on financial text.

Key features:
- Handles long documents (chunking for BERT's 512 token limit)
- Returns sentiment scores: negative, neutral, positive
- Aggregates scores across chunks
- Generates trading signals based on sentiment

Why FinBERT over generic BERT:
- Trained on financial news and analyst reports
- Understands domain-specific language ("headwinds", "tailwinds")
- Better at detecting financial sentiment nuances
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List
import json
import logging

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
                'compound': 0.0
            }
        
        chunks = self.chunk_text(text)
        chunk_scores = []
        
        for chunk in chunks:
            scores = self.analyze_chunk(chunk)
            chunk_scores.append(scores)
        
        # Aggregate scores
        chunk_lengths = [len(chunk.split()) for chunk in chunks]
        total_length = sum(chunk_lengths)
        
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
    
    def analyze_section_file(self, filepath: Path) -> Dict:
        """Analyze sentiment of a section file."""
        logger.info(f"Analyzing: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        scores = self.analyze_text(text)
        
        if scores['positive'] > scores['negative'] and scores['positive'] > scores['neutral']:
            label = 'positive'
        elif scores['negative'] > scores['positive'] and scores['negative'] > scores['neutral']:
            label = 'negative'
        else:
            label = 'neutral'
        
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
    
    def analyze_company(self, ticker: str) -> Dict:
        """Analyze all sections for a company."""
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING: {ticker}")
        logger.info(f"{'='*80}")
        
        sections_dir = Path("data/sections")
        section_files = {
            'item_1a': list(sections_dir.glob(f"{ticker}_*_item_1a.txt")),
            'item_7': list(sections_dir.glob(f"{ticker}_*_item_7.txt")),
            'item_8': list(sections_dir.glob(f"{ticker}_*_item_8.txt"))
        }
        
        results = {
            'ticker': ticker,
            'sections': {},
            'overall': {}
        }
        
        for section_name, files in section_files.items():
            if not files:
                continue
            
            filepath = files[0]
            section_result = self.analyze_section_file(filepath)
            results['sections'][section_name] = section_result
        
        # Calculate overall sentiment (weighted)
        if results['sections']:
            weights = {
                'item_1a': 0.25,
                'item_7': 0.60,
                'item_8': 0.15
            }
            
            weighted_compound = 0
            total_weight = 0
            
            for section, weight in weights.items():
                if section in results['sections']:
                    compound = results['sections'][section]['scores']['compound']
                    weighted_compound += compound * weight
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
        
        return results
    
    def save_results(self, results: Dict, ticker: str):
        """Save analysis results to JSON."""
        sections_dir = Path("data/sections")
        json_files = list(sections_dir.glob(f"{ticker}_*_sections.json"))
        
        if json_files:
            with open(json_files[0], 'r') as f:
                metadata = json.load(f)
                filing_date = metadata.get('filing_date', 'unknown')
        else:
            filing_date = 'unknown'
        
        output_path = self.output_dir / f"{ticker}_{filing_date}_sentiment.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved: {output_path}")
    
    def batch_analyze(self, tickers: List[str] = None) -> Dict:
        """Analyze sentiment for multiple companies."""
        sections_dir = Path("data/sections")
        
        if tickers is None:
            section_files = list(sections_dir.glob("*_item_*.txt"))
            tickers = list(set(f.stem.split('_')[0] for f in section_files))
        
        summary = {'total': len(tickers), 'results': {}}
        
        for ticker in sorted(tickers):
            try:
                results = self.analyze_company(ticker)
                self.save_results(results, ticker)
                
                if 'overall' in results:
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


if __name__ == "__main__":
    print("=" * 80)
    print("FINBERT SENTIMENT ANALYZER")
    print("=" * 80)
    
    analyzer = FinBERTAnalyzer()
    
    print("\n[TEST] Analyzing Apple...")
    results = analyzer.analyze_company("AAPL")
    analyzer.save_results(results, "AAPL")
    
    response = input("\nAnalyze all companies? (y/n): ")
    if response.lower() == 'y':
        analyzer.batch_analyze()