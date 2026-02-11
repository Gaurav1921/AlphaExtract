"""
FinBERT Sentiment Analyzer
---------------------------
Analyzes 10-K sections using FinBERT for financial sentiment scoring.
Generates per-section and weighted-overall trading signals.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Financial sentiment analysis using FinBERT."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Settings.SENTIMENT_MODEL
        self.max_length = Settings.SENTIMENT_MAX_TOKENS
        self.max_chunks = Settings.SENTIMENT_MAX_CHUNKS
        self.output_dir = Settings.SENTIMENT_DIR
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    def chunk_text(self, text: str) -> List[str]:
        """Split text into token-aware chunks that fit FinBERT's input window."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        stride = self.max_length - Settings.SENTIMENT_TOKEN_OVERLAP - 2  # -2 for [CLS]/[SEP]
        stride = max(stride, 1)

        chunks = []
        for start in range(0, len(tokens), stride):
            chunk_tokens = tokens[start : start + self.max_length - 2]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
            if len(chunks) >= self.max_chunks:
                break

        return chunks

    @torch.no_grad()
    def analyze_text(self, text: str) -> Dict:
        """Analyze a text block and return sentiment scores."""
        if not text or len(text.strip()) < 50:
            return {"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3, "compound": 0.0}

        chunks = self.chunk_text(text)
        if not chunks:
            return {"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3, "compound": 0.0}

        chunk_scores = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            neg, neu, pos = probs.tolist()

            # Weight by token count for proportional averaging
            token_count = len(self.tokenizer.encode(chunk, add_special_tokens=False))
            chunk_scores.append({
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "compound": pos - neg,
                "weight": token_count,
            })

        # Weighted average by token count
        total_weight = sum(s["weight"] for s in chunk_scores)
        avg = {
            "positive": sum(s["positive"] * s["weight"] for s in chunk_scores) / total_weight,
            "negative": sum(s["negative"] * s["weight"] for s in chunk_scores) / total_weight,
            "neutral": sum(s["neutral"] * s["weight"] for s in chunk_scores) / total_weight,
            "compound": sum(s["compound"] * s["weight"] for s in chunk_scores) / total_weight,
            "chunks_analyzed": len(chunk_scores),
        }

        return avg

    def analyze_section_file(self, filepath: Path) -> Dict:
        """Analyze a section file and return results with signal."""
        text = filepath.read_text(encoding="utf-8")
        scores = self.analyze_text(text)
        signal = Settings.generate_signal(scores["compound"])

        return {
            "scores": scores,
            "signal": signal,
            "word_count": len(text.split()),
        }

    def analyze_filing(self, ticker: str, filing_date: str) -> Optional[Dict]:
        """Analyze all sections for a single filing."""
        ticker = ticker.upper()
        sections_dir = Settings.SECTIONS_DIR

        section_files = sorted(sections_dir.glob(f"{ticker}_{filing_date}_item_*.txt"))
        if not section_files:
            logger.warning(f"No section files found for {ticker} {filing_date}")
            return None

        results = {"sections": {}, "overall": {}, "filing_date": filing_date, "ticker": ticker}

        section_scores = []
        for section_file in section_files:
            # Extract section key from filename (e.g., item_1a from AAPL_2024-01-01_item_1a.txt)
            stem_parts = section_file.stem.split("_")
            section_key_parts = []
            capture = False
            for part in stem_parts:
                if part == "item":
                    capture = True
                if capture:
                    section_key_parts.append(part)
            section_key = "_".join(section_key_parts) if section_key_parts else section_file.stem

            logger.info(f"Analyzing {ticker} {filing_date} {section_key}")
            section_result = self.analyze_section_file(section_file)
            results["sections"][section_key] = section_result

            weight = Settings.SECTION_WEIGHTS.get(section_key, 1.0 / 3)
            section_scores.append((section_result["scores"]["compound"], weight))

        # Weighted overall score
        if section_scores:
            total_weight = sum(w for _, w in section_scores)
            overall_compound = sum(score * w for score, w in section_scores) / total_weight
        else:
            overall_compound = 0.0

        results["overall"] = {
            "compound": overall_compound,
            "signal": Settings.generate_signal(overall_compound),
        }

        return results

    def analyze_and_save(self, ticker: str, filing_date: str) -> Optional[Dict]:
        """Analyze a filing and save results to disk."""
        results = self.analyze_filing(ticker, filing_date)
        if results is None:
            return None

        output_path = self.output_dir / f"{ticker}_{filing_date}_sentiment.json"
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(
            f"Saved sentiment for {ticker} {filing_date}: "
            f"{results['overall']['signal']} ({results['overall']['compound']:+.3f})"
        )
        return results

    def batch_analyze(self, ticker: str = None) -> Dict:
        """Analyze all unprocessed filings (optionally filtered by ticker)."""
        sections_dir = Settings.SECTIONS_DIR
        pattern = f"{ticker.upper()}_*_item_*.txt" if ticker else "*_item_*.txt"
        section_files = sorted(sections_dir.glob(pattern))

        # Collect unique (ticker, filing_date) pairs
        filing_set = set()
        for f in section_files:
            parts = f.stem.split("_")
            if len(parts) >= 2:
                filing_set.add((parts[0], parts[1]))

        # Filter out already-analyzed
        to_analyze = []
        for t, d in sorted(filing_set):
            sentiment_file = self.output_dir / f"{t}_{d}_sentiment.json"
            if not sentiment_file.exists():
                to_analyze.append((t, d))

        logger.info(f"Batch analyze: {len(to_analyze)} filings to process (of {len(filing_set)} total)")

        results = {"analyzed": [], "failed": []}
        for t, d in to_analyze:
            try:
                result = self.analyze_and_save(t, d)
                if result:
                    results["analyzed"].append(f"{t}_{d}")
                else:
                    results["failed"].append(f"{t}_{d}")
            except Exception as e:
                logger.error(f"Failed to analyze {t} {d}: {e}")
                results["failed"].append(f"{t}_{d}")

        return results
