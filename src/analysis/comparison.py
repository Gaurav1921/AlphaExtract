"""
Multi-Quarter Comparison Engine
--------------------------------
Compares 10-K filing data across multiple years for the same ticker.
Tracks sentiment evolution, keyword shifts, and signal changes over time.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config.settings import Settings

logger = logging.getLogger(__name__)


def load_all_filings(ticker: str) -> List[Dict]:
    """
    Load all available filing data (sentiment + ensemble) for a ticker,
    sorted chronologically.

    Returns list of dicts with keys:
        filing_date, year, sentiment, ensemble (if available)
    """
    ticker = ticker.upper()
    sentiment_dir = Settings.SENTIMENT_DIR

    sentiment_files = sorted(sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
    filings = []

    for sf in sentiment_files:
        parts = sf.stem.split("_")
        if len(parts) < 2:
            continue
        filing_date = parts[1]
        year = filing_date[:4]

        try:
            sentiment_data = json.loads(sf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {sf.name}: {e}")
            continue

        # Try to load ensemble data
        ensemble_file = sentiment_dir / f"{ticker}_{filing_date}_ensemble.json"
        ensemble_data = None
        if ensemble_file.exists():
            try:
                ensemble_data = json.loads(ensemble_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        filings.append({
            "filing_date": filing_date,
            "year": year,
            "sentiment": sentiment_data,
            "ensemble": ensemble_data,
        })

    return filings


def compare_quarters(ticker: str) -> Dict:
    """
    Generate a full multi-quarter comparison for a ticker.

    Returns:
        Dict with yearly_data, trends, deltas, and summary.
    """
    filings = load_all_filings(ticker)

    if len(filings) < 2:
        return {
            "ticker": ticker,
            "error": f"Need 2+ filings for comparison, found {len(filings)}",
            "filings_available": len(filings),
        }

    yearly_data = []
    for f in filings:
        sentiment = f["sentiment"]
        ensemble = f["ensemble"]
        sections = sentiment.get("sections", {})
        overall = sentiment.get("overall", {})

        entry = {
            "filing_date": f["filing_date"],
            "year": f["year"],
            "overall_score": overall.get("compound", 0),
            "overall_signal": overall.get("signal", "N/A"),
            "sections": {},
        }

        # Per-section scores
        for section_key in ["item_1a", "item_7", "item_8"]:
            section = sections.get(section_key, {})
            scores = section.get("scores", {})
            entry["sections"][section_key] = {
                "compound": scores.get("compound", 0),
                "positive": scores.get("positive", 0),
                "negative": scores.get("negative", 0),
                "neutral": scores.get("neutral", 0),
                "signal": section.get("signal", "N/A"),
                "word_count": section.get("word_count", 0),
            }

        # Ensemble data if available
        if ensemble:
            ens = ensemble.get("ensemble", {})
            signals = ensemble.get("signals", {})
            entry["ensemble"] = {
                "score": ens.get("score", 0),
                "signal": ens.get("signal", "N/A"),
                "components_agree": ens.get("components_agree", False),
                "finbert_score": signals.get("finbert", {}).get("score", 0),
                "keyword_score": signals.get("keywords", {}).get("score", 0),
                "llm_score": signals.get("llm", {}).get("score", 0),
            }

        yearly_data.append(entry)

    # Compute year-over-year deltas
    deltas = []
    for i in range(1, len(yearly_data)):
        curr = yearly_data[i]
        prev = yearly_data[i - 1]

        delta = {
            "from_year": prev["year"],
            "to_year": curr["year"],
            "from_date": prev["filing_date"],
            "to_date": curr["filing_date"],
            "overall_delta": round(curr["overall_score"] - prev["overall_score"], 4),
            "signal_changed": curr["overall_signal"] != prev["overall_signal"],
            "section_deltas": {},
        }

        for section_key in ["item_1a", "item_7", "item_8"]:
            curr_sec = curr["sections"].get(section_key, {})
            prev_sec = prev["sections"].get(section_key, {})
            delta["section_deltas"][section_key] = {
                "compound_delta": round(
                    curr_sec.get("compound", 0) - prev_sec.get("compound", 0), 4
                ),
                "word_count_delta": curr_sec.get("word_count", 0) - prev_sec.get("word_count", 0),
            }

        # Ensemble delta if both have it
        if curr.get("ensemble") and prev.get("ensemble"):
            delta["ensemble_delta"] = round(
                curr["ensemble"]["score"] - prev["ensemble"]["score"], 4
            )

        deltas.append(delta)

    # Compute overall trend
    scores = [d["overall_score"] for d in yearly_data]
    if len(scores) >= 2:
        first_half = scores[: len(scores) // 2]
        second_half = scores[len(scores) // 2 :]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_second - avg_first > 0.05:
            trend = "improving"
        elif avg_first - avg_second > 0.05:
            trend = "deteriorating"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    # Identify biggest shifts
    biggest_improvement = max(deltas, key=lambda d: d["overall_delta"]) if deltas else None
    biggest_decline = min(deltas, key=lambda d: d["overall_delta"]) if deltas else None

    return {
        "ticker": ticker,
        "filings_available": len(filings),
        "date_range": f"{filings[0]['filing_date']} to {filings[-1]['filing_date']}",
        "trend": trend,
        "yearly_data": yearly_data,
        "deltas": deltas,
        "summary": {
            "avg_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "score_range": round(max(scores) - min(scores), 4),
            "biggest_improvement": {
                "period": f"{biggest_improvement['from_year']} -> {biggest_improvement['to_year']}",
                "delta": biggest_improvement["overall_delta"],
            } if biggest_improvement else None,
            "biggest_decline": {
                "period": f"{biggest_decline['from_year']} -> {biggest_decline['to_year']}",
                "delta": biggest_decline["overall_delta"],
            } if biggest_decline else None,
        },
    }


def compare_tickers(tickers: List[str], year: str = None) -> Dict:
    """
    Compare multiple tickers side-by-side for the same year or latest data.

    Returns:
        Dict with per-ticker comparison data.
    """
    results = {}

    for ticker in tickers:
        ticker = ticker.upper()
        filings = load_all_filings(ticker)

        if not filings:
            results[ticker] = {"error": "No data available"}
            continue

        # Filter by year or take latest
        if year:
            matching = [f for f in filings if f["year"] == year]
            filing = matching[-1] if matching else filings[-1]
        else:
            filing = filings[-1]

        sentiment = filing["sentiment"]
        ensemble = filing.get("ensemble")
        overall = sentiment.get("overall", {})

        entry = {
            "filing_date": filing["filing_date"],
            "year": filing["year"],
            "overall_score": overall.get("compound", 0),
            "overall_signal": overall.get("signal", "N/A"),
        }

        # Section scores
        sections = sentiment.get("sections", {})
        for section_key in ["item_1a", "item_7", "item_8"]:
            section = sections.get(section_key, {})
            entry[section_key] = section.get("scores", {}).get("compound", 0)

        # Ensemble
        if ensemble:
            entry["ensemble_score"] = ensemble.get("ensemble", {}).get("score", 0)
            entry["ensemble_signal"] = ensemble.get("ensemble", {}).get("signal", "N/A")

        results[ticker] = entry

    return {
        "year_filter": year or "latest",
        "tickers": results,
    }


def section_deep_dive(ticker: str, section_key: str = "item_7") -> Dict:
    """
    Deep dive into a single section across all available years.

    Useful for tracking how MD&A, Risk Factors, or Financials sentiment
    evolves over time.
    """
    filings = load_all_filings(ticker)

    section_data = []
    for f in filings:
        sections = f["sentiment"].get("sections", {})
        section = sections.get(section_key, {})
        scores = section.get("scores", {})

        section_data.append({
            "filing_date": f["filing_date"],
            "year": f["year"],
            "compound": scores.get("compound", 0),
            "positive": scores.get("positive", 0),
            "negative": scores.get("negative", 0),
            "neutral": scores.get("neutral", 0),
            "signal": section.get("signal", "N/A"),
            "word_count": section.get("word_count", 0),
        })

    # Word count trend (indicator of disclosure expansion)
    word_counts = [d["word_count"] for d in section_data if d["word_count"] > 0]
    if len(word_counts) >= 2:
        wc_change = (word_counts[-1] - word_counts[0]) / word_counts[0] if word_counts[0] > 0 else 0
    else:
        wc_change = 0

    return {
        "ticker": ticker,
        "section": section_key,
        "data": section_data,
        "word_count_change_pct": round(wc_change * 100, 1),
    }
