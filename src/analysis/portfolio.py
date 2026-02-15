"""
Portfolio Signal Aggregation
-----------------------------
Combines sentiment and ensemble signals across multiple holdings into
portfolio-level metrics. Supports position-weighted and equal-weighted modes.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class HoldingSignal:
    """Signal data for a single portfolio holding."""
    ticker: str
    weight: float
    sentiment_score: Optional[float] = None
    sentiment_signal: Optional[str] = None
    ensemble_score: Optional[float] = None
    ensemble_signal: Optional[str] = None
    filing_date: Optional[str] = None
    sector: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PortfolioSignal:
    """Aggregated signal for an entire portfolio."""
    name: str
    holdings_count: int
    holdings_with_data: int
    coverage_pct: float
    weighted_sentiment: float
    weighted_ensemble: Optional[float]
    portfolio_signal: str
    sector_breakdown: Dict[str, Dict]
    signal_distribution: Dict[str, int]
    risk_concentration: Dict
    holdings: List[Dict]

    def to_dict(self) -> Dict:
        return asdict(self)


def _get_ticker_sector(ticker: str) -> Optional[str]:
    """Look up which sector a ticker belongs to."""
    for sector, tickers in Settings.SECTOR_TICKERS.items():
        if ticker in tickers:
            return sector
    return None


def _load_latest_json(ticker: str, suffix: str) -> Optional[Dict]:
    """
    Load the most recent JSON file matching {TICKER}_*_{suffix}.json.
    Shared loader for sentiment, ensemble, and other signal files.
    """
    pattern = f"{ticker.upper()}_*_{suffix}.json"
    files = sorted(Settings.SENTIMENT_DIR.glob(pattern))
    if not files:
        return None

    try:
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        # Extract filing date from filename (TICKER_DATE_type.json)
        parts = files[-1].stem.split("_")
        if len(parts) >= 2:
            data["_filing_date"] = parts[1]
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load {suffix} for {ticker}: {e}")
        return None


def _weighted_average(items: list, score_attr: str) -> Optional[float]:
    """Compute weighted average of a score attribute across holdings."""
    filtered = [h for h in items if getattr(h, score_attr) is not None]
    if not filtered:
        return None
    weight_sum = sum(h.weight for h in filtered)
    if weight_sum <= 0:
        return 0.0
    return sum(getattr(h, score_attr) * h.weight for h in filtered) / weight_sum


def _build_sector_breakdown(
    sector_scores: Dict[str, List[Tuple[float, float]]],
) -> Dict[str, Dict]:
    """Build sector-level aggregated scores."""
    breakdown = {}
    for sector, scores_weights in sector_scores.items():
        total_weight = sum(w for _, w in scores_weights)
        avg_score = (
            sum(s * w for s, w in scores_weights) / total_weight
            if total_weight > 0 else 0
        )
        breakdown[sector] = {
            "avg_score": round(avg_score, 4),
            "signal": Settings.generate_signal(avg_score),
            "weight": round(total_weight, 4),
            "count": len(scores_weights),
        }
    return breakdown


def _build_risk_concentration(
    holding_signals: List[HoldingSignal],
    sector_breakdown: Dict[str, Dict],
) -> Dict:
    """Compute risk concentration metrics."""
    max_holding = max(holding_signals, key=lambda h: h.weight) if holding_signals else None
    max_sector = (
        max(sector_breakdown.items(), key=lambda x: x[1]["weight"])
        if sector_breakdown else None
    )

    return {
        "max_single_holding": {
            "ticker": max_holding.ticker if max_holding else None,
            "weight": max_holding.weight if max_holding else 0,
        },
        "max_single_sector": {
            "sector": max_sector[0] if max_sector else None,
            "weight": max_sector[1]["weight"] if max_sector else 0,
        },
        "herfindahl_index": round(sum(h.weight ** 2 for h in holding_signals), 4),
    }


def aggregate_portfolio(
    holdings: Dict[str, float],
    name: str = "My Portfolio",
) -> PortfolioSignal:
    """
    Aggregate signals across a portfolio of holdings.

    Args:
        holdings: Dict mapping ticker -> weight (0-1). Weights are normalized.
        name: Portfolio name for display.

    Returns:
        PortfolioSignal with aggregated metrics.
    """
    # Normalize weights
    total_weight = sum(holdings.values()) or 1.0
    normalized = {t: w / total_weight for t, w in holdings.items()}

    holding_signals: List[HoldingSignal] = []
    sector_scores: Dict[str, List[Tuple[float, float]]] = {}

    for ticker, weight in normalized.items():
        ticker = ticker.upper()
        sector = _get_ticker_sector(ticker)

        sentiment = _load_latest_json(ticker, "sentiment")
        ensemble = _load_latest_json(ticker, "ensemble")

        hs = HoldingSignal(ticker=ticker, weight=round(weight, 4), sector=sector)

        if sentiment:
            overall = sentiment.get("overall", {})
            hs.sentiment_score = overall.get("compound")
            hs.sentiment_signal = overall.get("signal")
            hs.filing_date = sentiment.get("_filing_date")

        if ensemble:
            ens = ensemble.get("ensemble", {})
            hs.ensemble_score = ens.get("score")
            hs.ensemble_signal = ens.get("signal")

        holding_signals.append(hs)

        if sector and hs.sentiment_score is not None:
            sector_scores.setdefault(sector, []).append((hs.sentiment_score, weight))

    # Aggregated scores
    has_data = [h for h in holding_signals if h.sentiment_score is not None]
    coverage = len(has_data) / len(holding_signals) if holding_signals else 0

    weighted_sentiment = _weighted_average(holding_signals, "sentiment_score") or 0.0
    weighted_ensemble = _weighted_average(holding_signals, "ensemble_score")

    # Signal distribution
    signal_dist: Dict[str, int] = {}
    for h in has_data:
        sig = h.sentiment_signal or "N/A"
        signal_dist[sig] = signal_dist.get(sig, 0) + 1

    sector_breakdown = _build_sector_breakdown(sector_scores)
    risk_concentration = _build_risk_concentration(holding_signals, sector_breakdown)

    return PortfolioSignal(
        name=name,
        holdings_count=len(holding_signals),
        holdings_with_data=len(has_data),
        coverage_pct=round(coverage * 100, 1),
        weighted_sentiment=round(weighted_sentiment, 4),
        weighted_ensemble=round(weighted_ensemble, 4) if weighted_ensemble is not None else None,
        portfolio_signal=Settings.generate_signal(weighted_sentiment),
        sector_breakdown=sector_breakdown,
        signal_distribution=signal_dist,
        risk_concentration=risk_concentration,
        holdings=[h.to_dict() for h in holding_signals],
    )


def equal_weight_portfolio(
    tickers: List[str],
    name: str = "Equal Weight",
) -> PortfolioSignal:
    """Create an equal-weighted portfolio from a list of tickers."""
    weight = 1.0 / len(tickers) if tickers else 0
    return aggregate_portfolio({t: weight for t in tickers}, name=name)


def sector_portfolio(
    sector: str,
    name: str = None,
) -> PortfolioSignal:
    """Create a portfolio from all tickers in a given sector."""
    tickers = Settings.SECTOR_TICKERS.get(sector, [])
    if not tickers:
        logger.warning(f"Unknown sector: {sector}")
        return aggregate_portfolio({}, name=name or sector)
    return equal_weight_portfolio(tickers, name=name or f"{sector} Sector")


def multi_portfolio_comparison(
    portfolios: Dict[str, Dict[str, float]],
) -> Dict:
    """
    Compare multiple portfolios side by side.

    Args:
        portfolios: Dict mapping portfolio_name -> {ticker: weight, ...}

    Returns:
        Dict with comparison data for all portfolios.
    """
    results = {}
    for pname, holdings in portfolios.items():
        result = aggregate_portfolio(holdings, name=pname)
        results[pname] = {
            "sentiment": result.weighted_sentiment,
            "ensemble": result.weighted_ensemble,
            "signal": result.portfolio_signal,
            "holdings": result.holdings_count,
            "coverage": result.coverage_pct,
            "sector_breakdown": result.sector_breakdown,
        }

    ranked = sorted(results.items(), key=lambda x: x[1]["sentiment"], reverse=True)

    return {
        "portfolios": results,
        "ranking": [{"name": name, "score": data["sentiment"]} for name, data in ranked],
        "best": ranked[0][0] if ranked else None,
        "worst": ranked[-1][0] if ranked else None,
    }


def save_portfolio(result: PortfolioSignal, output_dir: Path = None) -> Path:
    """Save portfolio analysis to JSON."""
    output_dir = output_dir or Settings.DATA_DIR / "portfolios"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = result.name.lower().replace(" ", "_")
    filepath = output_dir / f"{safe_name}_portfolio.json"
    filepath.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    logger.info(f"Saved portfolio analysis: {filepath}")
    return filepath
