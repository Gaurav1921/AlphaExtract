"""
Portfolio Signal Aggregation
-----------------------------
Combines sentiment and ensemble signals across multiple holdings into
portfolio-level metrics. Supports position-weighted and equal-weighted modes.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
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


def _load_latest_sentiment(ticker: str) -> Optional[Dict]:
    """Load the most recent sentiment data for a ticker."""
    pattern = f"{ticker.upper()}_*_sentiment.json"
    files = sorted(Settings.SENTIMENT_DIR.glob(pattern))
    if not files:
        return None

    latest = files[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        # Extract filing date from filename
        parts = latest.stem.split("_")
        if len(parts) >= 2:
            data["_filing_date"] = parts[1]
        return data
    except Exception as e:
        logger.warning(f"Failed to load sentiment for {ticker}: {e}")
        return None


def _load_latest_ensemble(ticker: str) -> Optional[Dict]:
    """Load the most recent ensemble data for a ticker."""
    pattern = f"{ticker.upper()}_*_ensemble.json"
    files = sorted(Settings.SENTIMENT_DIR.glob(pattern))
    if not files:
        return None

    latest = files[-1]
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to load ensemble for {ticker}: {e}")
        return None


def aggregate_portfolio(
    holdings: Dict[str, float],
    name: str = "My Portfolio",
) -> PortfolioSignal:
    """
    Aggregate signals across a portfolio of holdings.

    Args:
        holdings: Dict mapping ticker -> weight (0-1). Weights should sum to 1.0.
                  If they don't, they'll be normalized.
        name: Portfolio name for display.

    Returns:
        PortfolioSignal with aggregated metrics.
    """
    # Normalize weights
    total_weight = sum(holdings.values())
    if total_weight <= 0:
        total_weight = 1.0
    normalized = {t: w / total_weight for t, w in holdings.items()}

    holding_signals: List[HoldingSignal] = []
    sector_scores: Dict[str, List[Tuple[float, float]]] = {}  # sector -> [(score, weight)]

    for ticker, weight in normalized.items():
        ticker = ticker.upper()
        sector = _get_ticker_sector(ticker)

        sentiment = _load_latest_sentiment(ticker)
        ensemble = _load_latest_ensemble(ticker)

        hs = HoldingSignal(
            ticker=ticker,
            weight=round(weight, 4),
            sector=sector,
        )

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

        # Collect sector data
        if sector and hs.sentiment_score is not None:
            if sector not in sector_scores:
                sector_scores[sector] = []
            sector_scores[sector].append((hs.sentiment_score, weight))

    # Compute aggregated scores
    has_data = [h for h in holding_signals if h.sentiment_score is not None]
    coverage = len(has_data) / len(holding_signals) if holding_signals else 0

    # Weighted sentiment
    if has_data:
        weight_sum = sum(h.weight for h in has_data)
        if weight_sum > 0:
            weighted_sentiment = sum(
                h.sentiment_score * h.weight for h in has_data
            ) / weight_sum
        else:
            weighted_sentiment = 0.0
    else:
        weighted_sentiment = 0.0

    # Weighted ensemble
    has_ensemble = [h for h in holding_signals if h.ensemble_score is not None]
    weighted_ensemble = None
    if has_ensemble:
        ens_weight_sum = sum(h.weight for h in has_ensemble)
        if ens_weight_sum > 0:
            weighted_ensemble = sum(
                h.ensemble_score * h.weight for h in has_ensemble
            ) / ens_weight_sum

    # Portfolio signal
    portfolio_signal = Settings.generate_signal(weighted_sentiment)

    # Signal distribution
    signal_dist: Dict[str, int] = {}
    for h in has_data:
        sig = h.sentiment_signal or "N/A"
        signal_dist[sig] = signal_dist.get(sig, 0) + 1

    # Sector breakdown
    sector_breakdown = {}
    for sector, scores_weights in sector_scores.items():
        weights_in_sector = sum(w for _, w in scores_weights)
        avg_score = (
            sum(s * w for s, w in scores_weights) / weights_in_sector
            if weights_in_sector > 0
            else 0
        )
        sector_breakdown[sector] = {
            "avg_score": round(avg_score, 4),
            "signal": Settings.generate_signal(avg_score),
            "weight": round(weights_in_sector, 4),
            "count": len(scores_weights),
        }

    # Risk concentration: largest single-ticker and single-sector exposure
    max_holding = max(holding_signals, key=lambda h: h.weight) if holding_signals else None
    max_sector = max(
        sector_breakdown.items(), key=lambda x: x[1]["weight"]
    ) if sector_breakdown else None

    risk_concentration = {
        "max_single_holding": {
            "ticker": max_holding.ticker if max_holding else None,
            "weight": max_holding.weight if max_holding else 0,
        },
        "max_single_sector": {
            "sector": max_sector[0] if max_sector else None,
            "weight": max_sector[1]["weight"] if max_sector else 0,
        },
        "herfindahl_index": round(
            sum(h.weight ** 2 for h in holding_signals), 4
        ),
    }

    return PortfolioSignal(
        name=name,
        holdings_count=len(holding_signals),
        holdings_with_data=len(has_data),
        coverage_pct=round(coverage * 100, 1),
        weighted_sentiment=round(weighted_sentiment, 4),
        weighted_ensemble=round(weighted_ensemble, 4) if weighted_ensemble is not None else None,
        portfolio_signal=portfolio_signal,
        sector_breakdown=sector_breakdown,
        signal_distribution=signal_dist,
        risk_concentration=risk_concentration,
        holdings=[h.to_dict() for h in holding_signals],
    )


def equal_weight_portfolio(
    tickers: List[str],
    name: str = "Equal Weight",
) -> PortfolioSignal:
    """
    Convenience: create an equal-weighted portfolio from a list of tickers.
    """
    weight = 1.0 / len(tickers) if tickers else 0
    holdings = {t: weight for t in tickers}
    return aggregate_portfolio(holdings, name=name)


def sector_portfolio(
    sector: str,
    name: str = None,
) -> PortfolioSignal:
    """
    Create a portfolio from all tickers in a given sector.
    """
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

    # Rank portfolios by sentiment
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
