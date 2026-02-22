"""
AlphaExtract REST API
---------------------
FastAPI endpoints for programmatic access to sentiment signals,
ensemble scores, portfolio analysis, options overlay, and filing alerts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config.settings import Settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AlphaExtract API",
    description="AI-Powered Financial Intelligence for SEC 10-K Filings",
    version=Settings.VERSION,
)


# ============================================================================
# Pydantic models
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    tickers_available: int


class SentimentResponse(BaseModel):
    ticker: str
    filing_date: Optional[str] = None
    overall_score: Optional[float] = None
    overall_signal: Optional[str] = None
    sections: Optional[Dict] = None


class EnsembleResponse(BaseModel):
    ticker: str
    filing_date: Optional[str] = None
    ensemble_score: Optional[float] = None
    ensemble_signal: Optional[str] = None
    components_agree: Optional[bool] = None
    signals: Optional[Dict] = None


class PortfolioRequest(BaseModel):
    holdings: Dict[str, float] = Field(
        ...,
        description="Ticker -> weight mapping (e.g. {'AAPL': 0.5, 'MSFT': 0.5})",
    )
    name: str = "API Portfolio"


class PortfolioResponse(BaseModel):
    name: str
    holdings_count: int
    holdings_with_data: int
    coverage_pct: float
    weighted_sentiment: float
    weighted_ensemble: Optional[float] = None
    portfolio_signal: str
    sector_breakdown: Dict
    signal_distribution: Dict
    risk_concentration: Dict
    holdings: List[Dict]


class OptionsResponse(BaseModel):
    ticker: str
    filing_sentiment: Optional[float] = None
    filing_signal: Optional[str] = None
    ensemble_score: Optional[float] = None
    ensemble_signal: Optional[str] = None
    options_score: float
    options_signal: str
    composite_score: float
    composite_signal: str
    filing_weight: float
    options_weight: float
    agreement: bool


class AlertsResponse(BaseModel):
    ticker: str
    alerts: List[Dict]
    total: int


class TickerListResponse(BaseModel):
    tickers: List[str]
    sectors: Dict[str, List[str]]
    total: int


# ============================================================================
# Helper: load JSON signal files
# ============================================================================

def _load_latest_json(ticker: str, suffix: str) -> Optional[Dict]:
    """Load the most recent {TICKER}_*_{suffix}.json from SENTIMENT_DIR."""
    pattern = f"{ticker.upper()}_*_{suffix}.json"
    files = sorted(Settings.SENTIMENT_DIR.glob(pattern))
    if not files:
        return None
    import json
    try:
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        parts = files[-1].stem.split("_")
        if len(parts) >= 2:
            data["_filing_date"] = parts[1]
        return data
    except (json.JSONDecodeError, OSError):
        return None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    sentiment_files = list(Settings.SENTIMENT_DIR.glob("*_sentiment.json"))
    tickers = {f.stem.split("_")[0] for f in sentiment_files}
    return HealthResponse(
        status="ok",
        version=Settings.VERSION,
        tickers_available=len(tickers),
    )


@app.get("/tickers", response_model=TickerListResponse)
def list_tickers():
    """List all available tickers and sectors."""
    return TickerListResponse(
        tickers=Settings.ALL_TICKERS,
        sectors=Settings.SECTOR_TICKERS,
        total=len(Settings.ALL_TICKERS),
    )


@app.get("/sentiment/{ticker}", response_model=SentimentResponse)
def get_sentiment(ticker: str):
    """Get latest FinBERT sentiment for a ticker."""
    ticker = ticker.upper()
    data = _load_latest_json(ticker, "sentiment")
    if not data:
        raise HTTPException(status_code=404, detail=f"No sentiment data for {ticker}")

    overall = data.get("overall", {})
    return SentimentResponse(
        ticker=ticker,
        filing_date=data.get("_filing_date"),
        overall_score=overall.get("compound"),
        overall_signal=overall.get("signal"),
        sections=data.get("sections"),
    )


@app.get("/ensemble/{ticker}", response_model=EnsembleResponse)
def get_ensemble(ticker: str):
    """Get latest ensemble score for a ticker."""
    ticker = ticker.upper()
    data = _load_latest_json(ticker, "ensemble")
    if not data:
        raise HTTPException(status_code=404, detail=f"No ensemble data for {ticker}")

    ens = data.get("ensemble", {})
    return EnsembleResponse(
        ticker=ticker,
        filing_date=data.get("_filing_date"),
        ensemble_score=ens.get("score"),
        ensemble_signal=ens.get("signal"),
        components_agree=ens.get("components_agree"),
        signals=data.get("signals"),
    )


@app.post("/portfolio", response_model=PortfolioResponse)
def compute_portfolio(request: PortfolioRequest):
    """Compute portfolio-level signal aggregation."""
    from src.analysis.portfolio import aggregate_portfolio

    if not request.holdings:
        raise HTTPException(status_code=400, detail="Holdings cannot be empty")

    result = aggregate_portfolio(request.holdings, name=request.name)
    return PortfolioResponse(
        name=result.name,
        holdings_count=result.holdings_count,
        holdings_with_data=result.holdings_with_data,
        coverage_pct=result.coverage_pct,
        weighted_sentiment=result.weighted_sentiment,
        weighted_ensemble=result.weighted_ensemble,
        portfolio_signal=result.portfolio_signal,
        sector_breakdown=result.sector_breakdown,
        signal_distribution=result.signal_distribution,
        risk_concentration=result.risk_concentration,
        holdings=result.holdings,
    )


@app.get("/options/{ticker}", response_model=OptionsResponse)
def get_options(
    ticker: str,
    filing_weight: float = Query(0.70, ge=0, le=1),
    options_weight: float = Query(0.30, ge=0, le=1),
):
    """Get composite filing + options signal for a ticker."""
    from src.market.options_overlay import OptionsOverlay

    ticker = ticker.upper()
    overlay = OptionsOverlay(filing_weight=filing_weight, options_weight=options_weight)
    signal = overlay.composite_signal(ticker)

    return OptionsResponse(
        ticker=signal.ticker,
        filing_sentiment=signal.filing_sentiment,
        filing_signal=signal.filing_signal,
        ensemble_score=signal.ensemble_score,
        ensemble_signal=signal.ensemble_signal,
        options_score=signal.options_score,
        options_signal=signal.options_signal,
        composite_score=signal.composite_score,
        composite_signal=signal.composite_signal,
        filing_weight=signal.filing_weight,
        options_weight=signal.options_weight,
        agreement=signal.agreement,
    )


@app.get("/alerts/{ticker}", response_model=AlertsResponse)
def check_alerts(
    ticker: str,
    days_back: int = Query(30, ge=1, le=365),
):
    """Check SEC EDGAR for recent 10-K filings."""
    from src.alerts.sec_monitor import SECMonitor

    ticker = ticker.upper()
    monitor = SECMonitor(watchlist=[ticker])
    alerts = monitor.check_once(days_back=days_back)

    return AlertsResponse(
        ticker=ticker,
        alerts=[a.to_dict() for a in alerts],
        total=len(alerts),
    )


@app.get("/signals/{ticker}")
def get_all_signals(ticker: str):
    """Get all available signals (sentiment + ensemble + options) for a ticker."""
    ticker = ticker.upper()
    result: Dict = {"ticker": ticker}

    sentiment = _load_latest_json(ticker, "sentiment")
    if sentiment:
        overall = sentiment.get("overall", {})
        result["sentiment"] = {
            "filing_date": sentiment.get("_filing_date"),
            "score": overall.get("compound"),
            "signal": overall.get("signal"),
        }

    ensemble = _load_latest_json(ticker, "ensemble")
    if ensemble:
        ens = ensemble.get("ensemble", {})
        result["ensemble"] = {
            "filing_date": ensemble.get("_filing_date"),
            "score": ens.get("score"),
            "signal": ens.get("signal"),
        }

    if not sentiment and not ensemble:
        raise HTTPException(status_code=404, detail=f"No signal data for {ticker}")

    return result
