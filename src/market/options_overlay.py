"""
Options Sentiment Overlay
--------------------------
Fetches options chain data (put/call ratios, implied volatility) and
combines with filing-based sentiment for a composite signal.
Uses yfinance for options data.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import Settings

logger = logging.getLogger(__name__)

# Thresholds for put/call ratio scoring
PC_RATIO_VERY_BEARISH = 1.5
PC_RATIO_BEARISH = 1.0
PC_RATIO_NEUTRAL = 0.7
PC_RATIO_BULLISH = 0.5

# Thresholds for IV skew scoring
IV_SKEW_BEARISH = 0.1
IV_SKEW_SLIGHT_BEARISH = 0.03
IV_SKEW_SLIGHT_BULLISH = -0.03
IV_SKEW_BULLISH = -0.1

# Threshold for direction classification
DIRECTION_THRESHOLD = 0.05


@dataclass
class OptionsSnapshot:
    """Options-derived sentiment for a single ticker."""
    ticker: str
    snapshot_date: str
    put_call_ratio: Optional[float] = None
    put_call_volume_ratio: Optional[float] = None
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_call_volume: int = 0
    total_put_volume: int = 0
    avg_call_iv: Optional[float] = None
    avg_put_iv: Optional[float] = None
    iv_skew: Optional[float] = None
    options_signal: str = "N/A"
    options_score: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompositeSignal:
    """Combined filing sentiment + options sentiment."""
    ticker: str
    filing_sentiment: Optional[float]
    filing_signal: Optional[str]
    ensemble_score: Optional[float]
    ensemble_signal: Optional[str]
    options_score: float
    options_signal: str
    composite_score: float
    composite_signal: str
    filing_weight: float
    options_weight: float
    agreement: bool
    filing_date: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _classify_direction(score: float) -> str:
    """Classify a score as bullish/bearish/neutral."""
    if score > DIRECTION_THRESHOLD:
        return "bullish"
    if score < -DIRECTION_THRESHOLD:
        return "bearish"
    return "neutral"


def _load_latest_json(ticker: str, suffix: str) -> Optional[Dict]:
    """Load the most recent {TICKER}_*_{suffix}.json file."""
    files = sorted(Settings.SENTIMENT_DIR.glob(f"{ticker}_*_{suffix}.json"))
    if not files:
        return None
    try:
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        parts = files[-1].stem.split("_")
        if len(parts) >= 2:
            data["_filing_date"] = parts[1]
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load {suffix} for {ticker}: {e}")
        return None


class OptionsOverlay:
    """
    Fetches options data and combines with filing sentiment.

    Options-derived signals:
    - Put/Call ratio (OI and volume)
    - IV skew (put IV vs call IV)
    - Composite score blending filing + options sentiment
    """

    def __init__(
        self,
        filing_weight: float = Settings.OPTIONS_FILING_WEIGHT,
        options_weight: float = Settings.OPTIONS_WEIGHT,
        cache_dir: Path = None,
    ):
        self.filing_weight = filing_weight
        self.options_weight = options_weight
        self.cache_dir = cache_dir or Settings.MARKET_DATA_DIR
        self._yf = None

    @property
    def yf(self):
        """Lazy import yfinance."""
        if self._yf is None:
            try:
                import yfinance
                self._yf = yfinance
            except ImportError:
                logger.error("yfinance not installed — run: pip install yfinance")
        return self._yf

    # ------------------------------------------------------------------
    # Options data fetching
    # ------------------------------------------------------------------

    def fetch_options(self, ticker: str) -> Optional[OptionsSnapshot]:
        """
        Fetch current options chain data for a ticker.

        Aggregates across nearest expiration dates to compute:
        - Put/call open interest ratio
        - Put/call volume ratio
        - Average implied volatility for calls and puts
        - IV skew (put IV - call IV)
        """
        if self.yf is None:
            return None

        ticker = ticker.upper()
        try:
            stock = self.yf.Ticker(ticker)
            expirations = stock.options

            if not expirations:
                logger.warning(f"No options data available for {ticker}")
                return None

            totals = self._aggregate_chains(stock, expirations[:3])
            return self._build_snapshot(ticker, totals)

        except (OSError, KeyError, ValueError) as e:
            logger.warning(f"Failed to fetch options for {ticker}: {e}")
            return None

    def _aggregate_chains(self, stock, expirations: list) -> Dict:
        """Aggregate options chain stats across expirations."""
        totals = {
            "call_oi": 0, "put_oi": 0,
            "call_vol": 0, "put_vol": 0,
            "call_iv_sum": 0.0, "put_iv_sum": 0.0,
            "call_iv_count": 0, "put_iv_count": 0,
        }

        for exp in expirations:
            try:
                chain = stock.option_chain(exp)
            except Exception:
                continue

            if not chain.calls.empty:
                totals["call_oi"] += int(chain.calls["openInterest"].sum())
                totals["call_vol"] += int(chain.calls["volume"].fillna(0).sum())
                iv_vals = chain.calls["impliedVolatility"].dropna()
                if not iv_vals.empty:
                    totals["call_iv_sum"] += iv_vals.sum()
                    totals["call_iv_count"] += len(iv_vals)

            if not chain.puts.empty:
                totals["put_oi"] += int(chain.puts["openInterest"].sum())
                totals["put_vol"] += int(chain.puts["volume"].fillna(0).sum())
                iv_vals = chain.puts["impliedVolatility"].dropna()
                if not iv_vals.empty:
                    totals["put_iv_sum"] += iv_vals.sum()
                    totals["put_iv_count"] += len(iv_vals)

        return totals

    def _build_snapshot(self, ticker: str, t: Dict) -> OptionsSnapshot:
        """Build OptionsSnapshot from aggregated totals."""
        pc_ratio = t["put_oi"] / t["call_oi"] if t["call_oi"] > 0 else None
        pc_vol_ratio = t["put_vol"] / t["call_vol"] if t["call_vol"] > 0 else None

        avg_call_iv = t["call_iv_sum"] / t["call_iv_count"] if t["call_iv_count"] > 0 else None
        avg_put_iv = t["put_iv_sum"] / t["put_iv_count"] if t["put_iv_count"] > 0 else None

        iv_skew = None
        if avg_put_iv is not None and avg_call_iv is not None:
            iv_skew = avg_put_iv - avg_call_iv

        options_score, options_signal = self._score_options(pc_ratio, pc_vol_ratio, iv_skew)

        if pc_ratio is not None and iv_skew is not None:
            logger.info(f"{ticker} options: P/C={pc_ratio:.2f}, IV skew={iv_skew:+.3f}, signal={options_signal}")
        else:
            logger.info(f"{ticker} options: limited data available")

        return OptionsSnapshot(
            ticker=ticker,
            snapshot_date=datetime.now().strftime("%Y-%m-%d"),
            put_call_ratio=round(pc_ratio, 4) if pc_ratio is not None else None,
            put_call_volume_ratio=round(pc_vol_ratio, 4) if pc_vol_ratio is not None else None,
            total_call_oi=t["call_oi"],
            total_put_oi=t["put_oi"],
            total_call_volume=t["call_vol"],
            total_put_volume=t["put_vol"],
            avg_call_iv=round(avg_call_iv, 4) if avg_call_iv is not None else None,
            avg_put_iv=round(avg_put_iv, 4) if avg_put_iv is not None else None,
            iv_skew=round(iv_skew, 4) if iv_skew is not None else None,
            options_signal=options_signal,
            options_score=round(options_score, 4),
        )

    @staticmethod
    def _score_pc_ratio(ratio: float) -> float:
        """Score put/call ratio component."""
        if ratio > PC_RATIO_VERY_BEARISH:
            return -0.8
        if ratio > PC_RATIO_BEARISH:
            return -0.3
        if ratio > PC_RATIO_NEUTRAL:
            return 0.0
        if ratio > PC_RATIO_BULLISH:
            return 0.3
        return 0.6

    @staticmethod
    def _score_volume_ratio(ratio: float) -> float:
        """Score put/call volume ratio component."""
        if ratio > PC_RATIO_VERY_BEARISH:
            return -0.6
        if ratio > PC_RATIO_BEARISH:
            return -0.2
        if ratio > PC_RATIO_NEUTRAL:
            return 0.0
        return 0.4

    @staticmethod
    def _score_iv_skew(skew: float) -> float:
        """Score IV skew component."""
        if skew > IV_SKEW_BEARISH:
            return -0.5
        if skew > IV_SKEW_SLIGHT_BEARISH:
            return -0.2
        if skew > IV_SKEW_SLIGHT_BULLISH:
            return 0.0
        if skew > IV_SKEW_BULLISH:
            return 0.2
        return 0.5

    @staticmethod
    def _score_options(
        pc_ratio: Optional[float],
        pc_volume_ratio: Optional[float],
        iv_skew: Optional[float],
    ) -> tuple:
        """
        Convert options metrics into a sentiment score [-1, +1].

        Logic:
        - High put/call ratio (>1.0) = bearish -> negative score
        - Low put/call ratio (<0.7) = bullish -> positive score
        - High IV skew (puts > calls) = bearish
        - Negative IV skew (calls > puts) = bullish
        """
        components = []

        if pc_ratio is not None:
            components.append(OptionsOverlay._score_pc_ratio(pc_ratio))
        if pc_volume_ratio is not None:
            components.append(OptionsOverlay._score_volume_ratio(pc_volume_ratio))
        if iv_skew is not None:
            components.append(OptionsOverlay._score_iv_skew(iv_skew))

        if not components:
            return 0.0, "N/A"

        score = max(-1.0, min(1.0, sum(components) / len(components)))
        return score, Settings.generate_signal(score)

    # ------------------------------------------------------------------
    # Composite signal
    # ------------------------------------------------------------------

    def composite_signal(
        self,
        ticker: str,
        options: OptionsSnapshot = None,
    ) -> CompositeSignal:
        """
        Combine filing sentiment with options sentiment for a composite score.

        Args:
            ticker: Stock ticker.
            options: Pre-fetched options snapshot (fetched if None).

        Returns:
            CompositeSignal with blended score and metadata.
        """
        ticker = ticker.upper()

        # Load filing data
        filing_sentiment = None
        filing_signal = None
        filing_date = None
        ensemble_score = None
        ensemble_signal = None

        sentiment_data = _load_latest_json(ticker, "sentiment")
        if sentiment_data:
            overall = sentiment_data.get("overall", {})
            filing_sentiment = overall.get("compound")
            filing_signal = overall.get("signal")
            filing_date = sentiment_data.get("_filing_date")

        ensemble_data = _load_latest_json(ticker, "ensemble")
        if ensemble_data:
            ens = ensemble_data.get("ensemble", {})
            ensemble_score = ens.get("score")
            ensemble_signal = ens.get("signal")

        # Fetch options if not provided
        if options is None:
            options = self.fetch_options(ticker)

        opt_score = options.options_score if options else 0.0
        opt_signal = options.options_signal if options else "N/A"

        # Composite: prefer ensemble over raw sentiment
        base_score = ensemble_score if ensemble_score is not None else (filing_sentiment or 0.0)

        if options and opt_signal != "N/A":
            composite = base_score * self.filing_weight + opt_score * self.options_weight
        else:
            composite = base_score

        composite = max(-1.0, min(1.0, composite))

        agreement = _classify_direction(base_score) == _classify_direction(opt_score)

        return CompositeSignal(
            ticker=ticker,
            filing_sentiment=round(filing_sentiment, 4) if filing_sentiment is not None else None,
            filing_signal=filing_signal,
            ensemble_score=round(ensemble_score, 4) if ensemble_score is not None else None,
            ensemble_signal=ensemble_signal,
            options_score=round(opt_score, 4),
            options_signal=opt_signal,
            composite_score=round(composite, 4),
            composite_signal=Settings.generate_signal(composite),
            filing_weight=self.filing_weight,
            options_weight=self.options_weight,
            agreement=agreement,
            filing_date=filing_date,
        )

    def batch_composite(self, tickers: List[str]) -> List[CompositeSignal]:
        """Compute composite signals for multiple tickers."""
        return [self.composite_signal(t) for t in tickers]

    def save_snapshot(self, snapshot: OptionsSnapshot, output_dir: Path = None) -> Path:
        """Save options snapshot to JSON."""
        output_dir = output_dir or Settings.MARKET_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"{snapshot.ticker}_{snapshot.snapshot_date}_options.json"
        filepath.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Saved options snapshot: {filepath}")
        return filepath

    def save_composite(self, signal: CompositeSignal, output_dir: Path = None) -> Path:
        """Save composite signal to JSON."""
        output_dir = output_dir or Settings.SENTIMENT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"{signal.ticker}_composite.json"
        filepath.write_text(json.dumps(signal.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Saved composite signal: {filepath}")
        return filepath
