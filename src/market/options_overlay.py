"""
Options Sentiment Overlay
--------------------------
Fetches options chain data (put/call ratios, implied volatility) and
combines with filing-based sentiment for a composite signal.
Uses yfinance for options data.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import Settings

logger = logging.getLogger(__name__)


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
        filing_weight: float = 0.70,
        options_weight: float = 0.30,
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

            # Use up to 3 nearest expirations for a robust snapshot
            use_expirations = expirations[:3]

            total_call_oi = 0
            total_put_oi = 0
            total_call_volume = 0
            total_put_volume = 0
            call_iv_sum = 0.0
            put_iv_sum = 0.0
            call_iv_count = 0
            put_iv_count = 0

            for exp in use_expirations:
                try:
                    chain = stock.option_chain(exp)
                except Exception:
                    continue

                calls = chain.calls
                puts = chain.puts

                if not calls.empty:
                    total_call_oi += int(calls["openInterest"].sum())
                    total_call_volume += int(calls["volume"].fillna(0).sum())
                    iv_vals = calls["impliedVolatility"].dropna()
                    if not iv_vals.empty:
                        call_iv_sum += iv_vals.sum()
                        call_iv_count += len(iv_vals)

                if not puts.empty:
                    total_put_oi += int(puts["openInterest"].sum())
                    total_put_volume += int(puts["volume"].fillna(0).sum())
                    iv_vals = puts["impliedVolatility"].dropna()
                    if not iv_vals.empty:
                        put_iv_sum += iv_vals.sum()
                        put_iv_count += len(iv_vals)

            # Compute ratios
            pc_ratio = (
                total_put_oi / total_call_oi
                if total_call_oi > 0
                else None
            )
            pc_volume_ratio = (
                total_put_volume / total_call_volume
                if total_call_volume > 0
                else None
            )

            avg_call_iv = call_iv_sum / call_iv_count if call_iv_count > 0 else None
            avg_put_iv = put_iv_sum / put_iv_count if put_iv_count > 0 else None

            iv_skew = None
            if avg_put_iv is not None and avg_call_iv is not None:
                iv_skew = avg_put_iv - avg_call_iv

            # Derive options signal
            options_score, options_signal = self._score_options(
                pc_ratio, pc_volume_ratio, iv_skew
            )

            snapshot = OptionsSnapshot(
                ticker=ticker,
                snapshot_date=datetime.now().strftime("%Y-%m-%d"),
                put_call_ratio=round(pc_ratio, 4) if pc_ratio is not None else None,
                put_call_volume_ratio=round(pc_volume_ratio, 4) if pc_volume_ratio is not None else None,
                total_call_oi=total_call_oi,
                total_put_oi=total_put_oi,
                total_call_volume=total_call_volume,
                total_put_volume=total_put_volume,
                avg_call_iv=round(avg_call_iv, 4) if avg_call_iv is not None else None,
                avg_put_iv=round(avg_put_iv, 4) if avg_put_iv is not None else None,
                iv_skew=round(iv_skew, 4) if iv_skew is not None else None,
                options_signal=options_signal,
                options_score=round(options_score, 4),
            )

            logger.info(
                f"{ticker} options: P/C={pc_ratio:.2f}, "
                f"IV skew={iv_skew:+.3f}, signal={options_signal}"
                if pc_ratio and iv_skew
                else f"{ticker} options: limited data available"
            )

            return snapshot

        except Exception as e:
            logger.warning(f"Failed to fetch options for {ticker}: {e}")
            return None

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

        # Put/Call OI ratio component
        if pc_ratio is not None:
            if pc_ratio > 1.5:
                components.append(-0.8)
            elif pc_ratio > 1.0:
                components.append(-0.3)
            elif pc_ratio > 0.7:
                components.append(0.0)
            elif pc_ratio > 0.5:
                components.append(0.3)
            else:
                components.append(0.6)

        # Put/Call volume ratio component
        if pc_volume_ratio is not None:
            if pc_volume_ratio > 1.5:
                components.append(-0.6)
            elif pc_volume_ratio > 1.0:
                components.append(-0.2)
            elif pc_volume_ratio > 0.7:
                components.append(0.0)
            else:
                components.append(0.4)

        # IV skew component
        if iv_skew is not None:
            if iv_skew > 0.1:
                components.append(-0.5)  # Puts more expensive = fear
            elif iv_skew > 0.03:
                components.append(-0.2)
            elif iv_skew > -0.03:
                components.append(0.0)
            elif iv_skew > -0.1:
                components.append(0.2)
            else:
                components.append(0.5)  # Calls more expensive = greed

        if not components:
            return 0.0, "N/A"

        score = sum(components) / len(components)
        score = max(-1.0, min(1.0, score))
        signal = Settings.generate_signal(score)

        return score, signal

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

        # Load filing sentiment
        filing_sentiment = None
        filing_signal = None
        filing_date = None
        ensemble_score = None
        ensemble_signal = None

        sentiment_files = sorted(Settings.SENTIMENT_DIR.glob(f"{ticker}_*_sentiment.json"))
        if sentiment_files:
            try:
                data = json.loads(sentiment_files[-1].read_text(encoding="utf-8"))
                overall = data.get("overall", {})
                filing_sentiment = overall.get("compound")
                filing_signal = overall.get("signal")
                parts = sentiment_files[-1].stem.split("_")
                if len(parts) >= 2:
                    filing_date = parts[1]
            except Exception as e:
                logger.warning(f"Failed to load sentiment for {ticker}: {e}")

        ensemble_files = sorted(Settings.SENTIMENT_DIR.glob(f"{ticker}_*_ensemble.json"))
        if ensemble_files:
            try:
                data = json.loads(ensemble_files[-1].read_text(encoding="utf-8"))
                ens = data.get("ensemble", {})
                ensemble_score = ens.get("score")
                ensemble_signal = ens.get("signal")
            except Exception as e:
                logger.warning(f"Failed to load ensemble for {ticker}: {e}")

        # Fetch options if not provided
        if options is None:
            options = self.fetch_options(ticker)

        options_score = options.options_score if options else 0.0
        options_signal_val = options.options_signal if options else "N/A"

        # Compute composite
        base_score = filing_sentiment if filing_sentiment is not None else 0.0
        # Prefer ensemble if available
        if ensemble_score is not None:
            base_score = ensemble_score

        if options and options_signal_val != "N/A":
            composite = (
                base_score * self.filing_weight
                + options_score * self.options_weight
            )
        else:
            # No options data: use filing only
            composite = base_score

        composite = max(-1.0, min(1.0, composite))
        composite_sig = Settings.generate_signal(composite)

        # Check agreement between filing and options
        filing_dir = "bullish" if (base_score or 0) > 0.05 else ("bearish" if (base_score or 0) < -0.05 else "neutral")
        options_dir = "bullish" if options_score > 0.05 else ("bearish" if options_score < -0.05 else "neutral")
        agreement = filing_dir == options_dir

        return CompositeSignal(
            ticker=ticker,
            filing_sentiment=round(filing_sentiment, 4) if filing_sentiment is not None else None,
            filing_signal=filing_signal,
            ensemble_score=round(ensemble_score, 4) if ensemble_score is not None else None,
            ensemble_signal=ensemble_signal,
            options_score=round(options_score, 4),
            options_signal=options_signal_val,
            composite_score=round(composite, 4),
            composite_signal=composite_sig,
            filing_weight=self.filing_weight,
            options_weight=self.options_weight,
            agreement=agreement,
            filing_date=filing_date,
        )

    def batch_composite(self, tickers: List[str]) -> List[CompositeSignal]:
        """Compute composite signals for multiple tickers."""
        results = []
        for ticker in tickers:
            result = self.composite_signal(ticker)
            results.append(result)
        return results

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
