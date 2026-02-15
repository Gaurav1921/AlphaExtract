"""
Backtesting Engine
-------------------
Runs backtest across historical filings: compares AlphaExtract signals
(both FinBERT-only and ensemble) against actual market returns.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

from src.config.settings import Settings
from src.market.price_data import MarketDataProvider
from src.backtesting import metrics as m

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for a complete backtest run."""

    def __init__(self, results: List[Dict], config: Dict):
        self.results = results
        self.config = config

    @property
    def hit_rate(self) -> float:
        return m.hit_rate(self.results)

    @property
    def directional_accuracy(self) -> float:
        return m.directional_accuracy(self.results)

    @property
    def sharpe(self) -> Optional[float]:
        return m.sharpe_ratio(self.results)

    @property
    def precision_by_signal(self) -> Dict:
        return m.precision_by_signal(self.results)

    @property
    def avg_return_by_signal(self) -> Dict:
        return m.avg_return_by_signal(self.results)

    @property
    def confusion_matrix(self) -> Dict:
        return m.confusion_matrix(self.results)

    def to_dict(self) -> Dict:
        return {
            "config": self.config,
            "summary": {
                "total_signals": len(self.results),
                "scorable": len([r for r in self.results if r.get("actual_direction") not in (None, "unknown")]),
                "hit_rate": round(self.hit_rate, 4),
                "directional_accuracy": round(self.directional_accuracy, 4),
                "sharpe_ratio": self.sharpe,
            },
            "precision_by_signal": self.precision_by_signal,
            "avg_return_by_signal": self.avg_return_by_signal,
            "confusion_matrix": self.confusion_matrix,
            "results": self.results,
        }


class Backtester:
    """
    Runs backtest comparing trading signals against actual market performance.

    For each (ticker, filing_date) with sentiment data:
      1. Get the signal (FinBERT-only and/or ensemble)
      2. Get the actual market return over a configurable window
      3. Score whether the signal predicted direction correctly
    """

    def __init__(self, return_window: int = 90, market_provider: MarketDataProvider = None):
        self.return_window = return_window
        self.market = market_provider or MarketDataProvider()

    def run(
        self,
        tickers: List[str],
        mode: str = "finbert",
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> BacktestResult:
        """
        Run backtest across multiple tickers.

        Args:
            tickers: List of ticker symbols.
            mode: "finbert" for FinBERT-only, "ensemble" for ensemble scores.
            progress_callback: Optional (message, percent) callback.

        Returns:
            BacktestResult with all individual results and aggregate metrics.
        """
        all_results = []
        total_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            ticker = ticker.upper()
            pct = int((i / total_tickers) * 100)
            if progress_callback:
                progress_callback(f"Backtesting {ticker}...", pct)
            logger.info(f"Backtesting {ticker} ({i+1}/{total_tickers})")

            ticker_results = self._backtest_ticker(ticker, mode)
            all_results.extend(ticker_results)

        if progress_callback:
            progress_callback("Backtest complete", 100)

        config = {
            "tickers": [t.upper() for t in tickers],
            "mode": mode,
            "return_window_days": self.return_window,
            "flat_threshold": Settings.MARKET_FLAT_THRESHOLD,
        }

        result = BacktestResult(all_results, config)
        logger.info(
            f"Backtest complete: {len(all_results)} signals, "
            f"hit_rate={result.hit_rate:.1%}, "
            f"directional_accuracy={result.directional_accuracy:.1%}"
        )
        return result

    def _backtest_ticker(self, ticker: str, mode: str) -> List[Dict]:
        """Backtest all filings for a single ticker."""
        results = []

        if mode == "ensemble":
            results = self._backtest_ensemble(ticker)
        else:
            results = self._backtest_finbert(ticker)

        return results

    def _backtest_finbert(self, ticker: str) -> List[Dict]:
        """Backtest using FinBERT-only sentiment signals."""
        results = []
        sentiment_files = sorted(Settings.SENTIMENT_DIR.glob(f"{ticker}_*_sentiment.json"))

        for sf in sentiment_files:
            parts = sf.stem.split("_")
            if len(parts) < 2:
                continue
            filing_date = parts[1]

            try:
                data = json.loads(sf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping {sf.name}: {e}")
                continue

            overall = data.get("overall", {})
            signal = overall.get("signal", "HOLD")
            score = overall.get("compound", 0.0)

            outcome = self.market.get_filing_outcome(ticker, filing_date, window=self.return_window)

            results.append({
                "ticker": ticker,
                "filing_date": filing_date,
                "model": "finbert",
                "signal": signal,
                "score": round(score, 4),
                "actual_return_pct": outcome.get("return_pct"),
                "actual_direction": outcome.get("direction"),
                "window_days": self.return_window,
            })

        return results

    def _backtest_ensemble(self, ticker: str) -> List[Dict]:
        """Backtest using ensemble signals."""
        results = []
        ensemble_files = sorted(Settings.SENTIMENT_DIR.glob(f"{ticker}_*_ensemble.json"))

        if not ensemble_files:
            logger.warning(f"No ensemble data for {ticker} — falling back to FinBERT")
            return self._backtest_finbert(ticker)

        for ef in ensemble_files:
            parts = ef.stem.split("_")
            if len(parts) < 2:
                continue
            filing_date = parts[1]

            try:
                data = json.loads(ef.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Skipping {ef.name}: {e}")
                continue

            ensemble = data.get("ensemble", {})
            signal = ensemble.get("signal", "HOLD")
            score = ensemble.get("score", 0.0)

            outcome = self.market.get_filing_outcome(ticker, filing_date, window=self.return_window)

            results.append({
                "ticker": ticker,
                "filing_date": filing_date,
                "model": "ensemble",
                "signal": signal,
                "score": round(score, 4),
                "actual_return_pct": outcome.get("return_pct"),
                "actual_direction": outcome.get("direction"),
                "window_days": self.return_window,
                "components_agree": ensemble.get("components_agree", False),
                "finbert_score": data.get("signals", {}).get("finbert", {}).get("score"),
                "keyword_score": data.get("signals", {}).get("keywords", {}).get("score"),
                "llm_score": data.get("signals", {}).get("llm", {}).get("score"),
            })

        return results

    def run_comparison(
        self,
        tickers: List[str],
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Run both FinBERT-only and ensemble backtests side by side.

        Returns comparison metrics showing which model performs better.
        """
        logger.info("Running comparison backtest: FinBERT vs Ensemble")
        finbert_result = self.run(tickers, mode="finbert", progress_callback=progress_callback)
        ensemble_result = self.run(tickers, mode="ensemble")

        comparison = m.compare_models(
            finbert_result.results,
            ensemble_result.results,
            label_a="FinBERT Only",
            label_b="Ensemble",
        )

        return {
            "comparison": comparison,
            "finbert": finbert_result.to_dict(),
            "ensemble": ensemble_result.to_dict(),
        }

    def save_results(self, result: BacktestResult, name: str = "backtest") -> Path:
        """Save backtest results to disk."""
        output_dir = Settings.BACKTEST_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{name}_{timestamp}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Saved backtest results: {path.name}")
        return path
