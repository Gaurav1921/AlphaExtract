"""
Calibrator
-----------
Optimizes signal thresholds, section weights, and ensemble weights
using backtest results to maximize prediction accuracy.
"""

import logging
from itertools import product
from typing import Dict, List, Optional, Tuple

from src.backtesting import metrics as m

logger = logging.getLogger(__name__)


class Calibrator:
    """Finds optimal parameters by grid-searching over backtest results."""

    def optimize_thresholds(
        self,
        results: List[Dict],
        score_key: str = "score",
    ) -> Dict:
        """
        Find optimal signal thresholds that maximize directional accuracy.

        Tests different boundary values for STRONG_BUY, BUY, SELL, STRONG_SELL.

        Returns:
            Dict with best thresholds and the accuracy they achieve.
        """
        if not results:
            return {"error": "No results to calibrate"}

        # Get score distribution to set search range
        scores = [r[score_key] for r in results if r.get(score_key) is not None]
        if len(scores) < 5:
            return {"error": "Too few data points to calibrate"}

        best_accuracy = 0.0
        best_thresholds = None

        # Grid search: test threshold combinations
        # STRONG_SELL < SELL < BUY < STRONG_BUY
        sell_range = [round(x * 0.05, 2) for x in range(-8, 0)]     # -0.40 to -0.05
        buy_range = [round(x * 0.05, 2) for x in range(1, 9)]       # 0.05 to 0.40

        for sell_th in sell_range:
            for buy_th in buy_range:
                strong_sell = sell_th - 0.15
                strong_buy = buy_th + 0.15

                # Re-classify each result with these thresholds
                reclassified = []
                for r in results:
                    score = r.get(score_key, 0)
                    if score >= strong_buy:
                        signal = "STRONG_BUY"
                    elif score >= buy_th:
                        signal = "BUY"
                    elif score >= sell_th:
                        signal = "HOLD"
                    elif score >= strong_sell:
                        signal = "SELL"
                    else:
                        signal = "STRONG_SELL"

                    reclassified.append({**r, "signal": signal})

                accuracy = m.directional_accuracy(reclassified)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_thresholds = {
                        "SIGNAL_STRONG_BUY": strong_buy,
                        "SIGNAL_BUY": buy_th,
                        "SIGNAL_SELL": sell_th,
                        "SIGNAL_STRONG_SELL": strong_sell,
                    }

        return {
            "best_thresholds": best_thresholds,
            "directional_accuracy": round(best_accuracy, 4),
            "original_accuracy": round(m.directional_accuracy(results), 4),
            "improvement": round(best_accuracy - m.directional_accuracy(results), 4),
            "data_points": len(scores),
        }

    def optimize_ensemble_weights(
        self,
        results: List[Dict],
        step: float = 0.05,
    ) -> Dict:
        """
        Find the best ensemble component weights (finbert, keywords, llm).

        Requires results with finbert_score, keyword_score, llm_score fields.
        """
        # Filter to results with all three scores
        complete = [
            r for r in results
            if r.get("finbert_score") is not None
            and r.get("keyword_score") is not None
            and r.get("llm_score") is not None
            and r.get("actual_direction") not in (None, "unknown")
        ]

        if len(complete) < 5:
            return {"error": f"Need 5+ complete results, have {len(complete)}"}

        best_accuracy = 0.0
        best_weights = None

        # Grid search: w1 + w2 + w3 = 1.0
        steps = [round(x * step, 2) for x in range(1, int(1.0 / step))]

        for w1 in steps:
            for w2 in steps:
                w3 = round(1.0 - w1 - w2, 2)
                if w3 < step:
                    continue

                reclassified = []
                for r in complete:
                    combined = (
                        w1 * r["finbert_score"]
                        + w2 * r["keyword_score"]
                        + w3 * r["llm_score"]
                    )
                    combined = max(-1.0, min(1.0, combined))

                    from src.config.settings import Settings
                    signal = Settings.generate_signal(combined)
                    reclassified.append({**r, "signal": signal, "score": combined})

                accuracy = m.directional_accuracy(reclassified)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = {"finbert": w1, "keywords": w2, "llm": w3}

        return {
            "best_weights": best_weights,
            "directional_accuracy": round(best_accuracy, 4),
            "original_accuracy": round(m.directional_accuracy(complete), 4),
            "improvement": round(best_accuracy - m.directional_accuracy(complete), 4),
            "data_points": len(complete),
        }

    def optimize_section_weights(
        self,
        results: List[Dict],
        step: float = 0.05,
    ) -> Dict:
        """
        Find optimal FinBERT section weights (item_1a, item_7, item_8).

        Requires results with section-level compound scores.
        """
        # We need to re-score from section-level data
        # This requires the original sentiment files — use what's in results
        scorable = []
        for r in results:
            if r.get("model") != "finbert":
                continue
            # We need section scores — load from sentiment files
            ticker = r.get("ticker")
            filing_date = r.get("filing_date")
            if not ticker or not filing_date:
                continue

            from src.config.settings import Settings
            sentiment_file = Settings.SENTIMENT_DIR / f"{ticker}_{filing_date}_sentiment.json"
            if not sentiment_file.exists():
                continue

            try:
                import json
                data = json.loads(sentiment_file.read_text(encoding="utf-8"))
                sections = data.get("sections", {})

                section_scores = {}
                for key, val in sections.items():
                    compound = val.get("scores", {}).get("compound")
                    if compound is not None:
                        section_scores[key] = compound

                if section_scores:
                    scorable.append({**r, "_section_scores": section_scores})
            except Exception:
                continue

        if len(scorable) < 5:
            return {"error": f"Need 5+ filings with section data, have {len(scorable)}"}

        best_accuracy = 0.0
        best_weights = None
        steps = [round(x * step, 2) for x in range(1, int(1.0 / step))]

        for w1a in steps:
            for w7 in steps:
                w8 = round(1.0 - w1a - w7, 2)
                if w8 < step:
                    continue

                weights = {"item_1a": w1a, "item_7": w7, "item_8": w8}

                reclassified = []
                for r in scorable:
                    section_scores = r["_section_scores"]
                    total_weight = 0.0
                    weighted_score = 0.0
                    for key, compound in section_scores.items():
                        w = weights.get(key, 0.0)
                        weighted_score += compound * w
                        total_weight += w

                    if total_weight > 0:
                        overall = weighted_score / total_weight
                    else:
                        overall = 0.0

                    from src.config.settings import Settings
                    signal = Settings.generate_signal(overall)
                    reclassified.append({**r, "signal": signal, "score": overall})

                accuracy = m.directional_accuracy(reclassified)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights

        return {
            "best_section_weights": best_weights,
            "directional_accuracy": round(best_accuracy, 4),
            "original_accuracy": round(m.directional_accuracy(scorable), 4),
            "improvement": round(best_accuracy - m.directional_accuracy(scorable), 4),
            "data_points": len(scorable),
        }

    def full_calibration(self, results: List[Dict]) -> Dict:
        """Run all calibration steps and return a combined recommendation."""
        logger.info(f"Running full calibration on {len(results)} results...")

        threshold_opt = self.optimize_thresholds(results)
        section_opt = self.optimize_section_weights(results)
        ensemble_opt = self.optimize_ensemble_weights(results)

        return {
            "thresholds": threshold_opt,
            "section_weights": section_opt,
            "ensemble_weights": ensemble_opt,
            "recommended_settings": self._build_recommendations(threshold_opt, section_opt, ensemble_opt),
        }

    @staticmethod
    def _build_recommendations(threshold_opt: Dict, section_opt: Dict, ensemble_opt: Dict) -> Dict:
        """Build a single recommended settings dict from optimization results."""
        rec = {}

        if "best_thresholds" in threshold_opt and threshold_opt.get("improvement", 0) > 0.01:
            rec.update(threshold_opt["best_thresholds"])

        if "best_section_weights" in section_opt and section_opt.get("improvement", 0) > 0.01:
            rec["SECTION_WEIGHTS"] = section_opt["best_section_weights"]

        if "best_weights" in ensemble_opt and ensemble_opt.get("improvement", 0) > 0.01:
            rec["ENSEMBLE_WEIGHT_FINBERT"] = ensemble_opt["best_weights"]["finbert"]
            rec["ENSEMBLE_WEIGHT_KEYWORDS"] = ensemble_opt["best_weights"]["keywords"]
            rec["ENSEMBLE_WEIGHT_LLM"] = ensemble_opt["best_weights"]["llm"]

        return rec
