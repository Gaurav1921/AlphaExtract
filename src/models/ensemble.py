"""
Ensemble Scoring Engine
------------------------
Combines three independent signal sources into a single trading signal:
  1. FinBERT sentiment score (existing)
  2. Keyword risk/opportunity signals (built on TRACKED_KEYWORDS)
  3. LLM qualitative analysis (Gemini)

Each source produces a score in [-1, +1]. The ensemble score is a weighted
average, and the final signal is derived from configurable thresholds.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import Settings
from src.models.anomaly import TRACKED_KEYWORDS

logger = logging.getLogger(__name__)


# --- Keyword sentiment polarity ---
# Bearish: terms whose emergence/spike signals risk (13 groups)
# Bullish: terms whose emergence/spike signals opportunity (7 groups)
_BEARISH_GROUPS = {
    "litigation", "regulatory", "sanctions", "antitrust",
    "liquidity", "debt", "impairment", "revenue_decline",
    "cybersecurity", "supply_chain", "workforce", "climate",
    "competition",
}
_BULLISH_GROUPS = {
    "acquisition", "ai_technology",
    "revenue_growth", "margin_expansion", "shareholder_returns",
    "innovation", "market_expansion",
}


class KeywordScorer:
    """Produces a keyword-based signal from section texts."""

    def score(
        self,
        current_texts: Dict[str, str],
        historical_texts: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """
        Score keyword signals from current filing text, optionally relative to history.

        Args:
            current_texts: {section_key: text} for the current filing.
            historical_texts: list of {section_key: text} dicts for prior filings.

        Returns:
            Dict with score [-1, +1], detail breakdown, and top risk/opportunity keywords.
        """
        current_full = " ".join(current_texts.values()).lower()
        current_counts = self._count_keywords(current_full)

        # If we have historical data, compute delta-based scoring
        if historical_texts and len(historical_texts) >= 2:
            hist_counts_list = [self._count_keywords(" ".join(h.values()).lower()) for h in historical_texts]
            return self._delta_score(current_counts, hist_counts_list)

        # Absolute scoring (no history to compare against)
        return self._absolute_score(current_counts)

    def _count_keywords(self, text: str) -> Dict[str, int]:
        """Count keyword group mentions in text."""
        counts = {}
        for category, groups in TRACKED_KEYWORDS.items():
            for group_name, terms in groups.items():
                counts[group_name] = sum(text.count(term.lower()) for term in terms)
        return counts

    def _absolute_score(self, counts: Dict[str, int]) -> Dict:
        """Score based on absolute keyword counts (no history).

        Uses average-per-group normalization so the 13 bearish groups
        aren't structurally overwhelmed by 2 high-frequency bullish groups.
        The raw score is also dampened (tanh-style) to avoid extreme signals
        from absolute counts alone — delta scoring is more reliable.
        """
        bearish_groups_active = [g for g in _BEARISH_GROUPS if counts.get(g, 0) > 0]
        bullish_groups_active = [g for g in _BULLISH_GROUPS if counts.get(g, 0) > 0]

        bearish_total = sum(counts.get(g, 0) for g in _BEARISH_GROUPS)
        bullish_total = sum(counts.get(g, 0) for g in _BULLISH_GROUPS)

        # Normalize: average mentions per active group (avoids 2 bullish groups
        # with high counts dominating 13 bearish groups)
        bearish_avg = bearish_total / len(_BEARISH_GROUPS) if _BEARISH_GROUPS else 0
        bullish_avg = bullish_total / len(_BULLISH_GROUPS) if _BULLISH_GROUPS else 0
        avg_total = bearish_avg + bullish_avg

        if avg_total == 0:
            score = 0.0
        else:
            # Net sentiment using normalized averages
            raw_score = (bullish_avg - bearish_avg) / avg_total
            # Dampen: absolute scoring should produce moderate signals (cap at ±0.5)
            # because without historical context, we can't distinguish "normal" from "elevated"
            score = raw_score * 0.5

        score = max(-1.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "method": "absolute",
            "bearish_mentions": bearish_total,
            "bullish_mentions": bullish_total,
            "bearish_groups_active": len(bearish_groups_active),
            "bullish_groups_active": len(bullish_groups_active),
            "top_risks": self._top_groups(counts, _BEARISH_GROUPS, n=3),
            "top_opportunities": self._top_groups(counts, _BULLISH_GROUPS, n=3),
        }

    def _delta_score(self, current: Dict[str, int], history: List[Dict[str, int]]) -> Dict:
        """Score based on change relative to historical average.

        Delta scoring is more informative than absolute, but still dampened
        to ±0.6 to avoid extreme signals from keyword changes alone.
        """
        # Compute historical average per keyword group
        avg_counts = {}
        for group in current:
            hist_vals = [h.get(group, 0) for h in history]
            avg_counts[group] = sum(hist_vals) / len(hist_vals) if hist_vals else 0

        # Compute weighted delta
        bearish_delta = 0.0
        bullish_delta = 0.0
        signals = []

        for group, curr_count in current.items():
            avg = avg_counts.get(group, 0)
            if avg == 0 and curr_count == 0:
                continue

            if avg == 0 and curr_count > 0:
                # New emergence
                delta = 1.0
            elif avg > 0:
                delta = (curr_count - avg) / avg
            else:
                delta = 0.0

            if group in _BEARISH_GROUPS and delta > 0.5:
                bearish_delta += min(delta, 3.0)  # Cap at 3x to avoid outliers
                signals.append({"group": group, "type": "risk_increase", "delta": round(delta, 2)})
            elif group in _BEARISH_GROUPS and delta < -0.3:
                bullish_delta += min(abs(delta), 2.0)  # Risk decreasing is bullish
                signals.append({"group": group, "type": "risk_decrease", "delta": round(delta, 2)})
            elif group in _BULLISH_GROUPS and delta > 0.5:
                bullish_delta += min(delta, 3.0)
                signals.append({"group": group, "type": "opportunity_increase", "delta": round(delta, 2)})
            elif group in _BULLISH_GROUPS and delta < -0.3:
                bearish_delta += min(abs(delta), 2.0)  # Opportunity decreasing is bearish
                signals.append({"group": group, "type": "opportunity_decrease", "delta": round(delta, 2)})

        total_signal = bullish_delta + bearish_delta
        if total_signal == 0:
            score = 0.0
        else:
            raw_score = (bullish_delta - bearish_delta) / total_signal
            # Dampen: delta scoring is more reliable than absolute but
            # keywords alone shouldn't produce extreme signals
            score = raw_score * 0.6

        score = max(-1.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "method": "delta",
            "bearish_delta": round(bearish_delta, 3),
            "bullish_delta": round(bullish_delta, 3),
            "signals": sorted(signals, key=lambda s: abs(s["delta"]), reverse=True)[:5],
            "top_risks": self._top_groups(current, _BEARISH_GROUPS, n=3),
            "top_opportunities": self._top_groups(current, _BULLISH_GROUPS, n=3),
        }

    @staticmethod
    def _top_groups(counts: Dict[str, int], group_set: set, n: int = 3) -> List[Dict]:
        relevant = [(g, counts.get(g, 0)) for g in group_set if counts.get(g, 0) > 0]
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [{"group": g, "mentions": c} for g, c in relevant[:n]]


class LLMScorer:
    """Produces a qualitative directional opinion on a filing using an LLM.

    Supports multiple providers via LLM_PROVIDER env var:
      - "groq"   — Free tier: 14,400 req/day, Llama 3.3 70B (recommended)
      - "ollama"  — Fully local, unlimited, no API key needed
      - "gemini"  — Google Gemini (free tier is very limited)
      - "auto"    — Auto-detect: tries Groq → Ollama → Gemini
    """

    def __init__(self):
        self._client = None
        self._provider = None
        self._model = None

    def _resolve_provider(self) -> Optional[str]:
        """Determine which LLM provider to use."""
        provider = Settings.LLM_PROVIDER.lower()
        if provider != "auto":
            return provider

        # Auto-detect: prefer Groq (generous free tier) → Gemini
        if Settings.GROQ_API_KEY:
            return "groq"
        if Settings.GEMINI_API_KEY:
            return "gemini"
        # Ollama doesn't need a key — check if it's reachable
        try:
            import urllib.request
            urllib.request.urlopen(f"{Settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
            return "ollama"
        except Exception:
            pass
        return None

    @property
    def client(self):
        if self._client is not None:
            return self._client

        provider = self._resolve_provider()
        if not provider:
            return None

        try:
            if provider == "groq":
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=Settings.GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                )
                self._model = Settings.GROQ_MODEL
                self._provider = "groq"
                logger.info(f"Groq client initialized: {self._model}")

            elif provider == "ollama":
                from openai import OpenAI
                self._client = OpenAI(
                    api_key="ollama",  # Ollama doesn't need a real key
                    base_url=f"{Settings.OLLAMA_BASE_URL}/v1",
                )
                self._model = Settings.OLLAMA_MODEL
                self._provider = "ollama"
                logger.info(f"Ollama client initialized: {self._model}")

            elif provider == "gemini":
                from google import genai
                self._client = genai.Client(api_key=Settings.GEMINI_API_KEY)
                self._model = Settings.GEMINI_MODEL
                self._provider = "gemini"
                logger.info(f"Gemini client initialized: {self._model}")

        except Exception as e:
            logger.warning(f"Could not initialize {provider} client: {e}")
        return self._client

    @property
    def provider(self) -> Optional[str]:
        # Ensure client is resolved first
        if self._provider is None:
            _ = self.client
        return self._provider

    @property
    def available(self) -> bool:
        return self.client is not None

    def _build_prompt(self, ticker: str, section_texts: Dict[str, str]) -> str:
        mda_text = section_texts.get("item_7", "")[:4000]
        risk_text = section_texts.get("item_1a", "")[:2000]
        return f"""You are a senior equity research analyst scoring {ticker}'s 10-K filing
for a quantitative trading system. Your score will be combined with other signals,
so accuracy matters more than caution.

## MD&A (Management Discussion & Analysis) — Excerpt:
{mda_text}

## Risk Factors — Excerpt:
{risk_text}

## Scoring rubric (follow this strictly):
- "bullish": Revenue AND margins growing, market position strengthening, few new risks
- "slightly_bullish": Mostly positive but some concerns (slowing growth, new competition)
- "neutral": Mixed signals — positives and negatives roughly balanced
- "slightly_bearish": Concerning trends — declining metrics, rising risks, vague guidance
- "bearish": Clear deterioration — revenue/margin decline, major new risks, weak outlook

## Important:
- Look at SPECIFIC NUMBERS in the MD&A (revenue growth %, margin changes, guidance)
- A company with 20%+ revenue growth is bullish even if risk factors are long
- A company with declining revenue is bearish even if management sounds optimistic
- Let the DATA drive your decision, not the tone of the writing

Respond in EXACTLY this JSON format (no other text):
{{
  "outlook": "bullish" | "slightly_bullish" | "neutral" | "slightly_bearish" | "bearish",
  "confidence": "high" | "medium" | "low",
  "bull_factors": ["positive factor 1", "positive factor 2"],
  "bear_factors": ["negative factor 1", "negative factor 2"],
  "one_line_summary": "brief summary"
}}"""

    def _call_openai_compatible(self, prompt: str) -> str:
        """Call Groq or Ollama via the OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini."""
        response = self.client.models.generate_content(
            model=self._model, contents=prompt
        )
        return response.text

    def score(self, ticker: str, section_texts: Dict[str, str]) -> Dict:
        """
        Ask the configured LLM for a directional opinion on the filing.

        Returns:
            Dict with score [-1, +1], outlook, confidence, and reasoning.
        """
        if not self.available:
            return {
                "score": 0.0,
                "outlook": "unavailable",
                "confidence": "none",
                "reasoning": "No LLM provider configured (set GROQ_API_KEY, GEMINI_API_KEY, or run Ollama)",
                "available": False,
            }

        prompt = self._build_prompt(ticker, section_texts)

        # Retry with exponential backoff for rate limits
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                if self._provider == "gemini":
                    text = self._call_gemini(prompt)
                else:
                    text = self._call_openai_compatible(prompt)
                return self._parse_response(text)
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "rate_limit" in error_str.lower()
                if is_rate_limit and attempt < max_retries:
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    logger.info(f"Rate limited ({self._provider}), retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue
                logger.warning(f"LLM scoring failed for {ticker} ({self._provider}): {e}")
                return {
                    "score": 0.0,
                    "outlook": "error",
                    "confidence": "none",
                    "reasoning": error_str,
                    "available": False,
                }

    def _parse_response(self, text: str) -> Dict:
        """Parse LLM JSON response into a normalized score."""
        # Strip markdown code blocks if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM response as JSON: {text[:200]}")
            return {"score": 0.0, "outlook": "parse_error", "confidence": "none", "reasoning": text[:300], "available": True}

        outlook = data.get("outlook", "neutral").lower().strip()
        confidence = data.get("confidence", "medium").lower().strip()

        # 5-point outlook scale — "slightly" variants produce moderate signals,
        # full "bullish"/"bearish" reserved for strong conviction
        outlook_base = {
            "bullish": 1.0,
            "slightly_bullish": 0.4,
            "neutral": 0.0,
            "slightly_bearish": -0.4,
            "bearish": -1.0,
        }.get(outlook, 0.0)

        # Conservative confidence multipliers — LLMs tend to be overconfident,
        # so we cap the maximum contribution (max LLM score: ±0.6)
        confidence_mult = {"high": 0.6, "medium": 0.5, "low": 0.25}.get(confidence, 0.4)
        score = outlook_base * confidence_mult

        return {
            "score": round(score, 4),
            "outlook": outlook,
            "confidence": confidence,
            "bull_factors": data.get("bull_factors", data.get("key_factors", [])),
            "bear_factors": data.get("bear_factors", []),
            "summary": data.get("one_line_summary", ""),
            "available": True,
        }


class EnsembleScorer:
    """
    Combines FinBERT, keyword, and LLM scores into a single trading signal.

    The ensemble score is a weighted average of three independent signals,
    each normalized to [-1, +1].
    """

    def __init__(
        self,
        weight_finbert: float = None,
        weight_keywords: float = None,
        weight_llm: float = None,
    ):
        self.weight_finbert = weight_finbert or Settings.ENSEMBLE_WEIGHT_FINBERT
        self.weight_keywords = weight_keywords or Settings.ENSEMBLE_WEIGHT_KEYWORDS
        self.weight_llm = weight_llm or Settings.ENSEMBLE_WEIGHT_LLM
        self.keyword_scorer = KeywordScorer()
        self.llm_scorer = LLMScorer()

    def score_filing(
        self,
        ticker: str,
        filing_date: str,
        sentiment_data: Optional[Dict] = None,
        section_texts: Optional[Dict[str, str]] = None,
        historical_texts: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """
        Produce an ensemble score for a single filing.

        Args:
            ticker: Stock ticker.
            filing_date: Filing date string (YYYY-MM-DD).
            sentiment_data: Existing FinBERT sentiment output (from sentiment.json).
            section_texts: {section_key: text} for current filing (for keyword + LLM scoring).
            historical_texts: List of prior filing {section_key: text} dicts.

        Returns:
            Full ensemble result with individual and combined scores.
        """
        ticker = ticker.upper()

        # --- Signal 1: FinBERT ---
        finbert_score = 0.0
        finbert_detail = {}
        if sentiment_data:
            compound = sentiment_data.get("overall", {}).get("compound", 0.0)
            # Compound is already roughly [-1, 1] but can exceed; clamp
            finbert_score = max(-1.0, min(1.0, compound))
            finbert_detail = {
                "compound": compound,
                "signal": sentiment_data.get("overall", {}).get("signal", "HOLD"),
                "sections": {
                    k: v.get("scores", {}).get("compound", 0)
                    for k, v in sentiment_data.get("sections", {}).items()
                },
            }

        # --- Signal 2: Keywords ---
        keyword_result = {"score": 0.0}
        if section_texts:
            keyword_result = self.keyword_scorer.score(section_texts, historical_texts)

        # --- Signal 3: LLM ---
        llm_result = {"score": 0.0, "available": False}
        if section_texts:
            llm_result = self.llm_scorer.score(ticker, section_texts)

        # --- Combine ---
        # If LLM is unavailable, redistribute its weight proportionally
        if not llm_result.get("available", False):
            effective_weights = self._redistribute_weights(exclude_llm=True)
        else:
            effective_weights = {
                "finbert": self.weight_finbert,
                "keywords": self.weight_keywords,
                "llm": self.weight_llm,
            }

        ensemble_score = (
            effective_weights["finbert"] * finbert_score
            + effective_weights["keywords"] * keyword_result["score"]
            + effective_weights["llm"] * llm_result["score"]
        )
        ensemble_score = max(-1.0, min(1.0, ensemble_score))

        # Check agreement
        scores = [finbert_score, keyword_result["score"]]
        if llm_result.get("available"):
            scores.append(llm_result["score"])
        all_agree = all(s > 0 for s in scores) or all(s < 0 for s in scores) or all(s == 0 for s in scores)

        ensemble_signal = Settings.generate_signal(ensemble_score)

        result = {
            "ticker": ticker,
            "filing_date": filing_date,
            "signals": {
                "finbert": {
                    "score": round(finbert_score, 4),
                    "signal": Settings.generate_signal(finbert_score),
                    "weight": effective_weights["finbert"],
                    "detail": finbert_detail,
                },
                "keywords": {
                    "score": round(keyword_result["score"], 4),
                    "signal": Settings.generate_signal(keyword_result["score"]),
                    "weight": effective_weights["keywords"],
                    "detail": {k: v for k, v in keyword_result.items() if k != "score"},
                },
                "llm": {
                    "score": round(llm_result["score"], 4),
                    "signal": Settings.generate_signal(llm_result["score"]) if llm_result.get("available") else "N/A",
                    "weight": effective_weights["llm"],
                    "detail": {k: v for k, v in llm_result.items() if k != "score"},
                },
            },
            "ensemble": {
                "score": round(ensemble_score, 4),
                "signal": ensemble_signal,
                "components_agree": all_agree,
            },
        }

        logger.info(
            f"Ensemble {ticker} {filing_date}: {ensemble_signal} ({ensemble_score:+.3f}) "
            f"[FB:{finbert_score:+.3f} KW:{keyword_result['score']:+.3f} LLM:{llm_result['score']:+.3f}]"
        )

        return result

    def _redistribute_weights(self, exclude_llm: bool = False) -> Dict[str, float]:
        """Redistribute weights when a component is unavailable."""
        if exclude_llm:
            total = self.weight_finbert + self.weight_keywords
            if total == 0:
                return {"finbert": 0.5, "keywords": 0.5, "llm": 0.0}
            return {
                "finbert": self.weight_finbert / total,
                "keywords": self.weight_keywords / total,
                "llm": 0.0,
            }
        return {
            "finbert": self.weight_finbert,
            "keywords": self.weight_keywords,
            "llm": self.weight_llm,
        }

    def save_result(self, result: Dict, output_dir: Path = None) -> Path:
        """Save ensemble result to disk."""
        output_dir = output_dir or Settings.SENTIMENT_DIR
        ticker = result["ticker"]
        filing_date = result["filing_date"]
        path = output_dir / f"{ticker}_{filing_date}_ensemble.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"Saved ensemble result: {path.name}")
        return path


def load_section_texts(ticker: str, filing_date: str) -> Dict[str, str]:
    """Load section text files for a filing."""
    ticker = ticker.upper()
    texts = {}
    for section_file in Settings.SECTIONS_DIR.glob(f"{ticker}_{filing_date}_item_*.txt"):
        parts = section_file.stem.split("_")
        # Reconstruct section key: everything after ticker_date
        key_parts = []
        capture = False
        for part in parts:
            if part == "item":
                capture = True
            if capture:
                key_parts.append(part)
        section_key = "_".join(key_parts) if key_parts else section_file.stem
        try:
            texts[section_key] = section_file.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"Cannot read {section_file.name}: {e}")
    return texts


def load_historical_texts(ticker: str, exclude_date: str) -> List[Dict[str, str]]:
    """Load section texts for all filings except the given date."""
    ticker = ticker.upper()
    # Find all unique filing dates
    dates = set()
    for f in Settings.SECTIONS_DIR.glob(f"{ticker}_*_item_*.txt"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            dates.add(parts[1])

    dates.discard(exclude_date)

    historical = []
    for date in sorted(dates):
        texts = load_section_texts(ticker, date)
        if texts:
            historical.append(texts)

    return historical
