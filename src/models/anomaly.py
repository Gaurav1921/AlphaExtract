"""
Anomaly Detection for 10-K Filings
------------------------------------
Detects unusual patterns by comparing current vs. historical filings.
Uses keyword frequency tracking and LLM-based contextual analysis.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.config.settings import Settings

logger = logging.getLogger(__name__)


# Keywords tracked across filings, organized by category
TRACKED_KEYWORDS = {
    "legal_regulatory": {
        "litigation": ["litigation", "lawsuit", "legal proceedings", "class action", "arbitration"],
        "regulatory": ["regulatory action", "SEC investigation", "consent decree", "compliance violation"],
        "sanctions": ["sanctions", "penalty", "fine", "enforcement action"],
        "antitrust": ["antitrust", "anti-competitive", "monopoly", "cartel"],
    },
    "financial_risk": {
        "liquidity": ["liquidity risk", "cash shortfall", "credit facility", "going concern"],
        "debt": ["debt covenant", "leverage ratio", "credit downgrade", "debt restructuring"],
        "impairment": ["goodwill impairment", "asset writedown", "restructuring charge"],
        "revenue": ["revenue decline", "customer concentration", "contract termination"],
    },
    "operational": {
        "cybersecurity": ["cybersecurity", "data breach", "ransomware", "security incident"],
        "supply_chain": ["supply chain disruption", "supplier risk", "component shortage"],
        "workforce": ["layoff", "workforce reduction", "restructuring", "headcount"],
        "climate": ["climate risk", "carbon emissions", "environmental regulation"],
    },
    "strategic": {
        "competition": ["competitive pressure", "market share loss", "disruptive technology"],
        "acquisition": ["acquisition", "merger", "divestiture", "joint venture"],
        "ai_technology": ["artificial intelligence", "machine learning", "generative AI"],
    },
}


class AnomalyDetector:
    """Detects anomalies in 10-K filings by comparing current to historical."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Settings.DATA_DIR
        self.sections_dir = self.data_dir / "sections"
        self.sentiment_dir = self.data_dir / "sentiment"
        self.anomalies_dir = Settings.ANOMALIES_DIR
        self._gemini_client = None

    @property
    def gemini_client(self):
        if self._gemini_client is None and Settings.GEMINI_API_KEY:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=Settings.GEMINI_API_KEY)
                logger.info(f"Gemini client initialized: {Settings.GEMINI_MODEL}")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
        return self._gemini_client

    def _get_sorted_filings(self, ticker: str) -> List[Path]:
        """Get all sentiment files for a ticker, sorted by date ascending."""
        files = list(self.sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
        return sorted(files, key=lambda f: f.stem.split("_")[1] if len(f.stem.split("_")) >= 2 else "")

    def _load_sentiment(self, filepath: Path) -> Optional[Dict]:
        """Load and validate a sentiment JSON file."""
        try:
            return json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {filepath.name}: {e}")
            return None

    def _load_section_text(self, ticker: str, filing_date: str) -> Dict[str, str]:
        """Load raw section text for keyword analysis."""
        texts = {}
        for section_file in self.sections_dir.glob(f"{ticker}_{filing_date}_item_*.txt"):
            section_key = "_".join(section_file.stem.split("_")[2:])
            try:
                texts[section_key] = section_file.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning(f"Cannot read {section_file.name}: {e}")
        return texts

    def detect_sentiment_shifts(self, current: Dict, previous: Dict) -> List[Dict]:
        """Compare sentiment scores between two filings."""
        anomalies = []
        current_sections = current.get("sections", {})
        previous_sections = previous.get("sections", {})

        for section_key in current_sections:
            if section_key not in previous_sections:
                continue

            curr_score = current_sections[section_key].get("scores", {}).get("compound", 0)
            prev_score = previous_sections[section_key].get("scores", {}).get("compound", 0)
            delta = curr_score - prev_score

            if abs(delta) >= Settings.ANOMALY_SENTIMENT_THRESHOLD:
                direction = "positive" if delta > 0 else "negative"
                severity = "high" if abs(delta) >= 0.3 else "medium"

                anomalies.append({
                    "type": "sentiment_shift",
                    "category": "Sentiment Analysis",
                    "title": f"Significant {direction} shift in {section_key.replace('_', ' ').title()}",
                    "description": f"Sentiment changed by {delta:+.3f} ({prev_score:+.3f} -> {curr_score:+.3f})",
                    "severity": severity,
                    "severity_reason": f"Delta of {abs(delta):.3f} exceeds threshold of {Settings.ANOMALY_SENTIMENT_THRESHOLD}",
                    "section": section_key,
                    "current_score": curr_score,
                    "previous_score": prev_score,
                    "delta": delta,
                })

        return anomalies

    def detect_keyword_anomalies(
        self, ticker: str, current_date: str, historical_dates: List[str]
    ) -> List[Dict]:
        """Detect unusual keyword frequency changes."""
        current_texts = self._load_section_text(ticker, current_date)
        if not current_texts:
            return []

        current_full_text = " ".join(current_texts.values()).lower()

        # Build historical baseline
        historical_counts: Dict[str, List[int]] = {}
        for hist_date in historical_dates:
            hist_texts = self._load_section_text(ticker, hist_date)
            hist_full = " ".join(hist_texts.values()).lower()
            for category, groups in TRACKED_KEYWORDS.items():
                for keyword_group, terms in groups.items():
                    key = f"{category}/{keyword_group}"
                    count = sum(hist_full.count(term.lower()) for term in terms)
                    historical_counts.setdefault(key, []).append(count)

        anomalies = []
        for category, groups in TRACKED_KEYWORDS.items():
            for keyword_group, terms in groups.items():
                key = f"{category}/{keyword_group}"
                current_count = sum(current_full_text.count(term.lower()) for term in terms)

                hist = historical_counts.get(key, [])
                if len(hist) < Settings.ANOMALY_MIN_HISTORICAL_COUNT:
                    continue

                avg_count = sum(hist) / len(hist)
                if avg_count == 0 and current_count > 0:
                    anomalies.append({
                        "type": "keyword_emergence",
                        "category": category.replace("_", " ").title(),
                        "title": f"New mentions of {keyword_group.replace('_', ' ')}",
                        "description": f"Found {current_count} mention(s) — absent in prior {len(hist)} filings",
                        "severity": "high" if current_count >= 5 else "medium",
                        "severity_reason": f"New keyword category with {current_count} mentions",
                        "current_count": current_count,
                        "average_count": avg_count,
                        "context": self._extract_context(current_full_text, terms),
                    })
                elif avg_count > 0:
                    ratio = current_count / avg_count
                    if ratio >= Settings.ANOMALY_FREQUENCY_MULTIPLIER:
                        anomalies.append({
                            "type": "keyword_spike",
                            "category": category.replace("_", " ").title(),
                            "title": f"Surge in {keyword_group.replace('_', ' ')} mentions",
                            "description": f"{current_count} mentions vs. avg {avg_count:.1f} ({ratio:.1f}x increase)",
                            "severity": "high" if ratio >= 5 else "medium",
                            "severity_reason": f"{ratio:.1f}x increase exceeds {Settings.ANOMALY_FREQUENCY_MULTIPLIER}x threshold",
                            "current_count": current_count,
                            "average_count": avg_count,
                            "ratio": ratio,
                            "context": self._extract_context(current_full_text, terms),
                        })
                    elif 0 < ratio <= 1 / Settings.ANOMALY_FREQUENCY_MULTIPLIER:
                        anomalies.append({
                            "type": "keyword_decline",
                            "category": category.replace("_", " ").title(),
                            "title": f"Sharp drop in {keyword_group.replace('_', ' ')} mentions",
                            "description": f"{current_count} mentions vs. avg {avg_count:.1f} ({1/ratio:.1f}x decrease)",
                            "severity": "medium",
                            "severity_reason": f"Mentions dropped to {1/ratio:.1f}x below threshold",
                            "current_count": current_count,
                            "average_count": avg_count,
                            "ratio": ratio,
                            "context": self._extract_context(current_full_text, terms),
                        })

        return anomalies

    def _extract_context(self, text: str, terms: List[str], max_sentences: int = 3) -> List[str]:
        """Extract sentences containing the tracked terms."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        matching = []
        for sentence in sentences:
            if any(term.lower() in sentence.lower() for term in terms):
                clean = sentence.strip()[:400]
                if clean and clean not in matching:
                    matching.append(clean)
                    if len(matching) >= max_sentences:
                        break
        return matching

    def enrich_with_llm(self, anomalies: List[Dict], ticker: str) -> List[Dict]:
        """Use Gemini to generate explanations for detected anomalies."""
        if not self.gemini_client or not anomalies:
            return anomalies

        for anomaly in anomalies:
            try:
                prompt = (
                    f"You are a financial analyst. A {anomaly['type']} anomaly was detected in "
                    f"{ticker}'s 10-K filing.\n\n"
                    f"Title: {anomaly['title']}\n"
                    f"Description: {anomaly['description']}\n"
                    f"Context: {'; '.join(anomaly.get('context', []))}\n\n"
                    f"In 2-3 sentences, explain why this matters for investors."
                )
                response = self.gemini_client.models.generate_content(
                    model=Settings.GEMINI_MODEL, contents=prompt
                )
                anomaly["explanation"] = response.text.strip()
            except Exception as e:
                logger.warning(f"LLM enrichment failed for anomaly: {e}")
                anomaly["explanation"] = "LLM analysis unavailable."

        return anomalies

    def analyze_ticker(self, ticker: str) -> Dict:
        """Full anomaly detection for a ticker."""
        ticker = ticker.upper()
        logger.info(f"Running anomaly detection for {ticker}")

        filings = self._get_sorted_filings(ticker)
        if len(filings) < 2:
            return {"ticker": ticker, "error": "Need 2+ filings for comparison", "anomalies": [], "total_anomalies": 0}

        current_file = filings[-1]
        previous_file = filings[-2]
        current_data = self._load_sentiment(current_file)
        previous_data = self._load_sentiment(previous_file)

        if not current_data or not previous_data:
            return {"ticker": ticker, "error": "Could not load sentiment data", "anomalies": [], "total_anomalies": 0}

        current_date = current_file.stem.split("_")[1]
        previous_date = previous_file.stem.split("_")[1]
        historical_dates = [f.stem.split("_")[1] for f in filings[:-1]]

        # Detect anomalies
        anomalies = []
        anomalies.extend(self.detect_sentiment_shifts(current_data, previous_data))
        anomalies.extend(self.detect_keyword_anomalies(ticker, current_date, historical_dates))

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        anomalies.sort(key=lambda a: severity_order.get(a.get("severity", "low"), 3))

        # Enrich with LLM explanations
        anomalies = self.enrich_with_llm(anomalies, ticker)

        report = {
            "ticker": ticker,
            "analysis_date": datetime.now().isoformat(),
            "current_filing_date": current_date,
            "compared_to": previous_date,
            "num_historical_filings": len(historical_dates),
            "total_anomalies": len(anomalies),
            "anomalies_by_severity": {
                "high": sum(1 for a in anomalies if a.get("severity") == "high"),
                "medium": sum(1 for a in anomalies if a.get("severity") == "medium"),
                "low": sum(1 for a in anomalies if a.get("severity") == "low"),
            },
            "anomalies": anomalies,
        }

        logger.info(f"Anomaly detection complete for {ticker}: {len(anomalies)} found")
        return report

    def save_report(self, report: Dict) -> Path:
        """Save anomaly report to disk."""
        ticker = report.get("ticker", "UNKNOWN")
        current_date = report.get("current_filing_date", "unknown")
        output_path = self.anomalies_dir / f"{ticker}_{current_date}_anomalies.json"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Saved anomaly report: {output_path.name}")
        return output_path
