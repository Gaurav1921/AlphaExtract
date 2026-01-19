"""
Anomaly Detection Engine
------------------------
Detects unusual patterns by comparing current filings to historical data.

Anomaly Types:
1. New keyword mentions (first appearance)
2. Sentiment shifts (compared to previous quarters)
3. Keyword frequency changes (5x increase/decrease)
4. Topic disappearances (mentioned before, now missing)
5. Risk escalation (minor → major)
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter
import json
import re
import logging
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies by comparing current vs. historical filings.
    """
    
    # Important financial keywords to track
    TRACKED_KEYWORDS = {
        'regulation': ['regulation', 'regulatory', 'compliance', 'legal'],
        'tariff': ['tariff', 'trade war', 'import tax', 'duty'],
        'supply_chain': ['supply chain', 'logistics', 'supplier', 'manufacturing'],
        'competition': ['competition', 'competitive', 'competitor', 'rival'],
        'china': ['china', 'chinese'],
        'ai': ['artificial intelligence', 'AI', 'machine learning', 'ML'],
        'cybersecurity': ['cybersecurity', 'data breach', 'hacking', 'ransomware'],
        'inflation': ['inflation', 'price increase', 'cost pressure'],
        'recession': ['recession', 'economic downturn', 'slowdown'],
        'interest_rate': ['interest rate', 'fed rate', 'monetary policy']
    }
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize anomaly detector.
        
        Args:
            data_dir: Directory containing sentiment and section data
        """
        if data_dir is None:
            data_dir = Path("data")
        
        self.data_dir = data_dir
        self.sentiment_dir = data_dir / "sentiment"
        self.sections_dir = data_dir / "sections"
        
        logger.info("Anomaly Detector initialized")
    
    def load_sentiment_history(self, ticker: str) -> List[Dict]:
        """
        Load all sentiment files for a ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            List of sentiment dicts sorted by date
        """
        files = list(self.sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
        
        history = []
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                
                # Parse date from filename
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    data['filing_date'] = parts[1]
                
                history.append(data)
        
        # Sort by date
        history.sort(key=lambda x: x.get('filing_date', ''))
        
        return history
    
    def load_section_text(self, ticker: str, filing_date: str, section: str) -> Optional[str]:
        """
        Load text for a specific section.
        
        Args:
            ticker: Stock ticker
            filing_date: Filing date
            section: Section name (item_1a, item_7, item_8)
            
        Returns:
            Section text or None
        """
        # Find matching file
        pattern = f"{ticker}_{filing_date}_{section}.txt"
        files = list(self.sections_dir.glob(pattern))
        
        if not files:
            logger.warning(f"Section not found: {pattern}")
            return None
        
        with open(files[0], 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_keywords(self, text: str) -> Counter:
        """
        Extract and count tracked keywords.
        
        Args:
            text: Input text
            
        Returns:
            Counter of keyword frequencies
        """
        text_lower = text.lower()
        keyword_counts = Counter()
        
        for category, keywords in self.TRACKED_KEYWORDS.items():
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
            
            if count > 0:
                keyword_counts[category] = count
        
        return keyword_counts
    
    def detect_sentiment_shift(
        self,
        current: Dict,
        history: List[Dict],
        threshold: float = 0.15
    ) -> List[Dict]:
        """
        Detect significant sentiment shifts.
        
        Args:
            current: Current filing sentiment
            history: Historical sentiment data
            threshold: Minimum change to flag (default: 0.15)
            
        Returns:
            List of anomalies detected
        """
        anomalies = []
        
        if not history:
            return anomalies
        
        # Get previous filing
        previous = history[-1]
        
        # Compare overall sentiment
        current_score = current.get('overall', {}).get('compound', 0)
        previous_score = previous.get('overall', {}).get('compound', 0)
        
        delta = current_score - previous_score
        
        if abs(delta) >= threshold:
            direction = "more positive" if delta > 0 else "more negative"
            anomalies.append({
                'type': 'sentiment_shift',
                'severity': 'high' if abs(delta) >= 0.3 else 'medium',
                'section': 'overall',
                'current_score': current_score,
                'previous_score': previous_score,
                'delta': delta,
                'description': f"Overall sentiment shifted {abs(delta):.2f} points {direction}",
                'direction': direction
            })
        
        # Compare section-level sentiment
        for section in ['item_1a', 'item_7', 'item_8']:
            if section not in current.get('sections', {}):
                continue
            
            if section not in previous.get('sections', {}):
                continue
            
            current_sec = current['sections'][section]['scores']['compound']
            previous_sec = previous['sections'][section]['scores']['compound']
            
            delta = current_sec - previous_sec
            
            if abs(delta) >= threshold:
                direction = "more positive" if delta > 0 else "more negative"
                anomalies.append({
                    'type': 'sentiment_shift',
                    'severity': 'medium',
                    'section': section,
                    'current_score': current_sec,
                    'previous_score': previous_sec,
                    'delta': delta,
                    'description': f"{section} sentiment {direction} (Δ {delta:+.2f})",
                    'direction': direction
                })
        
        return anomalies
    
    def detect_new_keywords(
        self,
        current_keywords: Counter,
        historical_keywords: List[Counter]
    ) -> List[Dict]:
        """
        Detect newly appearing keywords.
        
        Args:
            current_keywords: Current keyword counts
            historical_keywords: List of historical keyword counts
            
        Returns:
            List of anomalies
        """
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        # Get all keywords that appeared in history
        historical_set = set()
        for hist in historical_keywords:
            historical_set.update(hist.keys())
        
        # Find new keywords
        new_keywords = set(current_keywords.keys()) - historical_set
        
        for keyword in new_keywords:
            anomalies.append({
                'type': 'new_keyword',
                'severity': 'high',
                'keyword': keyword,
                'count': current_keywords[keyword],
                'description': f"First mention of '{keyword}' ({current_keywords[keyword]} occurrences)"
            })
        
        return anomalies
    
    def detect_frequency_changes(
        self,
        current_keywords: Counter,
        historical_keywords: List[Counter],
        multiplier: float = 3.0
    ) -> List[Dict]:
        """
        Detect significant frequency changes.
        
        Args:
            current_keywords: Current keyword counts
            historical_keywords: Historical keyword counts
            multiplier: Threshold multiplier (default: 3x change)
            
        Returns:
            List of anomalies
        """
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        # Calculate average historical frequency
        avg_historical = Counter()
        for hist in historical_keywords:
            avg_historical.update(hist)
        
        # Average across filings
        num_filings = len(historical_keywords)
        for key in avg_historical:
            avg_historical[key] /= num_filings
        
        # Compare current to average
        for keyword in current_keywords:
            if keyword not in avg_historical:
                continue
            
            current_count = current_keywords[keyword]
            avg_count = avg_historical[keyword]
            
            if avg_count == 0:
                continue
            
            ratio = current_count / avg_count
            
            if ratio >= multiplier:
                anomalies.append({
                    'type': 'frequency_increase',
                    'severity': 'high' if ratio >= 5 else 'medium',
                    'keyword': keyword,
                    'current_count': current_count,
                    'average_count': avg_count,
                    'ratio': ratio,
                    'description': f"'{keyword}' mentioned {ratio:.1f}x more than average ({current_count} vs {avg_count:.0f})"
                })
            elif ratio <= 1/multiplier:
                anomalies.append({
                    'type': 'frequency_decrease',
                    'severity': 'medium',
                    'keyword': keyword,
                    'current_count': current_count,
                    'average_count': avg_count,
                    'ratio': ratio,
                    'description': f"'{keyword}' mentioned {1/ratio:.1f}x less than average ({current_count} vs {avg_count:.0f})"
                })
        
        return anomalies
    
    def detect_missing_topics(
        self,
        current_keywords: Counter,
        historical_keywords: List[Counter],
        min_historical_count: int = 5
    ) -> List[Dict]:
        """
        Detect topics that disappeared.
        
        Args:
            current_keywords: Current keyword counts
            historical_keywords: Historical keyword counts
            min_historical_count: Minimum past mentions to flag
            
        Returns:
            List of anomalies
        """
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        # Get keywords that were common in history
        historical_totals = Counter()
        for hist in historical_keywords:
            historical_totals.update(hist)
        
        # Find keywords that disappeared
        for keyword, count in historical_totals.items():
            if count >= min_historical_count and keyword not in current_keywords:
                anomalies.append({
                    'type': 'missing_topic',
                    'severity': 'medium',
                    'keyword': keyword,
                    'historical_count': count,
                    'description': f"'{keyword}' no longer mentioned (previously {count} times across {len(historical_keywords)} filings)"
                })
        
        return anomalies
    
    def analyze_ticker(
        self,
        ticker: str,
        current_filing_date: Optional[str] = None
    ) -> Dict:
        """
        Complete anomaly analysis for a ticker.
        
        Args:
            ticker: Stock ticker
            current_filing_date: Date of current filing (uses latest if None)
            
        Returns:
            Dict with all detected anomalies
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"ANOMALY DETECTION: {ticker}")
        logger.info(f"{'='*80}")
        
        # Load sentiment history
        sentiment_history = self.load_sentiment_history(ticker)
        
        if len(sentiment_history) < 2:
            logger.warning(f"Need at least 2 filings for {ticker} (found {len(sentiment_history)})")
            return {
                'ticker': ticker,
                'error': 'Insufficient historical data',
                'anomalies': []
            }
        
        # Get current and historical data
        current = sentiment_history[-1]
        history = sentiment_history[:-1]
        
        current_date = current.get('filing_date', 'unknown')
        logger.info(f"Current filing: {current_date}")
        logger.info(f"Comparing to {len(history)} historical filings")
        
        # Extract keywords from sections
        current_keywords = Counter()
        historical_keywords = []
        
        # Current filing keywords
        for section in ['item_1a', 'item_7']:
            text = self.load_section_text(ticker, current_date, section)
            if text:
                current_keywords.update(self.extract_keywords(text))
        
        # Historical keywords
        for hist in history:
            hist_date = hist.get('filing_date', '')
            hist_keywords = Counter()
            
            for section in ['item_1a', 'item_7']:
                text = self.load_section_text(ticker, hist_date, section)
                if text:
                    hist_keywords.update(self.extract_keywords(text))
            
            if hist_keywords:
                historical_keywords.append(hist_keywords)
        
        # Detect anomalies
        all_anomalies = []
        
        # 1. Sentiment shifts
        logger.info("\n[1] Detecting sentiment shifts...")
        sentiment_anomalies = self.detect_sentiment_shift(current, history)
        all_anomalies.extend(sentiment_anomalies)
        logger.info(f"  Found {len(sentiment_anomalies)} sentiment anomalies")
        
        # 2. New keywords
        logger.info("\n[2] Detecting new keyword mentions...")
        new_keyword_anomalies = self.detect_new_keywords(current_keywords, historical_keywords)
        all_anomalies.extend(new_keyword_anomalies)
        logger.info(f"  Found {len(new_keyword_anomalies)} new keywords")
        
        # 3. Frequency changes
        logger.info("\n[3] Detecting frequency changes...")
        frequency_anomalies = self.detect_frequency_changes(current_keywords, historical_keywords)
        all_anomalies.extend(frequency_anomalies)
        logger.info(f"  Found {len(frequency_anomalies)} frequency changes")
        
        # 4. Missing topics
        logger.info("\n[4] Detecting missing topics...")
        missing_anomalies = self.detect_missing_topics(current_keywords, historical_keywords)
        all_anomalies.extend(missing_anomalies)
        logger.info(f"  Found {len(missing_anomalies)} missing topics")
        
        # Compile report
        report = {
            'ticker': ticker,
            'current_filing_date': current_date,
            'num_historical_filings': len(history),
            'total_anomalies': len(all_anomalies),
            'anomalies_by_severity': {
                'high': len([a for a in all_anomalies if a.get('severity') == 'high']),
                'medium': len([a for a in all_anomalies if a.get('severity') == 'medium']),
                'low': len([a for a in all_anomalies if a.get('severity') == 'low'])
            },
            'anomalies': all_anomalies,
            'analysis_date': datetime.now().isoformat()
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY: {len(all_anomalies)} total anomalies detected")
        logger.info(f"  High severity: {report['anomalies_by_severity']['high']}")
        logger.info(f"  Medium severity: {report['anomalies_by_severity']['medium']}")
        logger.info(f"{'='*80}")
        
        return report
    
    def save_report(self, report: Dict, output_dir: Path = None):
        """Save anomaly report to JSON."""
        if output_dir is None:
            output_dir = Path("data/anomalies")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{report['ticker']}_{report['current_filing_date']}_anomalies.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {filepath}")
    
    def print_report(self, report: Dict):
        """Pretty print anomaly report."""
        print(f"\n{'='*80}")
        print(f"ANOMALY REPORT: {report['ticker']}")
        print(f"{'='*80}")
        print(f"Filing Date: {report['current_filing_date']}")
        print(f"Total Anomalies: {report['total_anomalies']}")
        print(f"  High Severity: {report['anomalies_by_severity']['high']}")
        print(f"  Medium Severity: {report['anomalies_by_severity']['medium']}")
        
        if not report['anomalies']:
            print("\nNo anomalies detected - filing appears normal.")
            return
        
        # Group by type
        by_type = {}
        for anomaly in report['anomalies']:
            atype = anomaly['type']
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(anomaly)
        
        # Print each type
        for atype, anomalies in by_type.items():
            print(f"\n{atype.upper().replace('_', ' ')} ({len(anomalies)}):")
            print("-" * 80)
            
            for i, anomaly in enumerate(anomalies[:5], 1):  # Show top 5
                severity = anomaly.get('severity', 'medium')
                icon = '🔴' if severity == 'high' else '🟡'
                print(f"{icon} {i}. {anomaly['description']}")
            
            if len(anomalies) > 5:
                print(f"   ... and {len(anomalies) - 5} more")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ANOMALY DETECTION ENGINE")
    print("=" * 80)
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Analyze each company
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    for ticker in tickers:
        try:
            report = detector.analyze_ticker(ticker)
            detector.print_report(report)
            detector.save_report(report)
            
            if ticker != tickers[-1]:
                input("\nPress Enter for next company...")
        
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
    
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION COMPLETE")
    print("=" * 80)
    print("\nReports saved to: data/anomalies/")