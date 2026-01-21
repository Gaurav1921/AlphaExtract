"""
Enhanced Anomaly Detection Engine
---------------------------------
Detects unusual patterns by comparing current filings to historical data.
Now with rich context extraction and detailed comparison information.

Anomaly Types:
1. New keyword mentions (first appearance)
2. Sentiment shifts (compared to previous quarters)
3. Keyword frequency changes (5x increase/decrease)
4. Topic disappearances (mentioned before, now missing)
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
    Enhanced with context extraction and detailed descriptions.
    """
    
    TRACKED_KEYWORDS = {
        'regulation': ['regulation', 'regulatory', 'compliance', 'legal'],
        'tariff': ['tariff', 'trade war', 'import tax', 'duty', 'trade restriction'],
        'supply_chain': ['supply chain', 'logistics', 'supplier', 'manufacturing'],
        'competition': ['competition', 'competitive', 'competitor', 'rival'],
        'china': ['china', 'chinese', 'prc'],
        'ai': ['artificial intelligence', 'AI', 'machine learning', 'ML', 'deep learning', 'neural network'],
        'cybersecurity': ['cybersecurity', 'cyber security', 'data breach', 'hacking', 'ransomware', 'security incident'],
        'inflation': ['inflation', 'price increase', 'cost pressure', 'rising costs'],
        'recession': ['recession', 'economic downturn', 'slowdown', 'economic uncertainty'],
        'interest_rate': ['interest rate', 'fed rate', 'monetary policy', 'federal reserve']
    }
    
    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            data_dir = Path("data")
        
        self.data_dir = data_dir
        self.sentiment_dir = data_dir / "sentiment"
        self.sections_dir = data_dir / "sections"
        self.anomalies_dir = data_dir / "anomalies"
        self.anomalies_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Enhanced Anomaly Detector initialized")
    
    def load_sentiment_history(self, ticker: str) -> List[Dict]:
        """Load all sentiment files for a ticker."""
        files = list(self.sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
        
        history = []
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    data['filing_date'] = parts[1]
                history.append(data)
        
        history.sort(key=lambda x: x.get('filing_date', ''))
        return history
    
    def load_section_text(self, ticker: str, filing_date: str, section: str) -> Optional[str]:
        """Load text for a specific section."""
        pattern = f"{ticker}_{filing_date}_{section}.txt"
        files = list(self.sections_dir.glob(pattern))
        
        if not files:
            return None
        
        with open(files[0], 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_keywords_with_context(self, text: str, window_size: int = 100) -> Dict:
        """
        Extract keywords and their surrounding context.
        
        Returns:
            Dict with keyword counts and context snippets
        """
        text_lower = text.lower()
        results = {}
        
        for category, keywords in self.TRACKED_KEYWORDS.items():
            count = 0
            contexts = []
            
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    count += 1
                    # Extract context around the match
                    start = max(0, match.start() - window_size)
                    end = min(len(text), match.end() + window_size)
                    context = text[start:end].strip()
                    # Clean up context
                    context = ' '.join(context.split())
                    if len(contexts) < 5:  # Keep max 5 context snippets
                        contexts.append(f"...{context}...")
            
            if count > 0:
                results[category] = {
                    'count': count,
                    'contexts': contexts
                }
        
        return results
    
    def detect_sentiment_shift(
        self,
        current: Dict,
        history: List[Dict],
        threshold: float = 0.15
    ) -> List[Dict]:
        """Detect significant sentiment shifts with detailed comparison."""
        anomalies = []
        
        if not history:
            return anomalies
        
        previous = history[-1]
        historical_years = [h.get('filing_date', 'unknown')[:4] for h in history]
        
        current_score = current.get('overall', {}).get('compound', 0)
        previous_score = previous.get('overall', {}).get('compound', 0)
        
        delta = current_score - previous_score
        
        if abs(delta) >= threshold:
            direction = "more positive" if delta > 0 else "more negative"
            
            # Calculate historical average
            hist_scores = [h.get('overall', {}).get('compound', 0) for h in history]
            hist_avg = sum(hist_scores) / len(hist_scores) if hist_scores else 0
            
            anomalies.append({
                'type': 'sentiment_shift',
                'severity': 'high' if abs(delta) >= 0.3 else 'medium',
                'section': 'overall',
                'current_score': round(current_score, 3),
                'previous_score': round(previous_score, 3),
                'historical_average': round(hist_avg, 3),
                'delta': round(delta, 3),
                'description': f"Overall sentiment shifted {abs(delta):.2f} points {direction} (from {previous_score:+.2f} to {current_score:+.2f})",
                'comparison_details': f"Previous filing ({previous.get('filing_date', 'unknown')}): {previous_score:+.3f}\nHistorical average ({len(history)} filings): {hist_avg:+.3f}\nCurrent: {current_score:+.3f}",
                'historical_years': historical_years,
                'direction': direction
            })
        
        # Section-level analysis
        for section in ['item_1a', 'item_7', 'item_8']:
            if section not in current.get('sections', {}):
                continue
            if section not in previous.get('sections', {}):
                continue
            
            current_sec = current['sections'][section].get('scores', {}).get('compound', 0)
            previous_sec = previous['sections'][section].get('scores', {}).get('compound', 0)
            
            delta = current_sec - previous_sec
            
            if abs(delta) >= threshold:
                direction = "more positive" if delta > 0 else "more negative"
                section_name = {'item_1a': 'Risk Factors', 'item_7': 'MD&A', 'item_8': 'Financial Statements'}[section]
                
                anomalies.append({
                    'type': 'sentiment_shift',
                    'severity': 'medium',
                    'section': section,
                    'current_score': round(current_sec, 3),
                    'previous_score': round(previous_sec, 3),
                    'delta': round(delta, 3),
                    'description': f"{section_name} sentiment shifted {abs(delta):.2f} points {direction}",
                    'comparison_details': f"Previous: {previous_sec:+.3f} → Current: {current_sec:+.3f}",
                    'historical_years': historical_years,
                    'direction': direction
                })
        
        return anomalies
    
    def detect_new_keywords(
        self,
        current_keywords: Dict,
        historical_keywords: List[Dict],
        current_date: str
    ) -> List[Dict]:
        """Detect newly appearing keywords with context."""
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        historical_years = list(set(h.get('_filing_date', 'unknown')[:4] for h in historical_keywords if '_filing_date' in h))
        
        historical_set = set()
        for hist in historical_keywords:
            historical_set.update(hist.keys())
        historical_set.discard('_filing_date')
        
        current_set = set(current_keywords.keys())
        new_keywords = current_set - historical_set
        
        for keyword in new_keywords:
            kw_data = current_keywords[keyword]
            count = kw_data['count'] if isinstance(kw_data, dict) else kw_data
            contexts = kw_data.get('contexts', []) if isinstance(kw_data, dict) else []
            
            anomalies.append({
                'type': 'new_keyword',
                'severity': 'high',
                'keyword': keyword,
                'count': count,
                'description': f"First mention of '{keyword}' in {current_date[:4]} filing ({count} occurrences) - not found in {len(historical_keywords)} prior filings",
                'comparison_details': f"This keyword was NOT mentioned in any of the {len(historical_keywords)} historical filings analyzed.\nFirst appearance in {current_date} with {count} mentions.",
                'context_snippets': contexts[:3],
                'historical_years': historical_years
            })
        
        return anomalies
    
    def detect_frequency_changes(
        self,
        current_keywords: Dict,
        historical_keywords: List[Dict],
        current_date: str,
        multiplier: float = 3.0
    ) -> List[Dict]:
        """Detect significant frequency changes with context."""
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        historical_years = list(set(h.get('_filing_date', 'unknown')[:4] for h in historical_keywords if '_filing_date' in h))
        
        # Calculate average historical frequency
        avg_historical = Counter()
        for hist in historical_keywords:
            for key, value in hist.items():
                if key == '_filing_date':
                    continue
                count = value['count'] if isinstance(value, dict) else value
                avg_historical[key] += count
        
        num_filings = len(historical_keywords)
        for key in avg_historical:
            avg_historical[key] /= num_filings
        
        # Compare current to average
        for keyword, kw_data in current_keywords.items():
            if keyword not in avg_historical:
                continue
            
            current_count = kw_data['count'] if isinstance(kw_data, dict) else kw_data
            contexts = kw_data.get('contexts', []) if isinstance(kw_data, dict) else []
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
                    'average_count': round(avg_count, 1),
                    'ratio': round(ratio, 1),
                    'description': f"'{keyword}' mentioned {ratio:.1f}x more than historical average ({current_count} vs avg {avg_count:.0f}) in {current_date[:4]}",
                    'comparison_details': f"Current filing ({current_date}): {current_count} mentions\nHistorical average ({num_filings} filings from {', '.join(historical_years)}): {avg_count:.1f} mentions\nThis represents a {ratio:.1f}x increase",
                    'context_snippets': contexts[:3],
                    'historical_years': historical_years
                })
            elif ratio <= 1/multiplier:
                anomalies.append({
                    'type': 'frequency_decrease',
                    'severity': 'medium',
                    'keyword': keyword,
                    'current_count': current_count,
                    'average_count': round(avg_count, 1),
                    'ratio': round(ratio, 2),
                    'description': f"'{keyword}' mentioned {1/ratio:.1f}x less than historical average ({current_count} vs avg {avg_count:.0f})",
                    'comparison_details': f"Current: {current_count} mentions vs Historical average: {avg_count:.1f} mentions ({num_filings} filings)",
                    'context_snippets': contexts[:3],
                    'historical_years': historical_years
                })
        
        return anomalies
    
    def detect_missing_topics(
        self,
        current_keywords: Dict,
        historical_keywords: List[Dict],
        min_historical_count: int = 5
    ) -> List[Dict]:
        """Detect topics that disappeared."""
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        historical_years = list(set(h.get('_filing_date', 'unknown')[:4] for h in historical_keywords if '_filing_date' in h))
        
        historical_totals = Counter()
        for hist in historical_keywords:
            for key, value in hist.items():
                if key == '_filing_date':
                    continue
                count = value['count'] if isinstance(value, dict) else value
                historical_totals[key] += count
        
        current_set = set(current_keywords.keys())
        
        for keyword, count in historical_totals.items():
            if count >= min_historical_count and keyword not in current_set:
                anomalies.append({
                    'type': 'missing_topic',
                    'severity': 'medium',
                    'keyword': keyword,
                    'historical_count': count,
                    'description': f"'{keyword}' no longer mentioned - previously appeared {count} times across {len(historical_keywords)} filings",
                    'comparison_details': f"Total mentions in prior filings: {count}\nYears analyzed: {', '.join(historical_years)}\nCurrent filing: 0 mentions",
                    'historical_years': historical_years
                })
        
        return anomalies
    
    def analyze_ticker(
        self,
        ticker: str,
        current_filing_date: Optional[str] = None
    ) -> Dict:
        """Complete anomaly analysis for a ticker."""
        logger.info(f"\n{'='*80}")
        logger.info(f"ANOMALY DETECTION: {ticker}")
        logger.info(f"{'='*80}")
        
        sentiment_history = self.load_sentiment_history(ticker)
        
        if len(sentiment_history) < 2:
            logger.warning(f"Need at least 2 filings for {ticker} (found {len(sentiment_history)})")
            return {
                'ticker': ticker,
                'error': f'Insufficient historical data - found {len(sentiment_history)} filing(s), need at least 2',
                'anomalies': []
            }
        
        current = sentiment_history[-1]
        history = sentiment_history[:-1]
        
        current_date = current.get('filing_date', 'unknown')
        logger.info(f"Current filing: {current_date}")
        logger.info(f"Comparing to {len(history)} historical filings")
        
        # Extract keywords with context
        current_keywords = {}
        historical_keywords = []
        
        for section in ['item_1a', 'item_7']:
            text = self.load_section_text(ticker, current_date, section)
            if text:
                section_kw = self.extract_keywords_with_context(text)
                for kw, data in section_kw.items():
                    if kw in current_keywords:
                        current_keywords[kw]['count'] += data['count']
                        current_keywords[kw]['contexts'].extend(data['contexts'])
                    else:
                        current_keywords[kw] = data
        
        for hist in history:
            hist_date = hist.get('filing_date', '')
            hist_keywords = {'_filing_date': hist_date}
            
            for section in ['item_1a', 'item_7']:
                text = self.load_section_text(ticker, hist_date, section)
                if text:
                    section_kw = self.extract_keywords_with_context(text)
                    for kw, data in section_kw.items():
                        if kw in hist_keywords:
                            hist_keywords[kw]['count'] += data['count']
                        else:
                            hist_keywords[kw] = data
            
            if len(hist_keywords) > 1:
                historical_keywords.append(hist_keywords)
        
        # Detect anomalies
        all_anomalies = []
        
        logger.info("\n[1] Detecting sentiment shifts...")
        sentiment_anomalies = self.detect_sentiment_shift(current, history)
        all_anomalies.extend(sentiment_anomalies)
        
        logger.info("\n[2] Detecting new keyword mentions...")
        new_keyword_anomalies = self.detect_new_keywords(current_keywords, historical_keywords, current_date)
        all_anomalies.extend(new_keyword_anomalies)
        
        logger.info("\n[3] Detecting frequency changes...")
        frequency_anomalies = self.detect_frequency_changes(current_keywords, historical_keywords, current_date)
        all_anomalies.extend(frequency_anomalies)
        
        logger.info("\n[4] Detecting missing topics...")
        missing_anomalies = self.detect_missing_topics(current_keywords, historical_keywords)
        all_anomalies.extend(missing_anomalies)
        
        # Compile report
        report = {
            'ticker': ticker,
            'current_filing_date': current_date,
            'num_historical_filings': len(history),
            'historical_dates': [h.get('filing_date', 'unknown') for h in history],
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
        logger.info(f"{'='*80}")
        
        return report
    
    def save_report(self, report: Dict, output_dir: Path = None):
        """Save anomaly report to JSON."""
        if output_dir is None:
            output_dir = self.anomalies_dir
        
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
        print(f"Compared to: {report['num_historical_filings']} historical filings")
        print(f"Total Anomalies: {report['total_anomalies']}")
        
        if not report['anomalies']:
            print("\nNo anomalies detected - filing appears normal.")
            return
        
        by_type = {}
        for anomaly in report['anomalies']:
            atype = anomaly['type']
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(anomaly)
        
        for atype, anomalies in by_type.items():
            print(f"\n{atype.upper().replace('_', ' ')} ({len(anomalies)}):")
            print("-" * 80)
            
            for i, anomaly in enumerate(anomalies[:5], 1):
                severity = anomaly.get('severity', 'medium')
                icon = '🔴' if severity == 'high' else '🟡'
                print(f"{icon} {i}. {anomaly['description']}")
                if 'comparison_details' in anomaly:
                    print(f"   {anomaly['comparison_details']}")


if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED ANOMALY DETECTION ENGINE")
    print("=" * 80)
    
    detector = AnomalyDetector()
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    for ticker in tickers:
        try:
            report = detector.analyze_ticker(ticker)
            detector.print_report(report)
            detector.save_report(report)
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
    
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION COMPLETE")
    print("=" * 80)