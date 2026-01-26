"""
Enhanced Anomaly Detection Engine - v0.5.1
------------------------------------------
LOCATION: src/models/anomaly.py

Detects unusual patterns by comparing current filings to historical data.
Now includes detailed explanations and context extraction.

Anomaly Types:
1. New keyword mentions (first appearance)
2. Sentiment shifts (compared to previous quarters)
3. Keyword frequency changes (3x+ increase/decrease)
4. Topic disappearances (mentioned before, now missing)

CHANGES in v0.5.1:
- Removed year comparison (simplified to latest vs history)
- Added KEYWORD_EXPLANATIONS for why each keyword matters
- Added context extraction (actual sentences from filing)
- Enhanced anomaly descriptions with severity reasoning
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import json
import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies by comparing current vs. historical filings.
    Enhanced with explanations and context extraction.
    """
    
    # ========================================================================
    # KEYWORD DEFINITIONS WITH EXPLANATIONS
    # ========================================================================
    
    TRACKED_KEYWORDS = {
        'regulation': {
            'terms': ['regulation', 'regulatory', 'compliance', 'legal requirement'],
            'explanation': 'Increased regulatory discussion often signals compliance concerns, '
                          'potential fines, or new rules affecting operations.',
            'category': 'Regulatory Risk'
        },
        'tariff': {
            'terms': ['tariff', 'trade war', 'import tax', 'duty', 'trade restriction'],
            'explanation': 'Tariff mentions indicate exposure to trade policy changes, '
                          'which can impact margins and supply chain costs.',
            'category': 'Trade Policy'
        },
        'supply_chain': {
            'terms': ['supply chain', 'logistics', 'supplier', 'manufacturing disruption', 'shortage'],
            'explanation': 'Supply chain discussions highlight operational dependencies '
                          'and potential production or delivery risks.',
            'category': 'Operational Risk'
        },
        'competition': {
            'terms': ['competition', 'competitive', 'competitor', 'rival', 'market share'],
            'explanation': 'Increased competitive language may signal market pressure, '
                          'pricing challenges, or threats to market position.',
            'category': 'Market Risk'
        },
        'china': {
            'terms': ['china', 'chinese', 'beijing', 'prc'],
            'explanation': 'China-related mentions often involve supply chain exposure, '
                          'regulatory risks, or geopolitical concerns.',
            'category': 'Geopolitical Risk'
        },
        'ai': {
            'terms': ['artificial intelligence', ' ai ', 'machine learning', 'deep learning', 'generative ai'],
            'explanation': 'AI mentions reflect competitive positioning in technology, '
                          'R&D investment priorities, or disruption concerns.',
            'category': 'Technology'
        },
        'cybersecurity': {
            'terms': ['cybersecurity', 'cyber security', 'data breach', 'hacking', 'ransomware', 'cyber attack'],
            'explanation': 'Cybersecurity discussion indicates exposure to operational '
                          'disruption, data protection costs, and compliance requirements.',
            'category': 'Security Risk'
        },
        'inflation': {
            'terms': ['inflation', 'inflationary', 'price increase', 'cost pressure', 'rising costs'],
            'explanation': 'Inflation mentions signal margin pressure from rising input '
                          'costs, labor, or operational expenses.',
            'category': 'Economic Risk'
        },
        'recession': {
            'terms': ['recession', 'economic downturn', 'slowdown', 'economic uncertainty', 'demand weakness'],
            'explanation': 'Recession language indicates concern about consumer demand, '
                          'revenue stability, or macroeconomic exposure.',
            'category': 'Economic Risk'
        },
        'interest_rate': {
            'terms': ['interest rate', 'fed rate', 'monetary policy', 'borrowing cost', 'debt service'],
            'explanation': 'Interest rate discussion signals exposure to financing costs, '
                          'debt refinancing risks, or capital allocation challenges.',
            'category': 'Financial Risk'
        },
        'layoffs': {
            'terms': ['layoff', 'workforce reduction', 'restructuring', 'headcount', 'job cut', 'severance'],
            'explanation': 'Layoff mentions may indicate cost-cutting measures, '
                          'operational restructuring, or financial stress.',
            'category': 'Operational'
        },
        'acquisition': {
            'terms': ['acquisition', 'merger', 'acquire', 'takeover', 'deal', 'transaction'],
            'explanation': 'Acquisition discussion reflects growth strategy, integration '
                          'risks, or potential changes in company structure.',
            'category': 'Corporate Strategy'
        },
        'lawsuit': {
            'terms': ['lawsuit', 'litigation', 'legal proceedings', 'settlement', 'plaintiff', 'defendant'],
            'explanation': 'Litigation mentions indicate legal exposure, potential '
                          'financial liabilities, or reputational risks.',
            'category': 'Legal Risk'
        },
        'currency': {
            'terms': ['currency', 'foreign exchange', 'forex', 'fx risk', 'exchange rate', 'hedging'],
            'explanation': 'Currency discussion signals exposure to exchange rate '
                          'fluctuations affecting international revenue or costs.',
            'category': 'Financial Risk'
        },
        'climate': {
            'terms': ['climate', 'environmental', 'sustainability', 'carbon', 'emissions', 'esg'],
            'explanation': 'Climate and ESG mentions reflect regulatory compliance, '
                          'reputational positioning, or operational sustainability.',
            'category': 'ESG Risk'
        }
    }
    
    def __init__(self, data_dir: Path = None):
        """Initialize anomaly detector."""
        if data_dir is None:
            data_dir = Path("data")
        
        self.data_dir = data_dir
        self.sentiment_dir = data_dir / "sentiment"
        self.sections_dir = data_dir / "sections"
        self.output_dir = data_dir / "anomalies"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Enhanced Anomaly Detector initialized")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    def load_sentiment_history(self, ticker: str) -> List[Dict]:
        """Load all sentiment files for a ticker, sorted by date."""
        files = list(self.sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
        
        history = []
        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Parse date from filename
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    data['filing_date'] = parts[1]
                    data['_file'] = str(file)
                
                history.append(data)
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")
        
        # Sort by date (oldest first)
        history.sort(key=lambda x: x.get('filing_date', ''))
        
        return history
    
    def load_section_text(self, ticker: str, filing_date: str, section: str) -> Optional[str]:
        """Load text for a specific section."""
        # Try different filename formats
        patterns = [
            f"{ticker}_{filing_date}_{section}.txt",
            f"{ticker}_{filing_date}_item_{section.replace('item_', '')}.txt",
        ]
        
        for pattern in patterns:
            files = list(self.sections_dir.glob(pattern))
            if files:
                with open(files[0], 'r', encoding='utf-8') as f:
                    return f.read()
        
        return None
    
    # ========================================================================
    # KEYWORD EXTRACTION AND CONTEXT
    # ========================================================================
    
    def extract_keywords(self, text: str) -> Counter:
        """Extract and count tracked keywords."""
        text_lower = text.lower()
        keyword_counts = Counter()
        
        for category, config in self.TRACKED_KEYWORDS.items():
            count = 0
            for term in config['terms']:
                # Use word boundaries for accurate matching
                pattern = r'\b' + re.escape(term.strip().lower()) + r'\b'
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            
            if count > 0:
                keyword_counts[category] = count
        
        return keyword_counts
    
    def extract_context_sentences(
        self, 
        text: str, 
        keyword_category: str, 
        max_sentences: int = 2
    ) -> List[str]:
        """
        Extract sentences containing the keyword for context.
        
        Args:
            text: Full section text
            keyword_category: Category key from TRACKED_KEYWORDS
            max_sentences: Maximum sentences to return
            
        Returns:
            List of relevant sentences
        """
        if keyword_category not in self.TRACKED_KEYWORDS:
            return []
        
        terms = self.TRACKED_KEYWORDS[keyword_category]['terms']
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if any term appears in sentence
            for term in terms:
                if term.strip().lower() in sentence_lower:
                    # Clean up the sentence
                    clean_sentence = ' '.join(sentence.split())
                    if len(clean_sentence) > 50 and len(clean_sentence) < 500:
                        relevant_sentences.append(clean_sentence)
                        break
            
            if len(relevant_sentences) >= max_sentences:
                break
        
        return relevant_sentences
    
    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================
    
    def detect_sentiment_shift(
        self,
        current: Dict,
        previous: Dict,
        threshold: float = 0.15
    ) -> List[Dict]:
        """Detect significant sentiment shifts between filings."""
        anomalies = []
        
        # Compare overall sentiment
        current_score = current.get('overall', {}).get('compound', 0)
        previous_score = previous.get('overall', {}).get('compound', 0)
        
        delta = current_score - previous_score
        
        if abs(delta) >= threshold:
            direction = "improved" if delta > 0 else "declined"
            
            # Determine severity
            if abs(delta) >= 0.3:
                severity = 'high'
                severity_reason = f"Large shift of {abs(delta):.0%} indicates significant change in tone"
            else:
                severity = 'medium'
                severity_reason = f"Moderate shift of {abs(delta):.0%} suggests evolving outlook"
            
            anomalies.append({
                'type': 'sentiment_shift',
                'category': 'Sentiment Analysis',
                'severity': severity,
                'severity_reason': severity_reason,
                'title': f"Overall Sentiment {direction.title()}",
                'description': f"Overall sentiment {direction} by {abs(delta):.2f} points",
                'explanation': f"The company's overall tone in this filing is notably "
                              f"{'more optimistic' if delta > 0 else 'more cautious'} compared to "
                              f"the previous filing. This may reflect changes in business outlook, "
                              f"market conditions, or management confidence.",
                'current_score': round(current_score, 3),
                'previous_score': round(previous_score, 3),
                'delta': round(delta, 3),
                'direction': direction
            })
        
        return anomalies
    
    def detect_new_keywords(
        self,
        current_keywords: Counter,
        historical_keywords: List[Counter],
        section_text: str = None
    ) -> List[Dict]:
        """Detect newly appearing keywords."""
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        # Get all keywords that appeared in any historical filing
        historical_set = set()
        for hist in historical_keywords:
            historical_set.update(hist.keys())
        
        # Find new keywords
        new_keywords = set(current_keywords.keys()) - historical_set
        
        for keyword in new_keywords:
            config = self.TRACKED_KEYWORDS.get(keyword, {})
            category = config.get('category', 'General')
            explanation = config.get('explanation', 'This topic was not discussed in previous filings.')
            
            # Extract context if section text provided
            context = []
            if section_text:
                context = self.extract_context_sentences(section_text, keyword)
            
            anomalies.append({
                'type': 'new_keyword',
                'category': category,
                'severity': 'high',
                'severity_reason': 'First-time mention indicates new area of focus or concern',
                'title': f"NEW: {keyword.replace('_', ' ').title()}",
                'description': f"First mention of '{keyword.replace('_', ' ')}' ({current_keywords[keyword]} occurrences)",
                'explanation': explanation,
                'keyword': keyword,
                'count': current_keywords[keyword],
                'context': context
            })
        
        return anomalies
    
    def detect_frequency_changes(
        self,
        current_keywords: Counter,
        historical_keywords: List[Counter],
        section_text: str = None,
        multiplier: float = 3.0
    ) -> List[Dict]:
        """Detect significant keyword frequency changes."""
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        # Calculate average historical frequency
        avg_historical = Counter()
        for hist in historical_keywords:
            avg_historical.update(hist)
        
        num_filings = len(historical_keywords)
        for key in avg_historical:
            avg_historical[key] /= num_filings
        
        # Compare current to average
        for keyword in current_keywords:
            if keyword not in avg_historical:
                continue
            
            current_count = current_keywords[keyword]
            avg_count = avg_historical[keyword]
            
            if avg_count < 1:
                continue
            
            ratio = current_count / avg_count
            
            config = self.TRACKED_KEYWORDS.get(keyword, {})
            category = config.get('category', 'General')
            explanation = config.get('explanation', '')
            
            # Extract context
            context = []
            if section_text:
                context = self.extract_context_sentences(section_text, keyword)
            
            if ratio >= multiplier:
                # Significant increase
                if ratio >= 5:
                    severity = 'high'
                    severity_reason = f'{ratio:.1f}x increase suggests major new focus area'
                else:
                    severity = 'medium'
                    severity_reason = f'{ratio:.1f}x increase indicates growing importance'
                
                anomalies.append({
                    'type': 'frequency_increase',
                    'category': category,
                    'severity': severity,
                    'severity_reason': severity_reason,
                    'title': f"{keyword.replace('_', ' ').title()} - Frequency Spike",
                    'description': f"'{keyword.replace('_', ' ')}' mentioned {ratio:.1f}x more ({current_count} vs {avg_count:.0f} previously)",
                    'explanation': explanation,
                    'keyword': keyword,
                    'current_count': current_count,
                    'average_count': round(avg_count, 1),
                    'ratio': round(ratio, 1),
                    'context': context
                })
            
            elif ratio <= 1/multiplier:
                # Significant decrease
                anomalies.append({
                    'type': 'frequency_decrease',
                    'category': category,
                    'severity': 'low',
                    'severity_reason': f'Topic mentioned {1/ratio:.1f}x less than usual',
                    'title': f"{keyword.replace('_', ' ').title()} - Reduced Focus",
                    'description': f"'{keyword.replace('_', ' ')}' mentioned {1/ratio:.1f}x less ({current_count} vs {avg_count:.0f} previously)",
                    'explanation': f"Reduced discussion of this topic may indicate resolved concerns or shifted priorities.",
                    'keyword': keyword,
                    'current_count': current_count,
                    'average_count': round(avg_count, 1),
                    'ratio': round(ratio, 2),
                    'context': []
                })
        
        return anomalies
    
    def detect_missing_topics(
        self,
        current_keywords: Counter,
        historical_keywords: List[Counter],
        min_historical_count: int = 3
    ) -> List[Dict]:
        """Detect topics that disappeared from discussion."""
        anomalies = []
        
        if not historical_keywords:
            return anomalies
        
        # Get keywords consistently mentioned in history
        historical_totals = Counter()
        for hist in historical_keywords:
            historical_totals.update(hist)
        
        for keyword, count in historical_totals.items():
            if count >= min_historical_count and keyword not in current_keywords:
                config = self.TRACKED_KEYWORDS.get(keyword, {})
                category = config.get('category', 'General')
                
                anomalies.append({
                    'type': 'missing_topic',
                    'category': category,
                    'severity': 'low',
                    'severity_reason': 'Previously discussed topic no longer mentioned',
                    'title': f"{keyword.replace('_', ' ').title()} - No Longer Mentioned",
                    'description': f"'{keyword.replace('_', ' ')}' not mentioned (was {count} times across {len(historical_keywords)} previous filings)",
                    'explanation': f"This topic was regularly discussed in previous filings but is absent from the current one. "
                                  f"This could indicate resolved issues or de-prioritization.",
                    'keyword': keyword,
                    'historical_count': count,
                    'context': []
                })
        
        return anomalies
    
    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================
    
    def analyze_ticker(self, ticker: str) -> Dict:
        """
        Complete anomaly analysis for a ticker.
        Compares latest filing to historical average.
        
        Args:
            ticker: Stock ticker
            
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
                'error': f'Insufficient data: need 2+ filings, found {len(sentiment_history)}',
                'total_anomalies': 0,
                'anomalies': []
            }
        
        # Current = latest, Historical = all previous
        current = sentiment_history[-1]
        previous = sentiment_history[-2]  # Most recent previous
        history = sentiment_history[:-1]
        
        current_date = current.get('filing_date', 'unknown')
        previous_date = previous.get('filing_date', 'unknown')
        
        logger.info(f"Analyzing: {current_date} vs {len(history)} historical filings")
        logger.info(f"Most recent comparison: {current_date} vs {previous_date}")
        
        # Load section texts for context extraction
        current_text = ""
        for section in ['item_1a', 'item_7']:
            text = self.load_section_text(ticker, current_date, section)
            if text:
                current_text += text + "\n\n"
        
        # Extract keywords
        current_keywords = Counter()
        if current_text:
            current_keywords = self.extract_keywords(current_text)
        
        historical_keywords = []
        for hist in history:
            hist_date = hist.get('filing_date', '')
            hist_text = ""
            for section in ['item_1a', 'item_7']:
                text = self.load_section_text(ticker, hist_date, section)
                if text:
                    hist_text += text + "\n\n"
            
            if hist_text:
                historical_keywords.append(self.extract_keywords(hist_text))
        
        # Detect all anomaly types
        all_anomalies = []
        
        # 1. Sentiment shifts (vs previous filing)
        logger.info("\n[1] Checking sentiment shifts...")
        sentiment_anomalies = self.detect_sentiment_shift(current, previous)
        all_anomalies.extend(sentiment_anomalies)
        
        # 2. New keywords
        logger.info("[2] Checking for new keywords...")
        new_keyword_anomalies = self.detect_new_keywords(
            current_keywords, historical_keywords, current_text
        )
        all_anomalies.extend(new_keyword_anomalies)
        
        # 3. Frequency changes
        logger.info("[3] Checking frequency changes...")
        frequency_anomalies = self.detect_frequency_changes(
            current_keywords, historical_keywords, current_text
        )
        all_anomalies.extend(frequency_anomalies)
        
        # 4. Missing topics
        logger.info("[4] Checking for missing topics...")
        missing_anomalies = self.detect_missing_topics(
            current_keywords, historical_keywords
        )
        all_anomalies.extend(missing_anomalies)
        
        # Sort by severity (high first)
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        all_anomalies.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
        
        # Compile report
        report = {
            'ticker': ticker,
            'current_filing_date': current_date,
            'compared_to': previous_date,
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
        logger.info(f"SUMMARY: {len(all_anomalies)} anomalies detected")
        logger.info(f"  High: {report['anomalies_by_severity']['high']}")
        logger.info(f"  Medium: {report['anomalies_by_severity']['medium']}")
        logger.info(f"  Low: {report['anomalies_by_severity']['low']}")
        logger.info(f"{'='*80}")
        
        return report
    
    def save_report(self, report: Dict) -> Path:
        """Save anomaly report to JSON."""
        filename = f"{report['ticker']}_{report['current_filing_date']}_anomalies.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {filepath}")
        return filepath
    
    def print_report(self, report: Dict):
        """Pretty print anomaly report to console."""
        print(f"\n{'='*80}")
        print(f"ANOMALY REPORT: {report['ticker']}")
        print(f"{'='*80}")
        print(f"Current Filing: {report.get('current_filing_date', 'N/A')}")
        print(f"Compared To: {report.get('compared_to', 'N/A')}")
        print(f"Total Anomalies: {report.get('total_anomalies', 0)}")
        
        severity = report.get('anomalies_by_severity', {})
        print(f"  🔴 High: {severity.get('high', 0)}")
        print(f"  🟡 Medium: {severity.get('medium', 0)}")
        print(f"  🟢 Low: {severity.get('low', 0)}")
        
        if not report.get('anomalies'):
            print("\n✅ No anomalies detected - filing appears normal.")
            return
        
        print(f"\n{'='*80}")
        print("DETECTED ANOMALIES")
        print(f"{'='*80}")
        
        for i, anomaly in enumerate(report['anomalies'], 1):
            severity = anomaly.get('severity', 'medium')
            icon = '🔴' if severity == 'high' else ('🟡' if severity == 'medium' else '🟢')
            
            print(f"\n{icon} [{i}] {anomaly.get('title', 'Anomaly')}")
            print(f"    Category: {anomaly.get('category', 'N/A')}")
            print(f"    {anomaly.get('description', '')}")
            print(f"    ")
            print(f"    Why it matters: {anomaly.get('explanation', 'N/A')}")
            
            if anomaly.get('context'):
                print(f"    ")
                print(f"    Context from filing:")
                for ctx in anomaly['context'][:2]:
                    print(f"    \"{ctx[:200]}...\"")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED ANOMALY DETECTION ENGINE")
    print("=" * 80)
    
    detector = AnomalyDetector()
    
    # Test with Apple
    print("\n[TEST] Analyzing AAPL...")
    report = detector.analyze_ticker("AAPL")
    detector.print_report(report)
    detector.save_report(report)
    
    # Prompt for more
    response = input("\nAnalyze another ticker? Enter ticker or 'n': ").strip().upper()
    if response and response != 'N':
        report = detector.analyze_ticker(response)
        detector.print_report(report)
        detector.save_report(report)