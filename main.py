"""
AlphaExtract CLI
---------------
Command-line interface for running AlphaExtract components.

Usage:
    python main.py download AAPL GOOGL MSFT TSLA
    python main.py parse
    python main.py analyze
    python main.py dashboard
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloader import SECDownloader
from src.models.sentiment import SentimentAnalyzer
from src.models.embeddings import EmbeddingPipeline
from src.models.anomaly import AnomalyDetector


def download_command(args):
    """Download 10-K filings."""
    print(f"📥 Downloading filings for: {', '.join(args.tickers)}")
    
    downloader = SECDownloader(email=args.email)
    results = downloader.download_multiple(args.tickers)
    
    print(f"\n✓ Downloaded {len(results['successful'])} filings")
    if results['failed']:
        print(f"✗ Failed: {', '.join(results['failed'])}")


def parse_command(args):
    """Parse downloaded filings."""
    print("📄 Parsing 10-K filings with Docling...")
    
    # Import parser
    from src.data.parser import parse_all_filings
    
    results = parse_all_filings()
    print(f"✓ Parsed {results['success']} filings")


def analyze_command(args):
    """Run sentiment analysis."""
    print("🧠 Running sentiment analysis...")
    
    analyzer = SentimentAnalyzer()
    
    # Analyze all sections
    sections_dir = Path("data/sections")
    if not sections_dir.exists():
        print("✗ No sections found. Run 'split' command first.")
        return
    
    results = analyzer.analyze_directory(sections_dir)
    print(f"✓ Analyzed {results['total']} sections")


def index_command(args):
    """Index documents to OpenSearch."""
    print("🔍 Indexing documents to OpenSearch...")
    
    pipeline = EmbeddingPipeline()
    pipeline.process_directory(Path("data/sections"), recreate_index=args.recreate)
    
    print("✓ Indexing complete")


def anomaly_command(args):
    """Run anomaly detection."""
    print(f"🚨 Running anomaly detection for {args.ticker}...")
    
    detector = AnomalyDetector()
    report = detector.analyze_ticker(args.ticker)
    detector.save_report(report)
    detector.print_report(report)


def dashboard_command(args):
    """Launch Streamlit dashboard."""
    import os
    import subprocess
    
    print("🚀 Launching Streamlit dashboard...")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.port", str(args.port)
    ])


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="AlphaExtract - AI-Powered Financial Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download filings:     python main.py download AAPL GOOGL
  Parse filings:        python main.py parse
  Sentiment analysis:   python main.py analyze
  Index to OpenSearch:  python main.py index
  Anomaly detection:    python main.py anomaly TSLA
  Launch dashboard:     python main.py dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download 10-K filings')
    download_parser.add_argument('tickers', nargs='+', help='Stock tickers')
    download_parser.add_argument('--email', default='student@example.com', help='Your email')
    download_parser.set_defaults(func=download_command)
    
    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse downloaded filings')
    parse_parser.set_defaults(func=parse_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run sentiment analysis')
    analyze_parser.set_defaults(func=analyze_command)
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index to OpenSearch')
    index_parser.add_argument('--recreate', action='store_true', help='Recreate index')
    index_parser.set_defaults(func=index_command)
    
    # Anomaly command
    anomaly_parser = subparsers.add_parser('anomaly', help='Run anomaly detection')
    anomaly_parser.add_argument('ticker', help='Stock ticker')
    anomaly_parser.set_defaults(func=anomaly_command)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501, help='Port number')
    dashboard_parser.set_defaults(func=dashboard_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Run command
    args.func(args)


if __name__ == "__main__":
    main()