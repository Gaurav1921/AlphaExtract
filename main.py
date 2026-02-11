"""
AlphaExtract CLI
-----------------
Command-line interface for the AlphaExtract financial intelligence platform.

Usage:
    python main.py download AAPL GOOGL MSFT
    python main.py parse
    python main.py split
    python main.py analyze [--ticker AAPL]
    python main.py index [--recreate]
    python main.py anomaly TSLA
    python main.py dashboard [--port 8501]
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import Settings, setup_logging

logger = logging.getLogger(__name__)


def download_command(args):
    """Download 10-K filings from SEC EDGAR."""
    from src.data.downloader import SECDownloader

    downloader = SECDownloader(email=args.email)
    results = downloader.download_multiple(args.tickers, years=args.years)

    logger.info(f"Downloaded {len(results['successful'])} filings")
    if results["failed"]:
        logger.warning(f"Failed: {', '.join(results['failed'])}")


def parse_command(args):
    """Parse downloaded filings with Docling."""
    from src.data.parser import parse_all_filings

    results = parse_all_filings()
    logger.info(f"Parsed {results['success']}/{results['total']} filings")


def split_command(args):
    """Extract sections from parsed filings."""
    from src.data.splitter import SectionSplitter

    splitter = SectionSplitter()
    results = splitter.batch_process(Settings.PROCESSED_DIR)
    logger.info(f"Split {len(results['successful'])} filings")


def analyze_command(args):
    """Run sentiment analysis on extracted sections."""
    from src.models.sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    results = analyzer.batch_analyze(ticker=args.ticker)

    logger.info(f"Analyzed {len(results['analyzed'])} filings")
    if results["failed"]:
        logger.warning(f"Failed: {', '.join(results['failed'])}")


def index_command(args):
    """Index documents into OpenSearch for RAG."""
    from src.models.embeddings import EmbeddingPipeline

    pipeline = EmbeddingPipeline()
    if args.recreate:
        pipeline.os_client.create_index(recreate=True)

    result = pipeline.index_from_sections(ticker=args.ticker)
    logger.info(f"Indexed {result['indexed']} documents ({result['failed']} failed)")


def anomaly_command(args):
    """Run anomaly detection on a ticker."""
    from src.models.anomaly import AnomalyDetector

    detector = AnomalyDetector()
    report = detector.analyze_ticker(args.ticker)
    detector.save_report(report)

    total = report.get("total_anomalies", 0)
    high = report.get("anomalies_by_severity", {}).get("high", 0)
    logger.info(f"Anomaly detection for {args.ticker}: {total} anomalies ({high} high severity)")


def dashboard_command(args):
    """Launch the Streamlit dashboard."""
    import subprocess

    logger.info(f"Launching dashboard on port {args.port}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(args.port)],
        check=False,
    )


def pipeline_command(args):
    """Run full pipeline: download -> parse -> split -> analyze."""
    from src.pipeline.automated import AutomatedPipeline

    pipeline = AutomatedPipeline(email=args.email)
    result = pipeline.process_company(args.ticker, years=args.years)

    if result.success:
        logger.info(f"Pipeline complete: {result.message}")
    else:
        logger.error(f"Pipeline failed: {result.message}")
        for error in result.errors:
            logger.error(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="AlphaExtract - AI-Powered 10-K Financial Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py download AAPL GOOGL --years 5
  python main.py parse
  python main.py split
  python main.py analyze --ticker AAPL
  python main.py index --recreate
  python main.py anomaly TSLA
  python main.py pipeline AAPL --years 3
  python main.py dashboard --port 8501
        """,
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # download
    dl = subparsers.add_parser("download", help="Download 10-K filings from SEC EDGAR")
    dl.add_argument("tickers", nargs="+", help="Stock tickers (e.g. AAPL GOOGL)")
    dl.add_argument("--email", default=None, help="Email for SEC identification")
    dl.add_argument("--years", type=int, default=1, help="Number of years to download")
    dl.set_defaults(func=download_command)

    # parse
    p = subparsers.add_parser("parse", help="Parse downloaded filings with Docling")
    p.set_defaults(func=parse_command)

    # split
    sp = subparsers.add_parser("split", help="Extract sections (Item 1A, 7, 8)")
    sp.set_defaults(func=split_command)

    # analyze
    an = subparsers.add_parser("analyze", help="Run FinBERT sentiment analysis")
    an.add_argument("--ticker", default=None, help="Analyze specific ticker only")
    an.set_defaults(func=analyze_command)

    # index
    ix = subparsers.add_parser("index", help="Index into OpenSearch for RAG")
    ix.add_argument("--recreate", action="store_true", help="Recreate index from scratch")
    ix.add_argument("--ticker", default=None, help="Index specific ticker only")
    ix.set_defaults(func=index_command)

    # anomaly
    am = subparsers.add_parser("anomaly", help="Run anomaly detection")
    am.add_argument("ticker", help="Stock ticker")
    am.set_defaults(func=anomaly_command)

    # pipeline
    pl = subparsers.add_parser("pipeline", help="Full pipeline: download -> parse -> split -> analyze")
    pl.add_argument("ticker", help="Stock ticker")
    pl.add_argument("--email", default=None, help="Email for SEC identification")
    pl.add_argument("--years", type=int, default=5, help="Years of filings to process")
    pl.set_defaults(func=pipeline_command)

    # dashboard
    db = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    db.add_argument("--port", type=int, default=8501, help="Port number")
    db.set_defaults(func=dashboard_command)

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Print config warnings at startup
    for warning in Settings.validate():
        logger.warning(warning)

    if args.command is None:
        parser.print_help()
        return

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Command '{args.command}' failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
