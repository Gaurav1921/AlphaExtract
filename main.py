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


def ensemble_command(args):
    """Run ensemble scoring on existing sentiment data."""
    from src.models.ensemble import EnsembleScorer, load_section_texts, load_historical_texts
    from src.pipeline.automated import get_all_sentiment_data
    import json

    scorer = EnsembleScorer()
    ticker = args.ticker.upper()

    sentiment_data = get_all_sentiment_data(ticker)
    if not sentiment_data:
        logger.error(f"No sentiment data for {ticker} — run analyze first")
        return

    for filing_date, data in sentiment_data:
        ensemble_file = Settings.SENTIMENT_DIR / f"{ticker}_{filing_date}_ensemble.json"
        if ensemble_file.exists() and not args.force:
            logger.info(f"Skipping {filing_date} (ensemble exists)")
            continue

        section_texts = load_section_texts(ticker, filing_date)
        historical_texts = load_historical_texts(ticker, exclude_date=filing_date)

        result = scorer.score_filing(
            ticker=ticker,
            filing_date=filing_date,
            sentiment_data=data,
            section_texts=section_texts,
            historical_texts=historical_texts,
        )
        scorer.save_result(result)

    logger.info(f"Ensemble scoring complete for {ticker}")


def backtest_command(args):
    """Run backtest comparing signals against actual market returns."""
    from src.backtesting.backtester import Backtester
    from src.backtesting.report import generate_terminal_report, generate_comparison_report

    tickers = [t.upper() for t in args.tickers]
    backtester = Backtester(return_window=args.window)

    if args.compare:
        logger.info("Running comparison backtest: FinBERT vs Ensemble")
        comparison = backtester.run_comparison(tickers)
        print(generate_comparison_report(comparison))
        backtester.save_results(
            type("R", (), {"to_dict": lambda self: comparison})(),
            name="comparison",
        )
    else:
        result = backtester.run(tickers, mode=args.mode)
        print(generate_terminal_report(result.to_dict()))
        backtester.save_results(result, name=f"backtest_{args.mode}")


def calibrate_command(args):
    """Calibrate thresholds and weights using backtest results."""
    from src.backtesting.backtester import Backtester
    from src.backtesting.calibrator import Calibrator
    import json

    tickers = [t.upper() for t in args.tickers]
    backtester = Backtester(return_window=args.window)

    logger.info(f"Running backtest for calibration: {', '.join(tickers)}")
    result = backtester.run(tickers, mode=args.mode)

    calibrator = Calibrator()
    calibration = calibrator.full_calibration(result.results)

    print("\n" + "=" * 70)
    print("  CALIBRATION RESULTS")
    print("=" * 70)

    # Thresholds
    th = calibration.get("thresholds", {})
    if "best_thresholds" in th:
        print(f"\n  Signal Thresholds (improvement: +{th.get('improvement', 0):.1%})")
        for k, v in th["best_thresholds"].items():
            print(f"    {k}: {v}")
    else:
        print(f"\n  Thresholds: {th.get('error', 'no data')}")

    # Section weights
    sw = calibration.get("section_weights", {})
    if "best_section_weights" in sw:
        print(f"\n  Section Weights (improvement: +{sw.get('improvement', 0):.1%})")
        for k, v in sw["best_section_weights"].items():
            print(f"    {k}: {v}")
    else:
        print(f"\n  Section Weights: {sw.get('error', 'no data')}")

    # Ensemble weights
    ew = calibration.get("ensemble_weights", {})
    if "best_weights" in ew:
        print(f"\n  Ensemble Weights (improvement: +{ew.get('improvement', 0):.1%})")
        for k, v in ew["best_weights"].items():
            print(f"    {k}: {v}")
    else:
        print(f"\n  Ensemble Weights: {ew.get('error', 'no data')}")

    # Recommended settings
    rec = calibration.get("recommended_settings", {})
    if rec:
        print(f"\n  Recommended Settings Override:")
        print(f"    {json.dumps(rec, indent=4)}")

    # Save
    output = Settings.BACKTEST_DIR / "calibration_latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    logger.info(f"Saved calibration: {output}")


def pipeline_command(args):
    """Run full pipeline: download -> parse -> split -> analyze -> ensemble."""
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

    # ensemble
    en = subparsers.add_parser("ensemble", help="Run ensemble scoring (FinBERT + Keywords + LLM)")
    en.add_argument("ticker", help="Stock ticker")
    en.add_argument("--force", action="store_true", help="Re-score even if ensemble data exists")
    en.set_defaults(func=ensemble_command)

    # backtest
    bt = subparsers.add_parser("backtest", help="Backtest signals against market returns")
    bt.add_argument("tickers", nargs="+", help="Stock tickers to backtest")
    bt.add_argument("--mode", default="finbert", choices=["finbert", "ensemble"], help="Which model to backtest")
    bt.add_argument("--window", type=int, default=90, help="Return window in days (default 90)")
    bt.add_argument("--compare", action="store_true", help="Compare FinBERT vs Ensemble side by side")
    bt.set_defaults(func=backtest_command)

    # calibrate
    cal = subparsers.add_parser("calibrate", help="Calibrate thresholds and weights from backtest data")
    cal.add_argument("tickers", nargs="+", help="Stock tickers to use for calibration")
    cal.add_argument("--mode", default="finbert", choices=["finbert", "ensemble"], help="Which model to calibrate")
    cal.add_argument("--window", type=int, default=90, help="Return window in days")
    cal.set_defaults(func=calibrate_command)

    # pipeline
    pl = subparsers.add_parser("pipeline", help="Full pipeline: download -> parse -> split -> analyze -> ensemble")
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
