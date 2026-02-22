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
    python main.py alerts AAPL MSFT --days 30
    python main.py portfolio AAPL MSFT GOOGL
    python main.py options AAPL MSFT
    python main.py options-history AAPL --record
    python main.py api --port 8000
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
    tickers = [t.upper() for t in args.tickers]

    for ticker in tickers:
        sentiment_data = get_all_sentiment_data(ticker)
        if not sentiment_data:
            logger.error(f"No sentiment data for {ticker} — run analyze first")
            continue

        for filing_date, data in sentiment_data:
            ensemble_file = Settings.SENTIMENT_DIR / f"{ticker}_{filing_date}_ensemble.json"
            if ensemble_file.exists() and not args.force:
                logger.info(f"Skipping {ticker} {filing_date} (ensemble exists)")
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


def alerts_command(args):
    """Monitor SEC EDGAR for new 10-K filings."""
    from src.alerts.sec_monitor import SECMonitor
    from src.alerts.webhook import WebhookNotifier

    tickers = [t.upper() for t in args.tickers]

    # Set up webhook if URL provided
    webhook_notifier = None
    if args.webhook_url:
        webhook_notifier = WebhookNotifier()
        webhook_notifier.add_webhook(
            url=args.webhook_url,
            name=args.webhook_name,
            format=args.webhook_format,
        )
        logger.info(f"Webhook configured: {args.webhook_name} ({args.webhook_format})")

    monitor = SECMonitor(
        watchlist=tickers,
        poll_interval=args.interval,
        webhook_notifier=webhook_notifier,
    )

    def log_alert(alert):
        print(f"  NEW FILING: {alert.ticker} ({alert.company_name}) - {alert.filing_date}")
        print(f"  URL: {alert.filing_url}")

    monitor.on_new_filing(log_alert)

    if args.poll:
        logger.info(f"Starting continuous monitoring for {', '.join(tickers)}")
        monitor.poll(days_back=args.days)
    else:
        alerts = monitor.check_once(days_back=args.days)
        if alerts:
            logger.info(f"Found {len(alerts)} new filing(s)")
        else:
            logger.info("No new filings found")


def portfolio_command(args):
    """Aggregate signals across a portfolio of tickers."""
    from src.analysis.portfolio import (
        equal_weight_portfolio,
        sector_portfolio,
        save_portfolio,
    )

    if args.sector:
        result = sector_portfolio(args.sector)
    else:
        tickers = [t.upper() for t in args.tickers]
        result = equal_weight_portfolio(tickers, name=args.name)

    print(f"\n{'=' * 60}")
    print(f"  PORTFOLIO: {result.name}")
    print(f"{'=' * 60}")
    print(f"  Holdings: {result.holdings_count} ({result.holdings_with_data} with data)")
    print(f"  Coverage: {result.coverage_pct}%")
    print(f"  Weighted Sentiment: {result.weighted_sentiment:+.4f}")
    if result.weighted_ensemble is not None:
        print(f"  Weighted Ensemble:  {result.weighted_ensemble:+.4f}")
    print(f"  Signal: {result.portfolio_signal}")

    if result.sector_breakdown:
        print(f"\n  Sector Breakdown:")
        for sector, data in result.sector_breakdown.items():
            print(f"    {sector}: {data['avg_score']:+.4f} ({data['signal']}) [{data['weight']:.1%}]")

    print(f"\n  Signal Distribution:")
    for signal, count in result.signal_distribution.items():
        print(f"    {signal}: {count}")

    risk = result.risk_concentration
    print(f"\n  Risk Concentration:")
    print(f"    Max holding: {risk['max_single_holding']['ticker']} ({risk['max_single_holding']['weight']:.1%})")
    if risk["max_single_sector"]["sector"]:
        print(f"    Max sector:  {risk['max_single_sector']['sector']} ({risk['max_single_sector']['weight']:.1%})")
    print(f"    HHI: {risk['herfindahl_index']:.4f}")

    if args.save:
        save_portfolio(result)


def api_command(args):
    """Launch the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed — run: pip install uvicorn")
        return

    logger.info(f"Starting AlphaExtract API on {args.host}:{args.port}")
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def options_history_command(args):
    """Track historical options data over time."""
    from src.market.options_history import OptionsHistoryTracker

    tracker = OptionsHistoryTracker()
    tickers = [t.upper() for t in args.tickers]

    if args.record:
        logger.info(f"Recording options snapshots for {', '.join(tickers)}")
        entries = tracker.batch_fetch_and_record(tickers)
        logger.info(f"Recorded {len(entries)} snapshot(s)")
        for entry in entries:
            print(f"  {entry.ticker}: P/C={entry.put_call_ratio}, IV skew={entry.iv_skew}, score={entry.options_score}")
    else:
        for ticker in tickers:
            ts = tracker.get_history(ticker, limit=args.limit)

            print(f"\n{'=' * 60}")
            print(f"  OPTIONS HISTORY: {ticker}")
            print(f"{'=' * 60}")
            print(f"  Data Points: {ts.data_points}")

            if ts.date_range:
                print(f"  Range: {ts.date_range['first']} to {ts.date_range['last']}")

            if ts.trend:
                pc = ts.trend.get("put_call_ratio", {})
                iv = ts.trend.get("iv_skew", {})
                if pc:
                    print(f"  P/C Ratio: {pc.get('current', 'N/A')} (trend: {pc.get('direction', 'N/A')}, avg: {pc.get('avg', 'N/A')})")
                if iv:
                    print(f"  IV Skew:   {iv.get('current', 'N/A')} (trend: {iv.get('direction', 'N/A')}, avg: {iv.get('avg', 'N/A')})")
            elif ts.data_points == 0:
                print(f"  No data. Use --record to fetch and store snapshots.")


def options_command(args):
    """Fetch options data and compute composite signals."""
    from src.market.options_overlay import OptionsOverlay

    overlay = OptionsOverlay(
        filing_weight=args.filing_weight,
        options_weight=args.options_weight,
    )
    tickers = [t.upper() for t in args.tickers]

    for ticker in tickers:
        print(f"\n{'=' * 60}")
        print(f"  OPTIONS OVERLAY: {ticker}")
        print(f"{'=' * 60}")

        signal = overlay.composite_signal(ticker)

        if signal.filing_sentiment is not None:
            print(f"  Filing Sentiment: {signal.filing_sentiment:+.4f} ({signal.filing_signal})")
        else:
            print(f"  Filing Sentiment: N/A")
        if signal.ensemble_score is not None:
            print(f"  Ensemble Score:   {signal.ensemble_score:+.4f} ({signal.ensemble_signal})")

        print(f"  Options Score:    {signal.options_score:+.4f} ({signal.options_signal})")
        print(f"  Composite Score:  {signal.composite_score:+.4f} ({signal.composite_signal})")
        print(f"  Agreement:        {'Yes' if signal.agreement else 'No'}")
        print(f"  Weights:          Filing {signal.filing_weight:.0%} / Options {signal.options_weight:.0%}")

        if args.save:
            overlay.save_composite(signal)


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
  python main.py ensemble AAPL GOOGL MSFT --force
  python main.py pipeline AAPL --years 3
  python main.py dashboard --port 8501
  python main.py alerts AAPL MSFT --days 30
  python main.py alerts AAPL MSFT --poll --webhook-url https://hooks.slack.com/...
  python main.py portfolio AAPL MSFT GOOGL --save
  python main.py options AAPL MSFT --save
  python main.py options-history AAPL MSFT --record
  python main.py api --port 8000
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
    en.add_argument("tickers", nargs="+", help="Stock tickers (e.g. AAPL GOOGL MSFT)")
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

    # alerts
    al = subparsers.add_parser("alerts", help="Monitor SEC EDGAR for new 10-K filings")
    al.add_argument("tickers", nargs="+", help="Stock tickers to watch")
    al.add_argument("--days", type=int, default=30, help="Look back N days (default 30)")
    al.add_argument("--poll", action="store_true", help="Continuously poll (default: check once)")
    al.add_argument("--interval", type=int, default=3600, help="Poll interval in seconds (default 3600)")
    al.add_argument("--webhook-url", default=None, help="Webhook URL for notifications")
    al.add_argument("--webhook-name", default="cli-webhook", help="Webhook name (default: cli-webhook)")
    al.add_argument("--webhook-format", default="json", choices=["json", "slack"], help="Webhook payload format")
    al.set_defaults(func=alerts_command)

    # portfolio
    pf = subparsers.add_parser("portfolio", help="Aggregate signals across a portfolio")
    pf.add_argument("tickers", nargs="*", help="Stock tickers (equal-weighted)")
    pf.add_argument("--sector", default=None, help="Use all tickers from a sector instead")
    pf.add_argument("--name", default="My Portfolio", help="Portfolio name")
    pf.add_argument("--save", action="store_true", help="Save results to JSON")
    pf.set_defaults(func=portfolio_command)

    # options
    op = subparsers.add_parser("options", help="Options sentiment overlay")
    op.add_argument("tickers", nargs="+", help="Stock tickers")
    op.add_argument("--filing-weight", type=float, default=0.70, help="Filing signal weight (default 0.70)")
    op.add_argument("--options-weight", type=float, default=0.30, help="Options signal weight (default 0.30)")
    op.add_argument("--save", action="store_true", help="Save results to JSON")
    op.set_defaults(func=options_command)

    # options-history
    oh = subparsers.add_parser("options-history", help="Track options data over time")
    oh.add_argument("tickers", nargs="+", help="Stock tickers")
    oh.add_argument("--record", action="store_true", help="Fetch and record current snapshot")
    oh.add_argument("--limit", type=int, default=0, help="Max history entries to show (0=all)")
    oh.set_defaults(func=options_history_command)

    # api
    api = subparsers.add_parser("api", help="Launch the FastAPI REST API server")
    api.add_argument("--host", default=Settings.API_HOST, help="Host to bind (default 0.0.0.0)")
    api.add_argument("--port", type=int, default=Settings.API_PORT, help="Port number (default 8000)")
    api.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev mode)")
    api.set_defaults(func=api_command)

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
