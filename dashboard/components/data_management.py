"""
Data Management Components for Dashboard
-----------------------------------------
Reusable UI components for download prompts and data status displays.
Uses AutomatedPipeline for all processing — no subprocess calls.
"""

import logging
import streamlit as st
from pathlib import Path

logger = logging.getLogger(__name__)


def smart_download_prompt(ticker: str, current_count: int, required_count: int = 2):
    """
    Show download prompt when insufficient data exists for analysis.

    Args:
        ticker: Stock ticker symbol.
        current_count: Number of filings currently available.
        required_count: Minimum filings needed.
    """
    st.warning(f"Limited Data for {ticker}")
    st.info(f"Only {current_count} filing(s) available. Need {required_count}+ for anomaly detection.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download Last 5 Years", use_container_width=True, key=f"dl5_{ticker}"):
            _run_pipeline(ticker, years=5)

    with col2:
        if st.button("Download Last 10 Years", use_container_width=True, key=f"dl10_{ticker}"):
            _run_pipeline(ticker, years=10)


def _run_pipeline(ticker: str, years: int):
    """Run the automated pipeline with progress feedback."""
    try:
        from src.pipeline.automated import AutomatedPipeline
    except ImportError:
        st.error("Pipeline module not available. Check installation.")
        return

    pipeline = AutomatedPipeline()
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update(msg, pct):
        progress_bar.progress(min(pct / 100, 1.0))
        status_text.info(msg)

    pipeline.set_progress_callback(update)

    try:
        result = pipeline.process_company(ticker, years=years)
        if result.success:
            status_text.success(f"Done: {result.message}")
        else:
            status_text.error(f"Failed: {result.message}")
            for err in result.errors:
                st.warning(err)
    except Exception as e:
        logger.error(f"Pipeline error for {ticker}: {e}")
        status_text.error(f"Pipeline error: {e}")


def render_data_status(tickers: list):
    """
    Render data status overview for a list of tickers.

    Args:
        tickers: List of ticker symbols to show status for.
    """
    import pandas as pd
    from src.config.settings import Settings

    status_data = []
    for ticker in tickers:
        raw = list(Settings.RAW_DIR.glob(f"{ticker}_10K_*.*"))
        raw = [f for f in raw if f.suffix in (".html", ".txt", ".htm")]
        sections = list(Settings.SECTIONS_DIR.glob(f"{ticker}_*_item_*.txt"))
        sentiment = list(Settings.SENTIMENT_DIR.glob(f"{ticker}_*_sentiment.json"))

        status_data.append({
            "Company": ticker,
            "Downloaded": len(raw),
            "Sections": len(sections),
            "Analyzed": len(sentiment),
            "Complete": "Yes" if len(sentiment) > 0 and len(sentiment) == len(raw) else "No",
        })

    if status_data:
        st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)


def render_storage_stats():
    """Show storage usage for data directories."""
    from src.config.settings import Settings

    def _dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Raw Files", f"{_dir_size_mb(Settings.RAW_DIR):.1f} MB")
    with col2:
        st.metric("Processed", f"{_dir_size_mb(Settings.PROCESSED_DIR):.1f} MB")
    with col3:
        total = _dir_size_mb(Settings.DATA_DIR)
        st.metric("Total Data", f"{total:.1f} MB")
