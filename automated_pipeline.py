"""
Automated Data Management with Full Pipeline
--------------------------------------------
Complete automated workflow - user clicks, system does everything.
"""

import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.data.downloader import SECDownloader
from src.models.anomaly import AnomalyDetector


def get_filing_status(ticker: str) -> dict:
    """Get complete status of all filings for a ticker."""
    
    # Downloaded raw files
    raw_files = list(Path("data/raw").glob(f"{ticker}_10K_*.html")) + \
                list(Path("data/raw").glob(f"{ticker}_10K_*.txt"))
    
    raw_dates = set()
    for f in raw_files:
        parts = f.stem.split('_')
        if len(parts) >= 3:
            raw_dates.add(parts[2])  # Date part
    
    # Processed files
    processed_files = list(Path("data/processed").glob(f"{ticker}_*.md"))
    processed_dates = set()
    for f in processed_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            processed_dates.add(parts[1])
    
    # Section files
    section_files = list(Path("data/sections").glob(f"{ticker}_*_item_*.txt"))
    section_dates = set()
    for f in section_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            section_dates.add(parts[1])
    
    # Sentiment files
    sentiment_files = list(Path("data/sentiment").glob(f"{ticker}_*_sentiment.json"))
    sentiment_dates = set()
    for f in sentiment_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            sentiment_dates.add(parts[1])
    
    # Combine all dates
    all_dates = raw_dates | processed_dates | section_dates | sentiment_dates
    
    status_by_date = {}
    for date in sorted(all_dates, reverse=True):
        status_by_date[date] = {
            'downloaded': date in raw_dates,
            'parsed': date in processed_dates,
            'sections': date in section_dates,
            'sentiment': date in sentiment_dates,
            'complete': date in sentiment_dates  # Sentiment is final step
        }
    
    return status_by_date


def download_filing(ticker: str, filing_date: str, progress_callback=None):
    """Download a single filing."""
    downloader = SECDownloader()
    
    # This is simplified - actual implementation would fetch specific filing
    # For now, we'll download latest
    result = downloader.download_filing(ticker)
    
    if progress_callback:
        progress_callback("Downloaded", 100)
    
    return result is not None


def parse_filing(ticker: str, filing_date: str, progress_callback=None):
    """Parse downloaded filing with Docling."""
    from src.data.parser import parse_10k
    
    # Find the raw file
    raw_file = list(Path("data/raw").glob(f"{ticker}_10K_{filing_date}.*"))
    if not raw_file:
        return False
    
    if progress_callback:
        progress_callback("Parsing with Docling...", 0)
    
    try:
        # Parse the file
        result = parse_10k(raw_file[0])
        
        if progress_callback:
            progress_callback("Parsed", 100)
        
        return result is not None
    except Exception as e:
        st.error(f"Parse error: {e}")
        return False


def extract_sections(ticker: str, filing_date: str, progress_callback=None):
    """Extract sections from parsed filing."""
    from src.data.splitter import split_sections
    
    # Find parsed file
    parsed_file = list(Path("data/processed").glob(f"{ticker}_{filing_date}.md"))
    if not parsed_file:
        return False
    
    if progress_callback:
        progress_callback("Extracting sections...", 0)
    
    try:
        # Split sections
        sections = split_sections(parsed_file[0])
        
        if progress_callback:
            progress_callback("Sections extracted", 100)
        
        return sections is not None
    except Exception as e:
        st.error(f"Section extraction error: {e}")
        return False


def analyze_sentiment(ticker: str, filing_date: str, progress_callback=None):
    """Run sentiment analysis on sections."""
    from src.models.sentiment import FinBERTAnalyzer
    
    # Find section files
    section_files = list(Path("data/sections").glob(f"{ticker}_{filing_date}_item_*.txt"))
    if not section_files:
        return False
    
    if progress_callback:
        progress_callback("Analyzing sentiment...", 0)
    
    try:
        analyzer = FinBERTAnalyzer()
        
        results = {}
        for i, section_file in enumerate(section_files):
            with open(section_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            section_name = section_file.stem.split('_')[-1]  # item_1a, item_7, etc.
            scores = analyzer.analyze_text(text)
            results[section_name] = scores
            
            if progress_callback:
                progress = int((i + 1) / len(section_files) * 100)
                progress_callback(f"Analyzing {section_name}...", progress)
        
        # Save results
        output_file = Path("data/sentiment") / f"{ticker}_{filing_date}_sentiment.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if progress_callback:
            progress_callback("Sentiment analysis complete", 100)
        
        return True
        
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")
        return False


def index_to_opensearch(ticker: str, filing_date: str, progress_callback=None):
    """Index document chunks to OpenSearch."""
    from src.models.embeddings import EmbeddingPipeline
    from src.data.chunker import DocumentChunker
    
    # Find section files
    section_files = list(Path("data/sections").glob(f"{ticker}_{filing_date}_item_*.txt"))
    if not section_files:
        return False
    
    if progress_callback:
        progress_callback("Generating embeddings...", 0)
    
    try:
        pipeline = EmbeddingPipeline()
        chunker = DocumentChunker()
        
        for i, section_file in enumerate(section_files):
            with open(section_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Chunk text
            chunks = chunker.chunk_text(text, metadata={
                'ticker': ticker,
                'filing_date': filing_date,
                'section': section_file.stem.split('_')[-1]
            })
            
            # Index chunks
            pipeline.index_chunks(chunks)
            
            if progress_callback:
                progress = int((i + 1) / len(section_files) * 100)
                progress_callback(f"Indexing...", progress)
        
        if progress_callback:
            progress_callback("Indexed to OpenSearch", 100)
        
        return True
        
    except Exception as e:
        st.error(f"Indexing error: {e}")
        return False


def run_full_pipeline(ticker: str, filing_date: str, status_container, progress_bar):
    """Run complete pipeline for a single filing."""
    
    steps = [
        ("📥 Downloading", download_filing),
        ("📄 Parsing PDF", parse_filing),
        ("✂️ Extracting Sections", extract_sections),
        ("🧠 Analyzing Sentiment", analyze_sentiment),
        ("🔍 Indexing to OpenSearch", index_to_opensearch)
    ]
    
    for i, (step_name, step_func) in enumerate(steps):
        status_container.info(f"{step_name}...")
        
        def update_progress(msg, pct):
            progress_bar.progress(pct / 100)
            status_container.info(f"{step_name}: {msg}")
        
        success = step_func(ticker, filing_date, update_progress)
        
        if not success:
            status_container.error(f"❌ Failed at: {step_name}")
            return False
        
        status_container.success(f"✅ {step_name} complete")
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(0.5)  # Visual feedback
    
    status_container.success("🎉 Pipeline complete!")
    progress_bar.progress(1.0)
    return True


def render_data_management_page(companies: dict, selected_ticker: str):
    """Render the data management page with automated pipeline."""
    
    st.markdown('<h1 class="main-header">📥 Data Management</h1>', unsafe_allow_html=True)
    st.markdown("### Automated Download & Processing Pipeline")
    
    st.markdown("---")
    
    # Company selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        manage_ticker = st.selectbox(
            "Select Company to Manage",
            options=list(companies.keys()),
            format_func=lambda x: companies[x],
            index=list(companies.keys()).index(selected_ticker),
            key='manage_ticker'
        )
    
    with col2:
        st.metric("Selected", companies[manage_ticker])
    
    st.markdown("---")
    
    # Get current status
    status = get_filing_status(manage_ticker)
    
    st.markdown("#### 📊 Filing Status")
    
    if status:
        # Create status dataframe
        status_data = []
        for date, info in status.items():
            year = date[:4]
            status_icons = {
                'downloaded': '✓' if info['downloaded'] else '○',
                'parsed': '✓' if info['parsed'] else '○',
                'sections': '✓' if info['sections'] else '○',
                'sentiment': '✓' if info['sentiment'] else '○'
            }
            
            status_data.append({
                'Filing Date': date,
                'Year': year,
                'Downloaded': status_icons['downloaded'],
                'Parsed': status_icons['parsed'],
                'Sections': status_icons['sections'],
                'Sentiment': status_icons['sentiment'],
                'Complete': '✅' if info['complete'] else '⏳'
            })
        
        df = pd.DataFrame(status_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        complete_count = sum(1 for info in status.values() if info['complete'])
        total_count = len(status)
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric("Total Filings", total_count)
        
        with col_metric2:
            st.metric("Complete", complete_count)
        
        with col_metric3:
            st.metric("Pending", total_count - complete_count)
    
    else:
        st.warning(f"No filings found for {manage_ticker}")
        st.info("💡 Download filings below to get started!")
    
    st.markdown("---")
    
    # Download & Process Section
    st.markdown("#### 🚀 Download & Process")
    
    col1, col2 = st.columns(2)
    
    with col1:
        years_to_download = st.selectbox(
            "How many years?",
            options=[1, 3, 5, 10],
            format_func=lambda x: f"Last {x} year{'s' if x > 1 else ''}",
            index=2  # Default to 5 years
        )
    
    with col2:
        auto_process = st.checkbox(
            "Auto-process after download",
            value=True,
            help="Automatically parse, extract, analyze, and index"
        )
    
    if st.button("📥 Download & Process", type="primary", use_container_width=True):
        
        # Create progress containers
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        status_container.info(f"Starting download for {manage_ticker}...")
        
        # Step 1: Download filings
        downloader = SECDownloader()
        
        status_container.info("Fetching available filings from SEC...")
        
        # Get CIK
        cik = downloader.get_cik(manage_ticker)
        if not cik:
            status_container.error("❌ Ticker not found in SEC database")
            st.stop()
        
        # Download filings
        results = []
        for i in range(years_to_download):
            status_container.info(f"Downloading filing {i+1}/{years_to_download}...")
            progress_bar.progress((i + 1) / years_to_download * 0.3)  # 30% for downloads
            
            result = downloader.download_filing(manage_ticker)
            if result:
                results.append(result)
            
            time.sleep(0.2)  # Rate limit
        
        status_container.success(f"✅ Downloaded {len(results)} filings")
        
        if auto_process and results:
            # Process each filing
            total_filings = len(results)
            
            for idx, filepath in enumerate(results):
                filing_date = filepath.stem.split('_')[2]  # Extract date from filename
                
                status_container.info(f"Processing filing {idx+1}/{total_filings} ({filing_date})...")
                
                # Run pipeline
                success = run_full_pipeline(
                    manage_ticker,
                    filing_date,
                    status_container,
                    progress_bar
                )
                
                if not success:
                    break
            
            if success:
                status_container.success("🎉 All filings processed successfully!")
                st.balloons()
                time.sleep(2)
                st.rerun()
        
        else:
            status_container.success("✅ Download complete! Refresh to see new files.")
    
    st.markdown("---")
    
    # Process pending files
    if status:
        pending = {date: info for date, info in status.items() if not info['complete']}
        
        if pending:
            st.markdown("#### 🔄 Process Pending Files")
            st.info(f"{len(pending)} filing(s) need processing")
            
            if st.button("⚡ Process All Pending", use_container_width=True):
                
                status_container = st.empty()
                progress_bar = st.progress(0)
                
                total = len(pending)
                
                for idx, (date, info) in enumerate(pending.items()):
                    status_container.info(f"Processing {date} ({idx+1}/{total})...")
                    
                    success = run_full_pipeline(
                        manage_ticker,
                        date,
                        status_container,
                        progress_bar
                    )
                    
                    if not success:
                        break
                
                if success:
                    status_container.success("✅ All pending files processed!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
        else:
            st.success("✅ All filings are fully processed!")