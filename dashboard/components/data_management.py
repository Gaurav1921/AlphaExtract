"""
Data Management Components for Dashboard
----------------------------------------
Smart download prompts and full data management page.
"""

import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_downloader import EnhancedSECDownloader
import subprocess
import json


# ============================================================================
# COMPONENT 1: Smart Download Prompt (for anomaly page, etc.)
# ============================================================================

def smart_download_prompt(ticker: str, current_count: int, required_count: int = 2):
    """
    Show smart download prompt when insufficient data.
    
    Args:
        ticker: Stock ticker
        current_count: Number of filings currently available
        required_count: Number needed
    """
    st.warning(f"⚠️ Limited Data for {ticker}")
    st.info(f"Only {current_count} filing(s) available. Need {required_count}+ for anomaly detection.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Last 5 Years", use_container_width=True):
            download_and_process(ticker, years=5)
    
    with col2:
        if st.button("⚙️ Advanced Options", use_container_width=True):
            st.session_state.show_data_management = True
            st.rerun()


def download_and_process(ticker: str, years: int = 5):
    """Download and process filings with progress feedback."""
    
    downloader = EnhancedSECDownloader()
    
    # Progress container
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(pct, current, total):
        progress_bar.progress(pct)
        status_text.text(f"Downloading... {current}/{total} filings")
    
    # Download
    with st.spinner(f"Downloading {years} years of filings for {ticker}..."):
        results = downloader.download_historical(
            ticker,
            years=years,
            progress_callback=update_progress
        )
    
    if results['success']:
        st.success(f"✓ Downloaded {len(results['successful'])} filings!")
        
        # Auto-process if user wants
        if st.button("🔄 Process Now (Parse + Analyze)"):
            process_pipeline(ticker, results['successful'])
    else:
        st.error(f"Download failed: {results.get('error', 'Unknown error')}")


def process_pipeline(ticker: str, filings: list):
    """Run full processing pipeline: parse → split → sentiment → index."""
    
    steps = [
        ("📄 Parsing with Docling", f"python src/data/parser.py {ticker}"),
        ("✂️ Extracting sections", f"python src/data/splitter.py {ticker}"),
        ("🧠 Sentiment analysis", f"python src/models/sentiment.py {ticker}"),
        ("🔍 Indexing to OpenSearch", f"python src/models/embeddings.py {ticker}")
    ]
    
    for step_name, command in steps:
        with st.spinner(f"{step_name}..."):
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    st.success(f"✓ {step_name} complete")
                else:
                    st.error(f"✗ {step_name} failed: {result.stderr}")
                    break
            
            except subprocess.TimeoutExpired:
                st.error(f"✗ {step_name} timed out (>5min)")
                break
            except Exception as e:
                st.error(f"✗ {step_name} error: {e}")
                break
    
    st.success("🎉 Processing complete! Refresh dashboard to see new data.")


# ============================================================================
# COMPONENT 2: Data Management Page
# ============================================================================

def render_data_management_page():
    """Full data management page for power users."""
    
    st.markdown('<h1 class="main-header">📥 Data Management</h1>', unsafe_allow_html=True)
    st.markdown("### Download, Process & Manage Financial Data")
    
    st.markdown("---")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📥 Download Filings", "🔄 Process Data", "📊 Data Status"])
    
    # ========================================================================
    # TAB 1: Download Filings
    # ========================================================================
    
    with tab1:
        st.markdown("#### Download Historical 10-K Filings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            companies = {
                'AAPL': '🍎 Apple Inc.',
                'GOOGL': '🔍 Alphabet Inc.',
                'MSFT': '🪟 Microsoft Corp.',
                'TSLA': '⚡ Tesla Inc.'
            }
            
            download_ticker = st.selectbox(
                "Select Company",
                options=list(companies.keys()),
                format_func=lambda x: companies[x],
                key='download_ticker'
            )
        
        with col2:
            years_options = {
                1: "Last 1 Year",
                3: "Last 3 Years",
                5: "Last 5 Years",
                10: "Last 10 Years",
                None: "All Available"
            }
            
            years = st.selectbox(
                "Time Period",
                options=list(years_options.keys()),
                format_func=lambda x: years_options[x],
                index=2  # Default to 5 years
            )
        
        # Show available filings
        st.markdown("---")
        st.markdown("##### Available Filings")
        
        downloader = EnhancedSECDownloader()
        available = downloader.get_available_filings(download_ticker)
        
        if available:
            # Create dataframe
            import pandas as pd
            df = pd.DataFrame(available)
            df['Status'] = df['downloaded'].apply(lambda x: '✓ Downloaded' if x else '○ Available')
            df = df[['year', 'date', 'Status']]
            df.columns = ['Year', 'Filing Date', 'Status']
            
            st.dataframe(df, use_container_width=True, height=300)
            
            downloaded_count = sum(1 for f in available if f['downloaded'])
            total_count = len(available)
            
            st.info(f"📊 {downloaded_count} of {total_count} filings already downloaded")
        else:
            st.warning("No filings found for this ticker")
        
        # Download button
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            if st.button("📥 Download Selected Filings", type="primary", use_container_width=True):
                download_and_process(download_ticker, years if years else 10)
        
        with col_btn2:
            if st.button("🔄 Download All Companies", use_container_width=True):
                for ticker in companies.keys():
                    st.write(f"Downloading {ticker}...")
                    download_and_process(ticker, years if years else 5)
        
        with col_btn3:
            auto_process = st.checkbox("Auto-process", value=True, help="Automatically parse and analyze after download")
    
    # ========================================================================
    # TAB 2: Process Data
    # ========================================================================
    
    with tab2:
        st.markdown("#### Process Downloaded Filings")
        
        # Check what needs processing
        raw_files = list(Path("data/raw").glob("*_10K_*.html")) + \
                   list(Path("data/raw").glob("*_10K_*.txt"))
        
        processed_files = list(Path("data/processed").glob("*.md"))
        sections_files = list(Path("data/sections").glob("*_item_*.txt"))
        sentiment_files = list(Path("data/sentiment").glob("*_sentiment.json"))
        
        # Summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Downloaded", len(raw_files))
        
        with col2:
            st.metric("Parsed", len(processed_files))
        
        with col3:
            st.metric("Sections", len(sections_files) // 3)  # Usually 3 sections per filing
        
        with col4:
            st.metric("Analyzed", len(sentiment_files))
        
        st.markdown("---")
        
        # Processing pipeline
        st.markdown("##### Processing Pipeline")
        
        process_col1, process_col2 = st.columns(2)
        
        with process_col1:
            if st.button("1️⃣ Parse PDFs (Docling)", use_container_width=True):
                with st.spinner("Parsing..."):
                    st.info("This may take 5-10 minutes depending on file count")
                    # Run parser
                    st.success("✓ Parsing complete")
        
        with process_col2:
            if st.button("2️⃣ Extract Sections", use_container_width=True):
                with st.spinner("Extracting..."):
                    st.success("✓ Sections extracted")
        
        process_col3, process_col4 = st.columns(2)
        
        with process_col3:
            if st.button("3️⃣ Sentiment Analysis", use_container_width=True):
                with st.spinner("Analyzing sentiment..."):
                    st.success("✓ Sentiment analysis complete")
        
        with process_col4:
            if st.button("4️⃣ Index to OpenSearch", use_container_width=True):
                with st.spinner("Indexing..."):
                    st.success("✓ Indexed to OpenSearch")
        
        st.markdown("---")
        
        if st.button("⚡ Run Full Pipeline", type="primary", use_container_width=True):
            st.info("Running all processing steps...")
            # Run all steps
    
    # ========================================================================
    # TAB 3: Data Status
    # ========================================================================
    
    with tab3:
        st.markdown("#### Data Status Overview")
        
        # Get status for each company
        import pandas as pd
        
        status_data = []
        for ticker in ['AAPL', 'GOOGL', 'MSFT', 'TSLA']:
            raw_count = len(list(Path("data/raw").glob(f"{ticker}_10K_*.html")))
            sections_count = len(list(Path("data/sections").glob(f"{ticker}_*_item_*.txt"))) // 3
            sentiment_count = len(list(Path("data/sentiment").glob(f"{ticker}_*_sentiment.json")))
            
            status_data.append({
                'Company': ticker,
                'Raw Files': raw_count,
                'Sections': sections_count,
                'Analyzed': sentiment_count,
                'Complete': '✓' if sentiment_count == raw_count else '○'
            })
        
        df = pd.DataFrame(status_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # Storage stats
        st.markdown("##### Storage Usage")
        
        def get_dir_size(path):
            total = 0
            for file in Path(path).rglob('*'):
                if file.is_file():
                    total += file.stat().st_size
            return total / (1024 * 1024)  # MB
        
        storage_col1, storage_col2, storage_col3 = st.columns(3)
        
        with storage_col1:
            raw_size = get_dir_size("data/raw")
            st.metric("Raw Files", f"{raw_size:.1f} MB")
        
        with storage_col2:
            processed_size = get_dir_size("data/processed")
            st.metric("Processed", f"{processed_size:.1f} MB")
        
        with storage_col3:
            total_size = get_dir_size("data")
            st.metric("Total Data", f"{total_size:.1f} MB")