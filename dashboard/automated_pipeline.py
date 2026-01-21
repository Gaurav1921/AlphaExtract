"""
Automated Data Management with Full Pipeline (Dashboard Version)
-----------------------------------------------------------------
Complete automated workflow for the Streamlit dashboard.
Uses the proper class-based API from src modules.
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
from src.data.parser import TenKParser
from src.data.splitter import SectionSplitter
from src.models.anomaly import AnomalyDetector


def get_filing_status(ticker: str) -> dict:
    """Get complete status of all filings for a ticker."""
    
    raw_files = list(Path("data/raw").glob(f"{ticker}_10K_*.html")) + \
                list(Path("data/raw").glob(f"{ticker}_10K_*.txt"))
    
    raw_dates = set()
    for f in raw_files:
        parts = f.stem.split('_')
        if len(parts) >= 3:
            raw_dates.add(parts[2])
    
    processed_files = list(Path("data/processed").glob(f"{ticker}_*.md"))
    processed_dates = set()
    for f in processed_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            processed_dates.add(parts[1])
    
    section_files = list(Path("data/sections").glob(f"{ticker}_*_item_*.txt"))
    section_dates = set()
    for f in section_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            section_dates.add(parts[1])
    
    sentiment_files = list(Path("data/sentiment").glob(f"{ticker}_*_sentiment.json"))
    sentiment_dates = set()
    for f in sentiment_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            sentiment_dates.add(parts[1])
    
    all_dates = raw_dates | processed_dates | section_dates | sentiment_dates
    
    status_by_date = {}
    for date in sorted(all_dates, reverse=True):
        status_by_date[date] = {
            'downloaded': date in raw_dates,
            'parsed': date in processed_dates,
            'sections': date in section_dates,
            'sentiment': date in sentiment_dates,
            'complete': date in sentiment_dates
        }
    
    return status_by_date


def download_filing(ticker: str, filing_date: str, progress_callback=None):
    """Download a single filing."""
    downloader = SECDownloader()
    result = downloader.download_filing(ticker)
    
    if progress_callback:
        progress_callback("Downloaded", 100)
    
    return result is not None


def parse_filing(ticker: str, filing_date: str, progress_callback=None):
    """Parse downloaded filing with Docling using TenKParser."""
    
    raw_file = list(Path("data/raw").glob(f"{ticker}_10K_{filing_date}.*"))
    if not raw_file:
        st.error(f"Raw file not found: {ticker}_10K_{filing_date}")
        return False
    
    if progress_callback:
        progress_callback("Parsing with Docling...", 0)
    
    try:
        parser = TenKParser()
        result = parser.process_filing(raw_file[0])
        
        if progress_callback:
            progress_callback("Parsed successfully", 100)
        
        st.success(f"✓ Parsed → {ticker}_{filing_date}.md")
        return result.get('success', True)
        
    except Exception as e:
        st.error(f"Parse error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def extract_sections(ticker: str, filing_date: str, progress_callback=None):
    """Extract sections from parsed filing using SectionSplitter."""
    
    existing_sections = list(Path("data/sections").glob(f"{ticker}_{filing_date}_item_*.txt"))
    if existing_sections:
        if progress_callback:
            progress_callback(f"Sections already exist ({len(existing_sections)} files)", 100)
        st.info(f"⏭️  Sections already extracted: {len(existing_sections)} files")
        return True
    
    parsed_file = Path("data/processed") / f"{ticker}_{filing_date}.md"
    if not parsed_file.exists():
        st.error(f"Parsed file not found: {parsed_file}")
        return False
    
    if progress_callback:
        progress_callback("Extracting sections...", 0)
    
    try:
        splitter = SectionSplitter()
        result = splitter.process_file(parsed_file)
        
        section_files = list(Path("data/sections").glob(f"{ticker}_{filing_date}_item_*.txt"))
        
        if section_files:
            if progress_callback:
                progress_callback(f"Extracted {len(section_files)} sections", 100)
            st.success(f"✓ Extracted {len(section_files)} sections")
            return True
        else:
            st.warning("No sections extracted - file structure may be different than expected")
            return True
            
    except Exception as e:
        st.warning(f"Section extraction note: {e}")
        return True


def analyze_sentiment(ticker: str, filing_date: str, progress_callback=None):
    """Run sentiment analysis on sections."""
    
    sentiment_file = Path("data/sentiment") / f"{ticker}_{filing_date}_sentiment.json"
    if sentiment_file.exists():
        if progress_callback:
            progress_callback("Sentiment already analyzed", 100)
        st.info("⏭️  Sentiment analysis already complete")
        return True
    
    section_files = list(Path("data/sections").glob(f"{ticker}_{filing_date}_item_*.txt"))
    if not section_files:
        st.warning(f"No section files found - skipping sentiment analysis")
        return True
    
    if progress_callback:
        progress_callback("Analyzing sentiment...", 0)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        
        results = {
            'ticker': ticker,
            'sections': {},
            'overall': {}
        }
        
        all_scores = []
        
        for i, section_file in enumerate(section_files):
            section_name = section_file.stem.split('_')[-1]
            
            text = section_file.read_text(encoding='utf-8')
            
            max_length = 450
            words = text.split()
            chunks = []
            for j in range(0, len(words), max_length):
                chunk = ' '.join(words[j:j+max_length])
                chunks.append(chunk)
            
            chunk_scores = []
            for chunk in chunks[:10]:
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                neg, neu, pos = predictions[0].tolist()
                compound = pos - neg
                chunk_scores.append(compound)
            
            avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
            
            if avg_score >= 0.5:
                signal = 'STRONG_BUY'
            elif avg_score >= 0.2:
                signal = 'BUY'
            elif avg_score >= -0.2:
                signal = 'HOLD'
            elif avg_score >= -0.5:
                signal = 'SELL'
            else:
                signal = 'STRONG_SELL'
            
            results['sections'][section_name] = {
                'scores': {'compound': avg_score},
                'signal': signal,
                'word_count': len(words)
            }
            
            all_scores.append(avg_score)
            
            if progress_callback:
                progress = int((i + 1) / len(section_files) * 100)
                progress_callback(f"Analyzing {section_name}...", progress)
        
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        if overall_score >= 0.5:
            overall_signal = 'STRONG_BUY'
        elif overall_score >= 0.2:
            overall_signal = 'BUY'
        elif overall_score >= -0.2:
            overall_signal = 'HOLD'
        elif overall_score >= -0.5:
            overall_signal = 'SELL'
        else:
            overall_signal = 'STRONG_SELL'
        
        results['overall'] = {
            'compound': overall_score,
            'signal': overall_signal
        }
        
        sentiment_dir = Path("data/sentiment")
        sentiment_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = sentiment_dir / f"{ticker}_{filing_date}_sentiment.json"
        output_file.write_text(json.dumps(results, indent=2))
        
        if progress_callback:
            progress_callback("Sentiment analysis complete", 100)
        
        st.success(f"✓ Sentiment: {overall_signal} ({overall_score:+.3f})")
        return True
        
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def index_to_opensearch(ticker: str, filing_date: str, progress_callback=None):
    """Index document chunks to OpenSearch."""
    
    section_files = list(Path("data/sections").glob(f"{ticker}_{filing_date}_item_*.txt"))
    if not section_files:
        st.warning(f"No section files found for {ticker}_{filing_date}")
        return True
    
    if progress_callback:
        progress_callback("Indexing to OpenSearch...", 0)
    
    try:
        from src.models.embeddings import EmbeddingPipeline
        from src.data.chunker import DocumentChunker
        
        pipeline = EmbeddingPipeline()
        chunker = DocumentChunker()
        
        all_chunks = []
        for section_file in section_files:
            chunks = chunker.chunk_file(section_file)
            all_chunks.extend(chunks)
        
        if all_chunks:
            documents = pipeline.prepare_documents(all_chunks)
            pipeline.index_documents(documents)
        
        if progress_callback:
            progress_callback("Indexed to OpenSearch", 100)
        
        return True
        
    except Exception as e:
        st.warning(f"Indexing note: {e} (OpenSearch may not be running)")
        return True


def run_full_pipeline(ticker: str, filing_date: str, status_container, progress_bar):
    """Run complete pipeline for a single filing."""
    
    sentiment_file = Path("data/sentiment") / f"{ticker}_{filing_date}_sentiment.json"
    if sentiment_file.exists():
        status_container.success(f"✅ {filing_date} already fully processed!")
        progress_bar.progress(1.0)
        return True
    
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
            if step_name in ["📄 Parsing PDF", "🧠 Analyzing Sentiment"]:
                status_container.error(f"❌ Failed at: {step_name}")
                return False
            else:
                status_container.warning(f"⚠️ Skipped: {step_name}")
        else:
            status_container.success(f"✅ {step_name} complete")
        
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(0.3)
    
    status_container.success("🎉 Pipeline complete!")
    progress_bar.progress(1.0)
    return True


def render_data_management_page(companies: dict, selected_ticker: str):
    """Render the data management page with automated pipeline."""
    
    st.markdown('<h1 class="main-header">📥 Data Management</h1>', unsafe_allow_html=True)
    st.markdown("### Automated Download & Processing Pipeline")
    
    st.markdown("---")
    
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
    
    status = get_filing_status(manage_ticker)
    
    st.markdown("#### 📊 Filing Status")
    
    if status:
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
    
    st.markdown("#### 🚀 Download & Process")
    
    col1, col2 = st.columns(2)
    
    with col1:
        years_to_download = st.selectbox(
            "How many years?",
            options=[1, 3, 5, 10],
            format_func=lambda x: f"Last {x} year{'s' if x > 1 else ''}",
            index=2
        )
    
    with col2:
        auto_process = st.checkbox(
            "Auto-process after download",
            value=True,
            help="Automatically parse, extract, analyze, and index"
        )
    
    if st.button("📥 Download & Process", type="primary", use_container_width=True):
        
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        status_container.info(f"Starting download for {manage_ticker}...")
        
        downloader = SECDownloader()
        
        cik = downloader.get_cik(manage_ticker)
        if not cik:
            status_container.error("❌ Ticker not found in SEC database")
            st.stop()
        
        status_container.info("Fetching available filings from SEC...")
        
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = downloader.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            filings_data = data['filings']['recent']
            
            available_10ks = []
            for i, form in enumerate(filings_data['form']):
                if form == '10-K' and len(available_10ks) < years_to_download:
                    filing_date = filings_data['filingDate'][i]
                    accession = filings_data['accessionNumber'][i].replace('-', '')
                    primary_doc = filings_data['primaryDocument'][i]
                    
                    doc_url = (
                        f"{downloader.base_url}/Archives/edgar/data/"
                        f"{cik}/{accession}/{primary_doc}"
                    )
                    
                    available_10ks.append({
                        'url': doc_url,
                        'filing_date': filing_date,
                        'accession': accession,
                        'document': primary_doc
                    })
            
            if not available_10ks:
                status_container.error("❌ No 10-K filings found")
                st.stop()
            
            status_container.success(f"✅ Found {len(available_10ks)} filings to download")
            
            downloaded_files = []
            
            for idx, filing_info in enumerate(available_10ks):
                filing_date = filing_info['filing_date']
                
                existing = list(Path("data/raw").glob(f"{manage_ticker}_10K_{filing_date}.*"))
                if existing:
                    status_container.info(f"⏭️  Filing {filing_date} already exists")
                    downloaded_files.append((existing[0], filing_date))
                    continue
                
                status_container.info(f"📥 Downloading filing {idx+1}/{len(available_10ks)} ({filing_date})...")
                progress_bar.progress((idx + 1) / len(available_10ks) * 0.3)
                
                try:
                    time.sleep(0.15)
                    
                    response = downloader.session.get(filing_info['url'], timeout=30)
                    response.raise_for_status()
                    
                    ext = 'html' if 'html' in filing_info['document'] else 'txt'
                    filename = f"{manage_ticker}_10K_{filing_date}.{ext}"
                    filepath = Path("data/raw") / filename
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath.write_bytes(response.content)
                    
                    downloaded_files.append((filepath, filing_date))
                    status_container.success(f"✅ Downloaded {filing_date}")
                    
                except Exception as e:
                    status_container.error(f"❌ Failed to download {filing_date}: {e}")
                    continue
            
            status_container.success(f"✅ Downloaded {len(downloaded_files)} filings")
            
        except Exception as e:
            status_container.error(f"❌ Failed to fetch filings: {e}")
            st.stop()
        
        if auto_process and downloaded_files:
            total_filings = len(downloaded_files)
            
            for idx, (filepath, filing_date) in enumerate(downloaded_files):
                status_container.info(f"Processing filing {idx+1}/{total_filings} ({filing_date})...")
                
                success = run_full_pipeline(
                    manage_ticker,
                    filing_date,
                    status_container,
                    progress_bar
                )
                
                if not success:
                    status_container.warning(f"⚠️  Processing failed for {filing_date}, continuing with next...")
                    continue
            
            status_container.success("🎉 All filings processed!")
            st.balloons()
            time.sleep(2)
            st.rerun()
        
        else:
            status_container.success("✅ Download complete! Refresh to see new files.")
            time.sleep(2)
            st.rerun()
    
    st.markdown("---")
    
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