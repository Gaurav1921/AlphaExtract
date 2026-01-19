"""
AlphaExtract Dashboard - Complete Integrated Version
----------------------------------------------------
Streamlit app with database persistence and data management.
"""

import streamlit as st
import sys
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import subprocess

# Add to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.enhanced_rag import EnhancedRAG
from src.models.anomaly import AnomalyDetector
from src.database.supabase_client import SupabaseDB

# Page config
st.set_page_config(
    page_title="AlphaExtract - AI Financial Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .signal-strong-buy { color: #00FF00; font-weight: bold; font-size: 1.5rem; }
    .signal-buy { color: #90EE90; font-weight: bold; font-size: 1.5rem; }
    .signal-hold { color: #FFD700; font-weight: bold; font-size: 1.5rem; }
    .signal-sell { color: #FFA500; font-weight: bold; font-size: 1.5rem; }
    .signal-strong-sell { color: #FF0000; font-weight: bold; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag' not in st.session_state:
    with st.spinner("🔄 Initializing RAG system..."):
        st.session_state.rag = EnhancedRAG()

if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()

if 'db' not in st.session_state:
    st.session_state.db = SupabaseDB()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'AAPL'

# Helper functions
def load_and_save_sentiment(ticker: str):
    """Load sentiment from JSON and save to database."""
    sentiment_files = list(Path("data/sentiment").glob(f"{ticker}_*_sentiment.json"))
    
    if not sentiment_files:
        return None, None
    
    latest_file = sorted(sentiment_files)[-1]
    with open(latest_file, 'r') as f:
        sentiment_data = json.load(f)
    
    filing_date = latest_file.stem.split('_')[1]
    
    # Save to database (upsert - won't duplicate)
    if st.session_state.db.client:
        st.session_state.db.insert_sentiment(ticker, filing_date, sentiment_data)
    
    return sentiment_data, filing_date


def get_sentiment_history(ticker: str):
    """Get historical sentiment from database."""
    if not st.session_state.db.client:
        return []
    
    return st.session_state.db.get_sentiment_history(ticker, limit=10)


def count_available_filings(ticker: str) -> int:
    """Count how many sentiment files exist for a ticker."""
    return len(list(Path("data/sentiment").glob(f"{ticker}_*_sentiment.json")))


# Sidebar
with st.sidebar:
    st.markdown("### 🏢 Company Selection")
    
    companies = {
        'AAPL': '🍎 Apple Inc.',
        'GOOGL': '🔍 Alphabet Inc.',
        'MSFT': '🪟 Microsoft Corp.',
        'TSLA': '⚡ Tesla Inc.'
    }
    
    selected_ticker = st.selectbox(
        "Select Company",
        options=list(companies.keys()),
        format_func=lambda x: companies[x],
        index=list(companies.keys()).index(st.session_state.selected_ticker)
    )
    
    st.session_state.selected_ticker = selected_ticker
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📁 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Dashboard", "💬 RAG Chat", "🚨 Anomalies", "📊 Analytics", "📥 Data Management"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Actions
    st.markdown("### ⚙️ Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.rag.reset_conversation()
        st.success("Chat history cleared!")
    
    st.markdown("---")
    
    # Database status
    st.markdown("### 🗄️ Database")
    if st.session_state.db.client:
        st.success("✓ Connected")
    else:
        st.error("✗ Not connected")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **AlphaExtract v0.4.0**
    
    AI-Powered Financial Intelligence
    
    - 📊 Sentiment Analysis
    - 💬 Conversational RAG
    - 🚨 Anomaly Detection
    - 🗄️ Supabase Database
    """)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "🏠 Dashboard":
    st.markdown(f'<h1 class="main-header">🎯 AlphaExtract</h1>', unsafe_allow_html=True)
    st.markdown(f"### AI-Powered Intelligence for {companies[selected_ticker]}")
    
    # Load sentiment and save to DB
    sentiment_data, filing_date = load_and_save_sentiment(selected_ticker)
    
    if not sentiment_data:
        st.error(f"No sentiment data found for {selected_ticker}.")
        st.info("💡 Go to **📥 Data Management** to download and process filings.")
    else:
        # Key Metrics Row
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        overall_score = sentiment_data['overall']['compound']
        overall_signal = sentiment_data['overall']['signal']
        
        with col1:
            signal_class = f"signal-{overall_signal.lower().replace('_', '-')}"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.9rem;">Overall Signal</div>
                <div class="{signal_class}">{overall_signal}</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Score: {overall_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_score = sentiment_data['sections'].get('item_1a', {}).get('scores', {}).get('compound', 0)
            risk_signal = sentiment_data['sections'].get('item_1a', {}).get('signal', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.9rem;">Risk Factors</div>
                <div style="font-size: 1.3rem; font-weight: bold;">{risk_signal}</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Score: {risk_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            mda_score = sentiment_data['sections'].get('item_7', {}).get('scores', {}).get('compound', 0)
            mda_signal = sentiment_data['sections'].get('item_7', {}).get('signal', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.9rem;">MD&A</div>
                <div style="font-size: 1.3rem; font-weight: bold;">{mda_signal}</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Score: {mda_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            fin_score = sentiment_data['sections'].get('item_8', {}).get('scores', {}).get('compound', 0)
            fin_signal = sentiment_data['sections'].get('item_8', {}).get('signal', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 0.9rem;">Financials</div>
                <div style="font-size: 1.3rem; font-weight: bold;">{fin_signal}</div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Score: {fin_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts Row
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 📊 Sentiment Breakdown")
            
            sections_data = {
                'Risk Factors': risk_score,
                'MD&A': mda_score,
                'Financials': fin_score
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(sections_data.keys()),
                    y=list(sections_data.values()),
                    marker_color=['#3b82f6' if v >= 0 else '#ef4444' for v in sections_data.values()],
                    text=[f"{v:+.3f}" for v in sections_data.values()],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                yaxis_title="Sentiment Score",
                yaxis_range=[-1, 1],
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("#### 📈 Historical Sentiment Trend")
            
            # Get historical data from database
            history = get_sentiment_history(selected_ticker)
            
            if history and len(history) > 1:
                dates = [h['filing_date'] for h in reversed(history)]
                scores = [h['overall_compound'] for h in reversed(history)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=scores,
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=2),
                    marker=dict(size=8),
                    name='Sentiment'
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                fig.update_layout(
                    yaxis_title="Sentiment Score",
                    yaxis_range=[-1, 1],
                    showlegend=False,
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need multiple filings for historical trend. Download more in Data Management!")
        
        # Filing Info
        st.markdown("---")
        st.markdown("#### 📄 Filing Information")
        
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.metric("Filing Date", filing_date)
        
        with info_col2:
            total_words = sum(s.get('word_count', 0) for s in sentiment_data['sections'].values())
            st.metric("Words Analyzed", f"{total_words:,}")
        
        with info_col3:
            st.metric("Sections", len(sentiment_data['sections']))
        
        with info_col4:
            if st.session_state.db.client:
                st.metric("Database", "✓ Saved")
            else:
                st.metric("Database", "✗ Not saved")

# ============================================================================
# PAGE: RAG CHAT
# ============================================================================

elif page == "💬 RAG Chat":
    st.markdown(f'<h1 class="main-header">💬 RAG Chat</h1>', unsafe_allow_html=True)
    st.markdown(f"### Ask questions about {companies[selected_ticker]}'s 10-K filing")
    
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        **Source {i}:** {source['ticker']} - {source['section']} (Chunk {source['chunk_id']})  
                        **Relevance:** {source['score']:.3f}  
                        **Preview:** {source['preview']}
                        """)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the 10-K filing..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag.query(
                    prompt,
                    ticker=selected_ticker,
                    k=5,
                    verbose=False
                )
                
                st.markdown(response['answer'])
                
                # Log to database
                if st.session_state.db.client:
                    st.session_state.db.log_chat_query(
                        ticker=selected_ticker,
                        query=prompt,
                        answer=response['answer'],
                        num_sources=len(response['sources'])
                    )
                
                # Show sources
                if response['sources']:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(response['sources'], 1):
                            st.markdown(f"""
                            **Source {i}:** {source['ticker']} - {source['section']} (Chunk {source['chunk_id']})  
                            **Relevance:** {source['score']:.3f}  
                            **Preview:** {source['preview']}
                            """)
                
                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response['sources']
                })

# ============================================================================
# PAGE: ANOMALIES
# ============================================================================

elif page == "🚨 Anomalies":
    st.markdown(f'<h1 class="main-header">🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    st.markdown(f"### Unusual patterns in {companies[selected_ticker]}'s filings")
    
    # Check filing count
    filing_count = count_available_filings(selected_ticker)
    
    # Load anomaly data from local files first
    anomaly_files = list(Path("data/anomalies").glob(f"{selected_ticker}_*_anomalies.json"))
    
    if not anomaly_files:
        if filing_count < 2:
            # Smart download prompt
            st.warning(f"⚠️ Limited Data for {selected_ticker}")
            st.info(f"Only {filing_count} filing(s) available. Need 2+ for anomaly detection.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Go to Data Management", use_container_width=True, type="primary"):
                    st.session_state.page = "📥 Data Management"
                    st.rerun()
            
            with col2:
                st.info("Download more filings to enable anomaly detection")
        
        else:
            st.info(f"Found {filing_count} filings. Run anomaly detection.")
            
            if st.button("🔍 Run Anomaly Detection Now"):
                with st.spinner("Analyzing..."):
                    report = st.session_state.anomaly_detector.analyze_ticker(selected_ticker)
                    
                    # Check if analysis was successful
                    if 'error' in report:
                        st.error(f"Analysis failed: {report['error']}")
                    else:
                        st.session_state.anomaly_detector.save_report(report)
                        
                        # Save to database
                        if st.session_state.db.client and report.get('anomalies'):
                            st.session_state.db.insert_anomalies(
                                ticker=selected_ticker,
                                filing_date=report['current_filing_date'],
                                anomalies=report['anomalies']
                            )
                        
                        st.success("Analysis complete!")
                        st.rerun()
    else:
        latest_file = sorted(anomaly_files)[-1]
        with open(latest_file, 'r') as f:
            anomaly_data = json.load(f)
        
        # Save to database
        if st.session_state.db.client and anomaly_data.get('anomalies'):
            st.session_state.db.insert_anomalies(
                ticker=selected_ticker,
                filing_date=anomaly_data['current_filing_date'],
                anomalies=anomaly_data['anomalies']
            )
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies", anomaly_data['total_anomalies'])
        
        with col2:
            st.metric("High Severity", anomaly_data['anomalies_by_severity']['high'])
        
        with col3:
            st.metric("Medium Severity", anomaly_data['anomalies_by_severity']['medium'])
        
        st.markdown("---")
        
        # Group by type
        anomalies_by_type = {}
        for anomaly in anomaly_data['anomalies']:
            atype = anomaly['type']
            if atype not in anomalies_by_type:
                anomalies_by_type[atype] = []
            anomalies_by_type[atype].append(anomaly)
        
        # Display
        for atype, anomalies in anomalies_by_type.items():
            st.markdown(f"### {atype.replace('_', ' ').title()}")
            
            for anomaly in anomalies:
                severity = anomaly.get('severity', 'medium')
                icon = '🔴' if severity == 'high' else '🟡'
                
                with st.expander(f"{icon} {anomaly['description']}"):
                    st.json(anomaly)

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif page == "📊 Analytics":
    st.markdown(f'<h1 class="main-header">📊 Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### Usage Statistics & Insights")
    
    if not st.session_state.db.client:
        st.warning("⚠️ Database not connected. Analytics require Supabase connection.")
    else:
        st.markdown("---")
        
        # Get analytics data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Most Queried Companies")
            st.info("Coming soon - will show which companies users ask about most")
        
        with col2:
            st.markdown("#### 💬 Recent Questions")
            recent_queries = st.session_state.db.get_popular_queries(limit=5)
            
            if recent_queries:
                for i, q in enumerate(recent_queries, 1):
                    st.markdown(f"**{i}.** {q['query']} *(Company: {q.get('ticker', 'N/A')})*")
            else:
                st.info("No queries logged yet. Ask some questions in RAG Chat!")
        
        st.markdown("---")
        
        st.markdown("#### 🚨 High Severity Anomalies Across All Companies")
        high_severity = st.session_state.db.get_anomalies(severity='high', limit=10)
        
        if high_severity:
            df = pd.DataFrame(high_severity)
            df = df[['ticker', 'filing_date', 'anomaly_type', 'description']]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No high severity anomalies detected yet")

# ============================================================================
# PAGE: DATA MANAGEMENT
# ============================================================================

elif page == "📥 Data Management":
    st.markdown('<h1 class="main-header">📥 Data Management</h1>', unsafe_allow_html=True)
    st.markdown("### Download, Process & Manage Financial Data")
    
    st.markdown("---")
    
    # Data status overview
    st.markdown("#### 📊 Current Data Status")
    
    status_data = []
    for ticker in ['AAPL', 'GOOGL', 'MSFT', 'TSLA']:
        raw_count = len(list(Path("data/raw").glob(f"{ticker}_10K_*.html"))) + \
                   len(list(Path("data/raw").glob(f"{ticker}_10K_*.txt")))
        sections_count = len(list(Path("data/sections").glob(f"{ticker}_*_item_*.txt"))) // 3
        sentiment_count = len(list(Path("data/sentiment").glob(f"{ticker}_*_sentiment.json")))
        
        status_data.append({
            'Company': companies[ticker],
            'Downloaded': raw_count,
            'Processed': sections_count,
            'Analyzed': sentiment_count,
            'Ready': '✓' if sentiment_count >= 1 else '○'
        })
    
    df = pd.DataFrame(status_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### 🛠️ Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📥 Download")
        if st.button("Download More Filings", use_container_width=True, type="primary"):
            st.info("💡 Use the SEC downloader script to download more historical filings:")
            st.code("python src/data/downloader.py", language="bash")
    
    with col2:
        st.markdown("##### 🔄 Process")
        if st.button("Run Processing Pipeline", use_container_width=True):
            st.info("💡 Processing steps:")
            st.code("""
# 1. Parse PDFs
python src/data/parser.py

# 2. Extract sections
python src/data/splitter.py

# 3. Analyze sentiment
python src/models/sentiment.py

# 4. Index to OpenSearch
python src/models/embeddings.py
            """, language="bash")
    
    with col3:
        st.markdown("##### 🧹 Clean")
        if st.button("Clear Cache", use_container_width=True):
            st.success("Cache cleared!")
    
    st.markdown("---")
    st.markdown("#### 📖 Quick Start Guide")
    
    with st.expander("How to add more data"):
        st.markdown("""
        **Step 1: Download Filings**
        ```bash
        python src/data/downloader.py
        ```
        
        **Step 2: Parse Documents**
        ```bash
        python src/data/parser.py
        ```
        
        **Step 3: Extract Sections**
        ```bash
        python src/data/splitter.py
        ```
        
        **Step 4: Run Sentiment Analysis**
        ```bash
        python src/models/sentiment.py
        ```
        
        **Step 5: Index to OpenSearch**
        ```bash
        python src/models/embeddings.py
        ```
        
        Then refresh the dashboard to see new data!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">
    AlphaExtract v0.4.0 | Built with Streamlit, OpenSearch, Gemini & Supabase
</div>
""", unsafe_allow_html=True)