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

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.enhanced_rag import EnhancedRAG
from src.models.anomaly import AnomalyDetector
from src.database.supabase_client import SupabaseDB

st.set_page_config(
    page_title="AlphaExtract - AI Financial Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed CSS - Better contrast for signal colors on dark background
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
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .metric-score {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    /* Fixed signal colors - high contrast */
    .signal-strong-buy { 
        color: #22c55e !important; 
        font-weight: bold; 
        font-size: 1.5rem;
        text-shadow: 0 0 10px rgba(34, 197, 94, 0.5);
    }
    .signal-buy { 
        color: #4ade80 !important; 
        font-weight: bold; 
        font-size: 1.5rem;
        text-shadow: 0 0 10px rgba(74, 222, 128, 0.5);
    }
    .signal-hold { 
        color: #fbbf24 !important; 
        font-weight: bold; 
        font-size: 1.5rem;
        text-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
    }
    .signal-sell { 
        color: #fb923c !important; 
        font-weight: bold; 
        font-size: 1.5rem;
        text-shadow: 0 0 10px rgba(251, 146, 60, 0.5);
    }
    .signal-strong-sell { 
        color: #ef4444 !important; 
        font-weight: bold; 
        font-size: 1.5rem;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }
    .signal-na {
        color: #94a3b8 !important;
        font-weight: bold;
        font-size: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)


def get_signal_class(signal: str) -> str:
    """Get CSS class for signal."""
    if not signal or signal == 'N/A':
        return 'signal-na'
    return f"signal-{signal.lower().replace('_', '-')}"

if 'rag' not in st.session_state:
    with st.spinner("🔄 Initializing RAG system..."):
        try:
            st.session_state.rag = EnhancedRAG()
        except Exception as e:
            st.session_state.rag = None

if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()

if 'db' not in st.session_state:
    st.session_state.db = SupabaseDB()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'AAPL'

if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"


def load_and_save_sentiment(ticker: str):
    """Load sentiment from JSON and save to database."""
    sentiment_files = list(Path("data/sentiment").glob(f"{ticker}_*_sentiment.json"))
    
    if not sentiment_files:
        return None, None
    
    latest_file = sorted(sentiment_files)[-1]
    with open(latest_file, 'r') as f:
        sentiment_data = json.load(f)
    
    filing_date = latest_file.stem.split('_')[1]
    
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
    
    st.markdown("### 📁 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Dashboard", "💬 RAG Chat", "🚨 Anomalies", "📊 Analytics", "📥 Data Management"],
        index=["🏠 Dashboard", "💬 RAG Chat", "🚨 Anomalies", "📊 Analytics", "📥 Data Management"].index(st.session_state.current_page),
        label_visibility="collapsed",
        key="nav_radio"
    )
    
    # Update current page state
    st.session_state.current_page = page
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.rag:
            st.session_state.rag.reset_conversation()
        st.success("Chat history cleared!")
    
    st.markdown("---")
    
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

# PAGE: DASHBOARD
if page == "🏠 Dashboard":
    st.markdown(f'<h1 class="main-header">🎯 AlphaExtract</h1>', unsafe_allow_html=True)
    st.markdown(f"### AI-Powered Intelligence for {companies[selected_ticker]}")
    
    sentiment_data, filing_date = load_and_save_sentiment(selected_ticker)
    
    if not sentiment_data:
        st.error(f"No sentiment data found for {selected_ticker}.")
        st.info("💡 Go to **📥 Data Management** to download and process filings.")
    else:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        overall_score = sentiment_data['overall']['compound']
        overall_signal = sentiment_data['overall']['signal']
        
        with col1:
            signal_class = get_signal_class(overall_signal)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Signal</div>
                <div class="{signal_class}">{overall_signal}</div>
                <div class="metric-score">Score: {overall_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_score = sentiment_data['sections'].get('item_1a', {}).get('scores', {}).get('compound', 0)
            risk_signal = sentiment_data['sections'].get('item_1a', {}).get('signal', 'N/A')
            signal_class = get_signal_class(risk_signal)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Factors</div>
                <div class="{signal_class}">{risk_signal}</div>
                <div class="metric-score">Score: {risk_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            mda_score = sentiment_data['sections'].get('item_7', {}).get('scores', {}).get('compound', 0)
            mda_signal = sentiment_data['sections'].get('item_7', {}).get('signal', 'N/A')
            signal_class = get_signal_class(mda_signal)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MD&A</div>
                <div class="{signal_class}">{mda_signal}</div>
                <div class="metric-score">Score: {mda_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            fin_score = sentiment_data['sections'].get('item_8', {}).get('scores', {}).get('compound', 0)
            fin_signal = sentiment_data['sections'].get('item_8', {}).get('signal', 'N/A')
            signal_class = get_signal_class(fin_signal)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Financials</div>
                <div class="{signal_class}">{fin_signal}</div>
                <div class="metric-score">Score: {fin_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
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

# PAGE: RAG CHAT
elif page == "💬 RAG Chat":
    st.markdown(f'<h1 class="main-header">💬 RAG Chat</h1>', unsafe_allow_html=True)
    st.markdown(f"### Ask questions about {companies[selected_ticker]}'s 10-K filing")
    
    st.markdown("---")
    
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
    
    if prompt := st.chat_input("Ask a question about the 10-K filing..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag.query(
                    prompt,
                    ticker=selected_ticker,
                    k=5,
                    verbose=False
                )
                
                st.markdown(response['answer'])
                
                if st.session_state.db.client:
                    st.session_state.db.log_chat_query(
                        ticker=selected_ticker,
                        query=prompt,
                        answer=response['answer'],
                        num_sources=len(response['sources'])
                    )
                
                if response['sources']:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(response['sources'], 1):
                            st.markdown(f"""
                            **Source {i}:** {source['ticker']} - {source['section']} (Chunk {source['chunk_id']})  
                            **Relevance:** {source['score']:.3f}  
                            **Preview:** {source['preview']}
                            """)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response['sources']
                })

# PAGE: ANOMALIES
elif page == "🚨 Anomalies":
    st.markdown(f'<h1 class="main-header">🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    st.markdown(f"### Unusual patterns in {companies[selected_ticker]}'s filings")
    
    filing_count = count_available_filings(selected_ticker)
    
    anomaly_files = list(Path("data/anomalies").glob(f"{selected_ticker}_*_anomalies.json"))
    
    if not anomaly_files:
        if filing_count < 2:
            st.warning(f"⚠️ Limited Data for {selected_ticker}")
            st.info(f"Only {filing_count} filing(s) available. Need 2+ for anomaly detection.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Fixed: Actually navigate to Data Management
                if st.button("📥 Go to Data Management", use_container_width=True, type="primary"):
                    st.session_state.current_page = "📥 Data Management"
                    st.rerun()
            
            with col2:
                st.info("Download more filings to enable anomaly detection")
        
        else:
            st.info(f"Found {filing_count} filings. Run anomaly detection.")
            
            if st.button("🔍 Run Anomaly Detection Now"):
                with st.spinner("Analyzing..."):
                    report = st.session_state.anomaly_detector.analyze_ticker(selected_ticker)
                    
                    # Fixed: Check if analysis was successful
                    if 'error' in report:
                        st.error(f"Analysis failed: {report['error']}")
                        st.info("💡 Tip: You need at least 2 filings to detect anomalies. Download more filings for this company.")
                    else:
                        st.session_state.anomaly_detector.save_report(report)
                        
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
        
        # DON'T re-insert to database here - it causes duplicates on every page load!
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Anomalies", anomaly_data['total_anomalies'])
        
        with col2:
            st.metric("High Severity", anomaly_data['anomalies_by_severity']['high'])
        
        with col3:
            st.metric("Medium Severity", anomaly_data['anomalies_by_severity']['medium'])
        
        with col4:
            st.metric("Compared To", f"{anomaly_data.get('num_historical_filings', 'N/A')} prior filings")
        
        st.markdown("---")
        
        anomalies_by_type = {}
        for anomaly in anomaly_data['anomalies']:
            atype = anomaly['type']
            if atype not in anomalies_by_type:
                anomalies_by_type[atype] = []
            anomalies_by_type[atype].append(anomaly)
        
        for atype, anomalies in anomalies_by_type.items():
            st.markdown(f"### {atype.replace('_', ' ').title()}")
            
            for anomaly in anomalies:
                severity = anomaly.get('severity', 'medium')
                icon = '🔴' if severity == 'high' else '🟡' if severity == 'medium' else '🟢'
                
                with st.expander(f"{icon} {anomaly['description']}", expanded=True):
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.markdown(f"**Type:** {anomaly['type'].replace('_', ' ').title()}")
                        st.markdown(f"**Severity:** {severity.upper()}")
                        
                        # Show context if available
                        if 'context_snippets' in anomaly and anomaly['context_snippets']:
                            st.markdown("**📄 Context from Filing:**")
                            for snippet in anomaly['context_snippets'][:3]:
                                st.info(f"_{snippet}_")
                        
                        if 'comparison_details' in anomaly:
                            st.markdown("**📊 Comparison:**")
                            st.markdown(anomaly['comparison_details'])
                        
                        if 'historical_years' in anomaly:
                            st.markdown(f"**📅 Historical Years:** {', '.join(anomaly['historical_years'])}")
                    
                    with col_b:
                        if 'current_count' in anomaly:
                            st.metric("Current", anomaly['current_count'])
                        if 'average_count' in anomaly:
                            st.metric("Hist. Avg", f"{anomaly['average_count']:.1f}")
                        if 'ratio' in anomaly:
                            change = "↑" if anomaly['ratio'] > 1 else "↓"
                            st.metric("Change", f"{anomaly['ratio']:.1f}x {change}")
                    
                    with st.expander("📋 Raw Data", expanded=False):
                        st.json(anomaly)

# PAGE: ANALYTICS
elif page == "📊 Analytics":
    st.markdown(f'<h1 class="main-header">📊 Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### Usage Statistics & Insights")
    
    if not st.session_state.db.client:
        st.warning("⚠️ Database not connected. Analytics require Supabase connection.")
    else:
        st.markdown("---")
        
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
        
        st.markdown("#### 🚨 High Severity Anomalies (Deduplicated)")
        high_severity = st.session_state.db.get_anomalies(severity='high', limit=50)
        
        if high_severity:
            # Deduplicate by ticker + filing_date + anomaly_type + description
            seen = set()
            unique_anomalies = []
            for a in high_severity:
                key = (a.get('ticker', ''), a.get('filing_date', ''), a.get('anomaly_type', ''), a.get('description', ''))
                if key not in seen:
                    seen.add(key)
                    unique_anomalies.append(a)
            
            if unique_anomalies:
                df = pd.DataFrame(unique_anomalies[:10])
                display_cols = ['ticker', 'filing_date', 'anomaly_type', 'description']
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No unique high severity anomalies found")
        else:
            st.info("No high severity anomalies detected yet")

# PAGE: DATA MANAGEMENT
elif page == "📥 Data Management":
    from automated_pipeline import render_data_management_page
    render_data_management_page(companies, selected_ticker)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">
    AlphaExtract v0.4.0 | Built with Streamlit, OpenSearch, Gemini & Supabase
</div>
""", unsafe_allow_html=True)