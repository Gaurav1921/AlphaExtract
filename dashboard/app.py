"""
AlphaExtract Dashboard - v0.5.2
--------------------------------------
LOCATION: dashboard/app.py

FIXES in v0.5.2:
1. Force reload AnomalyDetector to pick up code changes
2. Auto-save anomaly reports to disk
3. Better debug output for anomaly fields
4. Improved anomaly card display with all v0.5.1 fields

Version: 0.5.2
"""

import streamlit as st
import sys
from pathlib import Path
import json
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import importlib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports with error handling
try:
    from src.rag.enhanced_rag import EnhancedRAG
except:
    EnhancedRAG = None

# FORCE RELOAD anomaly module to pick up changes
try:
    import src.models.anomaly as anomaly_module
    importlib.reload(anomaly_module)
    from src.models.anomaly import AnomalyDetector
except Exception as e:
    print(f"Failed to load AnomalyDetector: {e}")
    AnomalyDetector = None

try:
    from src.database.supabase_client import SupabaseDB
except:
    SupabaseDB = None

try:
    from src.data.company_search import CompanySearch
except:
    CompanySearch = None

try:
    from src.pipeline.automated import (
        AutomatedPipeline, 
        get_filing_status, 
        get_available_years,
        get_all_sentiment_data
    )
except:
    AutomatedPipeline = None
    get_filing_status = None
    get_available_years = lambda x: []
    get_all_sentiment_data = lambda x: []

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
        font-size: 2.5rem;
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
    .anomaly-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
    }
    .anomaly-high { border-left-color: #ef4444; }
    .anomaly-medium { border-left-color: #f59e0b; }
    .anomaly-low { border-left-color: #22c55e; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'company_search' not in st.session_state:
    st.session_state.company_search = CompanySearch() if CompanySearch else None

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'AAPL'

if 'selected_year' not in st.session_state:
    st.session_state.selected_year = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag' not in st.session_state:
    st.session_state.rag = None
    if EnhancedRAG:
        try:
            st.session_state.rag = EnhancedRAG()
        except:
            pass

# ALWAYS create fresh AnomalyDetector to pick up code changes
if 'anomaly_detector' not in st.session_state or st.session_state.get('_force_reload_detector'):
    st.session_state.anomaly_detector = AnomalyDetector() if AnomalyDetector else None
    st.session_state._force_reload_detector = False

if 'db' not in st.session_state:
    st.session_state.db = SupabaseDB() if SupabaseDB else None

if 'anomaly_report' not in st.session_state:
    st.session_state.anomaly_report = None

if 'anomaly_ticker' not in st.session_state:
    st.session_state.anomaly_ticker = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_companies() -> dict:
    """Get all available companies (defaults + watchlist)."""
    defaults = {
        'AAPL': '🍎 Apple Inc.',
        'GOOGL': '🔍 Alphabet Inc.',
        'MSFT': '🪟 Microsoft Corp.',
        'TSLA': '⚡ Tesla Inc.'
    }
    
    if st.session_state.company_search:
        return st.session_state.company_search.get_all_companies()
    
    return defaults


def normalize_sentiment_data(data: dict) -> dict:
    """
    Normalize sentiment JSON to use consistent keys.
    Handles both old format ("1a", "7", "8") and new format ("item_1a", etc.)
    """
    if not data or 'sections' not in data:
        return data
    
    sections = data['sections']
    normalized = {}
    
    # Key mapping: old -> new
    key_map = {
        '1a': 'item_1a',
        '1b': 'item_1b',
        '1': 'item_1',
        '7': 'item_7',
        '7a': 'item_7a',
        '8': 'item_8',
        '9': 'item_9',
    }
    
    for key, value in sections.items():
        # Already in correct format
        if key.startswith('item_'):
            normalized[key] = value
        # Legacy format
        elif key in key_map:
            normalized[key_map[key]] = value
        else:
            # Unknown key, try to normalize
            normalized[f"item_{key}"] = value
    
    data['sections'] = normalized
    return data


def load_sentiment_for_year(ticker: str, year: str = None) -> tuple:
    """Load sentiment data for a specific year or latest."""
    sentiment_dir = Path("data/sentiment")
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    
    ticker = ticker.upper()
    files = list(sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
    
    if not files:
        return None, None
    
    if year:
        # Filter files that match the year
        year_files = [f for f in files if f.stem.split('_')[1].startswith(year)]
        if year_files:
            files = year_files
    
    # Sort and get the appropriate file (latest within selection)
    latest_file = sorted(files, reverse=True)[0]
    
    with open(latest_file, 'r') as f:
        sentiment_data = json.load(f)
    
    # NORMALIZE KEYS
    sentiment_data = normalize_sentiment_data(sentiment_data)
    
    filing_date = latest_file.stem.split('_')[1]
    return sentiment_data, filing_date


def get_section_data(sentiment_data: dict, section_key: str) -> dict:
    """
    Safely get section data with fallback for missing keys.
    """
    if not sentiment_data:
        return {'scores': {'compound': 0}, 'signal': 'N/A', 'word_count': 0}
    
    sections = sentiment_data.get('sections', {})
    
    # Direct lookup (should work after normalization)
    if section_key in sections:
        return sections[section_key]
    
    # Return default
    return {'scores': {'compound': 0}, 'signal': 'N/A', 'word_count': 0}


def get_historical_sentiment(ticker: str) -> list:
    """Get all historical sentiment data for trend chart."""
    try:
        sentiment_data = get_all_sentiment_data(ticker)
    except:
        sentiment_data = []
    
    if not sentiment_data:
        # Fallback: manually load files
        sentiment_dir = Path("data/sentiment")
        files = sorted(sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
        
        sentiment_data = []
        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(f)
                    filing_date = f.stem.split('_')[1]
                    sentiment_data.append((filing_date, data))
            except:
                pass
    
    history = []
    for filing_date, data in sentiment_data:
        history.append({
            'filing_date': filing_date,
            'year': filing_date[:4],
            'overall_compound': data.get('overall', {}).get('compound', 0),
            'signal': data.get('overall', {}).get('signal', 'HOLD')
        })
    
    return history


def get_available_years_local(ticker: str) -> list:
    """Get available years for a ticker (fallback if pipeline not imported)."""
    try:
        return get_available_years(ticker)
    except:
        sentiment_dir = Path("data/sentiment")
        files = list(sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
        years = set()
        for f in files:
            parts = f.stem.split('_')
            if len(parts) >= 2:
                year = parts[1][:4]
                years.add(year)
        return sorted(years, reverse=True)


def get_filing_status_local(ticker: str) -> dict:
    """Get filing status (fallback)."""
    try:
        if get_filing_status:
            return get_filing_status(ticker)
    except:
        pass
    return {}


def render_anomaly_card(anomaly: dict):
    """
    Render a single anomaly with all v0.5.1 enhanced fields.
    Includes smart fallbacks for missing fields.
    """
    # Extract fields with smart fallbacks
    title = anomaly.get('title') or anomaly.get('keyword', '').replace('_', ' ').title() or 'Anomaly Detected'
    category = anomaly.get('category') or anomaly.get('type', 'General').replace('_', ' ').title()
    severity = anomaly.get('severity', 'medium')
    severity_reason = anomaly.get('severity_reason') or f"Severity: {severity}"
    description = anomaly.get('description', '')
    explanation = anomaly.get('explanation') or 'No additional context available.'
    context = anomaly.get('context', [])
    
    # Severity styling
    if severity == 'high':
        icon = '🔴'
    elif severity == 'medium':
        icon = '🟡'
    else:
        icon = '🟢'
    
    with st.expander(f"{icon} {title}", expanded=(severity == 'high')):
        # Category badge
        st.markdown(f"**Category:** {category}")
        
        # Main description
        if description:
            st.markdown(f"**What changed:** {description}")
        
        st.markdown("---")
        
        # Why it matters (the key enhancement in v0.5.1)
        st.markdown("**Why this matters:**")
        st.info(explanation)
        
        # Severity reasoning
        st.markdown(f"**Severity ({severity.upper()}):** {severity_reason}")
        
        # Context from filing (extracted sentences)
        if context and len(context) > 0:
            st.markdown("---")
            st.markdown("**📄 Context from filing:**")
            for i, ctx in enumerate(context[:3], 1):
                # Truncate long sentences
                display_text = ctx[:400] + "..." if len(ctx) > 400 else ctx
                st.markdown(f"> _{display_text}_")
        
        # Quantitative data if available
        if 'current_score' in anomaly or 'current_count' in anomaly:
            st.markdown("---")
            st.markdown("**📊 Data:**")
            
            if 'current_score' in anomaly:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current", f"{anomaly.get('current_score', 0):.3f}")
                with col2:
                    st.metric("Previous", f"{anomaly.get('previous_score', 0):.3f}")
                with col3:
                    delta = anomaly.get('delta', 0)
                    st.metric("Change", f"{delta:+.3f}")
            
            elif 'current_count' in anomaly:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Mentions", anomaly.get('current_count', 0))
                with col2:
                    avg = anomaly.get('average_count', anomaly.get('historical_count', 0))
                    label = "Historical Avg" if 'average_count' in anomaly else "Historical Total"
                    st.metric(label, f"{avg:.1f}" if isinstance(avg, float) else str(avg))
                with col3:
                    ratio = anomaly.get('ratio', 1.0)
                    if ratio >= 1:
                        st.metric("Change", f"{ratio:.1f}x more")
                    else:
                        st.metric("Change", f"{1/ratio:.1f}x less")
        
        # Debug: show raw data
        with st.expander("🔧 Raw JSON (debug)", expanded=False):
            st.json(anomaly)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Company Search
    if st.session_state.company_search:
        st.markdown("### 🔍 Search Company")
        
        search_query = st.text_input(
            "Ticker or name",
            placeholder="NVDA, NVIDIA...",
            key="search_input"
        )
        
        if search_query and len(search_query) >= 2:
            if search_query.upper() == search_query or len(search_query) <= 5:
                result = st.session_state.company_search.search_by_ticker(search_query)
                results = [result] if result else []
            else:
                results = st.session_state.company_search.search_by_name(search_query, limit=5)
            
            if results:
                for company in results:
                    if company:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"{company.ticker}")
                        with col2:
                            if st.button("➕", key=f"add_{company.ticker}"):
                                st.session_state.company_search.add_to_watchlist(company.ticker)
                                st.success(f"Added!")
            else:
                st.caption("No results")
        
        st.markdown("---")
    
    # Company Selector
    st.markdown("### 🏢 Select Company")
    companies = get_companies()
    
    selected_ticker = st.selectbox(
        "Company",
        options=list(companies.keys()),
        format_func=lambda x: companies[x],
        index=list(companies.keys()).index(st.session_state.selected_ticker) if st.session_state.selected_ticker in companies else 0,
        key="ticker_selector"
    )
    
    # Clear anomaly cache if ticker changed
    if selected_ticker != st.session_state.selected_ticker:
        st.session_state.anomaly_report = None
        st.session_state.anomaly_ticker = None
    
    st.session_state.selected_ticker = selected_ticker
    
    # Year Selector
    available_years = get_available_years_local(selected_ticker)
    
    if available_years:
        st.markdown("### 📅 Filing Year")
        
        year_options = ["Latest"] + available_years
        
        current_year = st.session_state.selected_year
        if current_year and current_year in available_years:
            default_index = year_options.index(current_year)
        else:
            default_index = 0
        
        selected_year = st.selectbox(
            "Year",
            options=year_options,
            index=default_index,
            key=f"year_selector_{selected_ticker}"
        )
        
        st.session_state.selected_year = None if selected_year == "Latest" else selected_year
    else:
        st.session_state.selected_year = None
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📁 Navigation")
    page = st.radio(
        "Page",
        ["🏠 Dashboard", "💬 RAG Chat", "🚨 Anomalies", "📊 Analytics", "📥 Data Management"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Watchlist
    if st.session_state.company_search:
        watchlist = st.session_state.company_search.get_watchlist()
        if watchlist:
            with st.expander(f"📋 Watchlist ({len(watchlist)})"):
                for item in watchlist:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(item['ticker'])
                    with col2:
                        if st.button("❌", key=f"rm_{item['ticker']}"):
                            st.session_state.company_search.remove_from_watchlist(item['ticker'])
    
    # Status
    st.markdown("---")
    status = get_filing_status_local(selected_ticker)
    if status:
        complete = sum(1 for s in status.values() if getattr(s, 'is_complete', False))
        st.metric("Processed", f"{complete}/{len(status)}")


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "🏠 Dashboard":
    st.markdown(f'<h1 class="main-header">🎯 AlphaExtract</h1>', unsafe_allow_html=True)
    st.markdown(f"### {companies.get(selected_ticker, selected_ticker)}")
    
    # Get year filter
    year_filter = st.session_state.selected_year
    
    # Load sentiment
    sentiment_data, filing_date = load_sentiment_for_year(selected_ticker, year_filter)
    
    if not sentiment_data:
        st.warning(f"No sentiment data for {selected_ticker}")
        st.info("💡 Go to **📥 Data Management** to process filings.")
    else:
        # Show year indicator
        if year_filter:
            st.success(f"📅 Showing **{year_filter}** data (Filed: {filing_date})")
        else:
            st.info(f"📅 Showing **Latest** filing ({filing_date})")
        
        st.markdown("---")
        
        # Get section data (handles both key formats)
        risk = get_section_data(sentiment_data, 'item_1a')
        mda = get_section_data(sentiment_data, 'item_7')
        fin = get_section_data(sentiment_data, 'item_8')
        
        overall = sentiment_data.get('overall', {})
        overall_score = overall.get('compound', 0)
        overall_signal = overall.get('signal', 'N/A')
        
        # Color mapping
        colors = {
            'STRONG_BUY': '#00C853', 
            'BUY': '#69F0AE', 
            'HOLD': '#FFD600', 
            'SELL': '#FF9100', 
            'STRONG_SELL': '#FF1744',
            'N/A': '#9E9E9E'
        }
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = colors.get(overall_signal, '#9E9E9E')
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">Overall Signal</div>
                <div style="color: {color}; font-weight: bold; font-size: 1.5rem;">{overall_signal}</div>
                <div style="color: #94a3b8;">Score: {overall_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_score = risk.get('scores', {}).get('compound', 0)
            risk_signal = risk.get('signal', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">Risk Factors</div>
                <div style="font-weight: bold; font-size: 1.3rem;">{risk_signal}</div>
                <div style="color: #94a3b8;">Score: {risk_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            mda_score = mda.get('scores', {}).get('compound', 0)
            mda_signal = mda.get('signal', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">MD&A</div>
                <div style="font-weight: bold; font-size: 1.3rem;">{mda_signal}</div>
                <div style="color: #94a3b8;">Score: {mda_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            fin_score = fin.get('scores', {}).get('compound', 0)
            fin_signal = fin.get('signal', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">Financials</div>
                <div style="font-weight: bold; font-size: 1.3rem;">{fin_signal}</div>
                <div style="color: #94a3b8;">Score: {fin_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 📊 Sentiment by Section")
            
            sections_chart = {
                'Risk Factors': risk_score,
                'MD&A': mda_score,
                'Financials': fin_score
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(sections_chart.keys()),
                    y=list(sections_chart.values()),
                    marker_color=['#3b82f6' if v >= 0 else '#ef4444' for v in sections_chart.values()],
                    text=[f"{v:+.3f}" for v in sections_chart.values()],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                yaxis_title="Score",
                yaxis_range=[-1, 1],
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"section_chart_{selected_ticker}_{year_filter}")
        
        with col_right:
            st.markdown("#### 📈 Historical Trend")
            
            history = get_historical_sentiment(selected_ticker)
            
            if len(history) >= 2:
                # Sort by date
                history.sort(key=lambda x: x['filing_date'])
                
                dates = [h['filing_date'] for h in history]
                scores = [h['overall_compound'] for h in history]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=scores,
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=10),
                    name='Sentiment'
                ))
                
                # Add reference line and zones
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_hrect(y0=0.2, y1=1, fillcolor="green", opacity=0.1, line_width=0)
                fig.add_hrect(y0=-1, y1=-0.2, fillcolor="red", opacity=0.1, line_width=0)
                
                fig.update_layout(
                    yaxis_title="Score",
                    yaxis_range=[-1, 1],
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f"trend_{selected_ticker}")
            else:
                st.info(f"📊 Need 2+ filings for trend chart. Currently have {len(history)}.")
                st.markdown("**Download more years in Data Management!**")


# ============================================================================
# PAGE: RAG CHAT
# ============================================================================

elif page == "💬 RAG Chat":
    st.markdown(f'<h1 class="main-header">💬 RAG Chat</h1>', unsafe_allow_html=True)
    st.markdown(f"### Ask about {companies.get(selected_ticker, selected_ticker)}")
    
    if not st.session_state.rag:
        st.warning("⚠️ RAG system not initialized. Check OpenSearch connection.")
        st.info("Make sure Docker is running: `docker-compose up -d`")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the 10-K filing..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag.query(
                            prompt,
                            ticker=selected_ticker,
                            k=5,
                            verbose=False
                        )
                        st.markdown(response['answer'])
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response['answer']
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")


# ============================================================================
# PAGE: ANOMALIES - ENHANCED v0.5.2
# ============================================================================

elif page == "🚨 Anomalies":
    st.markdown(f'<h1 class="main-header">🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    st.markdown(f"### Unusual patterns in {companies.get(selected_ticker, selected_ticker)}")
    
    available_years = get_available_years_local(selected_ticker)
    
    if len(available_years) < 2:
        st.warning(f"⚠️ Need 2+ years of data for anomaly detection.")
        st.info(f"Currently have {len(available_years)} year(s). Download more in Data Management.")
    else:
        st.markdown("---")
        
        # Control buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Available data:** {len(available_years)} years of filings")
        
        with col2:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        with col3:
            # Force reload detector button (for debugging)
            if st.button("🔄 Reload", use_container_width=True, help="Force reload anomaly detector"):
                st.session_state._force_reload_detector = True
                st.session_state.anomaly_report = None
                st.session_state.anomaly_ticker = None
                st.rerun()
        
        # Run analysis
        if analyze_btn and st.session_state.anomaly_detector:
            with st.spinner(f"Analyzing {selected_ticker}..."):
                # Create fresh detector instance to ensure latest code
                detector = AnomalyDetector()
                report = detector.analyze_ticker(selected_ticker)
                
                # SAVE REPORT TO DISK
                if 'error' not in report or report.get('total_anomalies', 0) > 0:
                    detector.save_report(report)
                    st.toast(f"Report saved to data/anomalies/", icon="💾")
                
                st.session_state.anomaly_report = report
                st.session_state.anomaly_ticker = selected_ticker
        
        # Display results
        report = st.session_state.anomaly_report
        
        if report and st.session_state.anomaly_ticker == selected_ticker:
            if 'error' in report and report.get('total_anomalies', 0) == 0:
                st.error(f"Error: {report['error']}")
            else:
                # Comparison info header
                current_date = report.get('current_filing_date', 'latest')
                compared_to = report.get('compared_to', 'previous')
                num_historical = report.get('num_historical_filings', 0)
                
                st.success(f"✅ Analysis complete!")
                st.markdown(f"""
                📅 **Current filing:** `{current_date}`  
                🔄 **Compared to:** `{compared_to}` (+ {num_historical} historical filing{'s' if num_historical != 1 else ''})
                """)
                
                st.markdown("---")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Anomalies", report.get('total_anomalies', 0))
                with col2:
                    st.metric("🔴 High", report.get('anomalies_by_severity', {}).get('high', 0))
                with col3:
                    st.metric("🟡 Medium", report.get('anomalies_by_severity', {}).get('medium', 0))
                with col4:
                    st.metric("🟢 Low", report.get('anomalies_by_severity', {}).get('low', 0))
                
                st.markdown("---")
                
                # Display anomalies
                anomalies = report.get('anomalies', [])
                
                if not anomalies:
                    st.success("✅ No anomalies detected - filing appears consistent with historical patterns!")
                else:
                    st.markdown("### 📋 Detected Anomalies")
                    st.caption("Click to expand each anomaly for full details and context.")
                    
                    # Group by severity
                    high = [a for a in anomalies if a.get('severity') == 'high']
                    medium = [a for a in anomalies if a.get('severity') == 'medium']
                    low = [a for a in anomalies if a.get('severity') == 'low']
                    
                    if high:
                        st.markdown("#### 🔴 High Severity")
                        for anomaly in high:
                            render_anomaly_card(anomaly)
                    
                    if medium:
                        st.markdown("#### 🟡 Medium Severity")
                        for anomaly in medium:
                            render_anomaly_card(anomaly)
                    
                    if low:
                        st.markdown("#### 🟢 Low Severity")
                        for anomaly in low:
                            render_anomaly_card(anomaly)
        else:
            st.info("👆 Click **Analyze** to detect anomalies in the latest filing")
            
            # Show if there's a cached report on disk
            anomaly_dir = Path("data/anomalies")
            cached_files = list(anomaly_dir.glob(f"{selected_ticker}_*_anomalies.json"))
            if cached_files:
                st.caption(f"📁 Found {len(cached_files)} cached report(s) on disk")
                if st.button("Load latest cached report"):
                    latest = sorted(cached_files)[-1]
                    with open(latest, 'r') as f:
                        st.session_state.anomaly_report = json.load(f)
                        st.session_state.anomaly_ticker = selected_ticker
                    st.rerun()


# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif page == "📊 Analytics":
    st.markdown(f'<h1 class="main-header">📊 Analytics</h1>', unsafe_allow_html=True)
    
    st.markdown("#### 📈 Sentiment Comparison (All Companies)")
    
    comparison = []
    for ticker in list(get_companies().keys()):
        sentiment, date = load_sentiment_for_year(ticker)
        if sentiment:
            comparison.append({
                'Company': ticker,
                'Date': date,
                'Score': sentiment.get('overall', {}).get('compound', 0),
                'Signal': sentiment.get('overall', {}).get('signal', 'N/A')
            })
    
    if comparison:
        df = pd.DataFrame(comparison)
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=df['Company'],
                y=df['Score'],
                marker_color=['#3b82f6' if s >= 0 else '#ef4444' for s in df['Score']],
                text=[f"{s:.2f}" for s in df['Score']],
                textposition='outside'
            )
        ])
        fig.update_layout(
            yaxis_range=[-1, 1],
            height=400,
            yaxis_title="Sentiment Score"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No sentiment data available. Process some filings first!")


# ============================================================================
# PAGE: DATA MANAGEMENT
# ============================================================================

elif page == "📥 Data Management":
    st.markdown(f'<h1 class="main-header">📥 Data Management</h1>', unsafe_allow_html=True)
    
    # Company selector
    col1, col2 = st.columns([3, 1])
    with col1:
        manage_ticker = st.selectbox(
            "Select Company",
            options=list(get_companies().keys()),
            format_func=lambda x: get_companies()[x],
            index=list(get_companies().keys()).index(selected_ticker) if selected_ticker in get_companies() else 0,
            key="manage_ticker_select"
        )
    with col2:
        st.metric("Selected", manage_ticker)
    
    st.markdown("---")
    
    # Status table
    st.markdown("#### 📊 Filing Status")
    
    status = get_filing_status_local(manage_ticker)
    
    if status:
        data = []
        for date, info in status.items():
            data.append({
                'Date': date,
                'Year': getattr(info, 'year', date[:4]),
                'Downloaded': '✓' if getattr(info, 'downloaded', False) else '○',
                'Parsed': '✓' if getattr(info, 'parsed', False) else '○',
                'Sections': '✓' if getattr(info, 'sections_extracted', False) else '○',
                'Sentiment': '✓' if getattr(info, 'sentiment_analyzed', False) else '○',
                'Complete': '✅' if getattr(info, 'is_complete', False) else '⏳'
            })
        
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        
        complete = sum(1 for s in status.values() if getattr(s, 'is_complete', False))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Filings", len(status))
        with col2:
            st.metric("Complete", complete)
        with col3:
            st.metric("Pending", len(status) - complete)
    else:
        st.info(f"No filings found for {manage_ticker}")
    
    st.markdown("---")
    
    # Download section
    st.markdown("#### 🚀 Download & Process")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        years = st.selectbox(
            "Years to download",
            [1, 2, 3, 5, 10],
            index=2,
            format_func=lambda x: f"Last {x} year{'s' if x > 1 else ''}",
            key="download_years"
        )
    
    with col2:
        auto = st.checkbox("Auto-process", value=True, key="auto_process_check")
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        start = st.button("📥 Download", type="primary", use_container_width=True)
    
    if start and AutomatedPipeline:
        pipeline = AutomatedPipeline()
        progress = st.progress(0)
        status_text = st.empty()
        
        def update(msg, pct):
            progress.progress(min(pct / 100, 1.0))
            status_text.info(f"📊 {msg}")
        
        pipeline.set_progress_callback(update)
        
        try:
            result = pipeline.process_company(manage_ticker, years=years, auto_process=auto)
            
            if result.success:
                status_text.success(f"✅ {result.message}")
                st.balloons()
            else:
                status_text.error(f"❌ {result.message}")
        except Exception as e:
            status_text.error(f"Error: {e}")
    elif start:
        st.error("Pipeline not available. Check imports.")


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #94a3b8;">AlphaExtract v0.5.2 | '
    'Built with Streamlit, FinBERT, OpenSearch & Gemini</div>',
    unsafe_allow_html=True
)