"""
AlphaExtract Dashboard - v2.0.0
================================
Full-featured Streamlit dashboard with:
- Sentiment overview & signal cards
- Multi-quarter comparison
- Ensemble & backtest visualization
- RAG chatbot (local, no Docker needed)
- Anomaly detection
- Sector analytics
- Data management pipeline
"""

import streamlit as st
import sys
from pathlib import Path
import json
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# IMPORTS WITH ERROR HANDLING
# ============================================================================

try:
    from src.rag.local_rag import LocalRAG
except Exception:
    LocalRAG = None

try:
    from src.rag.enhanced_rag import EnhancedRAG
except Exception:
    EnhancedRAG = None

try:
    from src.models.anomaly import AnomalyDetector
except Exception:
    AnomalyDetector = None

try:
    from src.database.supabase_client import SupabaseDB
except Exception:
    SupabaseDB = None

try:
    from src.data.company_search import CompanySearch
except Exception:
    CompanySearch = None

try:
    from src.pipeline.automated import (
        AutomatedPipeline,
        get_filing_status,
        get_available_years,
        get_all_sentiment_data,
    )
except Exception:
    AutomatedPipeline = None
    get_filing_status = None
    get_available_years = lambda x: []
    get_all_sentiment_data = lambda x: []

try:
    from src.analysis.comparison import (
        compare_quarters,
        compare_tickers,
        section_deep_dive,
        load_all_filings,
    )
except Exception:
    compare_quarters = None
    compare_tickers = None
    section_deep_dive = None
    load_all_filings = None

try:
    from src.config.settings import Settings
except Exception:
    Settings = None


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="AlphaExtract - AI Financial Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
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
    .sector-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "company_search" not in st.session_state:
    st.session_state.company_search = CompanySearch() if CompanySearch else None

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

if "selected_year" not in st.session_state:
    st.session_state.selected_year = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag" not in st.session_state:
    st.session_state.rag = None
    # Prefer local RAG (no Docker needed)
    if LocalRAG:
        try:
            st.session_state.rag = LocalRAG()
        except Exception as e:
            print(f"LocalRAG init failed: {e}")
    if st.session_state.rag is None and EnhancedRAG:
        try:
            st.session_state.rag = EnhancedRAG()
        except Exception:
            pass

if "anomaly_detector" not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector() if AnomalyDetector else None

if "anomaly_report" not in st.session_state:
    st.session_state.anomaly_report = None

if "anomaly_ticker" not in st.session_state:
    st.session_state.anomaly_ticker = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

SIGNAL_COLORS = {
    "STRONG_BUY": "#00C853",
    "BUY": "#69F0AE",
    "HOLD": "#FFD600",
    "SELL": "#FF9100",
    "STRONG_SELL": "#FF1744",
    "N/A": "#9E9E9E",
}


def get_companies() -> dict:
    """Get all available companies (defaults + watchlist + settings)."""
    if st.session_state.company_search:
        return st.session_state.company_search.get_all_companies()
    if Settings and hasattr(Settings, "DEFAULT_TICKERS"):
        return Settings.DEFAULT_TICKERS
    return {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corp.",
        "TSLA": "Tesla Inc.",
    }


def get_sector_for_ticker(ticker: str) -> str:
    """Find which sector a ticker belongs to."""
    if not Settings or not hasattr(Settings, "SECTOR_TICKERS"):
        return "Unknown"
    for sector, tickers in Settings.SECTOR_TICKERS.items():
        if ticker.upper() in tickers:
            return sector
    return "Other"


def normalize_sentiment_data(data: dict) -> dict:
    """Normalize sentiment JSON to use consistent keys."""
    if not data or "sections" not in data:
        return data
    sections = data["sections"]
    normalized = {}
    key_map = {"1a": "item_1a", "1b": "item_1b", "1": "item_1",
               "7": "item_7", "7a": "item_7a", "8": "item_8", "9": "item_9"}
    for key, value in sections.items():
        if key.startswith("item_"):
            normalized[key] = value
        elif key in key_map:
            normalized[key_map[key]] = value
        else:
            normalized[f"item_{key}"] = value
    data["sections"] = normalized
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
        year_files = [f for f in files if f.stem.split("_")[1].startswith(year)]
        if year_files:
            files = year_files
    latest_file = sorted(files, reverse=True)[0]
    with open(latest_file, "r") as f:
        sentiment_data = json.load(f)
    sentiment_data = normalize_sentiment_data(sentiment_data)
    filing_date = latest_file.stem.split("_")[1]
    return sentiment_data, filing_date


def load_ensemble_for_year(ticker: str, year: str = None) -> tuple:
    """Load ensemble data for a specific year or latest."""
    sentiment_dir = Path("data/sentiment")
    ticker = ticker.upper()
    files = list(sentiment_dir.glob(f"{ticker}_*_ensemble.json"))
    if not files:
        return None, None
    if year:
        year_files = [f for f in files if f.stem.split("_")[1].startswith(year)]
        if year_files:
            files = year_files
    latest_file = sorted(files, reverse=True)[0]
    with open(latest_file, "r") as f:
        data = json.load(f)
    filing_date = latest_file.stem.split("_")[1]
    return data, filing_date


def get_section_data(sentiment_data: dict, section_key: str) -> dict:
    """Safely get section data with fallback."""
    if not sentiment_data:
        return {"scores": {"compound": 0}, "signal": "N/A", "word_count": 0}
    sections = sentiment_data.get("sections", {})
    if section_key in sections:
        return sections[section_key]
    return {"scores": {"compound": 0}, "signal": "N/A", "word_count": 0}


def get_historical_sentiment(ticker: str) -> list:
    """Get all historical sentiment data for trend chart."""
    try:
        sentiment_data = get_all_sentiment_data(ticker)
    except Exception:
        sentiment_data = []
    if not sentiment_data:
        sentiment_dir = Path("data/sentiment")
        files = sorted(sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
        sentiment_data = []
        for f in files:
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                    filing_date = f.stem.split("_")[1]
                    sentiment_data.append((filing_date, data))
            except Exception:
                pass
    history = []
    for filing_date, data in sentiment_data:
        history.append({
            "filing_date": filing_date,
            "year": filing_date[:4],
            "overall_compound": data.get("overall", {}).get("compound", 0),
            "signal": data.get("overall", {}).get("signal", "HOLD"),
        })
    return history


def get_available_years_local(ticker: str) -> list:
    """Get available years for a ticker."""
    try:
        return get_available_years(ticker)
    except Exception:
        sentiment_dir = Path("data/sentiment")
        files = list(sentiment_dir.glob(f"{ticker}_*_sentiment.json"))
        years = set()
        for f in files:
            parts = f.stem.split("_")
            if len(parts) >= 2:
                years.add(parts[1][:4])
        return sorted(years, reverse=True)


def get_filing_status_local(ticker: str) -> dict:
    """Get filing status."""
    try:
        if get_filing_status:
            return get_filing_status(ticker)
    except Exception:
        pass
    return {}


def render_anomaly_card(anomaly: dict):
    """Render a single anomaly with enhanced fields."""
    title = anomaly.get("title") or anomaly.get("keyword", "").replace("_", " ").title() or "Anomaly Detected"
    category = anomaly.get("category") or anomaly.get("type", "General").replace("_", " ").title()
    severity = anomaly.get("severity", "medium")
    severity_reason = anomaly.get("severity_reason") or f"Severity: {severity}"
    description = anomaly.get("description", "")
    explanation = anomaly.get("explanation") or "No additional context available."
    context = anomaly.get("context", [])

    icon = {"high": "🔴", "medium": "🟡"}.get(severity, "🟢")

    with st.expander(f"{icon} {title}", expanded=(severity == "high")):
        st.markdown(f"**Category:** {category}")
        if description:
            st.markdown(f"**What changed:** {description}")
        st.markdown("---")
        st.markdown("**Why this matters:**")
        st.info(explanation)
        st.markdown(f"**Severity ({severity.upper()}):** {severity_reason}")

        if context:
            st.markdown("---")
            st.markdown("**Context from filing:**")
            for ctx in context[:3]:
                display = ctx[:400] + "..." if len(ctx) > 400 else ctx
                st.markdown(f"> _{display}_")

        if "current_score" in anomaly or "current_count" in anomaly:
            st.markdown("---")
            st.markdown("**Data:**")
            if "current_score" in anomaly:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Current", f"{anomaly.get('current_score', 0):.3f}")
                with c2:
                    st.metric("Previous", f"{anomaly.get('previous_score', 0):.3f}")
                with c3:
                    st.metric("Change", f"{anomaly.get('delta', 0):+.3f}")
            elif "current_count" in anomaly:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Current Mentions", anomaly.get("current_count", 0))
                with c2:
                    avg = anomaly.get("average_count", anomaly.get("historical_count", 0))
                    st.metric("Historical Avg", f"{avg:.1f}" if isinstance(avg, float) else str(avg))
                with c3:
                    ratio = anomaly.get("ratio", 1.0)
                    st.metric("Change", f"{ratio:.1f}x" if ratio >= 1 else f"{1/ratio:.1f}x less")


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Company Search
    if st.session_state.company_search:
        st.markdown("### Search Company")
        search_query = st.text_input("Ticker or name", placeholder="NVDA, NVIDIA...", key="search_input")
        if search_query and len(search_query) >= 2:
            if search_query.upper() == search_query or len(search_query) <= 5:
                result = st.session_state.company_search.search_by_ticker(search_query)
                results = [result] if result else []
            else:
                results = st.session_state.company_search.search_by_name(search_query, limit=5)
            if results:
                for company in results:
                    if company:
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.text(f"{company.ticker}")
                        with c2:
                            if st.button("+", key=f"add_{company.ticker}"):
                                st.session_state.company_search.add_to_watchlist(company.ticker)
                                st.success("Added!")
            else:
                st.caption("No results")
        st.markdown("---")

    # Company Selector
    st.markdown("### Select Company")
    companies = get_companies()
    selected_ticker = st.selectbox(
        "Company",
        options=list(companies.keys()),
        format_func=lambda x: f"{x} - {companies[x]}",
        index=list(companies.keys()).index(st.session_state.selected_ticker)
        if st.session_state.selected_ticker in companies else 0,
        key="ticker_selector",
    )

    if selected_ticker != st.session_state.selected_ticker:
        st.session_state.anomaly_report = None
        st.session_state.anomaly_ticker = None
    st.session_state.selected_ticker = selected_ticker

    # Sector badge
    sector = get_sector_for_ticker(selected_ticker)
    st.caption(f"Sector: **{sector}**")

    # Year Selector
    available_years = get_available_years_local(selected_ticker)
    if available_years:
        st.markdown("### Filing Year")
        year_options = ["Latest"] + available_years
        current_year = st.session_state.selected_year
        if current_year and current_year in available_years:
            default_index = year_options.index(current_year)
        else:
            default_index = 0
        selected_year = st.selectbox(
            "Year", options=year_options, index=default_index,
            key=f"year_selector_{selected_ticker}",
        )
        st.session_state.selected_year = None if selected_year == "Latest" else selected_year
    else:
        st.session_state.selected_year = None

    st.markdown("---")

    # Navigation
    st.markdown("### Navigation")
    page = st.radio(
        "Page",
        [
            "Dashboard",
            "Multi-Quarter",
            "Ensemble",
            "Backtesting",
            "RAG Chat",
            "Anomalies",
            "Sector Analytics",
            "Data Management",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Watchlist
    if st.session_state.company_search:
        watchlist = st.session_state.company_search.get_watchlist()
        if watchlist:
            with st.expander(f"Watchlist ({len(watchlist)})"):
                for item in watchlist:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.text(item["ticker"])
                    with c2:
                        if st.button("x", key=f"rm_{item['ticker']}"):
                            st.session_state.company_search.remove_from_watchlist(item["ticker"])

    # Status
    st.markdown("---")
    status = get_filing_status_local(selected_ticker)
    if status:
        complete = sum(1 for s in status.values() if getattr(s, "is_complete", False))
        st.metric("Processed", f"{complete}/{len(status)}")


# ============================================================================
# HELPER: BACKTEST DETAIL RENDERER (must be defined before page routing)
# ============================================================================


def _render_backtest_detail(data: dict, title: str):
    """Render detailed backtest results."""
    with st.expander(f"**{title} Detail**", expanded=True):
        config = data.get("config", {})
        summary = data.get("summary", {})

        st.markdown(f"Tickers: {', '.join(config.get('tickers', []))}")
        st.markdown(f"Mode: {config.get('mode', '?')} | Window: {config.get('return_window_days', '?')} days")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Signals", summary.get("total_signals", 0))
        with mc2:
            st.metric("Hit Rate", f"{summary.get('hit_rate', 0):.1%}")
        with mc3:
            st.metric("Dir. Accuracy", f"{summary.get('directional_accuracy', 0):.1%}")
        with mc4:
            sharpe = summary.get("sharpe_ratio")
            st.metric("Sharpe", f"{sharpe:.3f}" if sharpe else "N/A")

        # Precision by signal
        precision = data.get("precision_by_signal", {})
        if precision:
            st.markdown("**Precision by Signal:**")
            prec_data = []
            for sig in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]:
                if sig in precision:
                    s = precision[sig]
                    prec_data.append({
                        "Signal": sig,
                        "Correct": s["correct"],
                        "Total": s["total"],
                        "Precision": f"{s['precision']:.1%}",
                    })
            if prec_data:
                st.dataframe(pd.DataFrame(prec_data), width="stretch", hide_index=True)

        # Confusion matrix
        cm = data.get("confusion_matrix", {})
        if cm:
            st.markdown("**Confusion Matrix:**")
            cm_rows = []
            for predicted in ["bullish", "neutral", "bearish"]:
                if predicted in cm:
                    row = cm[predicted]
                    cm_rows.append({
                        "Predicted": predicted.title(),
                        "Up": row.get("up", 0),
                        "Flat": row.get("flat", 0),
                        "Down": row.get("down", 0),
                    })
            if cm_rows:
                st.dataframe(pd.DataFrame(cm_rows), width="stretch", hide_index=True)

        # Individual results
        results = data.get("results", [])
        if results:
            st.markdown("**Individual Signals:**")
            res_df = pd.DataFrame(results)
            cols_to_show = ["ticker", "filing_date", "signal", "score", "actual_return_pct", "actual_direction"]
            available_cols = [c for c in cols_to_show if c in res_df.columns]
            st.dataframe(res_df[available_cols], width="stretch", hide_index=True)


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "Dashboard":
    st.markdown('<h1 class="main-header">AlphaExtract</h1>', unsafe_allow_html=True)
    st.markdown(f"### {companies.get(selected_ticker, selected_ticker)}")

    year_filter = st.session_state.selected_year
    sentiment_data, filing_date = load_sentiment_for_year(selected_ticker, year_filter)

    if not sentiment_data:
        st.warning(f"No sentiment data for {selected_ticker}")
        st.info("Go to **Data Management** to process filings.")
    else:
        if year_filter:
            st.success(f"Showing **{year_filter}** data (Filed: {filing_date})")
        else:
            st.info(f"Showing **latest** filing ({filing_date})")

        st.markdown("---")

        risk = get_section_data(sentiment_data, "item_1a")
        mda = get_section_data(sentiment_data, "item_7")
        fin = get_section_data(sentiment_data, "item_8")
        overall = sentiment_data.get("overall", {})
        overall_score = overall.get("compound", 0)
        overall_signal = overall.get("signal", "N/A")

        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            color = SIGNAL_COLORS.get(overall_signal, "#9E9E9E")
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">Overall Signal</div>
                <div style="color: {color}; font-weight: bold; font-size: 1.5rem;">{overall_signal}</div>
                <div style="color: #94a3b8;">Score: {overall_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            rs = risk.get("scores", {}).get("compound", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">Risk Factors</div>
                <div style="font-weight: bold; font-size: 1.3rem;">{risk.get('signal', 'N/A')}</div>
                <div style="color: #94a3b8;">Score: {rs:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            ms = mda.get("scores", {}).get("compound", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">MD&A</div>
                <div style="font-weight: bold; font-size: 1.3rem;">{mda.get('signal', 'N/A')}</div>
                <div style="color: #94a3b8;">Score: {ms:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            fs = fin.get("scores", {}).get("compound", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">Financials</div>
                <div style="font-weight: bold; font-size: 1.3rem;">{fin.get('signal', 'N/A')}</div>
                <div style="color: #94a3b8;">Score: {fs:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Sentiment by Section")
            sections_chart = {"Risk Factors": rs, "MD&A": ms, "Financials": fs}
            fig = go.Figure(data=[go.Bar(
                x=list(sections_chart.keys()),
                y=list(sections_chart.values()),
                marker_color=["#3b82f6" if v >= 0 else "#ef4444" for v in sections_chart.values()],
                text=[f"{v:+.3f}" for v in sections_chart.values()],
                textposition="outside",
            )])
            fig.update_layout(yaxis_title="Score", yaxis_range=[-1, 1], height=350,
                              margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, width="stretch", key=f"section_chart_{selected_ticker}_{year_filter}")

        with col_right:
            st.markdown("#### Historical Trend")
            history = get_historical_sentiment(selected_ticker)
            if len(history) >= 2:
                history.sort(key=lambda x: x["filing_date"])
                dates = [h["filing_date"] for h in history]
                scores = [h["overall_compound"] for h in history]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=scores, mode="lines+markers",
                    line=dict(color="#3b82f6", width=3), marker=dict(size=10), name="Sentiment",
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_hrect(y0=0.2, y1=1, fillcolor="green", opacity=0.1, line_width=0)
                fig.add_hrect(y0=-1, y1=-0.2, fillcolor="red", opacity=0.1, line_width=0)
                fig.update_layout(yaxis_title="Score", yaxis_range=[-1, 1], height=350,
                                  margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                st.plotly_chart(fig, width="stretch", key=f"trend_{selected_ticker}")
            else:
                st.info(f"Need 2+ filings for trend chart. Currently have {len(history)}.")


# ============================================================================
# PAGE: MULTI-QUARTER COMPARISON
# ============================================================================

elif page == "Multi-Quarter":
    st.markdown('<h1 class="main-header">Multi-Quarter Comparison</h1>', unsafe_allow_html=True)
    st.markdown(f"### {companies.get(selected_ticker, selected_ticker)}")

    if compare_quarters is None:
        st.error("Comparison module not available. Check imports.")
    else:
        comparison = compare_quarters(selected_ticker)

        if "error" in comparison:
            st.warning(comparison["error"])
            st.info("Download more years in **Data Management** to enable comparison.")
        else:
            # Summary metrics
            summary = comparison["summary"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Filings", comparison["filings_available"])
            with col2:
                trend = comparison["trend"]
                trend_icon = {"improving": "up", "deteriorating": "down", "stable": "right"}.get(trend, "off")
                st.metric("Trend", trend.title(), delta=trend_icon if trend != "stable" else None)
            with col3:
                st.metric("Avg Score", f"{summary['avg_score']:+.3f}")
            with col4:
                st.metric("Score Range", f"{summary['score_range']:.3f}")

            st.markdown("---")

            # Yearly data table
            st.markdown("#### Yearly Sentiment Scores")
            yearly = comparison["yearly_data"]
            table_data = []
            for y in yearly:
                row = {
                    "Year": y["year"],
                    "Filing Date": y["filing_date"],
                    "Overall": round(y["overall_score"], 3),
                    "Signal": y["overall_signal"],
                    "Risk (1A)": round(y["sections"].get("item_1a", {}).get("compound", 0), 3),
                    "MD&A (7)": round(y["sections"].get("item_7", {}).get("compound", 0), 3),
                    "Financials (8)": round(y["sections"].get("item_8", {}).get("compound", 0), 3),
                }
                if y.get("ensemble"):
                    row["Ensemble"] = round(y["ensemble"]["score"], 3)
                    row["Ens. Signal"] = y["ensemble"]["signal"]
                table_data.append(row)

            st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

            st.markdown("---")

            # Multi-line chart: overall + per-section over time
            st.markdown("#### Sentiment Evolution")
            fig = go.Figure()

            dates = [y["filing_date"] for y in yearly]
            fig.add_trace(go.Scatter(
                x=dates, y=[y["overall_score"] for y in yearly],
                mode="lines+markers", name="Overall", line=dict(width=3, color="#3b82f6"),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=[y["sections"].get("item_1a", {}).get("compound", 0) for y in yearly],
                mode="lines+markers", name="Risk Factors", line=dict(width=2, dash="dash", color="#ef4444"),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=[y["sections"].get("item_7", {}).get("compound", 0) for y in yearly],
                mode="lines+markers", name="MD&A", line=dict(width=2, dash="dash", color="#22c55e"),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=[y["sections"].get("item_8", {}).get("compound", 0) for y in yearly],
                mode="lines+markers", name="Financials", line=dict(width=2, dash="dot", color="#f59e0b"),
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
            fig.update_layout(yaxis_title="Score", yaxis_range=[-1, 1], height=450,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, width="stretch")

            # Deltas table
            if comparison["deltas"]:
                st.markdown("#### Year-over-Year Changes")
                delta_data = []
                for d in comparison["deltas"]:
                    row = {
                        "Period": f"{d['from_year']} -> {d['to_year']}",
                        "Overall Delta": f"{d['overall_delta']:+.3f}",
                        "Signal Changed": "Yes" if d["signal_changed"] else "No",
                        "Risk Delta": f"{d['section_deltas'].get('item_1a', {}).get('compound_delta', 0):+.3f}",
                        "MD&A Delta": f"{d['section_deltas'].get('item_7', {}).get('compound_delta', 0):+.3f}",
                    }
                    if "ensemble_delta" in d:
                        row["Ensemble Delta"] = f"{d['ensemble_delta']:+.3f}"
                    delta_data.append(row)
                st.dataframe(pd.DataFrame(delta_data), width="stretch", hide_index=True)

            # Section deep dive
            st.markdown("---")
            st.markdown("#### Section Deep Dive")
            if section_deep_dive:
                section_choice = st.selectbox(
                    "Section", ["item_7", "item_1a", "item_8"],
                    format_func=lambda x: {"item_7": "MD&A", "item_1a": "Risk Factors", "item_8": "Financials"}[x],
                )
                dive = section_deep_dive(selected_ticker, section_choice)
                if dive["data"]:
                    dive_df = pd.DataFrame(dive["data"])
                    st.dataframe(dive_df, width="stretch", hide_index=True)
                    st.caption(f"Word count change: {dive['word_count_change_pct']:+.1f}% over the period")


# ============================================================================
# PAGE: ENSEMBLE
# ============================================================================

elif page == "Ensemble":
    st.markdown('<h1 class="main-header">Ensemble Scoring</h1>', unsafe_allow_html=True)
    st.markdown(f"### {companies.get(selected_ticker, selected_ticker)}")

    year_filter = st.session_state.selected_year
    ensemble_data, filing_date = load_ensemble_for_year(selected_ticker, year_filter)

    if not ensemble_data:
        st.warning(f"No ensemble data for {selected_ticker}")
        st.info("Run `python main.py ensemble {0}` to generate ensemble scores.".format(selected_ticker))
    else:
        st.info(f"Filing: {filing_date}")

        ens = ensemble_data.get("ensemble", {})
        signals = ensemble_data.get("signals", {})

        # Top-level ensemble result
        col1, col2, col3 = st.columns(3)
        with col1:
            color = SIGNAL_COLORS.get(ens.get("signal", "N/A"), "#9E9E9E")
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #64748b;">Ensemble Signal</div>
                <div style="color: {color}; font-weight: bold; font-size: 1.8rem;">{ens.get('signal', 'N/A')}</div>
                <div style="color: #94a3b8;">Score: {ens.get('score', 0):+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            agree = ens.get("components_agree", False)
            st.metric("Components Agree", "Yes" if agree else "No")
        with col3:
            st.metric("Filing Date", filing_date)

        st.markdown("---")

        # Individual signal breakdown
        st.markdown("#### Signal Components")
        sig_col1, sig_col2, sig_col3 = st.columns(3)

        for col, (name, label) in zip(
            [sig_col1, sig_col2, sig_col3],
            [("finbert", "FinBERT"), ("keywords", "Keywords"), ("llm", "LLM")],
        ):
            with col:
                sig = signals.get(name, {})
                sig_score = sig.get("score", 0)
                sig_signal = sig.get("signal", "N/A")
                sig_weight = sig.get("weight", 0)
                color = SIGNAL_COLORS.get(sig_signal, "#9E9E9E")
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #64748b;">{label} (w={sig_weight:.0%})</div>
                    <div style="color: {color}; font-weight: bold; font-size: 1.3rem;">{sig_signal}</div>
                    <div style="color: #94a3b8;">Score: {sig_score:+.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Weighted contribution chart
        st.markdown("#### Weighted Contribution")
        contrib_names = []
        contrib_vals = []
        contrib_colors = []
        for name, label in [("finbert", "FinBERT"), ("keywords", "Keywords"), ("llm", "LLM")]:
            sig = signals.get(name, {})
            weighted = sig.get("score", 0) * sig.get("weight", 0)
            contrib_names.append(label)
            contrib_vals.append(weighted)
            contrib_colors.append("#3b82f6" if weighted >= 0 else "#ef4444")

        fig = go.Figure(data=[go.Bar(
            x=contrib_names, y=contrib_vals,
            marker_color=contrib_colors,
            text=[f"{v:+.4f}" for v in contrib_vals],
            textposition="outside",
        )])
        fig.update_layout(yaxis_title="Weighted Score", height=350,
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, width="stretch")

        # Keyword detail
        kw_detail = signals.get("keywords", {}).get("detail", {})
        if kw_detail:
            st.markdown("---")
            st.markdown("#### Keyword Signal Detail")
            kcol1, kcol2 = st.columns(2)
            with kcol1:
                st.markdown("**Top Risks:**")
                for r in kw_detail.get("top_risks", []):
                    st.markdown(f"- {r['group'].replace('_', ' ').title()} ({r['mentions']} mentions)")
            with kcol2:
                st.markdown("**Top Opportunities:**")
                for o in kw_detail.get("top_opportunities", []):
                    st.markdown(f"- {o['group'].replace('_', ' ').title()} ({o['mentions']} mentions)")

        # LLM detail
        llm_detail = signals.get("llm", {}).get("detail", {})
        if llm_detail and llm_detail.get("available"):
            st.markdown("---")
            st.markdown("#### LLM Analysis")
            st.markdown(f"**Outlook:** {llm_detail.get('outlook', 'N/A')}")
            st.markdown(f"**Confidence:** {llm_detail.get('confidence', 'N/A')}")
            if llm_detail.get("summary"):
                st.info(llm_detail["summary"])
            lcol1, lcol2 = st.columns(2)
            with lcol1:
                st.markdown("**Bull Factors:**")
                for f_item in llm_detail.get("bull_factors", []):
                    st.markdown(f"- {f_item}")
            with lcol2:
                st.markdown("**Bear Factors:**")
                for f_item in llm_detail.get("bear_factors", []):
                    st.markdown(f"- {f_item}")


# ============================================================================
# PAGE: BACKTESTING
# ============================================================================

elif page == "Backtesting":
    st.markdown('<h1 class="main-header">Backtesting Results</h1>', unsafe_allow_html=True)

    backtest_dir = Path("data/backtest")
    backtest_files = sorted(backtest_dir.glob("*.json"), reverse=True) if backtest_dir.exists() else []

    if not backtest_files:
        st.warning("No backtest results found.")
        st.info("Run `python main.py backtest AAPL MSFT --compare` to generate results.")
    else:
        # File selector
        file_names = [f.name for f in backtest_files]
        selected_file = st.selectbox("Select Result", file_names)
        filepath = backtest_dir / selected_file

        with open(filepath, "r") as f:
            bt_data = json.load(f)

        # Check if comparison or single backtest
        if "comparison" in bt_data:
            st.markdown("#### Model Comparison: FinBERT vs Ensemble")
            comparison = bt_data["comparison"]

            for label, stats in comparison.items():
                st.markdown(f"**{label}:**")
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Signals", stats.get("n", 0))
                with mc2:
                    st.metric("Hit Rate", f"{stats.get('hit_rate', 0):.1%}")
                with mc3:
                    st.metric("Dir. Accuracy", f"{stats.get('directional_accuracy', 0):.1%}")
                with mc4:
                    sharpe = stats.get("sharpe")
                    st.metric("Sharpe", f"{sharpe:.3f}" if sharpe else "N/A")

            # Side-by-side bar chart
            labels = list(comparison.keys())
            if len(labels) == 2:
                fig = go.Figure(data=[
                    go.Bar(name=labels[0], x=["Hit Rate", "Dir. Accuracy"],
                           y=[comparison[labels[0]].get("hit_rate", 0),
                              comparison[labels[0]].get("directional_accuracy", 0)],
                           marker_color="#3b82f6"),
                    go.Bar(name=labels[1], x=["Hit Rate", "Dir. Accuracy"],
                           y=[comparison[labels[1]].get("hit_rate", 0),
                              comparison[labels[1]].get("directional_accuracy", 0)],
                           marker_color="#22c55e"),
                ])
                fig.update_layout(barmode="group", yaxis_title="Score", height=400,
                                  yaxis_range=[0, 1])
                st.plotly_chart(fig, width="stretch")

            # Show detailed results for each model
            for key in ["finbert", "ensemble"]:
                if key in bt_data:
                    _render_backtest_detail(bt_data[key], key.title())

        else:
            _render_backtest_detail(bt_data, "Backtest")


# ============================================================================
# PAGE: RAG CHAT
# ============================================================================

elif page == "RAG Chat":
    st.markdown('<h1 class="main-header">RAG Chat</h1>', unsafe_allow_html=True)
    st.markdown(f"### Ask about {companies.get(selected_ticker, selected_ticker)}")

    if not st.session_state.rag:
        st.warning("RAG system not initialized. Install sentence-transformers or set an LLM API key.")
    else:
        rag = st.session_state.rag

        # Index for current ticker if needed
        if hasattr(rag, "is_indexed") and not rag.is_indexed:
            with st.spinner(f"Indexing documents for {selected_ticker}..."):
                rag.index(selected_ticker)

        # Status
        if hasattr(rag, "status"):
            rag_status = rag.status()
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Documents", rag_status.get("documents", "?"))
            with sc2:
                st.metric("Embeddings", "Yes" if rag_status.get("has_embeddings") else "Keyword mode")
            with sc3:
                st.metric("LLM", rag_status.get("llm_provider", "None") or "None")

        st.markdown("---")

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
                        response = rag.query(prompt, ticker=selected_ticker, k=5, verbose=False)
                        answer = response["answer"]
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                        # Show sources
                        sources = response.get("sources", [])
                        if sources:
                            with st.expander("Sources"):
                                for s in sources:
                                    st.caption(
                                        f"{s['ticker']} | {s['filing_date']} | {s['section']} | "
                                        f"score: {s.get('score', 0):.3f}"
                                    )
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history.clear()
                if hasattr(rag, "clear_history"):
                    rag.clear_history()
                st.rerun()


# ============================================================================
# PAGE: ANOMALIES
# ============================================================================

elif page == "Anomalies":
    st.markdown('<h1 class="main-header">Anomaly Detection</h1>', unsafe_allow_html=True)
    st.markdown(f"### {companies.get(selected_ticker, selected_ticker)}")

    available_years = get_available_years_local(selected_ticker)

    if len(available_years) < 2:
        st.warning(f"Need 2+ years of data. Currently have {len(available_years)}.")
        st.info("Download more years in **Data Management**.")
    else:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Available data:** {len(available_years)} years")
        with col2:
            analyze_btn = st.button("Analyze", type="primary", width="stretch")
        with col3:
            if st.button("Reload", width="stretch"):
                st.session_state.anomaly_report = None
                st.session_state.anomaly_ticker = None
                st.rerun()

        if analyze_btn and st.session_state.anomaly_detector:
            with st.spinner(f"Analyzing {selected_ticker}..."):
                detector = AnomalyDetector()
                report = detector.analyze_ticker(selected_ticker)
                if "error" not in report or report.get("total_anomalies", 0) > 0:
                    detector.save_report(report)
                st.session_state.anomaly_report = report
                st.session_state.anomaly_ticker = selected_ticker

        report = st.session_state.anomaly_report
        if report and st.session_state.anomaly_ticker == selected_ticker:
            if "error" in report and report.get("total_anomalies", 0) == 0:
                st.error(f"Error: {report['error']}")
            else:
                st.success("Analysis complete!")
                st.markdown(f"Current: `{report.get('current_filing_date', '?')}` | "
                            f"Compared to: `{report.get('compared_to', '?')}`")

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Total", report.get("total_anomalies", 0))
                with mc2:
                    st.metric("High", report.get("anomalies_by_severity", {}).get("high", 0))
                with mc3:
                    st.metric("Medium", report.get("anomalies_by_severity", {}).get("medium", 0))
                with mc4:
                    st.metric("Low", report.get("anomalies_by_severity", {}).get("low", 0))

                anomalies = report.get("anomalies", [])
                if not anomalies:
                    st.success("No anomalies detected - filing is consistent with history!")
                else:
                    for severity_level in ["high", "medium", "low"]:
                        items = [a for a in anomalies if a.get("severity") == severity_level]
                        if items:
                            icon = {"high": "High", "medium": "Medium", "low": "Low"}[severity_level]
                            st.markdown(f"#### {icon} Severity")
                            for anomaly in items:
                                render_anomaly_card(anomaly)
        else:
            st.info("Click **Analyze** to detect anomalies.")
            anomaly_dir = Path("data/anomalies")
            cached = list(anomaly_dir.glob(f"{selected_ticker}_*_anomalies.json")) if anomaly_dir.exists() else []
            if cached:
                st.caption(f"Found {len(cached)} cached report(s)")
                if st.button("Load cached"):
                    latest = sorted(cached)[-1]
                    with open(latest, "r") as f:
                        st.session_state.anomaly_report = json.load(f)
                        st.session_state.anomaly_ticker = selected_ticker
                    st.rerun()


# ============================================================================
# PAGE: SECTOR ANALYTICS
# ============================================================================

elif page == "Sector Analytics":
    st.markdown('<h1 class="main-header">Sector Analytics</h1>', unsafe_allow_html=True)

    if not Settings or not hasattr(Settings, "SECTOR_TICKERS"):
        st.error("Sector configuration not available.")
    else:
        # Cross-company comparison
        st.markdown("#### Company Sentiment Comparison")

        comparison_data = []
        for ticker in list(get_companies().keys()):
            sentiment, date = load_sentiment_for_year(ticker)
            if sentiment:
                comparison_data.append({
                    "Company": ticker,
                    "Sector": get_sector_for_ticker(ticker),
                    "Date": date,
                    "Score": sentiment.get("overall", {}).get("compound", 0),
                    "Signal": sentiment.get("overall", {}).get("signal", "N/A"),
                })

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            # Bar chart colored by signal
            fig = go.Figure(data=[go.Bar(
                x=df["Company"],
                y=df["Score"],
                marker_color=[
                    "#3b82f6" if s >= 0 else "#ef4444" for s in df["Score"]
                ],
                text=[f"{s:.2f}" for s in df["Score"]],
                textposition="outside",
            )])
            fig.update_layout(yaxis_range=[-1, 1], height=400, yaxis_title="Sentiment Score")
            st.plotly_chart(fig, width="stretch")

            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No sentiment data available. Process some filings first!")

        st.markdown("---")

        # Sector breakdown
        st.markdown("#### Sector Breakdown")
        st.markdown("Available sectors and tickers for analysis:")

        for sector, tickers in Settings.SECTOR_TICKERS.items():
            with st.expander(f"**{sector}** ({len(tickers)} companies)"):
                # Check which have data
                sector_data = []
                for t in tickers:
                    sentiment_dir = Path("data/sentiment")
                    has_sentiment = len(list(sentiment_dir.glob(f"{t}_*_sentiment.json"))) > 0
                    has_ensemble = len(list(sentiment_dir.glob(f"{t}_*_ensemble.json"))) > 0
                    sector_data.append({
                        "Ticker": t,
                        "Sentiment Data": "Yes" if has_sentiment else "No",
                        "Ensemble Data": "Yes" if has_ensemble else "No",
                    })
                st.dataframe(pd.DataFrame(sector_data), width="stretch", hide_index=True)

        # Cross-sector comparison (only tickers with data)
        st.markdown("---")
        st.markdown("#### Cross-Sector Sentiment (Tickers with Data)")

        sector_scores = {}
        sentiment_dir = Path("data/sentiment")
        for sector, tickers in Settings.SECTOR_TICKERS.items():
            scores = []
            for t in tickers:
                files = list(sentiment_dir.glob(f"{t}_*_sentiment.json"))
                if files:
                    latest = sorted(files)[-1]
                    try:
                        with open(latest, "r") as f:
                            data = json.load(f)
                        score = data.get("overall", {}).get("compound", 0)
                        scores.append(score)
                    except Exception:
                        pass
            if scores:
                sector_scores[sector] = sum(scores) / len(scores)

        if sector_scores:
            fig = go.Figure(data=[go.Bar(
                x=list(sector_scores.keys()),
                y=list(sector_scores.values()),
                marker_color=[
                    "#3b82f6" if v >= 0 else "#ef4444" for v in sector_scores.values()
                ],
                text=[f"{v:+.3f}" for v in sector_scores.values()],
                textposition="outside",
            )])
            fig.update_layout(yaxis_title="Avg Sentiment", yaxis_range=[-1, 1], height=400)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No sector data available yet. Process filings across sectors to see comparison.")

        # Multi-ticker comparison tool
        st.markdown("---")
        st.markdown("#### Compare Specific Tickers")
        if compare_tickers:
            all_available = []
            for f in sentiment_dir.glob("*_*_sentiment.json"):
                ticker = f.stem.split("_")[0]
                if ticker not in all_available:
                    all_available.append(ticker)
            all_available.sort()

            if all_available:
                compare_selection = st.multiselect(
                    "Select tickers to compare",
                    all_available,
                    default=all_available[:min(4, len(all_available))],
                )

                if compare_selection and len(compare_selection) >= 2:
                    result = compare_tickers(compare_selection)
                    tickers_data = result.get("tickers", {})
                    rows = []
                    for t, data in tickers_data.items():
                        if "error" not in data:
                            rows.append({
                                "Ticker": t,
                                "Year": data.get("year", "?"),
                                "Overall": round(data.get("overall_score", 0), 3),
                                "Signal": data.get("overall_signal", "N/A"),
                                "Risk (1A)": round(data.get("item_1a", 0), 3),
                                "MD&A (7)": round(data.get("item_7", 0), 3),
                                "Financials (8)": round(data.get("item_8", 0), 3),
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("No tickers with data. Process some filings first!")


# ============================================================================
# PAGE: DATA MANAGEMENT
# ============================================================================

elif page == "Data Management":
    st.markdown('<h1 class="main-header">Data Management</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        manage_ticker = st.selectbox(
            "Select Company",
            options=list(get_companies().keys()),
            format_func=lambda x: f"{x} - {get_companies()[x]}",
            index=list(get_companies().keys()).index(selected_ticker)
            if selected_ticker in get_companies() else 0,
            key="manage_ticker_select",
        )
    with col2:
        st.metric("Selected", manage_ticker)

    st.markdown("---")

    # Status table
    st.markdown("#### Filing Status")
    status = get_filing_status_local(manage_ticker)
    if status:
        data = []
        for date, info in status.items():
            data.append({
                "Date": date,
                "Year": getattr(info, "year", date[:4]),
                "Downloaded": "Yes" if getattr(info, "downloaded", False) else "-",
                "Parsed": "Yes" if getattr(info, "parsed", False) else "-",
                "Sections": "Yes" if getattr(info, "sections_extracted", False) else "-",
                "Sentiment": "Yes" if getattr(info, "sentiment_analyzed", False) else "-",
                "Complete": "Done" if getattr(info, "is_complete", False) else "Pending",
            })
        st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)

        complete = sum(1 for s in status.values() if getattr(s, "is_complete", False))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Filings", len(status))
        with c2:
            st.metric("Complete", complete)
        with c3:
            st.metric("Pending", len(status) - complete)
    else:
        st.info(f"No filings found for {manage_ticker}")

    st.markdown("---")

    # Download section
    st.markdown("#### Download & Process")
    c1, c2, c3 = st.columns(3)
    with c1:
        years = st.selectbox(
            "Years to download", [1, 2, 3, 5, 10], index=2,
            format_func=lambda x: f"Last {x} year{'s' if x > 1 else ''}",
            key="download_years",
        )
    with c2:
        auto = st.checkbox("Auto-process", value=True, key="auto_process_check")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        start = st.button("Download", type="primary", width="stretch")

    if start and AutomatedPipeline:
        pipeline = AutomatedPipeline()
        progress = st.progress(0)
        status_text = st.empty()

        def update(msg, pct):
            progress.progress(min(pct / 100, 1.0))
            status_text.info(f"{msg}")

        pipeline.set_progress_callback(update)

        try:
            result = pipeline.process_company(manage_ticker, years=years, auto_process=auto)
            if result.success:
                status_text.success(f"Done: {result.message}")
                st.balloons()
            else:
                status_text.error(f"Failed: {result.message}")
        except Exception as e:
            status_text.error(f"Error: {e}")
    elif start:
        st.error("Pipeline not available. Check imports.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #94a3b8;">AlphaExtract v2.0.0 | '
    "Built with Streamlit, FinBERT, Ensemble Scoring & Local RAG</div>",
    unsafe_allow_html=True,
)
