# AlphaExtract

**AI-Powered Financial Intelligence System for SEC Filings**

Generate trading signals from 10-K documents using document intelligence, sentiment analysis, ensemble scoring, and backtesting.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

AlphaExtract is an end-to-end quantitative analysis pipeline that:

1. **Downloads** 10-K filings from SEC EDGAR
2. **Parses** complex XBRL documents with Docling
3. **Extracts** key sections (Risk Factors, MD&A, Financials)
4. **Analyzes** sentiment using FinBERT
5. **Scores** with ensemble model (FinBERT + Keywords + LLM)
6. **Backtests** signals against actual market returns
7. **Calibrates** thresholds and weights automatically
8. **Visualizes** everything in a Streamlit dashboard

### Why AlphaExtract?

- **Production-grade parsing** - Handles XBRL/iXBRL formats correctly
- **62+ tables extracted** from each 10-K automatically
- **1.4 seconds** to parse a 2MB document
- **$0 cost** - No Bloomberg Terminal needed
- **77.8% directional accuracy** on backtested signals
- **80 tickers across 8 sectors** supported

---

## Architecture

```
SEC EDGAR API --> Downloader --> Parser --> Splitter --> Sentiment (FinBERT)
                      |                                       |
                  Filing Alerts                         Ensemble Scoring
                  (SEC Monitor +                      (FinBERT + Keywords + LLM)
                   Webhooks)                                  |
                                    +-------+-------+-------+-------+-------+-------+
                                    |       |       |       |       |       |       |
                                Backtest  RAG    Anomaly  Portfolio Options  API   Dashboard
                                    |     Chat   Detect.  Signals  History       (11 pages)
                                Calibrate                  Overlay
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Document Processing | Docling (XBRL-aware) |
| Sentiment Analysis | FinBERT (ProsusAI) |
| Ensemble Scoring | FinBERT + Keyword Signals + LLM |
| RAG Chatbot | Local vector search (sentence-transformers + numpy) |
| LLM Providers | Groq (free) / Gemini / Ollama (local) |
| Market Data | yfinance |
| Options Data | yfinance options chains + historical tracking |
| Filing Alerts | SEC EDGAR submissions API + webhook notifications |
| Portfolio | Position-weighted signal aggregation + custom weight editor |
| REST API | FastAPI + Pydantic |
| Dashboard | Streamlit + Plotly |
| Database | Supabase (optional) |

---

## Installation

### Prerequisites

- Python 3.12+
- Anaconda (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/Gaurav1921/AlphaExtract.git
cd AlphaExtract

# Create environment
conda create -n alpha-extract python=3.12 -y
conda activate alpha-extract

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables (Optional)

```bash
# .env file
SEC_EMAIL=your@email.com          # Required for SEC EDGAR
GROQ_API_KEY=your_groq_key        # Free tier: 14,400 req/day
GEMINI_API_KEY=your_gemini_key     # Alternative LLM
```

---

## Quick Start

### CLI Commands

```bash
# Download 10-K filings
python main.py download AAPL MSFT GOOGL --years 5

# Parse documents
python main.py parse

# Extract sections (Item 1A, 7, 8)
python main.py split

# Run FinBERT sentiment analysis
python main.py analyze

# Run ensemble scoring (FinBERT + Keywords + LLM)
python main.py ensemble AAPL MSFT GOOGL

# Backtest signals against market returns
python main.py backtest AAPL MSFT --compare

# Calibrate thresholds and weights
python main.py calibrate AAPL MSFT

# Full pipeline (download -> parse -> split -> sentiment -> ensemble)
python main.py pipeline AAPL --years 5

# Launch dashboard
python main.py dashboard

# Monitor SEC for new filings
python main.py alerts AAPL MSFT GOOGL --days 30

# Continuous monitoring (polls every hour)
python main.py alerts AAPL MSFT --poll --interval 3600

# Alerts with webhook notifications (Slack/custom)
python main.py alerts AAPL MSFT --poll --webhook-url https://hooks.slack.com/... --webhook-format slack

# Portfolio signal aggregation
python main.py portfolio AAPL MSFT GOOGL JPM XOM --save
python main.py portfolio --sector Technology

# Options sentiment overlay
python main.py options AAPL MSFT --save

# Track options data over time
python main.py options-history AAPL MSFT --record
python main.py options-history AAPL --limit 30

# Launch REST API
python main.py api --port 8000
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard includes 11 pages:

| Page | What it does |
|------|-------------|
| **Dashboard** | Sentiment overview, signal cards, historical trend |
| **Multi-Quarter** | YoY comparison, sentiment evolution, section deep dive |
| **Ensemble** | Signal components, weighted contributions, keyword/LLM detail |
| **Backtesting** | Model comparison, precision tables, confusion matrix |
| **RAG Chat** | Ask questions about 10-K filings using local RAG |
| **Anomalies** | Detect unusual patterns vs historical filings |
| **Sector Analytics** | Cross-sector comparison, ticker universe browser |
| **Portfolio** | Multi-holding aggregation, sector breakdown, risk metrics |
| **Options Overlay** | Put/call ratios, IV skew, composite filing+options signal |
| **Filing Alerts** | SEC EDGAR monitoring for new 10-K filings |
| **Data Management** | Download pipeline with progress tracking |

---

## Features

### Ensemble Scoring (3-Signal Model)

| Signal | Source | Default Weight |
|--------|--------|---------------|
| FinBERT | Per-section sentiment [-1, +1] | 60% |
| Keywords | Bearish/bullish keyword frequency | 40% |
| LLM | Qualitative analysis (Groq/Gemini/Ollama) | 0% (opt-in) |

### Backtesting

- Compares signals against actual 90-day post-filing returns
- Hit rate, directional accuracy, Sharpe ratio
- Precision by signal type (STRONG_BUY through STRONG_SELL)
- Confusion matrix (predicted vs actual direction)

### RAG Chatbot

- Works locally without Docker or OpenSearch
- In-memory vector search with sentence-transformers
- Keyword-based fallback when no embedding model
- Multi-provider LLM support for answer generation
- Conversation history for follow-up questions

### Multi-Quarter Comparison

- Year-over-year sentiment delta tracking
- Section-level evolution (Risk Factors, MD&A, Financials)
- Trend detection (improving / stable / deteriorating)
- Word count change tracking

### Real-Time Filing Alerts

- Monitors SEC EDGAR for new 10-K filings via submissions API
- Ticker watchlist with automatic CIK resolution
- State persistence to avoid duplicate alerts
- Callback system for custom notifications
- Continuous polling mode or one-shot check
- CLI: `python main.py alerts AAPL MSFT --poll`

### Portfolio Signal Aggregation

- Combine signals across multiple holdings
- Position-weighted or equal-weighted modes
- Sector breakdown with weighted average scores
- Risk concentration metrics (HHI, max exposure)
- Signal distribution analysis
- Multi-portfolio comparison
- CLI: `python main.py portfolio AAPL MSFT JPM --save`

### Options Sentiment Overlay

- Fetches options chain data from yfinance
- Put/call open interest and volume ratios
- Implied volatility skew (put IV vs call IV)
- Options-derived sentiment score [-1, +1]
- Composite signal blending filing + options sentiment
- Configurable weighting (default: 70% filing / 30% options)
- Agreement detection between filing and options signals
- **Historical tracking**: Store snapshots over time, trend analysis
- CLI: `python main.py options AAPL MSFT --save`
- CLI: `python main.py options-history AAPL --record`

### REST API (FastAPI)

- Programmatic access to all signals via HTTP
- Endpoints: `/sentiment/{ticker}`, `/ensemble/{ticker}`, `/portfolio`, `/options/{ticker}`, `/alerts/{ticker}`, `/signals/{ticker}`
- Health check and ticker listing
- Pydantic request/response validation
- CLI: `python main.py api --port 8000`

### Webhook Notifications

- HTTP POST notifications when new 10-K filings are detected
- Slack-formatted messages (incoming webhooks)
- Generic JSON POST for custom endpoints
- Multiple webhook endpoints supported
- CLI: `python main.py alerts AAPL MSFT --poll --webhook-url https://...`

### Sector Coverage (80 Tickers, 8 Sectors)

| Sector | Example Tickers |
|--------|----------------|
| Technology | AAPL, MSFT, GOOGL, NVDA, AMD, META |
| Healthcare | JNJ, UNH, PFE, ABBV, MRK, LLY |
| Financials | JPM, BAC, GS, MS, WFC, BLK |
| Energy | XOM, CVX, COP, SLB, EOG |
| Consumer Discretionary | AMZN, TSLA, HD, MCD, NKE |
| Consumer Staples | PG, KO, PEP, COST, WMT |
| Industrials | CAT, HON, UPS, RTX, BA |
| Real Estate | AMT, PLD, CCI, EQIX, SPG |

---

## Project Structure

```
AlphaExtract/
├── main.py                          # CLI entry point
├── requirements.txt                 # Dependencies
├── dashboard/
│   ├── app.py                       # Streamlit dashboard (v3.0, 11 pages)
│   ├── components/
│   │   └── data_management.py       # Reusable UI components
│   ├── pages/
│   └── styles/
├── src/
│   ├── config/
│   │   └── settings.py              # Central configuration
│   ├── data/
│   │   ├── downloader.py            # SEC EDGAR download
│   │   ├── parser.py                # Docling-based parsing
│   │   ├── splitter.py              # Section extraction
│   │   ├── chunker.py               # Text chunking for RAG
│   │   └── company_search.py        # Company ticker search
│   ├── models/
│   │   ├── sentiment.py             # FinBERT sentiment analyzer
│   │   ├── ensemble.py              # 3-signal ensemble scoring
│   │   └── anomaly.py               # Anomaly detection
│   ├── alerts/
│   │   ├── sec_monitor.py           # SEC EDGAR filing alert monitor
│   │   └── webhook.py               # Webhook notifications (Slack/JSON)
│   ├── analysis/
│   │   ├── comparison.py            # Multi-quarter comparison
│   │   └── portfolio.py             # Portfolio signal aggregation
│   ├── api/
│   │   └── app.py                   # FastAPI REST API
│   ├── market/
│   │   ├── price_data.py            # Yahoo Finance market data
│   │   ├── options_overlay.py       # Options sentiment overlay
│   │   └── options_history.py       # Historical options tracking
│   ├── backtesting/
│   │   ├── backtester.py            # Backtest engine
│   │   ├── calibrator.py            # Parameter calibration
│   │   ├── metrics.py               # Accuracy metrics
│   │   └── report.py                # Report generation
│   ├── rag/
│   │   ├── local_rag.py             # Local RAG (no Docker)
│   │   └── enhanced_rag.py          # OpenSearch RAG
│   └── pipeline/
│       └── automated.py             # Pipeline orchestration
├── tests/
│   ├── test_downloader.py
│   ├── test_ensemble.py
│   ├── test_backtester.py
│   ├── test_price_data.py
│   ├── test_anomaly.py
│   ├── test_chunker.py
│   ├── test_pipeline.py
│   ├── test_settings.py
│   ├── test_local_rag.py
│   ├── test_comparison.py
│   ├── test_sec_monitor.py
│   ├── test_portfolio.py
│   ├── test_options_overlay.py
│   ├── test_options_history.py
│   ├── test_webhook.py
│   └── test_api.py
└── data/
    ├── raw/                         # Downloaded 10-K HTML
    ├── processed/                   # Parsed markdown + tables
    ├── sections/                    # Extracted sections
    ├── sentiment/                   # FinBERT + ensemble scores
    ├── anomalies/                   # Anomaly reports
    ├── backtest/                    # Backtest results
    ├── market/                      # Cached price data + options
    ├── options_history/             # Historical options tracking
    └── portfolios/                  # Portfolio analysis results
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific module
pytest tests/test_ensemble.py -v
```

---

## Roadmap

### Complete

- [x] SEC EDGAR downloader with rate limiting
- [x] Docling-based XBRL parser
- [x] Section extraction (Item 1A, 7, 8)
- [x] FinBERT sentiment analysis
- [x] Ensemble scoring (FinBERT + Keywords + LLM)
- [x] Backtesting engine with metrics
- [x] Automated calibration
- [x] Report generation (terminal + JSON)
- [x] Local RAG chatbot (no Docker required)
- [x] Multi-quarter comparison engine
- [x] Sector analytics (80 tickers, 8 sectors)
- [x] Streamlit dashboard v2.0 (8 pages)
- [x] Real-time filing alerts (SEC EDGAR monitor)
- [x] Portfolio-level signal aggregation
- [x] Options sentiment overlay
- [x] Streamlit dashboard v3.0 (11 pages)

- [x] REST API endpoint (FastAPI)
- [x] Webhook notifications for filing alerts (Slack/JSON)
- [x] Custom portfolio weight editor in dashboard
- [x] Historical options data tracking

---

## License

MIT License - See [LICENSE](LICENSE) file for details

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgments

- **Docling** by IBM Research for document intelligence
- **FinBERT** for financial sentiment analysis
- **SEC EDGAR** for free public company data
- **Groq** for free LLM API access

---

## Contact

Gaurav - gjs190201@gmail.com

Project Link: [https://github.com/Gaurav1921/AlphaExtract](https://github.com/Gaurav1921/AlphaExtract)

---

## Disclaimer

This project is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.

---

**Built with love by a quant engineer in training**
