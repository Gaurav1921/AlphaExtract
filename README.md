# AlphaExtract 📊

**AI-Powered Financial Intelligence System for SEC Filings**

Generate trading signals from 10-K documents using document intelligence, sentiment analysis, and LLM-powered anomaly detection.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

AlphaExtract is an end-to-end quantitative analysis pipeline that:

1. **Downloads** 10-K filings from SEC EDGAR
2. **Parses** complex XBRL documents with Docling
3. **Extracts** key sections (Risk Factors, MD&A, Financials)
4. **Analyzes** sentiment using FinBERT
5. **Generates** actionable trading signals
6. **Backtests** strategies with historical data

### Why AlphaExtract?

-  **Production-grade parsing** - Handles XBRL/iXBRL formats correctly
-  **62+ tables extracted** from each 10-K automatically
-  **1.4 seconds** to parse a 2MB document
-  **$0 cost** - No Bloomberg Terminal needed
-  **Fully customizable** - Open-source, extend as needed

---

## Architecture

```
SEC EDGAR API → Downloader → Parser → Analysis → Trading Signals
                    ↓           ↓         ↓            ↓
                  Raw HTML    Clean MD  Sentiment   BUY/SELL
                              + Tables   Scores
```

### Tech Stack

- **Document Processing:** Docling (XBRL-aware)
- **Sentiment Analysis:** FinBERT
- **RAG System:** LangChain + FAISS
- **LLM:** Claude/Llama (anomaly detection)
- **Database:** PostgreSQL
- **Dashboard:** Streamlit
- **Backtesting:** Backtrader

---

## Installation

### Prerequisites

- Python 3.12+
- Anaconda (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/AlphaExtract.git
cd AlphaExtract

# Create environment
conda create -n alpha-extract python=3.12 -y
conda activate alpha-extract

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Download 10-K Filings

```python
from sec_downloader import SECDownloader

downloader = SECDownloader(email="your@email.com")

# Single company
downloader.download_filing("TSLA")

# Batch download
downloader.download_multiple(["AAPL", "MSFT", "GOOGL"])
```

**Output:** `data/raw/TSLA_10K_2025-01-30.html` (2.5 MB)

### 2. Parse Documents

```python
from parse_10k import TenKParser

parser = TenKParser()
result = parser.process_filing("data/raw/TSLA_10K_2025-01-30.html")
```

**Output:**
- `data/processed/TSLA_2025-01-30.md` (51k words)
- `data/processed/TSLA_2025-01-30_table_*.csv` (62 tables)
- `data/processed/TSLA_2025-01-30_metadata.json`

### 3. Analyze Sentiment (Phase 2 - Coming Soon)

```python
from sentiment_analyzer import analyze_filing

signals = analyze_filing("TSLA")
# Output: {"sentiment": 0.72, "signal": "BULLISH"}
```

---

## Example Results

### Apple Inc. (AAPL) - Fiscal Year 2025

```json
{
  "ticker": "AAPL",
  "parse_time": 1.44,
  "total_words": 51521,
  "tables_extracted": 62,
  "sections": {
    "risk_factors": "5,234 words",
    "mda": "12,456 words",
    "financials": "62 tables"
  }
}
```

---

## Project Structure

```
AlphaExtract/
├── data/
│   ├── raw/              # Downloaded 10-K HTML files
│   └── processed/        # Parsed markdown + CSV tables
├── src/
│   ├── sec_downloader.py     # Phase 1: Download filings
│   ├── parse_10k.py          # Phase 1: Parse with Docling
│   ├── sentiment_analyzer.py # Phase 2: FinBERT analysis
│   └── signal_generator.py   # Phase 2: Trading signals
├── tests/
│   └── test_downloader.py
├── .gitignore
├── requirements.txt
├── README.md
└── PHASE_1_SUMMARY.md
```

---

## Roadmap

### Phase 1: Foundation (Complete)
- [x] SEC EDGAR downloader
- [x] Docling-based parser
- [x] Data pipeline setup

### Phase 2: Signal Generation (In Progress)
- [ ] Section splitter (Item 1A, 7, 8)
- [ ] FinBERT sentiment analysis
- [ ] Basic trading signals

### Phase 3: Advanced Intelligence (Planned)
- [ ] RAG chatbot for Q&A
- [ ] LLM anomaly detection
- [ ] Multi-quarter comparison

### Phase 4: Production System (Planned)
- [ ] Backtesting framework
- [ ] Streamlit dashboard
- [ ] Automated daily pipeline

---

## Documentation

- **[Phase 1 Summary](PHASE_1_SUMMARY.md)** - Detailed breakdown of what's built
- **[Architecture Decisions](docs/ARCHITECTURE.md)** - Why we chose each tool *(coming soon)*
- **[API Reference](docs/API.md)** - Function documentation *(coming soon)*

---

## Testing

```bash
# Run all tests
pytest

# Run specific component
pytest tests/test_downloader.py -v
```

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

---

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/YOUR_USERNAME/AlphaExtract](https://github.com/YOUR_USERNAME/AlphaExtract)

---

## Disclaimer

This project is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.

---

**Built with love by a quant engineer in training**