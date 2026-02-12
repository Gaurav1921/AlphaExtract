# Backtesting Framework + Ensemble Scoring — Implementation Plan

## Branch: `dev_2` (off current dev branch)

---

## Phase 1: Market Data Module
**File: `src/market/price_data.py`**

- `MarketDataProvider` class wrapping yfinance
- `get_price_at_date(ticker, date)` → closing price
- `get_returns(ticker, filing_date, windows=[30,60,90,180,365])` → dict of return %
- Local CSV cache under `data/market/` keyed by ticker — one fetch per ticker per session
- `get_filing_outcome(ticker, filing_date, window=90)` → {"return_pct": float, "direction": "up"|"down"|"flat"}
  - flat = abs(return) < 2%
- Handles edge cases: weekends (next trading day), delistings, splits (yfinance adjusts automatically)

**Dep:** `yfinance` added to requirements.txt

---

## Phase 2: Ensemble Scoring Engine
**File: `src/models/ensemble.py`**

Three independent signal sources combined into one score:

### Signal 1: FinBERT Sentiment (existing)
- Already produces compound score per section + overall
- Normalize to [-1, +1] range
- Weight: configurable (default 0.40)

### Signal 2: Keyword Signals (new, built on existing TRACKED_KEYWORDS)
- Score based on anomaly detection output:
  - keyword_emergence of negative terms → bearish signal
  - keyword_decline of negative terms → bullish signal
  - keyword_spike in risk terms → bearish
  - YoY keyword frequency delta matters more than absolute count
- Produce a keyword_score in [-1, +1]
- Weight: configurable (default 0.25)

### Signal 3: LLM Qualitative Analysis (new, Gemini)
- Send MD&A section (item_7) + risk factors summary to Gemini
- Structured prompt asking for:
  - Directional outlook: bullish / neutral / bearish
  - Confidence: high / medium / low
  - Key factors (2-3 bullet points)
- Parse response into score [-1, +1]: bullish+high=1.0, bearish+low=-0.3, etc.
- Weight: configurable (default 0.35)

### Ensemble Combination
- `ensemble_score = w1*finbert + w2*keywords + w3*llm`
- Generate signal from ensemble_score using (potentially recalibrated) thresholds
- Store both individual scores and ensemble score in output

**Output format:**
```json
{
  "ticker": "AAPL",
  "filing_date": "2024-01-15",
  "signals": {
    "finbert": {"score": 0.25, "signal": "BUY", "weight": 0.40},
    "keywords": {"score": -0.15, "signal": "HOLD", "weight": 0.25},
    "llm": {"score": 0.40, "signal": "BUY", "confidence": "medium", "weight": 0.35}
  },
  "ensemble": {
    "score": 0.20,
    "signal": "BUY",
    "components_agree": false
  }
}
```

---

## Phase 3: Backtesting Engine
**File: `src/backtesting/backtester.py`**

- `Backtester` class
- `run(tickers, years)` → runs full backtest:
  1. For each (ticker, filing_date) with sentiment data:
     - Get our signal (both old FinBERT-only and new ensemble)
     - Get actual market return at 30/60/90/180 days via MarketDataProvider
     - Record: signal, score, actual_return, direction_match
  2. Aggregate results

**File: `src/backtesting/metrics.py`**
- `hit_rate(results)` → % of correct direction calls
- `precision_by_signal(results)` → for each signal type, what % were correct
- `avg_return_by_signal(results)` → average actual return grouped by signal
- `sharpe_ratio(results)` → risk-adjusted return if you followed signals
- `confusion_matrix(results)` → predicted vs actual direction
- `compare_models(finbert_results, ensemble_results)` → side-by-side comparison

---

## Phase 4: Calibration
**File: `src/backtesting/calibrator.py`**

- `optimize_thresholds(backtest_results)` → find score cutoffs that maximize hit rate
  - Grid search over threshold values
  - Returns optimal STRONG_BUY/BUY/SELL/STRONG_SELL boundaries
- `optimize_weights(backtest_results)` → find best w1/w2/w3 for ensemble
  - Grid search over weight combinations (sum to 1.0)
  - Returns optimal weights
- `optimize_section_weights(backtest_results)` → find best item_1a/7/8 weights for FinBERT
- Output: recommended Settings overrides as JSON

---

## Phase 5: Report Generator
**File: `src/backtesting/report.py`**

- `generate_report(backtest_results, output_path)` → JSON + terminal summary
- Key sections:
  - Overall accuracy (hit rate, Sharpe)
  - Per-signal breakdown
  - Per-ticker breakdown
  - FinBERT-only vs Ensemble comparison
  - Recommended calibrated settings
  - Worst misses (biggest wrong calls — for debugging)

---

## Phase 6: Pipeline Integration
**Update: `src/pipeline/automated.py`**

- Add ensemble scoring as a pipeline stage after sentiment analysis
- New pipeline flow: Download → Parse → Split → Sentiment → **Ensemble Score** → Index
- Store ensemble results alongside sentiment in `data/sentiment/`

**Update: `main.py`**
- Add `backtest` CLI command: `python main.py backtest --tickers AAPL,MSFT --years 5`
- Add `calibrate` CLI command: `python main.py calibrate`

---

## Phase 7: Tests
- `tests/test_price_data.py` — mock yfinance, test caching, edge cases
- `tests/test_ensemble.py` — test score combination, weight normalization
- `tests/test_backtester.py` — test metrics calculation, direction matching
- `tests/test_calibrator.py` — test threshold optimization

---

## Execution Order

1. Market data module (foundation — everything needs price data)
2. Ensemble scoring engine (the analytical upgrade)
3. Backtesting engine + metrics (measurement tool)
4. Calibration (tune based on results)
5. Report generator (make results visible)
6. Pipeline integration + CLI commands
7. Tests throughout

---

## New Dependencies
- `yfinance>=0.2.30`

## New Settings (added to Settings class)
```python
# Ensemble weights
ENSEMBLE_WEIGHT_FINBERT = 0.40
ENSEMBLE_WEIGHT_KEYWORDS = 0.25
ENSEMBLE_WEIGHT_LLM = 0.35

# Market data
MARKET_DATA_DIR = DATA_DIR / "market"
MARKET_RETURN_WINDOWS = [30, 60, 90, 180, 365]
MARKET_FLAT_THRESHOLD = 0.02  # 2% = flat

# Backtest
BACKTEST_DIR = DATA_DIR / "backtest"
```
