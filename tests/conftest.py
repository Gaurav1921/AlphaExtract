"""
Shared test fixtures for AlphaExtract.
"""

import json
import pytest
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    dirs = ["raw", "processed", "sections", "sentiment", "anomalies", "logs"]
    for d in dirs:
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def sample_10k_markdown():
    """Return a sample 10-K markdown document with real section headers."""
    return """
# Company Annual Report 10-K

## Table of Contents

Item 1A. Risk Factors ........... 10
Item 7. Management's Discussion and Analysis ........... 40
Item 8. Financial Statements ........... 80

---

Item 1A. Risk Factors

Our business is subject to various risks and uncertainties that could have a material adverse
effect on our business, financial condition, results of operations, cash flows, and our ability
to execute on our current plans. The risk factors described below represent our principal risks.

We face intense competition in all aspects of our business. Our competitors include companies
with substantially greater financial, technical, and marketing resources. Competition has
intensified significantly over the past year with several new market entrants deploying
aggressive pricing strategies. This increasing competitive pressure may result in price
reductions, reduced margins, or loss of market share.

Supply chain disruptions continue to affect our operations. We rely on third-party suppliers
for critical components and raw materials. Geopolitical tensions, natural disasters, and
other events beyond our control may further disrupt these supply chains. We have experienced
increased lead times and cost escalation for certain components during the current fiscal year.

Cybersecurity threats pose an ongoing risk to our systems and data. Despite our investments
in security infrastructure, there can be no assurance that our defenses will be sufficient
to prevent unauthorized access to our systems. A significant breach could result in the loss
of confidential information, regulatory penalties, and reputational damage.

Climate change and environmental regulations present both risks and opportunities. We are
subject to increasingly stringent environmental regulations across jurisdictions. Compliance
with these regulations may require significant capital expenditure and operational changes.
Failure to comply could result in substantial fines, legal proceedings, and damage to our
reputation.

""" + "Additional risk details. " * 200 + """

Item 7. Management's Discussion and Analysis of Financial Condition

The following discussion should be read in conjunction with our consolidated financial
statements and related notes included elsewhere in this report. This discussion contains
forward-looking statements that involve risks and uncertainties.

Revenue for the fiscal year ended December 31, 2024 was $45.2 billion, representing a
12% increase from the prior year. This growth was driven primarily by strong demand in
our cloud computing segment, which grew 28% year-over-year. Our enterprise software
segment contributed $18.3 billion, up 8% from the prior period.

Operating expenses increased 7% to $32.1 billion, reflecting our continued investments
in research and development, particularly in artificial intelligence and machine learning
capabilities. We believe these investments position us well for future growth opportunities
in the rapidly evolving technology landscape.

Net income for the fiscal year was $8.7 billion, or $12.35 per diluted share, compared
to $7.2 billion, or $10.18 per diluted share in the prior year. This improvement reflects
both revenue growth and operational efficiency gains.

Our cash position remains strong with $23.4 billion in cash and short-term investments at
year end. During the year, we generated $14.2 billion in operating cash flow and returned
$6.8 billion to shareholders through dividends and share repurchases.

""" + "Further MD&A analysis. " * 200 + """

Item 8. Financial Statements and Supplementary Data

Consolidated Balance Sheet

Total Assets: $98.7 billion
Total Liabilities: $42.3 billion
Stockholders Equity: $56.4 billion

Revenue Recognition
We recognize revenue when control of promised goods or services is transferred to customers,
in an amount that reflects the consideration we expect to receive. Our revenue is primarily
derived from software licenses, cloud services subscriptions, and professional services.

""" + "Financial notes detail. " * 150 + """

Item 9. Changes in and Disagreements with Accountants

None.
"""


@pytest.fixture
def sample_sentiment_result():
    """Return a sample sentiment analysis result."""
    return {
        "sections": {
            "item_1a": {
                "scores": {"positive": 0.2, "negative": 0.5, "neutral": 0.3, "compound": -0.3},
                "signal": "SELL",
                "word_count": 1500,
            },
            "item_7": {
                "scores": {"positive": 0.6, "negative": 0.1, "neutral": 0.3, "compound": 0.5},
                "signal": "STRONG_BUY",
                "word_count": 2000,
            },
            "item_8": {
                "scores": {"positive": 0.4, "negative": 0.2, "neutral": 0.4, "compound": 0.2},
                "signal": "BUY",
                "word_count": 800,
            },
        },
        "overall": {"compound": 0.25, "signal": "BUY"},
        "ticker": "AAPL",
        "filing_date": "2024-01-15",
    }


@pytest.fixture
def sample_sentiment_previous():
    """Return a previous-year sentiment result for anomaly comparison."""
    return {
        "sections": {
            "item_1a": {
                "scores": {"positive": 0.3, "negative": 0.4, "neutral": 0.3, "compound": -0.1},
                "signal": "HOLD",
                "word_count": 1400,
            },
            "item_7": {
                "scores": {"positive": 0.5, "negative": 0.15, "neutral": 0.35, "compound": 0.35},
                "signal": "BUY",
                "word_count": 1800,
            },
            "item_8": {
                "scores": {"positive": 0.35, "negative": 0.25, "neutral": 0.4, "compound": 0.1},
                "signal": "HOLD",
                "word_count": 750,
            },
        },
        "overall": {"compound": 0.15, "signal": "HOLD"},
        "ticker": "AAPL",
        "filing_date": "2023-01-20",
    }


@pytest.fixture
def sentiment_files(tmp_data_dir, sample_sentiment_result, sample_sentiment_previous):
    """Create sentiment JSON files on disk for testing."""
    sentiment_dir = tmp_data_dir / "sentiment"
    current_path = sentiment_dir / "AAPL_2024-01-15_sentiment.json"
    previous_path = sentiment_dir / "AAPL_2023-01-20_sentiment.json"

    current_path.write_text(json.dumps(sample_sentiment_result), encoding="utf-8")
    previous_path.write_text(json.dumps(sample_sentiment_previous), encoding="utf-8")

    return {"current": current_path, "previous": previous_path}


@pytest.fixture
def section_files(tmp_data_dir):
    """Create section text files on disk for testing."""
    sections_dir = tmp_data_dir / "sections"

    risk_text = (
        "Our business is subject to intense competition. Competitors include firms with "
        "substantially greater resources. Cybersecurity threats pose ongoing risk. "
        "Supply chain disruption remains a concern. Litigation risks have increased. "
    ) * 20

    mda_text = (
        "Revenue grew 12% year-over-year driven by cloud computing. Operating expenses "
        "increased 7% reflecting R&D investment. Net income rose to $8.7 billion. "
        "Artificial intelligence investment is accelerating growth. "
    ) * 20

    fin_text = (
        "Total assets of $98.7 billion. Revenue recognition follows ASC 606. "
        "Goodwill impairment testing performed annually. No impairment recorded. "
    ) * 15

    files = {}
    for date in ["2024-01-15", "2023-01-20", "2022-01-18"]:
        for key, text in [("item_1a", risk_text), ("item_7", mda_text), ("item_8", fin_text)]:
            path = sections_dir / f"AAPL_{date}_{key}.txt"
            path.write_text(text, encoding="utf-8")
            files[f"{date}_{key}"] = path

    return files
