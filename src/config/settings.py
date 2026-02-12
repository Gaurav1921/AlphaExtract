"""
Central Configuration for AlphaExtract
--------------------------------------
All settings, paths, API configurations, and validation.
Single source of truth for the entire application.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Immutable application settings with validation."""

    PROJECT_NAME = "AlphaExtract"
    VERSION = "1.0.0"

    # --------------- Paths ---------------
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    SECTIONS_DIR = DATA_DIR / "sections"
    SENTIMENT_DIR = DATA_DIR / "sentiment"
    ANOMALIES_DIR = DATA_DIR / "anomalies"
    LOG_DIR = BASE_DIR / "logs"

    # --------------- API Keys ---------------
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # --------------- SEC EDGAR ---------------
    SEC_BASE_URL = "https://data.sec.gov"
    SEC_FILING_URL = "https://www.sec.gov"
    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
    SEC_TICKERS_FALLBACK_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_USER_EMAIL = os.getenv("SEC_EMAIL", "student@example.com")
    SEC_RATE_LIMIT_DELAY = 0.15  # seconds between requests (~6 req/sec)
    SEC_REQUEST_TIMEOUT = 30  # seconds
    SEC_MAX_RETRIES = 3
    SEC_RETRY_DELAY = 2  # seconds between retries

    # --------------- OpenSearch ---------------
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "alphaextract-docs")
    OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
    OPENSEARCH_BULK_TIMEOUT = 60  # seconds
    OPENSEARCH_BULK_BATCH_SIZE = 100

    # KNN / HNSW parameters
    HNSW_EF_SEARCH = 100
    HNSW_EF_CONSTRUCTION = 128
    HNSW_M = 24

    # --------------- Models ---------------
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM = 384
    EMBEDDING_BATCH_SIZE = 32
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")
    SENTIMENT_MAX_TOKENS = 512
    SENTIMENT_TOKEN_OVERLAP = 50
    SENTIMENT_MAX_CHUNKS = 50  # max chunks to process per section
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # --------------- Sentiment Signal Thresholds ---------------
    SIGNAL_STRONG_BUY = 0.5
    SIGNAL_BUY = 0.2
    SIGNAL_SELL = -0.2
    SIGNAL_STRONG_SELL = -0.5

    # Section weights for overall sentiment
    SECTION_WEIGHTS = {
        "item_1a": 0.25,  # Risk Factors
        "item_7": 0.60,   # MD&A — most important
        "item_8": 0.15,   # Financial Statements
    }

    # --------------- RAG ---------------
    CHUNK_SIZE = 500       # words per chunk
    CHUNK_OVERLAP = 50     # words overlap
    CHUNK_MIN_SIZE = 100   # minimum words for a chunk
    TOP_K = 5              # default retrieval count
    MAX_CONVERSATION_HISTORY = 5

    # --------------- Anomaly Detection ---------------
    ANOMALY_SENTIMENT_THRESHOLD = 0.15
    ANOMALY_FREQUENCY_MULTIPLIER = 3.0
    ANOMALY_MIN_HISTORICAL_COUNT = 3

    # --------------- Logging ---------------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # --------------- Ensemble Scoring ---------------
    ENSEMBLE_WEIGHT_FINBERT = 0.40
    ENSEMBLE_WEIGHT_KEYWORDS = 0.25
    ENSEMBLE_WEIGHT_LLM = 0.35

    # --------------- Market Data ---------------
    MARKET_DATA_DIR = DATA_DIR / "market"
    MARKET_RETURN_WINDOWS = [30, 60, 90, 180, 365]
    MARKET_FLAT_THRESHOLD = 0.02  # 2% absolute return = "flat"

    # --------------- Backtesting ---------------
    BACKTEST_DIR = DATA_DIR / "backtest"

    # --------------- Watchlist ---------------
    WATCHLIST_FILE = DATA_DIR / "watchlist.json"

    @classmethod
    def init_dirs(cls):
        """Create all required directories. Call once at startup."""
        for dir_path in [
            cls.DATA_DIR, cls.RAW_DIR, cls.PROCESSED_DIR,
            cls.SECTIONS_DIR, cls.SENTIMENT_DIR, cls.ANOMALIES_DIR,
            cls.MARKET_DATA_DIR, cls.BACKTEST_DIR, cls.LOG_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate configuration. Returns list of warnings (empty = all good).
        Does NOT raise — callers decide severity.
        """
        warnings = []
        if not cls.GEMINI_API_KEY:
            warnings.append("GEMINI_API_KEY not set — RAG and anomaly detection will be limited")
        if cls.SEC_USER_EMAIL == "student@example.com":
            warnings.append("SEC_EMAIL not set — using default; SEC may throttle requests")
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            warnings.append("SUPABASE_URL/SUPABASE_KEY not set — database persistence disabled")
        return warnings

    @classmethod
    def generate_signal(cls, compound: float) -> str:
        """Generate trading signal from compound sentiment score."""
        if compound >= cls.SIGNAL_STRONG_BUY:
            return "STRONG_BUY"
        elif compound >= cls.SIGNAL_BUY:
            return "BUY"
        elif compound >= cls.SIGNAL_SELL:
            return "HOLD"
        elif compound >= cls.SIGNAL_STRONG_SELL:
            return "SELL"
        else:
            return "STRONG_SELL"


def setup_logging(level: str = None) -> None:
    """
    Configure logging for the entire application.
    Call once at startup (main.py / dashboard).
    """
    log_level = getattr(logging, level or Settings.LOG_LEVEL, logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove existing handlers to avoid duplicates on re-import
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(Settings.LOG_FORMAT, datefmt=Settings.LOG_DATE_FORMAT))
    root.addHandler(console)

    # File handler (rotated externally or by size — keep simple)
    Settings.init_dirs()
    file_handler = logging.FileHandler(Settings.LOG_DIR / "alphaextract.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(Settings.LOG_FORMAT, datefmt=Settings.LOG_DATE_FORMAT))
    root.addHandler(file_handler)


# Initialize directories on import
Settings.init_dirs()

# Module-level singleton
settings = Settings()
