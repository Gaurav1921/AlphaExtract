"""
Central Configuration for AlphaExtract
--------------------------------------
All settings, paths, and API configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""
    
    PROJECT_NAME = "AlphaExtract"
    VERSION = "0.4.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    SECTIONS_DIR = DATA_DIR / "sections"
    SENTIMENT_DIR = DATA_DIR / "sentiment"
    ANOMALIES_DIR = DATA_DIR / "anomalies"
    
    # Ensure directories exist
    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, SECTIONS_DIR, SENTIMENT_DIR, ANOMALIES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # SEC
    SEC_BASE_URL = "https://data.sec.gov"
    SEC_USER_EMAIL = os.getenv("SEC_EMAIL", "student@example.com")
    
    # OpenSearch
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_INDEX = "alphaextract-docs"
    
    # Models
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    SENTIMENT_MODEL = "ProsusAI/finbert"
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    
    # RAG
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 5


settings = Settings()
