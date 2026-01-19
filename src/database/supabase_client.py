"""
Supabase Database Client
------------------------
Connects to Supabase and manages data persistence.

Tables:
- companies: Company metadata
- filings: 10-K filing records
- sentiment_scores: Historical sentiment analysis
- anomalies: Detected anomalies
- chat_queries: User questions (analytics)
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupabaseDB:
    """
    Supabase database client for AlphaExtract.
    """
    
    def __init__(self, url: str = None, key: str = None):
        """
        Initialize Supabase client.
        
        Note: Loads .env file automatically.
        
        Args:
            url: Supabase project URL
            key: Supabase API key
        """
        # Load environment variables
        load_dotenv()
        
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            logger.warning("Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY")
            self.client = None
            return
        
        try:
            self.client: Client = create_client(self.url, self.key)
            logger.info("✓ Connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self.client = None
    
    # ========================================================================
    # COMPANIES TABLE
    # ========================================================================
    
    def upsert_company(self, ticker: str, name: str, sector: str = None) -> bool:
        """
        Insert or update company record.
        
        Args:
            ticker: Stock ticker
            name: Company name
            sector: Industry sector
            
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            data = {
                'ticker': ticker.upper(),
                'name': name,
                'sector': sector,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            self.client.table('companies').upsert(data).execute()
            logger.info(f"✓ Upserted company: {ticker}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to upsert company {ticker}: {e}")
            return False
    
    # ========================================================================
    # FILINGS TABLE
    # ========================================================================
    
    def insert_filing(
        self,
        ticker: str,
        filing_date: str,
        filing_type: str = '10-K',
        url: str = None,
        file_path: str = None
    ) -> Optional[int]:
        """
        Insert filing record.
        
        Args:
            ticker: Stock ticker
            filing_date: Filing date (YYYY-MM-DD)
            filing_type: Type of filing (10-K, 10-Q, etc.)
            url: SEC URL
            file_path: Local file path
            
        Returns:
            Filing ID if successful
        """
        if not self.client:
            return None
        
        try:
            data = {
                'ticker': ticker.upper(),
                'filing_date': filing_date,
                'filing_type': filing_type,
                'url': url,
                'file_path': file_path,
                'created_at': datetime.utcnow().isoformat()
            }
            
            result = self.client.table('filings').insert(data).execute()
            filing_id = result.data[0]['id']
            
            logger.info(f"✓ Inserted filing: {ticker} {filing_date}")
            return filing_id
        
        except Exception as e:
            logger.error(f"Failed to insert filing: {e}")
            return None
    
    # ========================================================================
    # SENTIMENT SCORES TABLE
    # ========================================================================
    
    def insert_sentiment(
        self,
        ticker: str,
        filing_date: str,
        sentiment_data: Dict
    ) -> bool:
        """
        Insert sentiment analysis results.
        
        Args:
            ticker: Stock ticker
            filing_date: Filing date
            sentiment_data: Sentiment analysis dict (from FinBERT)
            
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            # Overall sentiment
            overall = sentiment_data.get('overall', {})
            
            # Section sentiments
            sections = sentiment_data.get('sections', {})
            
            data = {
                'ticker': ticker.upper(),
                'filing_date': filing_date,
                'overall_compound': overall.get('compound', 0),
                'overall_signal': overall.get('signal', 'HOLD'),
                'item_1a_compound': sections.get('item_1a', {}).get('scores', {}).get('compound'),
                'item_1a_signal': sections.get('item_1a', {}).get('signal'),
                'item_7_compound': sections.get('item_7', {}).get('scores', {}).get('compound'),
                'item_7_signal': sections.get('item_7', {}).get('signal'),
                'item_8_compound': sections.get('item_8', {}).get('scores', {}).get('compound'),
                'item_8_signal': sections.get('item_8', {}).get('signal'),
                'raw_data': sentiment_data,  # Store full JSON
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.client.table('sentiment_scores').insert(data).execute()
            logger.info(f"✓ Inserted sentiment: {ticker} {filing_date}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to insert sentiment: {e}")
            return False
    
    def get_sentiment_history(
        self,
        ticker: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get historical sentiment scores for a ticker.
        
        Args:
            ticker: Stock ticker
            limit: Number of records to return
            
        Returns:
            List of sentiment records
        """
        if not self.client:
            return []
        
        try:
            result = self.client.table('sentiment_scores')\
                .select('*')\
                .eq('ticker', ticker.upper())\
                .order('filing_date', desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data
        
        except Exception as e:
            logger.error(f"Failed to get sentiment history: {e}")
            return []
    
    # ========================================================================
    # ANOMALIES TABLE
    # ========================================================================
    
    def insert_anomalies(
        self,
        ticker: str,
        filing_date: str,
        anomalies: List[Dict]
    ) -> bool:
        """
        Insert anomaly detection results.
        
        Args:
            ticker: Stock ticker
            filing_date: Filing date
            anomalies: List of anomaly dicts
            
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            records = []
            for anomaly in anomalies:
                record = {
                    'ticker': ticker.upper(),
                    'filing_date': filing_date,
                    'anomaly_type': anomaly.get('type'),
                    'severity': anomaly.get('severity', 'medium'),
                    'description': anomaly.get('description'),
                    'details': anomaly,  # Store full JSON
                    'created_at': datetime.utcnow().isoformat()
                }
                records.append(record)
            
            if records:
                self.client.table('anomalies').insert(records).execute()
                logger.info(f"✓ Inserted {len(records)} anomalies: {ticker} {filing_date}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to insert anomalies: {e}")
            return False
    
    def get_anomalies(
        self,
        ticker: str = None,
        severity: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get anomalies with optional filtering.
        
        Args:
            ticker: Filter by ticker (optional)
            severity: Filter by severity (optional)
            limit: Number of records
            
        Returns:
            List of anomaly records
        """
        if not self.client:
            return []
        
        try:
            query = self.client.table('anomalies').select('*')
            
            if ticker:
                query = query.eq('ticker', ticker.upper())
            
            if severity:
                query = query.eq('severity', severity)
            
            result = query.order('created_at', desc=True).limit(limit).execute()
            return result.data
        
        except Exception as e:
            logger.error(f"Failed to get anomalies: {e}")
            return []
    
    # ========================================================================
    # CHAT QUERIES TABLE (Analytics)
    # ========================================================================
    
    def log_chat_query(
        self,
        ticker: str,
        query: str,
        answer: str,
        num_sources: int = 0
    ) -> bool:
        """
        Log user chat query for analytics.
        
        Args:
            ticker: Company queried about
            query: User's question
            answer: AI's answer
            num_sources: Number of sources retrieved
            
        Returns:
            True if successful
        """
        if not self.client:
            return False
        
        try:
            data = {
                'ticker': ticker.upper() if ticker else None,
                'query': query,
                'answer': answer,
                'num_sources': num_sources,
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.client.table('chat_queries').insert(data).execute()
            return True
        
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            return False
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict]:
        """Get most common user queries."""
        if not self.client:
            return []
        
        try:
            # This would need a view or aggregation in Supabase
            result = self.client.table('chat_queries')\
                .select('query, ticker')\
                .limit(limit)\
                .execute()
            
            return result.data
        
        except Exception as e:
            logger.error(f"Failed to get popular queries: {e}")
            return []


# ============================================================================
# SQL SCHEMA FOR SUPABASE
# ============================================================================

SQL_SCHEMA = """
-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Filings table
CREATE TABLE IF NOT EXISTS filings (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    filing_type VARCHAR(20) DEFAULT '10-K',
    url TEXT,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, filing_date, filing_type)
);

-- Sentiment scores table
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    overall_compound FLOAT,
    overall_signal VARCHAR(20),
    item_1a_compound FLOAT,
    item_1a_signal VARCHAR(20),
    item_7_compound FLOAT,
    item_7_signal VARCHAR(20),
    item_8_compound FLOAT,
    item_8_signal VARCHAR(20),
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, filing_date)
);

-- Anomalies table
CREATE TABLE IF NOT EXISTS anomalies (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    description TEXT,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat queries table (analytics)
CREATE TABLE IF NOT EXISTS chat_queries (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    query TEXT NOT NULL,
    answer TEXT,
    num_sources INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_scores(ticker, filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_ticker ON anomalies(ticker, filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_chat_ticker ON chat_queries(ticker);
CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_queries(created_at DESC);
"""


if __name__ == "__main__":
    print("=" * 70)
    print("SUPABASE DATABASE SETUP")
    print("=" * 70)
    
    print("\n📋 SQL Schema to run in Supabase SQL Editor:")
    print("-" * 70)
    print(SQL_SCHEMA)
    print("-" * 70)
    
    print("\n🔧 Testing connection...")
    db = SupabaseDB()
    
    if db.client:
        print("✓ Successfully connected to Supabase!")
        
        # Test insert
        print("\n🧪 Testing company insert...")
        success = db.upsert_company('AAPL', 'Apple Inc.', 'Technology')
        
        if success:
            print("✓ Test successful!")
        else:
            print("✗ Test failed")
    else:
        print("✗ Connection failed. Check your .env file:")
        print("  SUPABASE_URL=https://your-project.supabase.co")
        print("  SUPABASE_KEY=your-anon-key")