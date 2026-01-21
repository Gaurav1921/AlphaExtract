"""
Supabase Database Client
------------------------
Connects to Supabase and manages data persistence.
Now with upsert support to prevent duplicates.
"""

import os
from supabase import create_client, Client
from datetime import datetime
from typing import Dict, List, Optional
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupabaseDB:
    """Supabase database client for AlphaExtract."""
    
    def __init__(self, url: str = None, key: str = None):
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_KEY')
        
        if not self.url or not self.key:
            logger.warning("Supabase credentials not found")
            self.client = None
            return
        
        try:
            self.client: Client = create_client(self.url, self.key)
            logger.info("Connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self.client = None
    
    def upsert_company(self, ticker: str, name: str, sector: str = None) -> bool:
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
            return True
        except Exception as e:
            logger.error(f"Failed to upsert company {ticker}: {e}")
            return False
    
    def insert_sentiment(self, ticker: str, filing_date: str, sentiment_data: Dict) -> bool:
        if not self.client:
            return False
        
        try:
            overall = sentiment_data.get('overall', {})
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
                'raw_data': sentiment_data,
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.client.table('sentiment_scores').upsert(data, on_conflict='ticker,filing_date').execute()
            return True
        except Exception as e:
            if '23505' in str(e) or 'duplicate' in str(e).lower():
                return True
            logger.error(f"Failed to insert sentiment: {e}")
            return False
    
    def get_sentiment_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        if not self.client:
            return []
        
        try:
            result = self.client.table('sentiment_scores').select('*').eq('ticker', ticker.upper()).order('filing_date', desc=True).limit(limit).execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get sentiment history: {e}")
            return []
    
    def _generate_anomaly_hash(self, ticker: str, filing_date: str, anomaly: Dict) -> str:
        unique_str = f"{ticker}_{filing_date}_{anomaly.get('type', '')}_{anomaly.get('keyword', '')}_{anomaly.get('description', '')[:100]}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def insert_anomalies(self, ticker: str, filing_date: str, anomalies: List[Dict]) -> bool:
        if not self.client:
            return False
        
        try:
            for anomaly in anomalies:
                anomaly_hash = self._generate_anomaly_hash(ticker, filing_date, anomaly)
                
                record = {
                    'ticker': ticker.upper(),
                    'filing_date': filing_date,
                    'anomaly_type': anomaly.get('type'),
                    'severity': anomaly.get('severity', 'medium'),
                    'description': anomaly.get('description'),
                    'anomaly_hash': anomaly_hash,
                    'details': anomaly,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                try:
                    self.client.table('anomalies').upsert(record, on_conflict='anomaly_hash').execute()
                except Exception as inner_e:
                    if '23505' not in str(inner_e) and 'duplicate' not in str(inner_e).lower():
                        logger.warning(f"Anomaly upsert issue: {inner_e}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to insert anomalies: {e}")
            return False
    
    def get_anomalies(self, ticker: str = None, severity: str = None, limit: int = 50) -> List[Dict]:
        if not self.client:
            return []
        
        try:
            query = self.client.table('anomalies').select('*')
            
            if ticker:
                query = query.eq('ticker', ticker.upper())
            if severity:
                query = query.eq('severity', severity)
            
            result = query.order('created_at', desc=True).limit(limit).execute()
            
            # Deduplicate
            seen = set()
            unique = []
            for item in result.data:
                key = item.get('anomaly_hash') or f"{item.get('ticker')}_{item.get('filing_date')}_{item.get('description', '')[:30]}"
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            
            return unique
        except Exception as e:
            logger.error(f"Failed to get anomalies: {e}")
            return []
    
    def log_chat_query(self, ticker: str, query: str, answer: str, num_sources: int = 0) -> bool:
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
        if not self.client:
            return []
        
        try:
            result = self.client.table('chat_queries').select('query, ticker').limit(limit).execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get popular queries: {e}")
            return []


# SQL schema for Supabase - add anomaly_hash column and unique constraint
SQL_SCHEMA_UPDATE = """
-- Add anomaly_hash column if not exists
ALTER TABLE anomalies ADD COLUMN IF NOT EXISTS anomaly_hash VARCHAR(16);

-- Create unique index on anomaly_hash
CREATE UNIQUE INDEX IF NOT EXISTS idx_anomalies_hash ON anomalies(anomaly_hash);

-- Update existing records with hash (run once)
-- UPDATE anomalies SET anomaly_hash = md5(ticker || filing_date || anomaly_type || description)::varchar(16) WHERE anomaly_hash IS NULL;
"""

if __name__ == "__main__":
    print("Supabase Client")
    print("Run this SQL in Supabase to add deduplication support:")
    print(SQL_SCHEMA_UPDATE)
    
    db = SupabaseDB()
    if db.client:
        print("Connected successfully!")
    else:
        print("Connection failed - check credentials")