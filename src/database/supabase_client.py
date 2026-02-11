"""
Supabase Database Client
-------------------------
Manages data persistence for sentiment results and anomaly reports.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class SupabaseDB:
    """Client for Supabase PostgreSQL operations."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not Settings.SUPABASE_URL or not Settings.SUPABASE_KEY:
                logger.warning("Supabase credentials not configured — database operations disabled")
                return None
            try:
                from supabase import create_client
                self._client = create_client(Settings.SUPABASE_URL, Settings.SUPABASE_KEY)
                logger.info("Connected to Supabase")
            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}")
                return None
        return self._client

    @property
    def available(self) -> bool:
        return self.client is not None

    def save_sentiment(self, ticker: str, filing_date: str, data: Dict) -> bool:
        """Save or update sentiment analysis results."""
        if not self.available:
            return False
        try:
            record = {
                "ticker": ticker.upper(),
                "filing_date": filing_date,
                "overall_score": data.get("overall", {}).get("compound", 0),
                "overall_signal": data.get("overall", {}).get("signal", "HOLD"),
                "raw_data": json.dumps(data),
            }
            self.client.table("sentiment_results").upsert(
                record, on_conflict="ticker,filing_date"
            ).execute()
            logger.info(f"Saved sentiment to DB: {ticker} {filing_date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save sentiment for {ticker} {filing_date}: {e}")
            return False

    def save_anomalies(self, report: Dict) -> bool:
        """Save anomaly detection report."""
        if not self.available:
            return False
        try:
            ticker = report.get("ticker", "")
            filing_date = report.get("current_filing_date", "")

            for anomaly in report.get("anomalies", []):
                anomaly_hash = hashlib.sha256(
                    f"{ticker}:{filing_date}:{anomaly.get('type')}:{anomaly.get('title')}".encode()
                ).hexdigest()[:32]

                record = {
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "anomaly_hash": anomaly_hash,
                    "type": anomaly.get("type", ""),
                    "title": anomaly.get("title", ""),
                    "severity": anomaly.get("severity", "medium"),
                    "description": anomaly.get("description", ""),
                    "raw_data": json.dumps(anomaly),
                }
                self.client.table("anomalies").upsert(
                    record, on_conflict="anomaly_hash"
                ).execute()

            logger.info(f"Saved {len(report.get('anomalies', []))} anomalies to DB for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Failed to save anomalies: {e}")
            return False

    def get_sentiment_history(self, ticker: str) -> List[Dict]:
        """Get all sentiment records for a ticker, ordered by date."""
        if not self.available:
            return []
        try:
            result = (
                self.client.table("sentiment_results")
                .select("*")
                .eq("ticker", ticker.upper())
                .order("filing_date", desc=True)
                .limit(50)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get sentiment history for {ticker}: {e}")
            return []

    def get_anomalies(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Get anomaly records for a ticker."""
        if not self.available:
            return []
        try:
            result = (
                self.client.table("anomalies")
                .select("*")
                .eq("ticker", ticker.upper())
                .order("filing_date", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get anomalies for {ticker}: {e}")
            return []
