"""
Tests for src/data/downloader.py
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.downloader import SECDownloader


class TestGetCik:
    """Test CIK lookup from ticker symbols."""

    def test_rejects_invalid_ticker_format(self):
        downloader = SECDownloader(email="test@example.com")
        assert downloader.get_cik("123") is None
        assert downloader.get_cik("TOOLONGTICKER") is None
        assert downloader.get_cik("A1B") is None

    def test_successful_cik_lookup(self):
        downloader = SECDownloader(email="test@example.com")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(downloader.session, "get", return_value=mock_response):
            cik = downloader.get_cik("AAPL")
            assert cik == "0000320193"

    def test_fallback_endpoint(self):
        downloader = SECDownloader(email="test@example.com")

        # Primary returns empty
        primary_response = MagicMock()
        primary_response.json.return_value = {"data": []}
        primary_response.raise_for_status = MagicMock()

        # Fallback finds it
        fallback_response = MagicMock()
        fallback_response.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }
        fallback_response.raise_for_status = MagicMock()

        with patch.object(
            downloader.session,
            "get",
            side_effect=[primary_response, fallback_response],
        ):
            cik = downloader.get_cik("AAPL")
            assert cik == "0000320193"

    def test_network_error(self):
        import requests

        downloader = SECDownloader(email="test@example.com")
        with patch.object(
            downloader.session,
            "get",
            side_effect=requests.RequestException("timeout"),
        ):
            cik = downloader.get_cik("AAPL")
            assert cik is None


class TestGet10KFilings:
    """Test 10-K filing metadata retrieval."""

    def test_returns_filing_info(self):
        downloader = SECDownloader(email="test@example.com")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "10-K"],
                    "accessionNumber": [
                        "0001-23-456789",
                        "0001-23-999999",
                        "0001-22-456789",
                    ],
                    "filingDate": ["2024-01-15", "2024-06-01", "2023-01-20"],
                    "primaryDocument": ["filing.htm", "report.htm", "filing.htm"],
                }
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(downloader.session, "get", return_value=mock_response):
            filings = downloader.get_10k_filings("0000320193", limit=5)

        assert len(filings) == 2  # Only 10-K forms
        assert filings[0]["filing_date"] == "2024-01-15"
        assert filings[1]["filing_date"] == "2023-01-20"

    def test_respects_limit(self):
        downloader = SECDownloader(email="test@example.com")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-K", "10-K"],
                    "accessionNumber": ["0001-24-000001", "0001-23-000002", "0001-22-000003"],
                    "filingDate": ["2024-01-15", "2023-01-20", "2022-01-18"],
                    "primaryDocument": ["a.htm", "b.htm", "c.htm"],
                }
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(downloader.session, "get", return_value=mock_response):
            filings = downloader.get_10k_filings("0000320193", limit=1)

        assert len(filings) == 1

    def test_network_error_returns_empty(self):
        import requests

        downloader = SECDownloader(email="test@example.com")
        with patch.object(
            downloader.session,
            "get",
            side_effect=requests.RequestException("error"),
        ):
            filings = downloader.get_10k_filings("0000320193")
            assert filings == []


class TestDownloadFiling:
    """Test individual filing download."""

    def test_saves_file_to_disk(self, tmp_path):
        downloader = SECDownloader(email="test@example.com")
        downloader.raw_dir = tmp_path

        mock_response = MagicMock()
        mock_response.content = b"<html>10-K content</html>"
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        filing_info = {
            "url": "https://www.sec.gov/test/filing.htm",
            "filing_date": "2024-01-15",
            "accession": "0001234567",
        }

        with patch.object(downloader.session, "get", return_value=mock_response), \
             patch("src.data.downloader.time.sleep"):
            filepath = downloader.download_filing("AAPL", filing_info)

        assert filepath is not None
        assert filepath.exists()
        assert "AAPL_10K_2024-01-15" in filepath.name

    def test_saves_metadata_json(self, tmp_path):
        downloader = SECDownloader(email="test@example.com")
        downloader.raw_dir = tmp_path

        mock_response = MagicMock()
        mock_response.content = b"content"
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.raise_for_status = MagicMock()

        filing_info = {
            "url": "https://www.sec.gov/test/filing.txt",
            "filing_date": "2024-01-15",
            "accession": "0001234567",
        }

        with patch.object(downloader.session, "get", return_value=mock_response), \
             patch("src.data.downloader.time.sleep"):
            filepath = downloader.download_filing("AAPL", filing_info)

        metadata_path = filepath.with_suffix(".json")
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["ticker"] == "AAPL"

    def test_retry_on_failure(self, tmp_path):
        import requests

        downloader = SECDownloader(email="test@example.com")
        downloader.raw_dir = tmp_path

        # First call fails, second succeeds
        fail_response = requests.RequestException("timeout")
        ok_response = MagicMock()
        ok_response.content = b"ok"
        ok_response.headers = {"Content-Type": "text/html"}
        ok_response.raise_for_status = MagicMock()

        filing_info = {
            "url": "https://www.sec.gov/test/filing.htm",
            "filing_date": "2024-01-15",
            "accession": "001",
        }

        with patch.object(
            downloader.session,
            "get",
            side_effect=[fail_response, ok_response],
        ), patch("src.data.downloader.time.sleep"):
            filepath = downloader.download_filing("AAPL", filing_info)

        assert filepath is not None


class TestDownloadMultiple:
    """Test batch download."""

    def test_batch_download(self, tmp_path):
        downloader = SECDownloader(email="test@example.com")
        downloader.raw_dir = tmp_path

        # Mock CIK lookup
        mock_cik_response = MagicMock()
        mock_cik_response.json.return_value = {
            "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]]
        }
        mock_cik_response.raise_for_status = MagicMock()

        # Mock filing list
        mock_filings_response = MagicMock()
        mock_filings_response.json.return_value = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "accessionNumber": ["0001-24-000001"],
                    "filingDate": ["2024-01-15"],
                    "primaryDocument": ["filing.htm"],
                }
            }
        }
        mock_filings_response.raise_for_status = MagicMock()

        # Mock download
        mock_download_response = MagicMock()
        mock_download_response.content = b"<html>content</html>"
        mock_download_response.headers = {"Content-Type": "text/html"}
        mock_download_response.raise_for_status = MagicMock()

        with patch.object(
            downloader.session,
            "get",
            side_effect=[
                mock_cik_response,
                mock_filings_response,
                mock_download_response,
            ],
        ), patch("src.data.downloader.time.sleep"):
            results = downloader.download_multiple(["AAPL"], years=1)

        assert len(results["successful"]) == 1
        assert len(results["failed"]) == 0


class TestSECDownloaderInit:
    """Test downloader initialization."""

    def test_user_agent_header(self):
        downloader = SECDownloader(email="test@example.com")
        assert "test@example.com" in downloader.session.headers["User-Agent"]
        assert "AlphaExtract" in downloader.session.headers["User-Agent"]

    def test_default_email(self):
        downloader = SECDownloader()
        assert "User-Agent" in downloader.session.headers
