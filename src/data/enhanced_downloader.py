"""
Enhanced SEC Downloader with Historical Support
-------------------------------------------------
This module is kept for backward compatibility.
New code should use src.data.downloader.SECDownloader directly.
"""

from src.data.downloader import SECDownloader as EnhancedSECDownloader

__all__ = ["EnhancedSECDownloader"]
