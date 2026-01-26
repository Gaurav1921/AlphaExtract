"""
AlphaExtract Error Logger
-------------------------
Centralized error logging to errors.txt for easy debugging.

Usage:
    from src.utils.error_logger import log_error, get_error_log
    
    try:
        # some code
    except Exception as e:
        log_error("Module Name", "Function Name", e)
"""

import traceback
from pathlib import Path
from datetime import datetime
import sys

ERROR_LOG_PATH = Path("errors.txt")


def log_error(module: str, function: str, error: Exception, extra_info: str = ""):
    """
    Log an error to errors.txt with full traceback.
    
    Args:
        module: Module name where error occurred
        function: Function name where error occurred  
        error: The exception that was raised
        extra_info: Any additional context
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    error_entry = f"""
{'='*80}
TIMESTAMP: {timestamp}
MODULE: {module}
FUNCTION: {function}
ERROR TYPE: {type(error).__name__}
ERROR MESSAGE: {str(error)}
{f'EXTRA INFO: {extra_info}' if extra_info else ''}

TRACEBACK:
{traceback.format_exc()}
{'='*80}
"""
    
    # Append to error log
    with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(error_entry)
    
    # Also print to console
    print(f"[ERROR] {module}.{function}: {error}", file=sys.stderr)
    

def clear_error_log():
    """Clear the error log file."""
    if ERROR_LOG_PATH.exists():
        ERROR_LOG_PATH.unlink()
    print("Error log cleared.")


def get_error_log() -> str:
    """Read and return the error log contents."""
    if ERROR_LOG_PATH.exists():
        return ERROR_LOG_PATH.read_text(encoding='utf-8')
    return "No errors logged."


def get_recent_errors(n: int = 5) -> str:
    """Get the last N error entries."""
    if not ERROR_LOG_PATH.exists():
        return "No errors logged."
    
    content = ERROR_LOG_PATH.read_text(encoding='utf-8')
    entries = content.split('='*80)
    
    # Filter out empty entries
    entries = [e.strip() for e in entries if e.strip()]
    
    # Get last N
    recent = entries[-n:] if len(entries) >= n else entries
    
    return ('\n' + '='*80 + '\n').join(recent)


# Initialize error log on import
if not ERROR_LOG_PATH.exists():
    ERROR_LOG_PATH.write_text(f"# AlphaExtract Error Log\n# Created: {datetime.now()}\n\n")