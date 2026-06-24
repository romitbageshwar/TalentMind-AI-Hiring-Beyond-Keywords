"""Utility functions for the ranking system."""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
import json
import hashlib


def setup_logging(level: str = 'INFO') -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def hash_string(s: str) -> str:
    """Create a hash of a string."""
    return hashlib.md5(s.encode()).hexdigest()[:8]


def safe_get(data: Dict[str, Any], keys: list, default: Any = None) -> Any:
    """Safely get nested dictionary value."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, {})
        else:
            return default
    return data if data != {} else default


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            self.logger.info(f"{self.name} took {self.elapsed:.2f} seconds")


class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n
    
    def get_progress(self) -> float:
        """Get progress as fraction."""
        return self.current / self.total if self.total > 0 else 0
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_estimated_remaining(self) -> Optional[float]:
        """Get estimated remaining time in seconds."""
        if self.current == 0:
            return None
        elapsed = self.get_elapsed()
        progress = self.get_progress()
        if progress > 0:
            return (elapsed / progress) * (1 - progress)
        return None
    
    def get_summary(self) -> str:
        """Get progress summary string."""
        progress = self.get_progress() * 100
        elapsed = format_duration(self.get_elapsed())
        remaining = self.get_estimated_remaining()
        if remaining:
            return f"{self.description}: {progress:.1f}% ({elapsed} elapsed, ~{format_duration(remaining)} remaining)"
        return f"{self.description}: {progress:.1f}% ({elapsed} elapsed)"
