"""Load and parse candidate data from JSONL files."""

import gzip
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union

logger = logging.getLogger(__name__)


class CandidateLoader:
    """Load candidates from gzipped JSONL file."""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        self.is_gzipped = str(self.file_path).endswith('.gz')
        logger.info(f"Initialized loader for {self.file_path} (gzipped: {self.is_gzipped})")
    
    def load_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        candidates = []
        logger.info(f"Loading candidates from {self.file_path}")
        open_func = gzip.open if self.is_gzipped else open
        with open_func(self.file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                    candidates.append(candidate)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {i+1}: {e}")
                    continue
        logger.info(f"Loaded {len(candidates)} candidates")
        return candidates
    
    def stream(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        open_func = gzip.open if self.is_gzipped else open
        with open_func(self.file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    
    def get_count(self) -> int:
        count = 0
        open_func = gzip.open if self.is_gzipped else open
        with open_func(self.file_path, 'rt', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count

    # The rest of the static methods remain unchanged
