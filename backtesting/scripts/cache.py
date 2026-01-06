"""
Cache system for backtesting - stores search results to avoid redundant API calls.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(question_id: int) -> Path:
    """Get the cache file path for a question."""
    return CACHE_DIR / f"q_{question_id}.json"


def save_search_cache(
    question_id: int,
    question_title: str,
    search_query: str,
    search_date: str,
    resolution_date: str,
    search_results: list[dict],
    crawled_pages: list[dict],
    metadata: dict = None
) -> Path:
    """
    Save search results to cache.
    
    Args:
        question_id: Metaculus question ID
        question_title: Question title
        search_query: The query used for search
        search_date: Date used for search cutoff (before resolution)
        resolution_date: When the question resolved
        search_results: Raw search results from Exa
        crawled_pages: Any crawled pages
        metadata: Additional metadata
    
    Returns:
        Path to the cache file
    """
    ensure_cache_dir()
    
    cache_data = {
        "question_id": question_id,
        "question_title": question_title,
        "search_query": search_query,
        "search_date": search_date,
        "resolution_date": resolution_date,
        "search_results": search_results,
        "crawled_pages": crawled_pages,
        "metadata": metadata or {},
        "cached_at": datetime.now().isoformat()
    }
    
    cache_path = get_cache_path(question_id)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, default=str)
    
    print(f"[Cache] Saved search cache for question {question_id}")
    return cache_path


def load_search_cache(question_id: int) -> Optional[dict]:
    """
    Load cached search results for a question.
    
    Returns:
        Cached data dict, or None if not cached
    """
    cache_path = get_cache_path(question_id)
    
    if not cache_path.exists():
        return None
    
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_cached(question_id: int) -> bool:
    """Check if a question has cached search data."""
    return get_cache_path(question_id).exists()


def list_cached_questions() -> list[int]:
    """Get list of all cached question IDs."""
    ensure_cache_dir()
    cached = []
    for f in CACHE_DIR.glob("q_*.json"):
        try:
            qid = int(f.stem.split("_")[1])
            cached.append(qid)
        except (ValueError, IndexError):
            continue
    return cached


def clear_cache():
    """Clear all cached data."""
    ensure_cache_dir()
    for f in CACHE_DIR.glob("q_*.json"):
        f.unlink()
    print("[Cache] Cleared all cached data")
