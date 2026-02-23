"""Built-in dataset loaders for popular benchmarks."""

from __future__ import annotations

import random
from pathlib import Path

from trustgate.types import Question


def subsample(
    questions: list[Question],
    n: int,
    seed: int = 42,
) -> list[Question]:
    """Randomly subsample *n* questions with a fixed seed for reproducibility."""
    if n >= len(questions):
        return questions
    rng = random.Random(seed)
    return rng.sample(questions, n)


def _datasets_cache_dir() -> Path:
    """Return the datasets cache directory (~/.trustgate/datasets/)."""
    d = Path.home() / ".trustgate" / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_jsonl(url: str, name: str) -> Path:
    """Download a JSONL file from a URL, caching it locally.

    Uses httpx for the download. Returns the local file path.
    """
    cache_dir = _datasets_cache_dir()
    local_path = cache_dir / name

    if local_path.exists():
        return local_path

    import httpx

    resp = httpx.get(url, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return local_path
