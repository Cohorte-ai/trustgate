"""Disk cache with SHA-256 hashed filenames."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class DiskCache:
    """File-based response cache keyed by SHA-256 hash.

    Each cached entry is a JSON file named ``{hash}.json`` inside the cache
    directory.  Writes are atomic (write to tempfile, then rename) so
    concurrent processes won't see partially-written files.
    """

    def __init__(self, cache_dir: str = ".trustgate_cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def key(
        url: str,
        provider: str,
        model: str,
        prompt: str,
        temperature: float | None,
        index: int,
    ) -> str:
        """Deterministic SHA-256 hash of all request parameters."""
        blob = json.dumps(
            {
                "url": url,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "index": index,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def _path(self, cache_key: str) -> Path:
        return self._dir / f"{cache_key}.json"

    def has(self, cache_key: str) -> bool:
        """Check if a cache entry exists."""
        return self._path(cache_key).exists()

    def get(self, cache_key: str) -> str | None:
        """Return the cached response string, or ``None`` if missing."""
        path = self._path(cache_key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data["response"])

    def put(
        self,
        cache_key: str,
        response: str,
        *,
        provider: str = "",
        model: str = "",
        temperature: float | None = 0.0,
        index: int = 0,
    ) -> None:
        """Write a response to the cache (atomic via tempfile + rename)."""
        payload = {
            "provider": provider,
            "model": model,
            "prompt_hash": cache_key,
            "temperature": temperature,
            "index": index,
            "response": response,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2)

        # Atomic write: write to a temp file in the same dir, then rename.
        fd, tmp_path = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            os.write(fd, data.encode("utf-8"))
            os.close(fd)
            os.replace(tmp_path, self._path(cache_key))
        except BaseException:
            os.close(fd) if not _fd_closed(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        """Return cache statistics."""
        entries = list(self._dir.glob("*.json"))
        if not entries:
            return {
                "total_entries": 0,
                "total_size_bytes": 0,
                "oldest": None,
                "newest": None,
            }

        sizes = [e.stat().st_size for e in entries]
        mtimes = [e.stat().st_mtime for e in entries]

        return {
            "total_entries": len(entries),
            "total_size_bytes": sum(sizes),
            "oldest": datetime.fromtimestamp(min(mtimes), tz=timezone.utc).isoformat(),
            "newest": datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat(),
        }

    def clear(self) -> int:
        """Delete all cached entries.  Return the number deleted."""
        entries = list(self._dir.glob("*.json"))
        for entry in entries:
            entry.unlink()
        return len(entries)


def _fd_closed(fd: int) -> bool:
    """Check if a file descriptor is already closed."""
    try:
        os.fstat(fd)
        return False
    except OSError:
        return True
