"""Tests for the disk cache."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from theaios.trustgate.cache import DiskCache

_URL = "https://api.openai.com/v1/chat/completions"
_URL2 = "https://api.anthropic.com/v1/messages"


@pytest.fixture()
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache(cache_dir=str(tmp_path / "cache"))


class TestDiskCacheKey:
    def test_deterministic(self, cache: DiskCache) -> None:
        k1 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        k2 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        assert k1 == k2

    def test_different_prompt(self, cache: DiskCache) -> None:
        k1 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        k2 = cache.key(_URL, "openai", "gpt-4.1", "world", 0.7, 0)
        assert k1 != k2

    def test_different_model(self, cache: DiskCache) -> None:
        k1 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        k2 = cache.key(_URL, "openai", "gpt-4.1-mini", "hello", 0.7, 0)
        assert k1 != k2

    def test_different_temperature(self, cache: DiskCache) -> None:
        k1 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        k2 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.9, 0)
        assert k1 != k2

    def test_different_index(self, cache: DiskCache) -> None:
        k1 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        k2 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 1)
        assert k1 != k2

    def test_different_provider(self, cache: DiskCache) -> None:
        k1 = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        k2 = cache.key(_URL2, "anthropic", "gpt-4.1", "hello", 0.7, 0)
        assert k1 != k2

    def test_different_url(self, cache: DiskCache) -> None:
        k1 = cache.key(_URL, "generic", "m1", "hello", 0.7, 0)
        k2 = cache.key("https://other.example.com/ask", "generic", "m1", "hello", 0.7, 0)
        assert k1 != k2

    def test_none_temperature(self, cache: DiskCache) -> None:
        k1 = cache.key(_URL, "generic_http", "", "hello", None, 0)
        k2 = cache.key(_URL, "generic_http", "", "hello", None, 0)
        assert k1 == k2
        # None temperature differs from 0.7
        k3 = cache.key(_URL, "generic_http", "", "hello", 0.7, 0)
        assert k1 != k3

    def test_is_hex_string(self, cache: DiskCache) -> None:
        k = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        assert len(k) == 64  # SHA-256 hex
        int(k, 16)  # should not raise


class TestDiskCachePutGet:
    def test_round_trip(self, cache: DiskCache) -> None:
        k = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        cache.put(k, "The answer is 42.")
        assert cache.get(k) == "The answer is 42."

    def test_get_missing_returns_none(self, cache: DiskCache) -> None:
        assert cache.get("0" * 64) is None

    def test_has_returns_true(self, cache: DiskCache) -> None:
        k = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        cache.put(k, "response")
        assert cache.has(k) is True

    def test_has_returns_false(self, cache: DiskCache) -> None:
        assert cache.has("0" * 64) is False

    def test_overwrite(self, cache: DiskCache) -> None:
        k = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        cache.put(k, "first")
        cache.put(k, "second")
        assert cache.get(k) == "second"

    def test_unicode_response(self, cache: DiskCache) -> None:
        k = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        cache.put(k, "La reponse est 42")
        assert cache.get(k) == "La reponse est 42"

    def test_cache_file_is_valid_json(self, cache: DiskCache) -> None:
        k = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        cache.put(k, "response", provider="openai", model="gpt-4.1")
        path = cache.cache_dir / f"{k}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["response"] == "response"
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4.1"
        assert "cached_at" in data

    def test_put_with_metadata(self, cache: DiskCache) -> None:
        k = cache.key(_URL2, "anthropic", "claude", "prompt", 0.5, 2)
        cache.put(k, "resp", provider="anthropic", model="claude", temperature=0.5, index=2)
        path = cache.cache_dir / f"{k}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["provider"] == "anthropic"
        assert data["temperature"] == 0.5
        assert data["index"] == 2


class TestDiskCacheStats:
    def test_empty_cache(self, cache: DiskCache) -> None:
        stats = cache.stats()
        assert stats["total_entries"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["oldest"] is None
        assert stats["newest"] is None

    def test_with_entries(self, cache: DiskCache) -> None:
        for i in range(5):
            k = cache.key(_URL, "openai", "gpt-4.1", f"prompt_{i}", 0.7, 0)
            cache.put(k, f"response_{i}")
        stats = cache.stats()
        assert stats["total_entries"] == 5
        assert stats["total_size_bytes"] > 0
        assert stats["oldest"] is not None
        assert stats["newest"] is not None


class TestDiskCacheClear:
    def test_clear_empty(self, cache: DiskCache) -> None:
        assert cache.clear() == 0

    def test_clear_removes_all(self, cache: DiskCache) -> None:
        for i in range(3):
            k = cache.key(_URL, "openai", "gpt-4.1", f"prompt_{i}", 0.7, 0)
            cache.put(k, f"response_{i}")
        assert cache.clear() == 3
        assert cache.stats()["total_entries"] == 0

    def test_cache_usable_after_clear(self, cache: DiskCache) -> None:
        k = cache.key(_URL, "openai", "gpt-4.1", "hello", 0.7, 0)
        cache.put(k, "first")
        cache.clear()
        assert cache.get(k) is None
        cache.put(k, "second")
        assert cache.get(k) == "second"


class TestDiskCacheConcurrency:
    def test_concurrent_writes(self, cache: DiskCache) -> None:
        """Multiple threads writing to different keys should not corrupt data."""
        errors: list[Exception] = []

        def write(i: int) -> None:
            try:
                k = cache.key(_URL, "openai", "gpt-4.1", f"prompt_{i}", 0.7, 0)
                cache.put(k, f"response_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert cache.stats()["total_entries"] == 20

        # Verify all values are readable
        for i in range(20):
            k = cache.key(_URL, "openai", "gpt-4.1", f"prompt_{i}", 0.7, 0)
            assert cache.get(k) == f"response_{i}"
