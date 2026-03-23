# tests/test_storage/test_cache.py
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from storage.cache import QueryCache
from shared.models import RAGResult


@pytest.fixture
def cache(db):
    return QueryCache(db, ttl_hours=24)


def _result(answer: str = "42") -> RAGResult:
    return RAGResult(answer=answer, sources=["doc.md"], web_references=[], from_cache=False)


def test_cache_miss_returns_none(cache):
    assert cache.get("what is leave policy?") is None


def test_cache_set_and_get(cache):
    cache.set("what is leave policy?", _result("20 days"))
    result = cache.get("what is leave policy?")
    assert result is not None
    assert result.answer == "20 days"
    assert result.from_cache is True


def test_cache_hit_preserves_sources(cache):
    cache.set("query", _result("ans"))
    result = cache.get("query")
    assert result.sources == ["doc.md"]


def test_cache_ttl_expiry_returns_none(cache):
    cache.set("old query", _result("stale"))
    # Patch datetime to simulate TTL expiry
    future = datetime.now(timezone.utc) + timedelta(hours=25)
    with patch("storage.cache.datetime") as mock_dt:
        mock_dt.now.return_value = future
        mock_dt.fromisoformat = datetime.fromisoformat
        result = cache.get("old query")
    assert result is None


def test_clear_all_removes_entries(cache):
    cache.set("q1", _result())
    cache.set("q2", _result())
    cache.clear_all()
    assert cache.get("q1") is None
    assert cache.get("q2") is None


def test_web_references_round_trip(cache):
    result = RAGResult(
        answer="ans", sources=[], web_references=[{"url": "http://x.com", "title": "X"}], from_cache=False
    )
    cache.set("web query", result)
    loaded = cache.get("web query")
    assert loaded.web_references == [{"url": "http://x.com", "title": "X"}]
