# tests/test_storage/test_vector_store.py
import pytest
from storage.vector_store import VectorStore


@pytest.fixture
def store(db):
    return VectorStore(db)


def _fake_vec(seed: float = 0.1) -> list[float]:
    """384-dim vector for testing."""
    return [seed] * 384


def test_insert_and_search_returns_chunk(store):
    store.insert("policy.md", 0, "Annual leave is 20 days.", _fake_vec(0.1))
    results = store.search(_fake_vec(0.1), top_k=1)
    assert len(results) == 1
    text, doc_name = results[0]
    assert text == "Annual leave is 20 days."
    assert doc_name == "policy.md"


def test_search_returns_top_k(store):
    for i in range(5):
        store.insert("doc.md", i, f"chunk {i}", _fake_vec(0.1 + i * 0.01))
    results = store.search(_fake_vec(0.1), top_k=3)
    assert len(results) == 3


def test_search_empty_store_returns_empty(store):
    results = store.search(_fake_vec(0.5), top_k=3)
    assert results == []


def test_delete_by_doc_removes_chunks(store):
    store.insert("to_delete.md", 0, "will be removed", _fake_vec(0.2))
    store.insert("keep.md", 0, "stays here", _fake_vec(0.3))
    store.delete_by_doc("to_delete.md")
    results = store.search(_fake_vec(0.2), top_k=5)
    doc_names = [r[1] for r in results]
    assert "to_delete.md" not in doc_names


def test_has_doc_returns_true_when_present(store):
    store.insert("exists.md", 0, "content", _fake_vec(0.1))
    assert store.has_doc("exists.md") is True


def test_has_doc_returns_false_when_absent(store):
    assert store.has_doc("missing.md") is False
