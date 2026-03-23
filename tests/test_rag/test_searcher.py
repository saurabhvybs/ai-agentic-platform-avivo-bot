import pytest
from unittest.mock import MagicMock
from rag.retrieval.searcher import RAGSearcher
from rag.ingestion.embedder import TextEmbedder
from storage.vector_store import VectorStore


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=TextEmbedder)
    embedder.embed.return_value = [0.1] * 384
    return embedder


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.search.return_value = [
        ("Annual leave is 20 days.", "leave_policy"),
        ("Sick leave is 10 days.", "leave_policy"),
    ]
    return store


@pytest.fixture
def searcher(mock_embedder, mock_vector_store):
    return RAGSearcher(mock_embedder, mock_vector_store, top_k=2)


async def test_retrieve_returns_chunks(searcher):
    results = await searcher.retrieve("what is leave policy?")
    assert len(results) == 2
    assert results[0][0] == "Annual leave is 20 days."
    assert results[0][1] == "leave_policy"


async def test_retrieve_calls_embed_with_query(searcher, mock_embedder):
    await searcher.retrieve("my query")
    mock_embedder.embed.assert_called_once_with("my query")


async def test_retrieve_calls_search_with_top_k(searcher, mock_vector_store):
    await searcher.retrieve("query")
    mock_vector_store.search.assert_called_once_with([0.1] * 384, 2)


async def test_retrieve_empty_results(mock_embedder):
    store = MagicMock(spec=VectorStore)
    store.search.return_value = []
    searcher = RAGSearcher(mock_embedder, store, top_k=3)
    results = await searcher.retrieve("unknown topic")
    assert results == []
