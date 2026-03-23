"""Unit tests for scripts/ingest.py — verifies idempotency and --force behavior."""
from unittest.mock import MagicMock, patch
import scripts.ingest as ingest_module


def _setup_mocks(has_doc=False, docs=None, chunk_count=2):
    """Return configured mocks for all ingest.py dependencies."""
    if docs is None:
        docs = [("policy", "policy content here")]
    mock_vs = MagicMock()
    mock_vs.has_doc.return_value = has_doc
    mock_cache = MagicMock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = docs
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [f"chunk {i}" for i in range(chunk_count)]
    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1] * 384] * chunk_count
    return mock_vs, mock_cache, mock_loader, mock_chunker, mock_embedder


def _run_main(force=False, has_doc=False, docs=None, chunk_count=2):
    mock_vs, mock_cache, mock_loader, mock_chunker, mock_embedder = _setup_mocks(
        has_doc=has_doc, docs=docs, chunk_count=chunk_count
    )
    mock_db = MagicMock()
    with (
        patch.object(ingest_module, "DBManager") as mock_db_cls,
        patch.object(ingest_module, "VectorStore", return_value=mock_vs),
        patch.object(ingest_module, "QueryCache", return_value=mock_cache),
        patch.object(ingest_module, "DocumentLoader", return_value=mock_loader),
        patch.object(ingest_module, "TextChunker", return_value=mock_chunker),
        patch.object(ingest_module, "TextEmbedder", return_value=mock_embedder),
        patch.object(ingest_module, "configure_logging"),
        patch.object(ingest_module, "settings"),
    ):
        mock_db_cls.get_instance.return_value = mock_db
        ingest_module.main(force=force)
    return mock_vs, mock_cache


def test_skips_already_ingested_doc_when_no_force():
    mock_vs, _ = _run_main(force=False, has_doc=True)
    mock_vs.insert.assert_not_called()


def test_force_clears_cache_and_deletes_doc_before_reingest():
    mock_vs, mock_cache = _run_main(force=True, has_doc=True, chunk_count=2)
    mock_cache.clear_all.assert_called_once()
    mock_vs.delete_by_doc.assert_called_with("policy")
    assert mock_vs.insert.call_count == 2


def test_inserts_chunks_when_not_already_ingested():
    mock_vs, _ = _run_main(force=False, has_doc=False, chunk_count=3)
    assert mock_vs.insert.call_count == 3


def test_empty_docs_list_does_not_insert():
    mock_vs, _ = _run_main(force=False, docs=[])
    mock_vs.insert.assert_not_called()
