# tests/test_bot/test_integration.py
"""End-to-end integration test: ingest docs → embed → search → generate answer.

Uses real sqlite-vec (no mocks). Requires OPENAI_API_KEY in environment.
Skipped automatically if OPENAI_API_KEY is not set.
"""
import asyncio
import os
import pytest
from pathlib import Path

from storage.db import DBManager
from storage.vector_store import VectorStore
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.chunker import TextChunker
from rag.ingestion.embedder import TextEmbedder
from rag.retrieval.searcher import RAGSearcher
from rag.generation.generator import RAGGenerator
from shared.config import settings


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "placeholder",
    reason="OPENAI_API_KEY not set — skipping integration test",
)


@pytest.fixture
def integration_db(tmp_path):
    DBManager.reset()
    instance = DBManager(str(tmp_path / "integration.db"))
    yield instance
    DBManager.reset()


async def test_full_rag_pipeline(integration_db, tmp_path):
    """Ingest a small doc, retrieve chunks, generate answer — real sqlite-vec."""
    # Write a tiny knowledge base
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "leave_policy.md").write_text(
        "Annual leave policy: all employees get 20 days of paid leave per year."
    )

    # Ingest
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=200, overlap=30)
    embedder = TextEmbedder()
    vector_store = VectorStore(integration_db)

    docs = loader.load(kb_dir)
    for doc_name, content in docs:
        chunks = chunker.chunk(content)
        embeddings = embedder.embed_batch(chunks)
        for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
            vector_store.insert(doc_name, idx, text, emb)

    # Retrieve
    searcher = RAGSearcher(embedder, vector_store, top_k=2)
    results = await searcher.retrieve("how many leave days do I get?")
    assert len(results) > 0
    assert any("20 days" in text for text, _doc in results)

    # Generate
    generator = RAGGenerator(settings)
    rag_result = await generator.generate(
        query="How many leave days do I get?",
        retrieved_chunks=results,
        history=[],
    )
    assert "20" in rag_result.answer
    assert rag_result.from_cache is False
    await generator.close()
