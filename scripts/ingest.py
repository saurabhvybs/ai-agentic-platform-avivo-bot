#!/usr/bin/env python3
"""One-time ingestion script: chunk + embed + store all knowledge base docs.

Usage:
    python scripts/ingest.py               # Skip already-ingested docs
    python scripts/ingest.py --force       # Re-ingest all docs, clear cache
"""
import sys
from pathlib import Path

# Add packages/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from shared.config import settings
from shared.logger import configure_logging, logger
from storage.db import DBManager
from storage.vector_store import VectorStore
from storage.cache import QueryCache
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.chunker import TextChunker
from rag.ingestion.embedder import TextEmbedder


def main(force: bool = False) -> None:
    configure_logging(settings.LOG_LEVEL)

    db = DBManager.get_instance(settings.DB_PATH)
    vector_store = VectorStore(db)
    query_cache = QueryCache(db, settings.CACHE_TTL_HOURS)
    loader = DocumentLoader()
    chunker = TextChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    embedder = TextEmbedder(settings.EMBEDDING_MODEL)

    kb_dir = Path(__file__).parent.parent / "data" / "knowledge_base"
    docs = loader.load(kb_dir)

    if not docs:
        logger.warning("No .md files found in data/knowledge_base/")
        return

    total_chunks = 0
    ingested_docs = 0

    if force:
        logger.info("--force: clearing query cache before re-ingestion")
        query_cache.clear_all()

    for doc_name, content in docs:
        if not force and vector_store.has_doc(doc_name):
            logger.info(f"Skipping {doc_name} (already ingested). Use --force to re-ingest.")
            continue

        # NOTE: delete + insert is not atomic. A failure mid-insert leaves this doc
        # partially or fully absent from the vector store. Use --force to recover.
        if force:
            vector_store.delete_by_doc(doc_name)

        chunks = chunker.chunk(content)
        embeddings = embedder.embed_batch(chunks)

        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            vector_store.insert(doc_name, idx, chunk_text, embedding)

        logger.info(f"Ingested {doc_name}: {len(chunks)} chunks")
        total_chunks += len(chunks)
        ingested_docs += 1

    print(f"\nIngested {ingested_docs} docs, {total_chunks} chunks total.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
