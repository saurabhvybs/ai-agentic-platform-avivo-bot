#!/usr/bin/env python3
"""CLI loop to test the RAG pipeline without running the Telegram bot.

Usage:
    python scripts/test_rag_cli.py

Reads a query from stdin, runs the full RAG pipeline via asyncio.run(),
prints the answer + sources. Exit with empty input or Ctrl+C.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from shared.config import settings
from shared.logger import configure_logging
from storage.db import DBManager
from storage.vector_store import VectorStore
from rag.ingestion.embedder import TextEmbedder
from rag.retrieval.searcher import RAGSearcher
from rag.generation.generator import RAGGenerator


async def run_session() -> None:
    db = DBManager.get_instance(settings.DB_PATH)
    vector_store = VectorStore(db)
    embedder = TextEmbedder(settings.EMBEDDING_MODEL)
    searcher = RAGSearcher(embedder, vector_store, settings.TOP_K)
    generator = RAGGenerator(settings)

    try:
        while True:
            try:
                query = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                break

            chunks = await searcher.retrieve(query)
            result = await generator.generate(query, chunks, history=[])

            print(f"\n📄 Answer:\n{result.answer}")
            if result.sources:
                print(f"\n📚 Sources: {', '.join(result.sources)}")
    finally:
        await generator.close()


def main() -> None:
    configure_logging("WARNING")  # Suppress info logs in CLI
    print("Avivo Bot — RAG CLI Test (Ctrl+C or empty input to exit)\n")
    asyncio.run(run_session())


if __name__ == "__main__":
    main()
