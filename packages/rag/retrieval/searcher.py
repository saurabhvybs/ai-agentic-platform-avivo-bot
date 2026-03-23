import asyncio

from rag.ingestion.embedder import TextEmbedder
from storage.vector_store import VectorStore


class RAGSearcher:
    def __init__(
        self, embedder: TextEmbedder, vector_store: VectorStore, top_k: int
    ) -> None:
        # top_k injected from settings.TOP_K
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k

    async def retrieve(self, query: str) -> list[tuple[str, str]]:
        """Embed query and search vector store. asyncio.to_thread wrapping lives here."""
        vector = await asyncio.to_thread(self._embedder.embed, query)
        results = await asyncio.to_thread(self._vector_store.search, vector, self._top_k)
        return results
