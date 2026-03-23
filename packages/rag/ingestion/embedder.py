import threading

from sentence_transformers import SentenceTransformer

# Lazy module-level singleton — loaded on first call, not at import time.
# This avoids slow imports during test runs that don't use embeddings.
# NOTE: Singleton is keyed on the first model_name used. All callers are
# expected to use the same model (settings.EMBEDDING_MODEL). A different
# model_name on a later call returns the already-loaded model silently.
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                _model = SentenceTransformer(model_name)
    return _model


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name

    def embed(self, text: str) -> list[float]:
        return _get_model(self._model_name).encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return _get_model(self._model_name).encode(texts, batch_size=32).tolist()
