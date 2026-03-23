import numpy as np

import rag.ingestion.embedder as embedder_module
from rag.ingestion.embedder import TextEmbedder


def test_embed_returns_384_dim_vector():
    embedder = TextEmbedder()
    vec = embedder.embed("Hello world")
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


def test_embed_batch_returns_correct_shape():
    embedder = TextEmbedder()
    vecs = embedder.embed_batch(["Hello", "World", "Test"])
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


def test_singleton_not_reloaded():
    """Model object should be the same Python object on repeated calls."""
    embedder_module._model = None  # Reset for clean test
    embedder = TextEmbedder()
    embedder.embed("first call")
    model_1 = embedder_module._model
    embedder.embed("second call")
    model_2 = embedder_module._model
    assert model_1 is model_2


def test_similar_texts_have_closer_vectors():
    embedder = TextEmbedder()

    v1 = np.array(embedder.embed("leave policy annual days"))
    v2 = np.array(embedder.embed("vacation days per year"))
    v3 = np.array(embedder.embed("python programming language"))

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cosine(v1, v2) > cosine(v1, v3)
