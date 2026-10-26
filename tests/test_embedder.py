"""Unit tests for person_a.name_embedder."""

import numpy as np
import pytest

from person_a.name_embedder import get_name_embedder, FallbackTfidfEmbedder


def test_embedder_dimension_and_shape():
    embedder = get_name_embedder()
    tokens = ["customer", "identifier"]
    vec = embedder.embed_tokens(tokens)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_embedder_l2_normalization():
    embedder = get_name_embedder()
    vec = embedder.embed_tokens(["order", "amount"])
    norm = float(np.linalg.norm(vec))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embedder_batch_shape():
    embedder = get_name_embedder()
    batch = [
        ["customer", "id"],
        ["order", "date"],
        ["total", "amount"],
        ["phone", "number"],
    ]
    matrix = embedder.embed_batch(batch)
    assert matrix.shape == (4, 384)
    for row in matrix:
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-5)


def test_embedder_cache_consistency():
    embedder = get_name_embedder()
    tokens = ["employee", "department"]
    vec1 = embedder.embed_tokens(tokens)
    vec2 = embedder.embed_tokens(tokens)
    assert np.array_equal(vec1, vec2)


def test_fallback_tfidf_embedder():
    fallback = FallbackTfidfEmbedder(dim=384)
    texts = ["customer identifier", "order quantity"]
    matrix = fallback.encode(texts)

    assert matrix.shape == (2, 384)
    for row in matrix:
        norm = float(np.linalg.norm(row))
        assert norm == pytest.approx(1.0, abs=1e-5)
