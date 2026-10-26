"""Unit tests for person_a.name_similarity."""

import numpy as np
import pytest

from person_a.name_similarity import (
    compute_cosine_similarity,
    compute_name_similarity_matrix,
    format_similarity_matrix,
)


def test_cosine_similarity_identical_and_orthogonal():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    assert compute_cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-5)
    assert compute_cosine_similarity(a, c) == pytest.approx(0.0, abs=1e-5)
    # Negative cosine is clipped to 0.0
    assert compute_cosine_similarity(a, d) == pytest.approx(0.0, abs=1e-5)


def test_matrix_dimensions_and_bounds():
    src_embs = {
        "c1": np.array([1.0, 0.5, 0.0], dtype=np.float32),
        "c2": np.array([0.0, 1.0, 0.2], dtype=np.float32),
        "c3": np.array([0.5, 0.5, 0.5], dtype=np.float32),
    }
    tgt_embs = {
        "t1": np.array([1.0, 0.5, 0.0], dtype=np.float32),
        "t2": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    matrix = compute_name_similarity_matrix(src_embs, tgt_embs)
    assert matrix.shape == (3, 2)
    assert (matrix >= 0.0).all() and (matrix <= 1.0).all()
    # c1 and t1 are identical
    assert matrix[0, 0] == pytest.approx(1.0, abs=1e-4)


def test_format_similarity_matrix():
    mat = np.array([[0.95, 0.12], [0.08, 0.88]], dtype=np.float32)
    formatted = format_similarity_matrix(mat, ["src_a", "src_b"], ["tgt_1", "tgt_2"])
    assert "src_a" in formatted
    assert "tgt_1" in formatted
    assert "0.950" in formatted
