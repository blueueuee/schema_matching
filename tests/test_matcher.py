"""Unit tests for matching.matcher."""

import pytest
import numpy as np

from matching.matcher import build_similarity_matrix, match_schemas
from evaluation.benchmark import load_synthetic_pairs
from evaluation.baselines import compute_name_similarity
from value_pipeline.similarity import value_similarity


def test_build_similarity_matrix():
    pairs = load_synthetic_pairs()
    assert len(pairs) > 0, "Synthetic pairs must be generated"

    pair_1 = pairs[0]
    profiles_a = pair_1["profiles_a"]
    profiles_b = pair_1["profiles_b"]

    matrix = build_similarity_matrix(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=0.5
    )

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(profiles_a), len(profiles_b))
    assert (matrix >= 0.0).all() and (matrix <= 1.0).all()


def test_match_schemas_synthetic_pair_1():
    pairs = load_synthetic_pairs()
    pair_1 = pairs[0]
    assert pair_1["pair_id"] == "pair_1"

    matches = match_schemas(
        pair_1["profiles_a"],
        pair_1["profiles_b"],
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=0.5,
        threshold=0.3
    )

    # In synthetic pair 1 (exact match), all 4 columns should match with score > 0.8
    assert len(matches) == 4
    for m in matches:
        assert m["accepted"] is True
        assert m["score"] > 0.8
        # Source ID and target ID should have matching suffix
        col_a_suffix = m["column_a"].split(".")[-1]
        col_b_suffix = m["column_b"].split(".")[-1]
        assert col_a_suffix == col_b_suffix


def test_match_schemas_empty_input():
    matches = match_schemas([], [])
    assert matches == []
