"""Baseline matchers for comparative schema-matching evaluation.

Implements:
1. baseline_name_only: pure semantic name embedding cosine similarity (alpha=1.0).
2. baseline_value_only: pure value distribution profiling similarity (alpha=0.0).
3. baseline_fixed_fusion: static fixed-weight fusion (alpha=0.5).
4. baseline_adaptive_fusion: adaptive per-pair alpha fusion.
"""

from typing import List, Union
import numpy as np

from matching.matcher import match_schemas
from value_pipeline.similarity import value_similarity
from person_a.name_similarity import compute_cosine_similarity
from person_a.normalize_names import normalize_column_name
from person_a.name_embedder import get_name_embedder


def compute_name_similarity(profile_a: dict, profile_b: dict) -> float:
    """Computes cosine similarity between name_embedding vectors of two column profiles.

    If embeddings are empty, generates them via person_a embedding model.
    Falls back to token Jaccard if embedder is unavailable.
    """
    emb_a = profile_a.get("name_embedding", [])
    emb_b = profile_b.get("name_embedding", [])

    if not emb_a or not emb_b:
        tokens_a = profile_a.get("normalized_tokens") or normalize_column_name(profile_a.get("raw_name", ""))
        tokens_b = profile_b.get("normalized_tokens") or normalize_column_name(profile_b.get("raw_name", ""))
        embedder = get_name_embedder()
        emb_a = embedder.embed_tokens(tokens_a)
        emb_b = embedder.embed_tokens(tokens_b)

    return compute_cosine_similarity(emb_a, emb_b)


def baseline_name_only(
    profiles_a: List[dict],
    profiles_b: List[dict],
    threshold: float = 0.3,
) -> List[dict]:
    """Name-only baseline: Uses only cosine similarity on name_embedding (alpha=1.0)."""
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=1.0,
        threshold=threshold,
        method="name_only",
    )


def baseline_value_only(
    profiles_a: List[dict],
    profiles_b: List[dict],
    threshold: float = 0.3,
) -> List[dict]:
    """Value-only baseline: Uses only value_similarity (alpha=0.0)."""
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=0.0,
        threshold=threshold,
        method="value_only",
    )


def baseline_fixed_fusion(
    profiles_a: List[dict],
    profiles_b: List[dict],
    alpha: float = 0.5,
    threshold: float = 0.3,
) -> List[dict]:
    """Fixed fusion baseline: Uses static alpha (default 0.5) combining name and value similarity."""
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=alpha,
        threshold=threshold,
        method="fixed",
    )


def baseline_adaptive_fusion(
    profiles_a: List[dict],
    profiles_b: List[dict],
    threshold: float = 0.3,
) -> List[dict]:
    """Adaptive fusion method: Uses per-column-pair adaptive alpha based on signal reliability."""
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        threshold=threshold,
        method="adaptive",
    )
