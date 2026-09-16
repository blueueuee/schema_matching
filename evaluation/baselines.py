"""
Baseline matchers for comparative schema-matching evaluation.
Implements:
1. baseline_name_only: pure semantic name embedding cosine similarity.
2. baseline_value_only: pure value distribution profiling similarity (alpha=0.0).
3. baseline_fixed_fusion: static fixed-weight fusion (alpha=0.5).
"""

from typing import List
import numpy as np

from matching.matcher import match_schemas
from value_pipeline.similarity import value_similarity


def compute_name_similarity(profile_a: dict, profile_b: dict) -> float:
    """
    Computes cosine similarity between name_embedding vectors of two column profiles.
    Falls back to token Jaccard similarity when embeddings are empty list [].
    """
    emb_a = np.array(profile_a.get("name_embedding", []), dtype=np.float64)
    emb_b = np.array(profile_b.get("name_embedding", []), dtype=np.float64)

    if emb_a.size > 0 and emb_b.size > 0 and emb_a.shape == emb_b.shape:
        denom = float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        if denom > 0:
            cos = float(np.dot(emb_a, emb_b) / denom)
            return float(np.clip(cos, 0.0, 1.0))

    # Token overlap fallback for mocked [] embeddings
    tokens_a = set(profile_a.get("normalized_tokens") or [])
    tokens_b = set(profile_b.get("normalized_tokens") or [])

    if not tokens_a and profile_a.get("raw_name"):
        tokens_a = set(str(profile_a["raw_name"]).lower().replace("-", "_").split("_"))
    if not tokens_b and profile_b.get("raw_name"):
        tokens_b = set(str(profile_b["raw_name"]).lower().replace("-", "_").split("_"))

    if tokens_a and tokens_b:
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return float(len(intersection) / len(union)) if union else 0.0

    return 0.0


def baseline_name_only(
    profiles_a: List[dict],
    profiles_b: List[dict],
    threshold: float = 0.3
) -> List[dict]:
    """
    Name-only baseline: Uses only cosine similarity on name_embedding (alpha=1.0).
    """
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=1.0,
        threshold=threshold
    )


def baseline_value_only(
    profiles_a: List[dict],
    profiles_b: List[dict],
    threshold: float = 0.3
) -> List[dict]:
    """
    Value-only baseline: Uses only value_similarity (alpha=0.0).
    """
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=0.0,
        threshold=threshold
    )


def baseline_fixed_fusion(
    profiles_a: List[dict],
    profiles_b: List[dict],
    alpha: float = 0.5,
    threshold: float = 0.3
) -> List[dict]:
    """
    Fixed fusion baseline: Uses static alpha (default 0.5) combining name and value similarity.
    """
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=alpha,
        threshold=threshold
    )
