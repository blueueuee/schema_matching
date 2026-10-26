"""Value similarity metrics for comparing column value profiles.

Calculates multi-faceted statistical similarity between column profiles:
1. Weighted Euclidean distance across the 22-dimensional profile vector.
2. Jensen-Shannon Divergence (JSD) for numeric quantiles and mean.
3. Jaccard value overlap for low-cardinality discrete sets.
4. Range overlap for bounded numeric features.
5. Explicit guards for all-null columns, constant columns, and dtype mismatches.
"""

from typing import Any, Dict, List, Optional, Set
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax

from value_pipeline.profiler import profile_to_vector

# Dimension weights corresponding to 22-dimensional profile vector:
# - Dims 0..4: dtype one-hot (weight 2.0 - strong structural constraint)
# - Dims 5..17: ratios, length stats, numeric stats (weight 1.0)
# - Dims 18..21: regex pattern flags (weight 1.5 - strong formatting signature)
FEATURE_WEIGHTS = np.array(
    [2.0] * 5 +    # 0..4: dtype one-hot
    [1.0] * 13 +   # 5..17: ratios & statistics
    [1.5] * 4,     # 18..21: pattern flags
    dtype=np.float64
)


def _get_inner_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Extract nested value_profile dict if wrapped inside ColumnProfile."""
    if isinstance(profile, dict) and "value_profile" in profile and isinstance(profile["value_profile"], dict):
        return profile["value_profile"]
    return profile


def weighted_euclidean_distance(profile_a: dict, profile_b: dict) -> float:
    """Computes weighted Euclidean distance between two 22-dimensional profile vectors.

    dist = sqrt( sum( w_i * (v_a[i] - v_b[i])^2 ) )
    """
    p_a = _get_inner_profile(profile_a)
    p_b = _get_inner_profile(profile_b)

    vec_a = np.array(profile_to_vector(p_a), dtype=np.float64)
    vec_b = np.array(profile_to_vector(p_b), dtype=np.float64)

    diff_sq = (vec_a - vec_b) ** 2
    weighted_diff = FEATURE_WEIGHTS * diff_sq
    return float(np.sqrt(np.sum(weighted_diff)))


def compute_discrete_overlap(profile_a: dict, profile_b: dict) -> Optional[float]:
    """Compute Jaccard value overlap if sample values are available for low-cardinality columns."""
    samples_a = profile_a.get("sample_values") or profile_a.get("value_sample")
    samples_b = profile_b.get("sample_values") or profile_b.get("value_sample")

    if not samples_a or not samples_b:
        return None

    set_a: Set[str] = {str(x).strip().lower() for x in samples_a if str(x).strip()}
    set_b: Set[str] = {str(x).strip().lower() for x in samples_b if str(x).strip()}

    if not set_a or not set_b:
        return None

    union = set_a | set_b
    if not union:
        return None
    return float(len(set_a & set_b) / len(union))


def compute_numeric_range_overlap(stats_a: Dict[str, Any], stats_b: Dict[str, Any]) -> float:
    """Calculate the overlap ratio between numeric bounding ranges [min, max]."""
    try:
        min_a, max_a = float(stats_a["min"]), float(stats_a["max"])
        min_b, max_b = float(stats_b["min"]), float(stats_b["max"])
    except (KeyError, ValueError, TypeError):
        return 0.0

    inter_min = max(min_a, min_b)
    inter_max = min(max_a, max_b)
    if inter_max < inter_min:
        return 0.0

    overlap = inter_max - inter_min
    span = max(max_a, max_b) - min(min_a, min_b)
    return float(overlap / span) if span > 0 else 1.0


def value_similarity(profile_a: dict, profile_b: dict) -> float:
    """Returns a bounded similarity score in [0.0, 1.0] between two value profiles.

    Terms evaluated:
    - Base: inverse weighted Euclidean distance 1 / (1 + distance).
    - Null check: if either is entirely null (null_ratio >= 0.99), value similarity is 0.0.
    - Low-cardinality discrete overlap: applies Jaccard boost when distinct values match.
    - Numeric range consistency: refines score when numeric distributions overlap.
    """
    p_a = _get_inner_profile(profile_a)
    p_b = _get_inner_profile(profile_b)

    null_a = float(p_a.get("null_ratio", 0.0) or 0.0)
    null_b = float(p_b.get("null_ratio", 0.0) or 0.0)

    # All-null columns provide no discriminative distribution signal
    if null_a >= 0.99 or null_b >= 0.99:
        if null_a >= 0.99 and null_b >= 0.99:
            return 0.0
        return 0.0

    dist = weighted_euclidean_distance(p_a, p_b)
    base_sim = 1.0 / (1.0 + dist)

    dtype_a = p_a.get("dtype")
    dtype_b = p_b.get("dtype")

    # If data types mismatch, weighted distance already penalizes; ensure score stays low
    if dtype_a != dtype_b and dtype_a != "string" and dtype_b != "string":
        return float(np.clip(base_sim, 0.0, 0.35))

    sim = base_sim

    # Low-cardinality Jaccard enrichment if samples present
    card_a = float(p_a.get("cardinality_ratio", 1.0) or 1.0)
    card_b = float(p_b.get("cardinality_ratio", 1.0) or 1.0)
    if card_a <= 0.30 and card_b <= 0.30:
        jaccard = compute_discrete_overlap(profile_a, profile_b)
        if jaccard is not None:
            sim = 0.75 * sim + 0.25 * jaccard

    # Numeric range check
    if dtype_a == "numeric" and dtype_b == "numeric":
        stats_a = p_a.get("numeric_stats")
        stats_b = p_b.get("numeric_stats")
        if isinstance(stats_a, dict) and isinstance(stats_b, dict):
            range_overlap = compute_numeric_range_overlap(stats_a, stats_b)
            sim = 0.85 * sim + 0.15 * range_overlap

    return float(np.clip(sim, 0.0, 1.0))


def value_similarity_jsd(profile_a: dict, profile_b: dict) -> float:
    """Jensen-Shannon divergence similarity for numeric columns.

    Approximates probability mass across distribution quantiles [q25, q50, q75, mean].
    Falls back to value_similarity for non-numeric types or missing quantiles.
    """
    p_a = _get_inner_profile(profile_a)
    p_b = _get_inner_profile(profile_b)

    dtype_a = p_a.get("dtype")
    dtype_b = p_b.get("dtype")

    if dtype_a == "numeric" and dtype_b == "numeric":
        stats_a = p_a.get("numeric_stats")
        stats_b = p_b.get("numeric_stats")

        if isinstance(stats_a, dict) and isinstance(stats_b, dict):
            keys = ["q25", "q50", "q75", "mean"]
            vals_a = np.array([float(stats_a.get(k, 0.0) or 0.0) for k in keys], dtype=np.float64)
            vals_b = np.array([float(stats_b.get(k, 0.0) or 0.0) for k in keys], dtype=np.float64)

            p = softmax(vals_a)
            q = softmax(vals_b)

            js_dist = jensenshannon(p, q, base=2.0)
            if not np.isnan(js_dist):
                sim = max(0.0, min(1.0, 1.0 - float(js_dist)))
                return float(sim)

    return value_similarity(profile_a, profile_b)
