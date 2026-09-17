"""
Stage 5: Schema Matcher using Hungarian algorithm (maximum weight bipartite matching).
Blends semantic name similarity and value similarity into a similarity matrix,
then finds the globally optimal one-to-one column correspondence.
"""

from typing import Callable, Dict, List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from value_pipeline.similarity import value_similarity


def _get_column_identifier(profile: dict, fallback_idx: int) -> str:
    """Extract unique column ID or raw name from profile."""
    return profile.get("column_id") or profile.get("raw_name") or f"col_{fallback_idx}"


def _get_value_profile(profile: dict) -> dict:
    """Extract value_profile dict from ColumnProfile schema if nested."""
    return profile.get("value_profile", profile)


def build_similarity_matrix(
    profiles_a: List[dict],
    profiles_b: List[dict],
    name_sim_fn: Optional[Callable[[dict, dict], float]] = None,
    value_sim_fn: Optional[Callable[[dict, dict], float]] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Build an (|A| x |B|) matrix where entry [i][j] =
        alpha * name_sim_fn(a_i, b_j) + (1 - alpha) * value_sim_fn(a_i, b_j)

    Parameters
    ----------
    profiles_a : List[dict]
        Source schema column profiles.
    profiles_b : List[dict]
        Target schema column profiles.
    name_sim_fn : Callable[[dict, dict], float], optional
        Function returning semantic name similarity in [0, 1].
    value_sim_fn : Callable[[dict, dict], float], optional
        Function returning value distribution similarity in [0, 1].
    alpha : float, default=0.5
        Weight for name similarity (1 - alpha for value similarity).

    Returns
    -------
    np.ndarray
        Matrix of size (len(profiles_a), len(profiles_b)) with similarity scores in [0, 1].
    """
    n_a = len(profiles_a)
    n_b = len(profiles_b)

    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float64)

    v_fn = value_sim_fn if value_sim_fn is not None else value_similarity
    matrix = np.zeros((n_a, n_b), dtype=np.float64)

    for i, p_a in enumerate(profiles_a):
        vp_a = _get_value_profile(p_a)
        for j, p_b in enumerate(profiles_b):
            vp_b = _get_value_profile(p_b)

            # Compute name similarity if weighted
            if alpha > 0.0 and name_sim_fn is not None:
                n_sim = float(name_sim_fn(p_a, p_b))
            else:
                n_sim = 0.0

            # Compute value similarity if weighted
            if alpha < 1.0:
                v_sim = float(v_fn(vp_a, vp_b))
            else:
                v_sim = 0.0

            score = alpha * n_sim + (1.0 - alpha) * v_sim
            matrix[i, j] = float(np.clip(score, 0.0, 1.0))

    return matrix


def match_schemas(
    profiles_a: List[dict],
    profiles_b: List[dict],
    name_sim_fn: Optional[Callable[[dict, dict], float]] = None,
    value_sim_fn: Optional[Callable[[dict, dict], float]] = None,
    alpha: float = 0.5,
    threshold: float = 0.3
) -> List[dict]:
    """
    Run Hungarian algorithm on the similarity matrix to obtain optimal
    one-to-one column matches.

    Parameters
    ----------
    profiles_a : List[dict]
        Source schema column profiles.
    profiles_b : List[dict]
        Target schema column profiles.
    name_sim_fn : Callable[[dict, dict], float], optional
        Semantic name similarity function.
    value_sim_fn : Callable[[dict, dict], float], optional
        Value similarity function.
    alpha : float, default=0.5
        Fusion weight between name and value signals.
    threshold : float, default=0.3
        Confidence threshold for accepted match.

    Returns
    -------
    List[dict]
        Sorted list of match records:
        [
          {
            "column_a": "DB1.Customers.cust_id",
            "column_b": "DB2.Clients.client_no",
            "score": 0.87,
            "accepted": True
          },
          ...
        ]
    """
    if not profiles_a or not profiles_b:
        return []

    sim_matrix = build_similarity_matrix(
        profiles_a,
        profiles_b,
        name_sim_fn=name_sim_fn,
        value_sim_fn=value_sim_fn,
        alpha=alpha
    )

    # Scipy minimizes cost, so invert similarities
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        score = float(sim_matrix[r, c])
        col_a_id = _get_column_identifier(profiles_a[r], r)
        col_b_id = _get_column_identifier(profiles_b[c], c)

        matches.append({
            "column_a": col_a_id,
            "column_b": col_b_id,
            "score": round(score, 4),
            "accepted": bool(score >= threshold)
        })

    # Sort matches by score descending
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches
