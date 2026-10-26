"""Schema Matcher using Hungarian algorithm (maximum weight bipartite matching).

Blends semantic name similarity and value profiling similarity into a global
similarity matrix, then finds optimal one-to-one column correspondence.
Supports:
1. Adaptive per-pair fusion (method="adaptive")
2. Static fixed-weight fusion (method="fixed", alpha=0.5)
3. Name-only matching (method="name_only", alpha=1.0)
4. Value-only matching (method="value_only", alpha=0.0)
5. Threshold gating for unmatched columns
6. Rectangular matrix support (|A| != |B|)
7. Explainability metadata on every match record
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from shared.contracts import Match, MatchExplanation
from value_pipeline.similarity import value_similarity
from person_a.name_similarity import compute_cosine_similarity
from person_a.fusion import compute_fused_similarity_matrix, FusedMatchScore


def _get_column_identifier(profile: Dict[str, Any], fallback_idx: int) -> str:
    """Extract unique column ID or raw name from profile."""
    return profile.get("column_id") or profile.get("raw_name") or f"col_{fallback_idx}"


def _get_value_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Extract value_profile dict from ColumnProfile schema if nested."""
    return profile.get("value_profile", profile)


def _ensure_name_embedding(profile: Dict[str, Any]) -> List[float]:
    """Ensure column profile contains a name embedding, generating on the fly if needed."""
    emb = profile.get("name_embedding")
    if isinstance(emb, (list, np.ndarray)) and len(emb) > 0:
        return [float(x) for x in emb]

    # Lazy generation via person_a pipeline
    from person_a.normalize_names import normalize_column_name
    from person_a.name_embedder import get_name_embedder

    tokens = profile.get("normalized_tokens")
    if not tokens:
        raw_name = profile.get("raw_name") or profile.get("column_id") or ""
        tokens = normalize_column_name(raw_name)

    embedder = get_name_embedder()
    vec = embedder.embed_tokens(tokens)
    vec_list = [float(x) for x in vec]
    profile["name_embedding"] = vec_list
    profile["normalized_tokens"] = tokens
    return vec_list


def build_similarity_matrix(
    profiles_a: List[Dict[str, Any]],
    profiles_b: List[Dict[str, Any]],
    name_sim_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
    value_sim_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
    alpha: float = 0.5,
    method: str = "fixed",
) -> np.ndarray:
    """Build an (|A| x |B|) matrix where entry [i][j] = similarity(a_i, b_j) in [0, 1]."""
    n_a = len(profiles_a)
    n_b = len(profiles_b)

    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float64)

    v_fn = value_sim_fn if value_sim_fn is not None else value_similarity

    # If adaptive method requested, delegate to person_a.fusion
    if method == "adaptive":
        name_matrix = np.zeros((n_a, n_b), dtype=np.float32)
        value_matrix = np.zeros((n_a, n_b), dtype=np.float32)

        for i, p_a in enumerate(profiles_a):
            emb_a = _ensure_name_embedding(p_a)
            vp_a = _get_value_profile(p_a)
            for j, p_b in enumerate(profiles_b):
                emb_b = _ensure_name_embedding(p_b)
                vp_b = _get_value_profile(p_b)

                if name_sim_fn is not None:
                    name_matrix[i, j] = float(name_sim_fn(p_a, p_b))
                else:
                    name_matrix[i, j] = compute_cosine_similarity(emb_a, emb_b)

                value_matrix[i, j] = float(v_fn(vp_a, vp_b))

        fused_matrix, _, _ = compute_fused_similarity_matrix(
            name_matrix=name_matrix,
            value_matrix=value_matrix,
            source_profiles=profiles_a,
            target_profiles=profiles_b,
        )
        return fused_matrix.astype(np.float64)

    # Fixed or single-signal methods
    matrix = np.zeros((n_a, n_b), dtype=np.float64)
    for i, p_a in enumerate(profiles_a):
        vp_a = _get_value_profile(p_a)
        emb_a = _ensure_name_embedding(p_a) if alpha > 0.0 else []
        for j, p_b in enumerate(profiles_b):
            vp_b = _get_value_profile(p_b)
            emb_b = _ensure_name_embedding(p_b) if alpha > 0.0 else []

            if alpha > 0.0:
                if name_sim_fn is not None:
                    n_sim = float(name_sim_fn(p_a, p_b))
                else:
                    n_sim = compute_cosine_similarity(emb_a, emb_b)
            else:
                n_sim = 0.0

            if alpha < 1.0:
                v_sim = float(v_fn(vp_a, vp_b))
            else:
                v_sim = 0.0

            score = alpha * n_sim + (1.0 - alpha) * v_sim
            matrix[i, j] = float(np.clip(score, 0.0, 1.0))

    return matrix


def match_schemas(
    profiles_a: List[Dict[str, Any]],
    profiles_b: List[Dict[str, Any]],
    name_sim_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
    value_sim_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
    alpha: float = 0.5,
    threshold: float = 0.3,
    method: str = "fixed",
) -> List[Match]:
    """Run Hungarian algorithm on similarity matrix to obtain optimal 1-to-1 matches."""
    if not profiles_a or not profiles_b:
        return []

    n_a = len(profiles_a)
    n_b = len(profiles_b)
    v_fn = value_sim_fn if value_sim_fn is not None else value_similarity

    # Compute name and value components
    name_matrix = np.zeros((n_a, n_b), dtype=np.float32)
    value_matrix = np.zeros((n_a, n_b), dtype=np.float32)

    for i, p_a in enumerate(profiles_a):
        emb_a = _ensure_name_embedding(p_a)
        vp_a = _get_value_profile(p_a)
        for j, p_b in enumerate(profiles_b):
            emb_b = _ensure_name_embedding(p_b)
            vp_b = _get_value_profile(p_b)

            if name_sim_fn is not None:
                name_matrix[i, j] = float(name_sim_fn(p_a, p_b))
            else:
                name_matrix[i, j] = compute_cosine_similarity(emb_a, emb_b)

            value_matrix[i, j] = float(v_fn(vp_a, vp_b))

    explanations_grid: Optional[List[List[FusedMatchScore]]] = None
    alpha_matrix = np.full((n_a, n_b), alpha, dtype=np.float32)

    if method == "adaptive":
        sim_matrix, alpha_matrix, explanations_grid = compute_fused_similarity_matrix(
            name_matrix=name_matrix,
            value_matrix=value_matrix,
            source_profiles=profiles_a,
            target_profiles=profiles_b,
        )
    elif method == "name_only":
        sim_matrix = name_matrix.copy()
        alpha_matrix.fill(1.0)
    elif method == "value_only":
        sim_matrix = value_matrix.copy()
        alpha_matrix.fill(0.0)
    else:  # fixed
        sim_matrix = alpha * name_matrix + (1.0 - alpha) * value_matrix

    # Hungarian algorithm: Scipy minimizes cost, so negate similarity
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[Match] = []
    for r, c in zip(row_ind, col_ind):
        score = float(sim_matrix[r, c])
        col_a_id = _get_column_identifier(profiles_a[r], r)
        col_b_id = _get_column_identifier(profiles_b[c], c)
        n_sim = float(name_matrix[r, c])
        v_sim = float(value_matrix[r, c])
        pair_alpha = float(alpha_matrix[r, c])

        if explanations_grid is not None:
            fused_score_obj = explanations_grid[r][c]
            explanation = MatchExplanation(
                source_column=col_a_id,
                target_column=col_b_id,
                name_similarity=n_sim,
                value_similarity=v_sim,
                alpha=pair_alpha,
                fused_score=score,
                reasons=fused_score_obj.reasons,
                source_dtype=fused_score_obj.source_dtype,
                target_dtype=fused_score_obj.target_dtype,
            )
        else:
            explanation = MatchExplanation(
                source_column=col_a_id,
                target_column=col_b_id,
                name_similarity=n_sim,
                value_similarity=v_sim,
                alpha=pair_alpha,
                fused_score=score,
                reasons=[f"Static method={method}"],
            )

        match = Match(
            column_a=col_a_id,
            column_b=col_b_id,
            score=round(score, 4),
            accepted=bool(score >= threshold),
            name_similarity=round(n_sim, 4),
            value_similarity=round(v_sim, 4),
            alpha=round(pair_alpha, 4),
            explanation=explanation,
        )
        matches.append(match)

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches
