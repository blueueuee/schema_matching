"""Core service layer for schema matching.

Provides the single entry point `match_tables` consumed by both CLI evaluation
and Streamlit UI without code duplication.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from shared.contracts import Match, ColumnProfile, validate_profile
from value_pipeline.sampler import sample_column
from value_pipeline.profiler import build_value_profile
from value_pipeline.similarity import value_similarity
from person_a.normalize_names import normalize_column_name
from person_a.name_embedder import get_name_embedder
from person_a.name_similarity import compute_name_similarity_matrix
from person_a.fusion import compute_fused_similarity_matrix
from matching.matcher import match_schemas


@dataclass
class MatchResult:
    matches: List[Match]
    unmatched_source: List[str]
    unmatched_target: List[str]
    source_profiles: List[Dict[str, Any]]
    target_profiles: List[Dict[str, Any]]
    similarity_matrix: np.ndarray
    name_matrix: np.ndarray
    value_matrix: np.ndarray
    alpha_matrix: np.ndarray
    source_columns: List[str]
    target_columns: List[str]
    backend_name: str
    method: str
    threshold: float


def build_column_profiles(
    df: pd.DataFrame,
    table_prefix: str = "",
    sample_size: int = 500,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Build standardized ColumnProfile dictionaries for all columns in a DataFrame."""
    embedder = get_name_embedder()
    profiles: List[Dict[str, Any]] = []

    # Batch encode normalized names for efficiency
    col_names = [str(col) for col in df.columns]
    token_lists = [normalize_column_name(col) for col in col_names]
    embeddings = embedder.embed_batch(token_lists)

    for i, col in enumerate(col_names):
        raw_series = df[col].tolist()
        total_count = len(raw_series)
        sampled_vals = sample_column(raw_series, n=sample_size, random_state=random_state)
        val_profile = build_value_profile(sampled_vals, total_count=total_count)

        col_id = f"{table_prefix}.{col}" if table_prefix else col
        emb = [float(x) for x in embeddings[i]]

        profile_dict = {
            "column_id": col_id,
            "raw_name": col,
            "normalized_tokens": token_lists[i],
            "data_type": val_profile["dtype"],
            "sample_values": sampled_vals[:20],
            "name_embedding": emb,
            "value_profile": val_profile,
        }
        profiles.append(profile_dict)

    return profiles


def match_tables(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    method: str = "adaptive",
    threshold: float = 0.35,
    sample_size: int = 500,
    alpha: float = 0.5,
) -> MatchResult:
    """Run full schema-matching pipeline between two pandas DataFrames.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source schema table.
    target_df : pd.DataFrame
        Target schema table.
    method : str, default="adaptive"
        One of: 'adaptive', 'name_only', 'value_only', 'fixed'.
    threshold : float, default=0.35
        Confidence threshold below which matches are marked unaccepted.
    sample_size : int, default=500
        Maximum sample size for value profiling.
    alpha : float, default=0.5
        Fixed alpha when method='fixed'.

    Returns
    -------
    MatchResult
        Complete matching outcomes, matrices, profiles, and unassigned columns.
    """
    if source_df is None or target_df is None:
        raise ValueError("source_df and target_df must not be None")
    if source_df.empty and len(source_df.columns) == 0:
        raise ValueError("source_df has no columns")
    if target_df.empty and len(target_df.columns) == 0:
        raise ValueError("target_df has no columns")

    embedder = get_name_embedder()
    backend_name = embedder.backend_name

    source_profiles = build_column_profiles(source_df, table_prefix="", sample_size=sample_size)
    target_profiles = build_column_profiles(target_df, table_prefix="", sample_size=sample_size)

    src_cols = [p["raw_name"] for p in source_profiles]
    tgt_cols = [p["raw_name"] for p in target_profiles]

    n_src = len(source_profiles)
    n_tgt = len(target_profiles)

    # 1. Compute name similarity matrix
    src_embs = np.array([p["name_embedding"] for p in source_profiles], dtype=np.float32)
    tgt_embs = np.array([p["name_embedding"] for p in target_profiles], dtype=np.float32)
    name_matrix = compute_name_similarity_matrix(src_embs, tgt_embs)

    # 2. Compute value similarity matrix
    value_matrix = np.zeros((n_src, n_tgt), dtype=np.float32)
    for i, p_src in enumerate(source_profiles):
        for j, p_tgt in enumerate(target_profiles):
            value_matrix[i, j] = float(value_similarity(p_src["value_profile"], p_tgt["value_profile"]))

    # 3. Compute fused similarity matrix & alphas
    if method == "adaptive":
        sim_matrix, alpha_matrix, explanations_grid = compute_fused_similarity_matrix(
            name_matrix=name_matrix,
            value_matrix=value_matrix,
            source_profiles=source_profiles,
            target_profiles=target_profiles,
        )
    elif method == "name_only":
        sim_matrix = name_matrix.copy()
        alpha_matrix = np.ones((n_src, n_tgt), dtype=np.float32)
        explanations_grid = None
    elif method == "value_only":
        sim_matrix = value_matrix.copy()
        alpha_matrix = np.zeros((n_src, n_tgt), dtype=np.float32)
        explanations_grid = None
    else:  # fixed
        sim_matrix = (alpha * name_matrix + (1.0 - alpha) * value_matrix).astype(np.float32)
        alpha_matrix = np.full((n_src, n_tgt), alpha, dtype=np.float32)
        explanations_grid = None

    # 4. Hungarian assignment
    matches = match_schemas(
        profiles_a=source_profiles,
        profiles_b=target_profiles,
        method=method,
        alpha=alpha,
        threshold=threshold,
    )

    # 5. Determine unmatched columns
    matched_src = {m.column_a for m in matches if m.accepted}
    matched_tgt = {m.column_b for m in matches if m.accepted}

    unmatched_src = [c for c in src_cols if c not in matched_src]
    unmatched_tgt = [c for c in tgt_cols if c not in matched_tgt]

    return MatchResult(
        matches=matches,
        unmatched_source=unmatched_src,
        unmatched_target=unmatched_tgt,
        source_profiles=source_profiles,
        target_profiles=target_profiles,
        similarity_matrix=sim_matrix,
        name_matrix=name_matrix,
        value_matrix=value_matrix,
        alpha_matrix=alpha_matrix,
        source_columns=src_cols,
        target_columns=tgt_cols,
        backend_name=backend_name,
        method=method,
        threshold=threshold,
    )
