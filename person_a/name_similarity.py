"""Vectorized cosine similarity calculation for column name embeddings."""

from typing import Dict, List, Tuple, Union
import numpy as np


def compute_cosine_similarity(
    embedding_a: Union[List[float], np.ndarray],
    embedding_b: Union[List[float], np.ndarray],
) -> float:
    """Compute cosine similarity between two embedding vectors, clipped to [0.0, 1.0].

    Note on clipping:
    Cosine similarity mathematically ranges from -1.0 to 1.0. In semantic schema matching,
    unrelated concepts produce negative or near-zero cosine values; negative similarity represents
    zero semantic alignment. Scores are thus bounded in [0.0, 1.0] with np.clip.
    """
    a = np.asarray(embedding_a, dtype=np.float32).ravel()
    b = np.asarray(embedding_b, dtype=np.float32).ravel()

    if a.size == 0 or b.size == 0:
        return 0.0

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(sim, 0.0, 1.0))


def compute_name_similarity_matrix(
    source_embeddings: Union[Dict[str, np.ndarray], np.ndarray, List[np.ndarray]],
    target_embeddings: Union[Dict[str, np.ndarray], np.ndarray, List[np.ndarray]],
) -> np.ndarray:
    """Vectorized calculation of (|source| x |target|) similarity matrix via matrix multiplication.

    Operates in O(1) vectorized matrix multiplication without per-pair Python loops.
    """
    if isinstance(source_embeddings, dict):
        source_arr = np.asarray(list(source_embeddings.values()), dtype=np.float32)
    else:
        source_arr = np.asarray(source_embeddings, dtype=np.float32)

    if isinstance(target_embeddings, dict):
        target_arr = np.asarray(list(target_embeddings.values()), dtype=np.float32)
    else:
        target_arr = np.asarray(target_embeddings, dtype=np.float32)

    if source_arr.size == 0 or target_arr.size == 0:
        n_src = source_arr.shape[0] if source_arr.ndim > 0 else 0
        n_tgt = target_arr.shape[0] if target_arr.ndim > 0 else 0
        return np.zeros((n_src, n_tgt), dtype=np.float32)

    if source_arr.ndim == 1:
        source_arr = source_arr.reshape(1, -1)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(1, -1)

    # Normalize rows to unit L2 norm
    s_norms = np.linalg.norm(source_arr, axis=1, keepdims=True)
    t_norms = np.linalg.norm(target_arr, axis=1, keepdims=True)
    s_norms[s_norms == 0.0] = 1.0
    t_norms[t_norms == 0.0] = 1.0

    s_unit = source_arr / s_norms
    t_unit = target_arr / t_norms

    # Vectorized cosine dot product (N x D) @ (D x M) -> (N x M)
    matrix = s_unit @ t_unit.T
    return np.clip(matrix, 0.0, 1.0).astype(np.float32)


def format_similarity_matrix(
    matrix: np.ndarray,
    source_cols: List[str],
    target_cols: List[str],
    decimals: int = 3,
) -> str:
    """Format similarity matrix as a readable aligned ASCII table."""
    lines: List[str] = []
    header = "Source".ljust(25)
    for col in target_cols:
        header += col[:15].ljust(18)
    lines.append(header)
    lines.append("-" * len(header))

    for i, source_col in enumerate(source_cols):
        row = source_col[:24].ljust(25)
        for j in range(len(target_cols)):
            score = matrix[i, j] if i < matrix.shape[0] and j < matrix.shape[1] else 0.0
            row += f"{score:.{decimals}f}".ljust(18)
        lines.append(row)

    return "\n".join(lines)