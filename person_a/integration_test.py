"""Integration test for Person A name-matching pipeline."""

from typing import List
import numpy as np

from person_a.normalize_names import normalize_column_name
from person_a.name_embedder import get_name_embedder
from person_a.name_similarity import (
    compute_name_similarity_matrix,
    format_similarity_matrix,
)


def run_name_pipeline(source_cols: List[str], target_cols: List[str]) -> np.ndarray:
    print("=" * 70)
    print("PERSON A - NAME MATCHING PIPELINE")
    print("=" * 70)

    # 1. Normalize names
    print("\n[1] NORMALIZATION")
    source_normalized = {col: normalize_column_name(col) for col in source_cols}
    for col in source_cols:
        print(f"{col:25s} -> {source_normalized[col]}")

    target_normalized = {col: normalize_column_name(col) for col in target_cols}
    for col in target_cols:
        print(f"{col:25s} -> {target_normalized[col]}")

    # 2. Embeddings
    print("\n[2] EMBEDDINGS")
    embedder = get_name_embedder()
    source_embeddings = {col: embedder.embed_tokens(tokens) for col, tokens in source_normalized.items()}
    target_embeddings = {col: embedder.embed_tokens(tokens) for col, tokens in target_normalized.items()}

    print(f"Backend: {embedder.backend_name}")
    print(f"Source count: {len(source_embeddings)}, Target count: {len(target_embeddings)}, Dim: {embedder.embedding_dim}")

    # 3. Similarity matrix
    print("\n[3] NAME SIMILARITY")
    matrix = compute_name_similarity_matrix(source_embeddings, target_embeddings)
    print(format_similarity_matrix(matrix, source_cols, target_cols))

    # 4. Best match per row
    print("\n[4] BEST NAME MATCHES")
    for i, source_col in enumerate(source_cols):
        best_idx = int(np.argmax(matrix[i]))
        target_col = target_cols[best_idx]
        score = matrix[i, best_idx]
        print(f"{source_col:25s} -> {target_col:25s} score={score:.4f}")

    return matrix


if __name__ == "__main__":
    src = ["customer_id", "customer_name", "dob", "phone_no", "order_amount"]
    tgt = ["cust_identifier", "full_customer_name", "date_of_birth", "telephone_number", "total_amt"]
    run_name_pipeline(src, tgt)