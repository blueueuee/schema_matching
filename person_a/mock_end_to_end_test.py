"""End-to-end pipeline test combining name embeddings and real profile similarity."""

from typing import List, Tuple
import numpy as np

from person_a.normalize_names import normalize_column_name
from person_a.name_embedder import get_name_embedder
from person_a.name_similarity import (
    compute_name_similarity_matrix,
    format_similarity_matrix,
)
from person_a.fusion import compute_fused_similarity_matrix
from person_a.mock_value_profiles import create_sample_profile
from matching.matcher import match_schemas


def run() -> List[Tuple[str, str, float]]:
    source_cols = ["customer_id", "customer_name", "dob", "phone_no", "order_amount"]
    target_cols = ["cust_identifier", "full_customer_name", "date_of_birth", "telephone_number", "total_amt"]

    # 1. Normalization
    source_tokens = {col: normalize_column_name(col) for col in source_cols}
    target_tokens = {col: normalize_column_name(col) for col in target_cols}

    # 2. Embeddings
    embedder = get_name_embedder()
    source_embs = {col: embedder.embed_tokens(tokens) for col, tokens in source_tokens.items()}
    target_embs = {col: embedder.embed_tokens(tokens) for col, tokens in target_tokens.items()}

    name_matrix = compute_name_similarity_matrix(source_embs, target_embs)

    # 3. Value Profiles using real profiler
    categories = ["id", "string", "date", "phone", "amount"]
    source_profiles = [create_sample_profile(col, cat) for col, cat in zip(source_cols, categories)]
    target_profiles = [create_sample_profile(col, cat) for col, cat in zip(target_cols, categories)]

    # 4. Fused Matrix
    from value_pipeline.similarity import value_similarity

    n_src = len(source_profiles)
    n_tgt = len(target_profiles)
    value_matrix = np.zeros((n_src, n_tgt), dtype=np.float32)
    for i in range(n_src):
        for j in range(n_tgt):
            value_matrix[i, j] = value_similarity(source_profiles[i]["value_profile"], target_profiles[j]["value_profile"])

    fused_matrix, alphas, explanations = compute_fused_similarity_matrix(
        name_matrix=name_matrix,
        value_matrix=value_matrix,
        source_profiles=source_profiles,
        target_profiles=target_profiles,
    )

    print("\n" + "=" * 70)
    print("FUSED SIMILARITY MATRIX")
    print("=" * 70)
    print(format_similarity_matrix(fused_matrix, source_cols, target_cols))

    matches = match_schemas(
        source_profiles,
        target_profiles,
        alpha=0.5,
        threshold=0.3,
    )

    results = []
    print("\n" + "=" * 70)
    print("MATCHES (via Hungarian algorithm)")
    print("=" * 70)
    for m in matches:
        print(f"{m['column_a']:25s} -> {m['column_b']:25s} score={m['score']:.4f} accepted={m['accepted']}")
        results.append((m["column_a"], m["column_b"], m["score"]))

    return results


if __name__ == "__main__":
    run()