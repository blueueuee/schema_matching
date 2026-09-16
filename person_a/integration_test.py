import numpy as np

from normalize_names import normalize_column_name
from name_embedder import NameEmbedder
from name_similarity import (
    compute_name_similarity_matrix,
    format_similarity_matrix
)


def run_name_pipeline(source_cols, target_cols):

    print("=" * 70)
    print("PERSON A - NAME MATCHING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Normalize names
    # --------------------------------------------------

    print("\n[1] NORMALIZATION")

    source_normalized = {}

    for col in source_cols:
        source_normalized[col] = normalize_column_name(col)

        print(
            f"{col:25s} -> "
            f"{source_normalized[col]}"
        )

    target_normalized = {}

    for col in target_cols:
        target_normalized[col] = normalize_column_name(col)

        print(
            f"{col:25s} -> "
            f"{target_normalized[col]}"
        )

    # --------------------------------------------------
    # 2. Create embeddings
    # --------------------------------------------------

    print("\n[2] EMBEDDINGS")

    embedder = NameEmbedder()

    source_embeddings = {}

    for col, tokens in source_normalized.items():

        source_embeddings[col] = (
            embedder.embed_tokens(tokens)
        )

    target_embeddings = {}

    for col, tokens in target_normalized.items():

        target_embeddings[col] = (
            embedder.embed_tokens(tokens)
        )

    print(
        f"Source embeddings: {len(source_embeddings)}"
    )

    print(
        f"Target embeddings: {len(target_embeddings)}"
    )

    print(
        f"Dimension: {embedder.embedding_dim}"
    )

    # --------------------------------------------------
    # 3. Similarity matrix
    # --------------------------------------------------

    print("\n[3] NAME SIMILARITY")

    matrix = compute_name_similarity_matrix(
        source_embeddings,
        target_embeddings
    )

    format_similarity_matrix(
        matrix,
        source_cols,
        target_cols
    )

    # --------------------------------------------------
    # 4. Best match
    # --------------------------------------------------

    print("\n[4] BEST NAME MATCHES")

    for i, source_col in enumerate(source_cols):

        best_index = np.argmax(matrix[i])

        target_col = target_cols[best_index]

        score = matrix[i][best_index]

        print(
            f"{source_col:25s} -> "
            f"{target_col:25s} "
            f"score={score:.4f}"
        )

    return matrix


if __name__ == "__main__":

    source_columns = [
        "customer_id",
        "customer_name",
        "dob",
        "phone_no",
        "order_amount"
    ]

    target_columns = [
        "cust_identifier",
        "full_customer_name",
        "date_of_birth",
        "telephone_number",
        "total_amt"
    ]

    run_name_pipeline(
        source_columns,
        target_columns
    )