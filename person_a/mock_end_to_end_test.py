import numpy as np

from normalize_names import normalize_column_name
from name_embedder import NameEmbedder
from name_similarity import (
    compute_name_similarity_matrix,
    format_similarity_matrix
)
from fusion import (
    compute_fused_similarity_matrix
)
from mock_value_profiles import (
    create_mock_value_similarity_matrix
)


def run():

    source_cols = [
        "customer_id",
        "customer_name",
        "dob",
        "phone_no",
        "order_amount"
    ]

    target_cols = [
        "cust_identifier",
        "full_customer_name",
        "date_of_birth",
        "telephone_number",
        "total_amt"
    ]

    # ==================================================
    # NAME PIPELINE
    # ==================================================

    print("=" * 70)
    print("1. NORMALIZING COLUMN NAMES")
    print("=" * 70)

    source_tokens = {
        col: normalize_column_name(col)
        for col in source_cols
    }

    target_tokens = {
        col: normalize_column_name(col)
        for col in target_cols
    }

    for col in source_cols:
        print(
            f"{col:25s} -> "
            f"{source_tokens[col]}"
        )

    for col in target_cols:
        print(
            f"{col:25s} -> "
            f"{target_tokens[col]}"
        )

    # ==================================================
    # EMBEDDINGS
    # ==================================================

    print("\n" + "=" * 70)
    print("2. GENERATING NAME EMBEDDINGS")
    print("=" * 70)

    embedder = NameEmbedder()

    source_embeddings = {}

    for col in source_cols:

        source_embeddings[col] = (
            embedder.embed_tokens(
                source_tokens[col]
            )
        )

    target_embeddings = {}

    for col in target_cols:

        target_embeddings[col] = (
            embedder.embed_tokens(
                target_tokens[col]
            )
        )

    # ==================================================
    # NAME SIMILARITY
    # ==================================================

    print("\n" + "=" * 70)
    print("3. NAME SIMILARITY")
    print("=" * 70)

    name_matrix = (
        compute_name_similarity_matrix(
            source_embeddings,
            target_embeddings
        )
    )

    format_similarity_matrix(
        name_matrix,
        source_cols,
        target_cols
    )

    # ==================================================
    # MOCK VALUE SIMILARITY
    # ==================================================

    print("\n" + "=" * 70)
    print("4. MOCK VALUE SIMILARITY")
    print("=" * 70)

    value_matrix = (
        create_mock_value_similarity_matrix(
            source_cols,
            target_cols
        )
    )

    format_similarity_matrix(
        value_matrix,
        source_cols,
        target_cols
    )

    # ==================================================
    # FUSION
    # ==================================================

    print("\n" + "=" * 70)
    print("5. ADAPTIVE FUSION")
    print("=" * 70)

    fused_matrix = (
        compute_fused_similarity_matrix(
            name_matrix,
            value_matrix,
            source_cols,
            target_cols
        )
    )

    format_similarity_matrix(
        fused_matrix,
        source_cols,
        target_cols
    )

    # ==================================================
    # FINAL MATCHES
    # ==================================================

    print("\n" + "=" * 70)
    print("6. FINAL MATCHES")
    print("=" * 70)

    matches = []

    for i, source in enumerate(source_cols):

        best_index = np.argmax(
            fused_matrix[i]
        )

        target = target_cols[
            best_index
        ]

        score = fused_matrix[
            i,
            best_index
        ]

        matches.append(
            (
                source,
                target,
                float(score)
            )
        )

        print(
            f"{source:25s} -> "
            f"{target:25s} "
            f"score={score:.4f}"
        )

    return matches


if __name__ == "__main__":

    run()