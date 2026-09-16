"""
Similarity calculation for column name embeddings.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(
    embedding_a,
    embedding_b
):
    """
    Compute cosine similarity between two embeddings.
    """

    a = np.asarray(embedding_a).reshape(1, -1)
    b = np.asarray(embedding_b).reshape(1, -1)

    similarity = cosine_similarity(a, b)[0][0]

    # Convert [-1, 1] to [0, 1]
    similarity = (similarity + 1) / 2

    return float(similarity)


def compute_name_similarity_matrix(
    source_embeddings,
    target_embeddings
):
    """
    Compute similarity between every source
    and target column.
    """

    source_cols = list(source_embeddings.keys())
    target_cols = list(target_embeddings.keys())

    matrix = np.zeros(
        (len(source_cols), len(target_cols)),
        dtype=np.float32
    )

    for i, source_col in enumerate(source_cols):

        for j, target_col in enumerate(target_cols):

            matrix[i][j] = compute_cosine_similarity(
                source_embeddings[source_col],
                target_embeddings[target_col]
            )

    return matrix


def format_similarity_matrix(
    matrix,
    source_cols,
    target_cols,
    decimals=3
):

    print()

    header = "Source".ljust(25)

    for col in target_cols:
        header += col[:15].ljust(18)

    print(header)
    print("-" * len(header))

    for i, source_col in enumerate(source_cols):

        row = source_col.ljust(25)

        for j in range(len(target_cols)):

            score = matrix[i][j]

            row += f"{score:.{decimals}f}".ljust(18)

        print(row)


if __name__ == "__main__":

    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    c = np.array([0, 1, 0])

    print(
        "Similarity A-B:",
        compute_cosine_similarity(a, b)
    )

    print(
        "Similarity A-C:",
        compute_cosine_similarity(a, c)
    )