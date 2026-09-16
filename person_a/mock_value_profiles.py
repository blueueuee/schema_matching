import numpy as np


def create_mock_value_profile(
    column_name
):

    name = column_name.lower()

    profile = np.zeros(
        21,
        dtype=np.float32
    )

    # ID-like
    if any(
        word in name
        for word in [
            "id",
            "no",
            "number",
            "code"
        ]
    ):

        profile[0] = 1.0
        profile[7] = 0.95
        profile[10] = 1.0

    # Date-like
    elif any(
        word in name
        for word in [
            "date",
            "dt",
            "dob",
            "time"
        ]
    ):

        profile[3] = 1.0
        profile[7] = 0.8

    # Status-like
    elif any(
        word in name
        for word in [
            "status",
            "flag",
            "indicator"
        ]
    ):

        profile[2] = 1.0
        profile[7] = 0.05

    # Amount-like
    elif any(
        word in name
        for word in [
            "amount",
            "amt",
            "price",
            "cost",
            "total"
        ]
    ):

        profile[1] = 1.0
        profile[7] = 0.7

    else:

        profile[2] = 1.0
        profile[7] = 0.4

    return profile


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def create_mock_value_similarity_matrix(
    source_cols,
    target_cols
):

    source_profiles = {
        col: create_mock_value_profile(col)
        for col in source_cols
    }

    target_profiles = {
        col: create_mock_value_profile(col)
        for col in target_cols
    }

    matrix = np.zeros(
        (len(source_cols), len(target_cols)),
        dtype=np.float32
    )

    for i, source in enumerate(source_cols):

        for j, target in enumerate(target_cols):

            similarity = cosine_similarity(
                source_profiles[source],
                target_profiles[target]
            )

            matrix[i][j] = (
                similarity
            )

    return matrix


if __name__ == "__main__":

    source = [
        "customer_id",
        "dob",
        "amount"
    ]

    target = [
        "cust_id",
        "date_of_birth",
        "total_amount"
    ]

    matrix = create_mock_value_similarity_matrix(
        source,
        target
    )

    print(matrix)