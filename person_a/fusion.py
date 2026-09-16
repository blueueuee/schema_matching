"""
Adaptive fusion of name and value similarity.

Final score:

    final = alpha * name_similarity
          + (1-alpha) * value_similarity
"""

import numpy as np


GENERIC_NAMES = {
    "col",
    "column",
    "field",
    "data",
    "value",
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "v1",
    "v2",
    "v3",
    "f1",
    "f2",
    "attr",
    "property"
}


def is_generic_name(name):

    name = name.lower().strip()

    return (
        name in GENERIC_NAMES
        or len(name) < 3
    )


def adaptive_alpha(
    name_sim,
    value_sim,
    source_name,
    target_name,
    debug=False
):

    source_generic = is_generic_name(
        source_name
    )

    target_generic = is_generic_name(
        target_name
    )

    # --------------------------------------------------
    # Base trust
    # --------------------------------------------------

    if source_generic or target_generic:
        alpha = 0.3
    else:
        alpha = 0.7

    # --------------------------------------------------
    # Adjust based on name similarity
    # --------------------------------------------------

    if name_sim > 0.85:

        alpha = min(
            0.9,
            alpha + 0.15
        )

    elif name_sim < 0.2:

        alpha = max(
            0.1,
            alpha - 0.3
        )

    elif name_sim > 0.6:

        alpha = min(
            0.85,
            alpha + 0.1
        )

    # --------------------------------------------------
    # Check disagreement
    # --------------------------------------------------

    disagreement = abs(
        name_sim - value_sim
    )

    if disagreement > 0.5:

        alpha = 0.5

    if debug:

        print(
            f"\n{source_name} vs {target_name}"
        )

        print(
            f"name similarity  = {name_sim:.3f}"
        )

        print(
            f"value similarity = {value_sim:.3f}"
        )

        print(
            f"disagreement     = {disagreement:.3f}"
        )

        print(
            f"alpha            = {alpha:.3f}"
        )

    return alpha


def fused_similarity(
    name_sim,
    value_sim,
    source_name,
    target_name
):

    alpha = adaptive_alpha(
        name_sim,
        value_sim,
        source_name,
        target_name
    )

    final_score = (
        alpha * name_sim
        +
        (1 - alpha) * value_sim
    )

    return float(
        np.clip(
            final_score,
            0.0,
            1.0
        )
    )


def compute_fused_similarity_matrix(
    name_matrix,
    value_matrix,
    source_cols,
    target_cols
):

    if name_matrix.shape != value_matrix.shape:

        raise ValueError(
            "Name and value matrices "
            "must have the same shape."
        )

    fused_matrix = np.zeros_like(
        name_matrix,
        dtype=np.float32
    )

    for i, source_col in enumerate(source_cols):

        for j, target_col in enumerate(target_cols):

            fused_matrix[i][j] = (
                fused_similarity(
                    name_matrix[i][j],
                    value_matrix[i][j],
                    source_col,
                    target_col
                )
            )

    return fused_matrix


if __name__ == "__main__":

    print("=" * 60)
    print("ADAPTIVE FUSION TEST")
    print("=" * 60)

    test_cases = [

        (
            0.90,
            0.30,
            "customer_id",
            "cust_id"
        ),

        (
            0.20,
            0.90,
            "col_x",
            "data_y"
        ),

        (
            0.50,
            0.50,
            "order_date",
            "order_dt"
        ),

        (
            0.95,
            0.10,
            "full_name",
            "cust_name"
        )
    ]

    for case in test_cases:

        name_sim = case[0]
        value_sim = case[1]
        source = case[2]
        target = case[3]

        score = fused_similarity(
            name_sim,
            value_sim,
            source,
            target
        )

        alpha = adaptive_alpha(
            name_sim,
            value_sim,
            source,
            target
        )

        print()
        print(
            f"{source} -> {target}"
        )

        print(
            f"Name similarity : {name_sim}"
        )

        print(
            f"Value similarity: {value_sim}"
        )

        print(
            f"Alpha           : {alpha:.3f}"
        )

        print(
            f"Final score     : {score:.3f}"
        )