"""
Evaluation metrics for schema matching.
Computes Precision, Recall, and F1 score against ground truth mappings.
"""

from typing import Dict, List, Set, Tuple


def compute_metrics(
    predicted_matches: List[dict],
    ground_truth: List[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Computes precision, recall, and F1 score for schema matching predictions.

    Parameters
    ----------
    predicted_matches : List[dict]
        Output list of match dictionaries from match_schemas.
        Only matches with accepted=True will be evaluated.
    ground_truth : List[Tuple[str, str]]
        List of correct (column_a_id, column_b_id) pairs.

    Returns
    -------
    dict
        {
          "precision": float,
          "recall": float,
          "f1": float,
          "tp": int,
          "fp": int,
          "fn": int
        }
    """
    # Filter to accepted matches only
    pred_pairs: Set[Tuple[str, str]] = {
        (m["column_a"], m["column_b"])
        for m in predicted_matches
        if m.get("accepted", True)
    }

    gt_pairs: Set[Tuple[str, str]] = set(ground_truth)

    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
