"""End-to-end integration tests for full benchmark evaluation pipeline."""

from pathlib import Path
import pandas as pd
import pytest

from app.service import match_tables
from evaluation.run_evaluation import run_evaluation, load_all_schema_pairs
from evaluation.metrics import compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"


def test_end_to_end_all_pairs_macro_f1():
    """Verify adaptive fusion runs without error across all 5 schema pairs and exceeds F1 floor."""
    pairs = load_all_schema_pairs()
    assert len(pairs) >= 5

    f1_scores = []
    for pair in pairs:
        res = match_tables(
            source_df=pair["df_source"],
            target_df=pair["df_target"],
            method="adaptive",
            threshold=0.35,
        )
        assert res is not None
        assert len(res.matches) > 0

        pred_tuples = [m.to_dict() for m in res.matches]
        metrics = compute_metrics(pred_tuples, pair["ground_truth"])
        f1_scores.append(metrics["f1"])

    macro_f1 = sum(f1_scores) / len(f1_scores)
    # The requirement requires >= conservative floor (0.60)
    assert macro_f1 >= 0.60, f"Macro F1 {macro_f1:.4f} below minimum floor of 0.60"
    assert macro_f1 >= 0.85, f"Macro F1 {macro_f1:.4f} expected high performance"


def test_rectangular_matcher_input():
    """Verify Hungarian matching correctly handles asymmetric rectangular matrices (|A| != |B|)."""
    pairs = load_all_schema_pairs()
    pair_5 = next((p for p in pairs if p["pair_id"] == "schema_pair_5"), None)
    assert pair_5 is not None

    df_src = pair_5["df_source"]
    df_tgt = pair_5["df_target"]

    # Pair 5 has 9 source columns and 10 target columns
    assert df_src.shape[1] == 9
    assert df_tgt.shape[1] == 10

    res = match_tables(
        source_df=df_src,
        target_df=df_tgt,
        method="adaptive",
        threshold=0.35,
    )
    # Hungarian matcher produces min(9, 10) = 9 one-to-one pairs
    assert len(res.matches) == 9
    # Accepted matches must be unique on both sides
    accepted = [m for m in res.matches if m.accepted]
    matched_srcs = [m.column_a for m in accepted]
    matched_tgts = [m.column_b for m in accepted]
    assert len(matched_srcs) == len(set(matched_srcs))
    assert len(matched_tgts) == len(set(matched_tgts))


def test_cli_evaluation_produces_reports():
    """Verify run_evaluation produces all required report CSVs with non-empty rows."""
    comp_df, long_df, details_df = run_evaluation()

    comp_file = REPORTS_DIR / "comparison_table.csv"
    long_file = REPORTS_DIR / "results_long.csv"
    details_file = REPORTS_DIR / "per_match_details.csv"

    assert comp_file.exists()
    assert long_file.exists()
    assert details_file.exists()

    assert len(comp_df) == 4, f"Expected 4 methods in comparison_table, found {len(comp_df)}"
    assert len(long_df) == 20, f"Expected 4 methods * 5 pairs = 20 rows, found {len(long_df)}"
    assert len(details_df) > 0, "per_match_details must have match records"
