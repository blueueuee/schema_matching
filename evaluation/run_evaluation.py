"""Comprehensive evaluation suite comparing schema matching methods.

Evaluates 4 methods across all benchmark schema pairs:
1. Name-Only Baseline (alpha = 1.0)
2. Value-Only Baseline (alpha = 0.0)
3. Fixed-Weight Fusion (alpha = 0.5)
4. Adaptive Fusion (per-column-pair adaptive alpha)

Threshold policy:
A fixed, documented threshold of 0.35 is applied uniformly across all 4 methods.
This ensures a strictly fair comparison with no test-set tuning or ground-truth leakage.

Produces:
- reports/comparison_table.csv (wide-format comparative table)
- reports/results_long.csv (tidy long-format results)
- reports/per_match_details.csv (fine-grained match decisions and explanations)
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pandas as pd
import numpy as np

from app.service import match_tables
from evaluation.metrics import compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SCHEMAS_DIR = REPO_ROOT / "data" / "test_schemas"
REPORTS_DIR = REPO_ROOT / "reports"

METHODS = [
    ("Name-Only", "name_only"),
    ("Value-Only", "value_only"),
    ("Fixed-Fusion (0.5)", "fixed"),
    ("Adaptive Fusion", "adaptive"),
]

FIXED_THRESHOLD = 0.35


def load_all_schema_pairs() -> List[Dict[str, Any]]:
    """Load all schema pairs and ground truths from data/test_schemas."""
    if not TEST_SCHEMAS_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {TEST_SCHEMAS_DIR}. Run generate_synthetic.py first.")

    pairs: List[Dict[str, Any]] = []
    pair_dirs = sorted([d for d in TEST_SCHEMAS_DIR.iterdir() if d.is_dir() and d.name.startswith("schema_pair_")])

    for pair_dir in pair_dirs:
        src_path = pair_dir / "source.csv"
        tgt_path = pair_dir / "target.csv"
        gt_path = pair_dir / "ground_truth.csv"

        if not (src_path.exists() and tgt_path.exists() and gt_path.exists()):
            continue

        df_src = pd.read_csv(src_path)
        df_tgt = pd.read_csv(tgt_path)
        df_gt = pd.read_csv(gt_path)

        gt_tuples = list(zip(df_gt["source_column"].astype(str), df_gt["target_column"].astype(str)))

        pairs.append({
            "pair_id": pair_dir.name,
            "df_source": df_src,
            "df_target": df_tgt,
            "ground_truth": gt_tuples,
        })

    return pairs


def run_evaluation() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute evaluation across all methods and schema pairs, generating reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    schema_pairs = load_all_schema_pairs()
    if not schema_pairs:
        raise RuntimeError("No schema pairs found to evaluate.")

    results_long_rows: List[Dict[str, Any]] = []
    per_match_rows: List[Dict[str, Any]] = []

    # Map: method_label -> {pair_id: f1}
    comparison_map: Dict[str, Dict[str, float]] = {label: {} for label, _ in METHODS}
    precision_map: Dict[str, List[float]] = {label: [] for label, _ in METHODS}
    recall_map: Dict[str, List[float]] = {label: [] for label, _ in METHODS}
    f1_map: Dict[str, List[float]] = {label: [] for label, _ in METHODS}

    for pair in schema_pairs:
        pair_id = pair["pair_id"]
        df_src: pd.DataFrame = pair["df_source"]
        df_tgt: pd.DataFrame = pair["df_target"]
        gt_tuples: List[Tuple[str, str]] = pair["ground_truth"]
        gt_set: Set[Tuple[str, str]] = set(gt_tuples)

        for method_label, method_key in METHODS:
            match_res = match_tables(
                source_df=df_src,
                target_df=df_tgt,
                method=method_key,
                threshold=FIXED_THRESHOLD,
                sample_size=500,
                alpha=0.5,
            )

            # Record per-match decisions
            for m in match_res.matches:
                pair_tuple = (m.column_a, m.column_b)
                is_correct = bool(pair_tuple in gt_set) if m.accepted else False
                expl_text = m.explanation.explanation_text if m.explanation else ""

                per_match_rows.append({
                    "pair_id": pair_id,
                    "method": method_label,
                    "source_column": m.column_a,
                    "target_column": m.column_b,
                    "confidence_score": m.score,
                    "accepted": m.accepted,
                    "name_similarity": m.name_similarity,
                    "value_similarity": m.value_similarity,
                    "alpha": m.alpha,
                    "is_in_ground_truth": bool(pair_tuple in gt_set),
                    "is_correct_accepted_match": is_correct,
                    "explanation": expl_text,
                })

            # Calculate metrics
            pred_matches = [m.to_dict() for m in match_res.matches]
            metrics = compute_metrics(pred_matches, gt_tuples)

            p = metrics["precision"]
            r = metrics["recall"]
            f = metrics["f1"]

            comparison_map[method_label][pair_id] = f
            precision_map[method_label].append(p)
            recall_map[method_label].append(r)
            f1_map[method_label].append(f)

            results_long_rows.append({
                "pair_id": pair_id,
                "method": method_label,
                "precision": p,
                "recall": r,
                "f1": f,
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "threshold": FIXED_THRESHOLD,
            })

    # Build comparison wide table
    pair_ids = [p["pair_id"] for p in schema_pairs]
    comp_rows: List[Dict[str, Any]] = []

    for method_label, _ in METHODS:
        row: Dict[str, Any] = {"method": method_label}
        for pid in pair_ids:
            row[f"{pid}_f1"] = comparison_map[method_label].get(pid, 0.0)

        row["macro_precision"] = round(float(np.mean(precision_map[method_label])), 4)
        row["macro_recall"] = round(float(np.mean(recall_map[method_label])), 4)
        row["macro_f1"] = round(float(np.mean(f1_map[method_label])), 4)
        comp_rows.append(row)

    df_comp = pd.DataFrame(comp_rows)
    df_long = pd.DataFrame(results_long_rows)
    df_details = pd.DataFrame(per_match_rows)

    # Save reports
    df_comp.to_csv(REPORTS_DIR / "comparison_table.csv", index=False)
    df_long.to_csv(REPORTS_DIR / "results_long.csv", index=False)
    df_details.to_csv(REPORTS_DIR / "per_match_details.csv", index=False)

    return df_comp, df_long, df_details


def print_evaluation_summary(df_comp: pd.DataFrame, df_long: pd.DataFrame) -> None:
    """Print readable terminal output for evaluation results."""
    print("\n" + "=" * 90)
    print(" " * 28 + "SCHEMA MATCHING BENCHMARK REPORT")
    print("=" * 90)
    print(f"Threshold Policy: Fixed threshold = {FIXED_THRESHOLD} across all methods")
    print("-" * 90)

    f1_cols = [c for c in df_comp.columns if c.endswith("_f1") and c != "macro_f1"]
    header = f"{'Method':<22} | " + " | ".join(f"{c.replace('_f1', ''):<10}" for c in f1_cols) + f" | {'Macro P':<8} | {'Macro R':<8} | {'Macro F1':<8}"
    print(header)
    print("-" * len(header))

    for _, row in df_comp.iterrows():
        line = f"{row['method']:<22} | "
        for c in f1_cols:
            line += f"{row[c]:<10.4f} | "
        line += f"{row['macro_precision']:<8.4f} | {row['macro_recall']:<8.4f} | {row['macro_f1']:<8.4f}"
        print(line)

    print("=" * len(header))
    print(f"Reports saved to: {REPORTS_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    comp, long_df, details = run_evaluation()
    print_evaluation_summary(comp, long_df)
