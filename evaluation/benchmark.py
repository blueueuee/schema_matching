"""
Benchmarking harness for schema matching.
Loads benchmark pairs, executes matchers, and computes comparative performance tables.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional
import json
import numpy as np

from evaluation.metrics import compute_metrics
from evaluation.baselines import (
    baseline_name_only,
    baseline_value_only,
    baseline_fixed_fusion
)

DEFAULT_SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def load_synthetic_pairs(synthetic_dir: Optional[str] = None) -> List[dict]:
    """
    Load the synthetic schema pairs from data/synthetic/.
    Each pair is a JSON file with keys: profiles_a, profiles_b, ground_truth,
    and optional description / pair_id.
    """
    target_dir = Path(synthetic_dir) if synthetic_dir else DEFAULT_SYNTHETIC_DIR
    if not target_dir.exists():
        return []

    pairs = []
    # Sort file paths so pair_1, pair_2, etc. are in order
    files = sorted(list(target_dir.glob("*.json")))
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            # Ensure ground_truth entries are tuples
            if "ground_truth" in data:
                data["ground_truth"] = [tuple(item) for item in data["ground_truth"]]
            if "file_name" not in data:
                data["file_name"] = f.name
            pairs.append(data)

    return pairs


def run_benchmark(
    schema_pairs: List[dict],
    matcher_fn: Callable,
    **matcher_kwargs
) -> dict:
    """
    Runs matcher_fn on every schema pair, collects metrics per pair,
    and returns aggregate mean precision / recall / F1 across all pairs.
    """
    if not schema_pairs:
        return {
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "pair_results": []
        }

    precisions = []
    recalls = []
    f1s = []
    pair_results = []

    for pair in schema_pairs:
        profiles_a = pair.get("profiles_a", [])
        profiles_b = pair.get("profiles_b", [])
        gt = pair.get("ground_truth", [])

        matches = matcher_fn(profiles_a, profiles_b, **matcher_kwargs)
        metrics = compute_metrics(matches, gt)

        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])

        pair_results.append({
            "pair_id": pair.get("pair_id", pair.get("file_name", "unknown")),
            "description": pair.get("description", ""),
            "metrics": metrics,
            "matches": matches
        })

    return {
        "mean_precision": round(float(np.mean(precisions)), 4),
        "mean_recall": round(float(np.mean(recalls)), 4),
        "mean_f1": round(float(np.mean(f1s)), 4),
        "pair_results": pair_results
    }


def print_comparison_table(results: Dict[str, dict]):
    """Print an ASCII comparison table across all evaluated baseline matchers."""
    print("\n" + "=" * 78)
    print(" " * 22 + "SCHEMA MATCHING BENCHMARK RESULTS")
    print("=" * 78)
    print(f"{'Matcher Baseline':<25} | {'Mean Precision':<15} | {'Mean Recall':<15} | {'Mean F1':<15}")
    print("-" * 78)
    for name, res in results.items():
        p = f"{res['mean_precision']:.4f}"
        r = f"{res['mean_recall']:.4f}"
        f = f"{res['mean_f1']:.4f}"
        print(f"{name:<25} | {p:<15} | {r:<15} | {f:<15}")
    print("=" * 78)


if __name__ == "__main__":
    pairs = load_synthetic_pairs()
    if not pairs:
        print("No synthetic pairs found. Run generate_synthetic.py first.")
    else:
        print(f"Loaded {len(pairs)} synthetic benchmark schema pairs.")

        baseline_results = {
            "1. Name-Only Baseline": run_benchmark(pairs, baseline_name_only),
            "2. Value-Only Baseline": run_benchmark(pairs, baseline_value_only),
            "3. Fixed-Fusion (alpha=0.5)": run_benchmark(pairs, baseline_fixed_fusion),
        }

        print_comparison_table(baseline_results)
