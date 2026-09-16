"""
Evaluation package for schema matching benchmarks, baselines, and metrics.
"""

from evaluation.metrics import compute_metrics
from evaluation.baselines import (
    compute_name_similarity,
    baseline_name_only,
    baseline_value_only,
    baseline_fixed_fusion,
)
from evaluation.benchmark import (
    load_synthetic_pairs,
    run_benchmark,
    print_comparison_table,
)

__all__ = [
    "compute_metrics",
    "compute_name_similarity",
    "baseline_name_only",
    "baseline_value_only",
    "baseline_fixed_fusion",
    "load_synthetic_pairs",
    "run_benchmark",
    "print_comparison_table",
]
