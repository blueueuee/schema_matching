"""Tests for dataset generator and ground truth integrity."""

from pathlib import Path
import pandas as pd
import pytest

from shared.contracts import validate_profile
from generate_synthetic import build_pair_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SCHEMAS_DIR = REPO_ROOT / "data" / "test_schemas"


def test_schema_pairs_exist():
    assert TEST_SCHEMAS_DIR.exists(), "test_schemas directory must exist"
    pair_dirs = sorted([d for d in TEST_SCHEMAS_DIR.iterdir() if d.is_dir() and d.name.startswith("schema_pair_")])
    assert len(pair_dirs) >= 5, f"Expected at least 5 schema pairs, found {len(pair_dirs)}"


@pytest.mark.parametrize("pair_idx", [1, 2, 3, 4, 5])
def test_ground_truth_headers_and_integrity(pair_idx: int):
    pair_dir = TEST_SCHEMAS_DIR / f"schema_pair_{pair_idx}"
    assert pair_dir.exists()

    src_csv = pair_dir / "source.csv"
    tgt_csv = pair_dir / "target.csv"
    gt_csv = pair_dir / "ground_truth.csv"

    assert src_csv.exists(), f"Missing source.csv in {pair_dir.name}"
    assert tgt_csv.exists(), f"Missing target.csv in {pair_dir.name}"
    assert gt_csv.exists(), f"Missing ground_truth.csv in {pair_dir.name}"

    df_src = pd.read_csv(src_csv)
    df_tgt = pd.read_csv(tgt_csv)
    df_gt = pd.read_csv(gt_csv)

    # Check non-empty datasets
    assert len(df_src) > 0
    assert len(df_tgt) > 0
    assert len(df_gt) > 0

    # Check ground truth schema: only source_column, target_column
    assert list(df_gt.columns) == ["source_column", "target_column"]

    src_headers = set(df_src.columns)
    tgt_headers = set(df_tgt.columns)

    # CRITICAL: Assert every ground-truth column exists in the corresponding CSV headers
    for _, row in df_gt.iterrows():
        sc = row["source_column"]
        tc = row["target_column"]
        assert sc in src_headers, f"Ground-truth source '{sc}' not found in {pair_dir.name}/source.csv headers"
        assert tc in tgt_headers, f"Ground-truth target '{tc}' not found in {pair_dir.name}/target.csv headers"

    # Assert 1-to-1 ground truth uniqueness
    assert len(df_gt["source_column"]) == len(df_gt["source_column"].unique()), f"Duplicate source column in GT: {pair_dir.name}"
    assert len(df_gt["target_column"]) == len(df_gt["target_column"].unique()), f"Duplicate target column in GT: {pair_dir.name}"


def test_generated_profiles_comply_with_contract():
    df = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "signup_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "balance": [100.5, 250.0, 39.9],
    })
    profiles = build_pair_profile(df, "Test")
    assert len(profiles) == 3
    for p in profiles:
        assert validate_profile(p) is True
