"""Unit tests for person_a.fusion adaptive alpha rules and explainability."""

import pytest
from person_a.fusion import compute_alpha, fused_similarity, is_generic_token_or_name


def test_base_alpha_and_explanation():
    alpha, eff_val_sim, reasons = compute_alpha(
        source_col="customer_identifier",
        target_col="client_code",
        name_sim=0.70,
        value_sim=0.70,
        source_profile={"data_type": "string", "normalized_tokens": ["customer", "identifier"]},
        target_profile={"data_type": "string", "normalized_tokens": ["client", "code"]},
    )
    # Default specific names should be around base alpha 0.6
    assert 0.55 <= alpha <= 0.65
    assert eff_val_sim == pytest.approx(0.70, abs=1e-3)


def test_generic_name_decreases_alpha():
    alpha, _, reasons = compute_alpha(
        source_col="id",
        target_col="code",
        name_sim=0.40,
        value_sim=0.90,
        source_profile={"data_type": "string", "normalized_tokens": ["id"]},
        target_profile={"data_type": "string", "normalized_tokens": ["code"]},
    )
    # Generic names decrease alpha toward value profile
    assert alpha <= 0.35
    assert any("generic" in r for r in reasons)


def test_sparse_values_increase_alpha():
    # Null ratio > 0.8 means value distribution is unreliable
    alpha, _, reasons = compute_alpha(
        source_col="order_amount",
        target_col="total_due",
        name_sim=0.75,
        value_sim=0.50,
        source_profile={"data_type": "numeric", "value_profile": {"null_ratio": 0.95}},
        target_profile={"data_type": "numeric", "value_profile": {"null_ratio": 0.05}},
    )
    # Unreliable values push alpha up toward name signal
    assert alpha >= 0.70
    assert any("sparse or high-null" in r for r in reasons)


def test_high_name_sim_and_matching_dtype_boosts_alpha():
    alpha, _, reasons = compute_alpha(
        source_col="date_of_birth",
        target_col="birth_date",
        name_sim=0.92,
        value_sim=0.88,
        source_profile={"data_type": "date", "normalized_tokens": ["date", "birth"]},
        target_profile={"data_type": "date", "normalized_tokens": ["birth", "date"]},
    )
    assert alpha >= 0.70
    assert any("matching" in r and "date" in r for r in reasons)


def test_dtype_mismatch_penalty():
    _, eff_val_sim, reasons = compute_alpha(
        source_col="created_at",
        target_col="created_at",
        name_sim=1.0,
        value_sim=0.80,
        source_profile={"data_type": "date"},
        target_profile={"data_type": "numeric"},
    )
    # Incompatible dtypes penalize value similarity
    assert eff_val_sim < 0.20
    assert any("data type mismatch" in r for r in reasons)


def test_false_friend_generic_name_suppression():
    # Generic name with incompatible dtypes (e.g. status string vs HTTP code int)
    alpha, eff_val_sim, reasons = compute_alpha(
        source_col="status",
        target_col="status",
        name_sim=1.0,
        value_sim=0.50,
        source_profile={"data_type": "string", "normalized_tokens": ["status"]},
        target_profile={"data_type": "numeric", "normalized_tokens": ["status"]},
    )
    # Should clamp alpha to lower bound 0.15 to avoid false friend match
    assert alpha == pytest.approx(0.15, abs=1e-3)
    assert any("false-friend" in r for r in reasons)


def test_alpha_clamping():
    # Test lower bound clamp
    alpha_low, _, _ = compute_alpha(
        source_col="x",
        target_col="y",
        name_sim=0.0,
        value_sim=1.0,
        source_profile={"data_type": "numeric", "normalized_tokens": ["x"]},
        target_profile={"data_type": "date", "normalized_tokens": ["y"]},
    )
    assert alpha_low >= 0.15

    # Test upper bound clamp
    alpha_high, _, _ = compute_alpha(
        source_col="exact_custom_field_identifier",
        target_col="exact_custom_field_identifier",
        name_sim=1.0,
        value_sim=0.0,
        source_profile={"data_type": "string", "value_profile": {"null_ratio": 1.0}},
        target_profile={"data_type": "string", "value_profile": {"null_ratio": 1.0}},
    )
    assert alpha_high <= 0.85


def test_fused_similarity_explanation_object():
    score_obj = fused_similarity(
        name_sim=0.85,
        value_sim=0.75,
        source_col="cust_id",
        target_col="customer_id",
        source_profile={"data_type": "string", "normalized_tokens": ["customer", "identifier"]},
        target_profile={"data_type": "string", "normalized_tokens": ["customer", "identifier"]},
    )
    assert 0.0 <= score_obj.fused <= 1.0
    assert 0.15 <= score_obj.alpha <= 0.85
    assert len(score_obj.explanation) > 0
    assert "Score" in score_obj.explanation or "Fused" in score_obj.explanation
