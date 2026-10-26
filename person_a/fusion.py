"""Adaptive fusion of semantic name and value profiling signals.

Computes a per-column-pair adaptive alpha weight based on:
1. Informativeness of column names (generic vs specific)
2. Reliability of value samples (null ratio, sample depth)
3. Agreement between data types and signal consistency
4. Generates human-readable, audit-ready explanations for every match decision.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

GENERIC_NAMES: Set[str] = {
    "id",
    "identifier",
    "name",
    "date",
    "dt",
    "value",
    "val",
    "code",
    "cd",
    "type",
    "num",
    "number",
    "data",
    "col",
    "field",
    "item",
    "record",
    "text",
    "info",
    "status",
    "attr",
    "property",
    "age",
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "v1",
    "v2",
    "f1",
    "f2",
}


@dataclass
class FusedMatchScore:
    source_col: str
    target_col: str
    name_sim: float
    value_sim: float
    alpha: float
    fused: float
    reasons: List[str] = field(default_factory=list)
    source_dtype: str = "unknown"
    target_dtype: str = "unknown"

    @property
    def explanation(self) -> str:
        base = (
            f"Score {self.fused:.3f} = {self.alpha:.2f} * {self.name_sim:.3f} (name) + "
            f"{1.0 - self.alpha:.2f} * {self.value_sim:.3f} (value)."
        )
        if self.reasons:
            return base + " Adjustments: " + "; ".join(self.reasons) + "."
        return base


def is_generic_token_or_name(name_or_tokens: Any) -> bool:
    """Check if a column name or its tokens lack distinct semantic specificity."""
    if isinstance(name_or_tokens, (list, tuple)):
        tokens = [str(t).lower().strip() for t in name_or_tokens if str(t).strip()]
        if len(tokens) == 0:
            return True
        if len(tokens) == 1 and (tokens[0] in GENERIC_NAMES or len(tokens[0]) <= 3):
            return True
        joined = "".join(tokens)
        if len(joined) <= 3:
            return True
        return False

    name = str(name_or_tokens).lower().strip()
    if "." in name:
        name = name.split(".")[-1]
    cleaned = name.replace("_", "").replace("-", "")
    return name in GENERIC_NAMES or len(cleaned) <= 3


def _extract_profile_metadata(profile: Optional[Dict[str, Any]]) -> Tuple[str, float, int]:
    """Extract dtype, null_ratio, and sample count from profile or value_profile."""
    if not profile or not isinstance(profile, dict):
        return "string", 0.0, 0

    vp = profile.get("value_profile") if isinstance(profile.get("value_profile"), dict) else profile
    dtype = str(vp.get("dtype") or profile.get("data_type") or "string").lower()
    null_ratio = float(vp.get("null_ratio", 0.0) or 0.0)
    samples = profile.get("sample_values") or vp.get("sample_values") or []
    sample_count = len(samples)
    return dtype, null_ratio, sample_count


def _are_dtypes_incompatible(dtype_a: str, dtype_b: str) -> bool:
    """Determine whether two data types are fundamentally incompatible."""
    if dtype_a == dtype_b:
        return False
    text_types = {"string", "categorical", "id_code"}
    # Two text subtypes can be partially compatible
    if dtype_a in text_types and dtype_b in text_types:
        return False
    # Numeric vs non-numeric, or Date vs non-date are incompatible
    return True


def compute_alpha(
    source_col: str,
    target_col: str,
    name_sim: float,
    value_sim: float,
    source_profile: Optional[Dict[str, Any]] = None,
    target_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, List[str]]:
    """Compute adaptive alpha in [0.15, 0.85] and effective value similarity with explanatory reasons.

    Rules:
    - Base alpha: 0.60 (semantic names are generally primary indicators in schemas)
    - Name informativeness: decrease alpha when either column name is short/generic
    - Value reliability: increase alpha when value samples are sparse or null-heavy
    - Agreement: increase alpha when name_sim is very high and dtypes agree
    - Disagreement: move alpha toward the more reliable signal
    - Dtype mismatch: applies a documented penalty to value_sim
    - False-friend suppression: generic names with incompatible dtypes clamp alpha to minimum (0.15)
    - Clamping: enforced to [0.15, 0.85] to guarantee neither signal is completely silenced.
    """
    reasons: List[str] = []
    alpha = 0.60

    src_dtype, src_null_ratio, src_samples = _extract_profile_metadata(source_profile)
    tgt_dtype, tgt_null_ratio, tgt_samples = _extract_profile_metadata(target_profile)

    src_tokens = source_profile.get("normalized_tokens") if source_profile else None
    tgt_tokens = target_profile.get("normalized_tokens") if target_profile else None

    src_generic = is_generic_token_or_name(src_tokens or source_col)
    tgt_generic = is_generic_token_or_name(tgt_tokens or target_col)

    # 1. Generic name penalty
    if src_generic and tgt_generic:
        alpha -= 0.30
        reasons.append("both column names are generic; relying primarily on value profile")
    elif src_generic or tgt_generic:
        alpha -= 0.15
        reasons.append("uninformative/generic column name detected; reduced name weight")

    # 2. Value sample reliability check
    val_unreliable = (
        src_null_ratio > 0.80
        or tgt_null_ratio > 0.80
        or (src_samples > 0 and src_samples < 3)
        or (tgt_samples > 0 and tgt_samples < 3)
    )
    if val_unreliable:
        alpha += 0.20
        reasons.append("sparse or high-null values; increased reliance on name signal")

    # 3. Dtype incompatibility penalty
    effective_value_sim = float(value_sim)
    dtypes_incompatible = _are_dtypes_incompatible(src_dtype, tgt_dtype)

    if dtypes_incompatible:
        effective_value_sim = float(value_sim * 0.15)
        reasons.append(f"data type mismatch ({src_dtype} vs {tgt_dtype}); penalized value similarity")

        # False-friend suppression: generic name + incompatible dtypes -> suppress name weight
        if src_generic or tgt_generic:
            alpha = 0.15
            reasons.append("generic name with incompatible data types; suppressed false-friend name weight")

    # 4. High name similarity + dtype agreement boost
    elif name_sim >= 0.80 and src_dtype == tgt_dtype:
        alpha += 0.10
        reasons.append(f"high semantic name match and matching {src_dtype} data type")

    # 5. Signal disagreement resolution
    disagreement = abs(name_sim - effective_value_sim)
    if disagreement > 0.45 and not dtypes_incompatible:
        if (src_generic or tgt_generic) and effective_value_sim > name_sim:
            alpha -= 0.10
            reasons.append("signals disagree; shifted toward value profile due to generic names")
        elif not src_generic and not tgt_generic and name_sim > effective_value_sim:
            alpha += 0.10
            reasons.append("signals disagree; shifted toward distinct semantic name match")
        else:
            reasons.append(f"signals disagree ({name_sim:.2f} name vs {effective_value_sim:.2f} value)")

    clamped_alpha = float(np.clip(alpha, 0.15, 0.85))
    return clamped_alpha, effective_value_sim, reasons


def fused_similarity(
    name_sim: float,
    value_sim: float,
    source_col: str,
    target_col: str,
    source_profile: Optional[Dict[str, Any]] = None,
    target_profile: Optional[Dict[str, Any]] = None,
) -> FusedMatchScore:
    """Compute fused similarity score returning full structured explanation."""
    alpha, eff_val_sim, reasons = compute_alpha(
        source_col=source_col,
        target_col=target_col,
        name_sim=name_sim,
        value_sim=value_sim,
        source_profile=source_profile,
        target_profile=target_profile,
    )

    fused_score = alpha * name_sim + (1.0 - alpha) * eff_val_sim
    clamped_fused = float(np.clip(fused_score, 0.0, 1.0))

    src_dtype, _, _ = _extract_profile_metadata(source_profile)
    tgt_dtype, _, _ = _extract_profile_metadata(target_profile)

    return FusedMatchScore(
        source_col=source_col,
        target_col=target_col,
        name_sim=round(float(name_sim), 4),
        value_sim=round(float(eff_val_sim), 4),
        alpha=round(float(alpha), 4),
        fused=round(clamped_fused, 4),
        reasons=reasons,
        source_dtype=src_dtype,
        target_dtype=tgt_dtype,
    )


def compute_fused_similarity_matrix(
    name_matrix: np.ndarray,
    value_matrix: np.ndarray,
    source_profiles: List[Dict[str, Any]],
    target_profiles: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[List[FusedMatchScore]]]:
    """Compute full (|A| x |B|) fused similarity matrix, alpha matrix, and detailed explanations grid."""
    n_src = len(source_profiles)
    n_tgt = len(target_profiles)

    fused_matrix = np.zeros((n_src, n_tgt), dtype=np.float32)
    alpha_matrix = np.zeros((n_src, n_tgt), dtype=np.float32)
    scores_grid: List[List[FusedMatchScore]] = []

    for i, p_src in enumerate(source_profiles):
        row_scores: List[FusedMatchScore] = []
        src_name = p_src.get("column_id") or p_src.get("raw_name") or f"src_{i}"

        for j, p_tgt in enumerate(target_profiles):
            tgt_name = p_tgt.get("column_id") or p_tgt.get("raw_name") or f"tgt_{j}"
            n_sim = float(name_matrix[i, j]) if i < name_matrix.shape[0] and j < name_matrix.shape[1] else 0.0
            v_sim = float(value_matrix[i, j]) if i < value_matrix.shape[0] and j < value_matrix.shape[1] else 0.0

            match_score = fused_similarity(
                name_sim=n_sim,
                value_sim=v_sim,
                source_col=src_name,
                target_col=tgt_name,
                source_profile=p_src,
                target_profile=p_tgt,
            )

            fused_matrix[i, j] = match_score.fused
            alpha_matrix[i, j] = match_score.alpha
            row_scores.append(match_score)

        scores_grid.append(row_scores)

    return fused_matrix, alpha_matrix, scores_grid