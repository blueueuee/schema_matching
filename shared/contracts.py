"""Shared data contracts and validation for column profiles and match results."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from jsonschema import ValidationError, validate

SCHEMA_PATH = Path(__file__).resolve().parent / "column_profile_schema.json"


def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_column_profile(profile: Dict[str, Any]) -> bool:
    """Validate a profile dictionary against shared JSON schema."""
    schema = load_schema()
    validate(instance=profile, schema=schema)
    return True


# Alias for concise import
validate_profile = validate_column_profile


@dataclass
class ColumnProfile:
    column_id: str
    raw_name: str
    normalized_tokens: List[str]
    data_type: str
    sample_values: List[Any]
    name_embedding: List[float] = field(default_factory=list)
    value_profile: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_id": self.column_id,
            "raw_name": self.raw_name,
            "normalized_tokens": list(self.normalized_tokens),
            "data_type": self.data_type,
            "sample_values": list(self.sample_values),
            "name_embedding": [float(x) for x in self.name_embedding],
            "value_profile": self.value_profile,
        }

    def validate(self) -> bool:
        return validate_profile(self.to_dict())


@dataclass
class MatchExplanation:
    source_column: str
    target_column: str
    name_similarity: float
    value_similarity: float
    alpha: float
    fused_score: float
    reasons: List[str]
    source_dtype: str = "unknown"
    target_dtype: str = "unknown"

    @property
    def explanation_text(self) -> str:
        base = (
            f"Score {self.fused_score:.3f} = {self.alpha:.2f} * {self.name_similarity:.3f} (name) + "
            f"{1.0 - self.alpha:.2f} * {self.value_similarity:.3f} (value)."
        )
        if self.reasons:
            return base + " Adjustments: " + "; ".join(self.reasons) + "."
        return base


@dataclass
class Match:
    column_a: str
    column_b: str
    score: float
    accepted: bool
    name_similarity: float = 0.0
    value_similarity: float = 0.0
    alpha: float = 0.5
    explanation: Optional[MatchExplanation] = None

    def __getitem__(self, item: str) -> Any:
        if item == "column_a":
            return self.column_a
        if item == "column_b":
            return self.column_b
        if item == "score":
            return self.score
        if item == "accepted":
            return self.accepted
        if item == "name_similarity":
            return self.name_similarity
        if item == "value_similarity":
            return self.value_similarity
        if item == "alpha":
            return self.alpha
        if item == "explanation":
            return self.explanation.explanation_text if self.explanation else ""
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_a": self.column_a,
            "column_b": self.column_b,
            "score": round(float(self.score), 4),
            "accepted": bool(self.accepted),
            "name_similarity": round(float(self.name_similarity), 4),
            "value_similarity": round(float(self.value_similarity), 4),
            "alpha": round(float(self.alpha), 4),
            "explanation": self.explanation.explanation_text if self.explanation else "",
        }