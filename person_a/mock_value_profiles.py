"""Profile generator using real sample data and the actual value profiler.

Builds realistic value profiles for tests without hardcoded vector indices.
"""

from typing import Any, Dict, List
from shared.contracts import ColumnProfile
from value_pipeline.profiler import build_value_profile

SAMPLE_DATASETS: Dict[str, List[Any]] = {
    "id": ["CUST-001", "CUST-002", "CUST-003", "CUST-004", "CUST-005"],
    "date": ["2023-01-15", "2023-03-22", "2023-06-01", "2023-09-18", "2023-11-30"],
    "amount": [49.99, 129.50, 9.99, 899.00, 24.50],
    "phone": ["+1-555-0192", "+1-555-0193", "+1-555-0194", "+1-555-0195", "+1-555-0196"],
    "email": ["user1@example.com", "user2@corp.org", "user3@mail.net", "user4@test.com", "user5@domain.io"],
    "categorical": ["ACTIVE", "ACTIVE", "PENDING", "ACTIVE", "INACTIVE"],
    "string": ["Quick brown fox", "Lorem ipsum dolor", "Database integration", "Semantic matching", "System admin"],
}


def create_sample_profile(column_name: str, sample_category: str = "id") -> Dict[str, Any]:
    """Build a valid ColumnProfile dictionary using real sample data and the actual value profiler."""
    samples = SAMPLE_DATASETS.get(sample_category, SAMPLE_DATASETS["string"])
    val_profile = build_value_profile(samples, total_count=len(samples))

    profile = ColumnProfile(
        column_id=f"test.{column_name}",
        raw_name=column_name,
        normalized_tokens=column_name.lower().split("_"),
        data_type=val_profile["dtype"],
        sample_values=samples,
        name_embedding=[],
        value_profile=val_profile,
    )
    return profile.to_dict()