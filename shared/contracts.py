import json
from pathlib import Path
from jsonschema import validate


SCHEMA_PATH = Path(__file__).parent / "column_profile_schema.json"


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_column_profile(profile):
    schema = load_schema()
    validate(instance=profile, schema=schema)
    return True


if __name__ == "__main__":
    test_profile = {
        "column_id": "test.customer.customer_id",
        "raw_name": "cust_id",
        "normalized_tokens": ["customer", "identifier"],
        "data_type": "string",
        "sample_values": ["C001", "C002"],
        "name_embedding": [0.1, 0.2, 0.3]
    }

    try:
        validate_column_profile(test_profile)
        print("✓ Column profile is valid")
    except Exception as e:
        print("✗ Validation failed")
        print(e)