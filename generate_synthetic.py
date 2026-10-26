"""Synthetic benchmark schema generator for schema matching evaluation.

Generates 5 realistic schema pairs of increasing difficulty:
1. schema_pair_1: Easy - Abbreviations and case variations
2. schema_pair_2: Medium - Synonyms and terminology changes
3. schema_pair_3: Hard - Generic names (id, code, val) where names mislead and value profiles decide
4. schema_pair_4: Hard - False friend columns with identical names but conflicting semantics/dtypes
5. schema_pair_5: Real-World - Asymmetric tables with unmatched columns, null-heavy fields, and reordered columns

Emits CSV files into data/test_schemas/schema_pair_*/:
  - source.csv
  - target.csv
  - ground_truth.csv (columns: source_column, target_column)
And JSON representations in data/synthetic/pair_*.json for legacy test compatibility.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import random
import pandas as pd
import numpy as np

from value_pipeline.profiler import build_value_profile
from person_a.normalize_names import normalize_column_name
from person_a.name_embedder import get_name_embedder

REPO_ROOT = Path(__file__).resolve().parent
TEST_SCHEMAS_DIR = REPO_ROOT / "data" / "test_schemas"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"


def build_pair_profile(df: pd.DataFrame, table_prefix: str) -> List[Dict[str, Any]]:
    """Build ColumnProfile objects for all columns in a DataFrame."""
    embedder = get_name_embedder()
    profiles: List[Dict[str, Any]] = []

    for col in df.columns:
        raw_values = df[col].tolist()
        val_profile = build_value_profile(raw_values, total_count=len(raw_values))
        tokens = normalize_column_name(str(col))
        emb = embedder.embed_tokens(tokens)

        col_id = f"{table_prefix}.{col}"
        profiles.append({
            "column_id": col_id,
            "raw_name": str(col),
            "normalized_tokens": tokens,
            "data_type": val_profile["dtype"],
            "sample_values": [v for v in raw_values[:15] if pd.notna(v)],
            "name_embedding": [float(x) for x in emb],
            "value_profile": val_profile,
        })
    return profiles


def generate_all_datasets() -> List[Dict[str, Any]]:
    random.seed(42)
    np.random.seed(42)

    n_rows = 60
    pairs: List[Dict[str, Any]] = []

    # =========================================================================
    # Pair 1: Easy - Abbreviations and case variations
    # =========================================================================
    c_ids = [f"CUST-{1000 + i}" for i in range(n_rows)]
    fnames = random.choices(["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda"], k=n_rows)
    dobs = [f"{random.randint(1970, 2002):04d}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(n_rows)]
    phones = [f"+1-555-01{i:02d}" for i in range(n_rows)]
    qtys = [random.randint(1, 25) for _ in range(n_rows)]
    amounts = [round(float(q * random.uniform(15.0, 95.0)), 2) for q in qtys]

    df1_src = pd.DataFrame({
        "cust_id": c_ids,
        "emp_fname": fnames,
        "dob": dobs,
        "phone_num": phones,
        "order_qty": qtys,
        "total_amt": amounts,
    })
    df1_tgt = pd.DataFrame({
        "customer_id": c_ids,
        "employee_first_name": fnames,
        "date_of_birth": dobs,
        "telephone_number": phones,
        "order_quantity": qtys,
        "total_amount": amounts,
    })
    gt1 = [
        ("cust_id", "customer_id"),
        ("emp_fname", "employee_first_name"),
        ("dob", "date_of_birth"),
        ("phone_num", "telephone_number"),
        ("order_qty", "order_quantity"),
        ("total_amt", "total_amount"),
    ]
    pairs.append({
        "pair_id": "schema_pair_1",
        "description": "Easy: Abbreviations and naming style variations",
        "df_source": df1_src,
        "df_target": df1_tgt,
        "ground_truth": gt1,
    })

    # =========================================================================
    # Pair 2: Medium - Synonyms and domain terminology
    # =========================================================================
    client_codes = [f"CLT-{2000 + i}" for i in range(n_rows)]
    emails = [f"user_{i}@enterprise-{random.choice(['corp', 'net', 'org'])}.com" for i in range(n_rows)]
    signup_dates = [f"202{random.randint(0, 4)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(n_rows)]
    credit_scores = [float(random.randint(580, 840)) for _ in range(n_rows)]
    addresses = [f"{random.randint(100, 999)} {random.choice(['Maple', 'Oak', 'Pine', 'Cedar'])} St" for _ in range(n_rows)]
    balances = [round(random.uniform(50.0, 8500.0), 2) for _ in range(n_rows)]

    df2_src = pd.DataFrame({
        "client_code": client_codes,
        "contact_mail": emails,
        "signup_date": signup_dates,
        "credit_rating": credit_scores,
        "billing_address": addresses,
        "account_balance": balances,
    })
    df2_tgt = pd.DataFrame({
        "customer_identifier": client_codes,
        "user_email": emails,
        "registration_date": signup_dates,
        "credit_score": credit_scores,
        "residential_address": addresses,
        "outstanding_balance": balances,
    })
    gt2 = [
        ("client_code", "customer_identifier"),
        ("contact_mail", "user_email"),
        ("signup_date", "registration_date"),
        ("credit_rating", "credit_score"),
        ("billing_address", "residential_address"),
        ("account_balance", "outstanding_balance"),
    ]
    pairs.append({
        "pair_id": "schema_pair_2",
        "description": "Medium: Synonyms and domain terminology mapping",
        "df_source": df2_src,
        "df_target": df2_tgt,
        "ground_truth": gt2,
    })

    # =========================================================================
    # Pair 3: Hard - Generic names where names mislead and values decide
    # =========================================================================
    # Target columns are intentionally named with inverted generic labels (code <-> id, label <-> name)
    states = random.choices(["CA", "NY", "TX", "WA", "FL", "IL", "CO", "MA"], k=n_rows)
    companies = [f"Enterprise Vendor {i}" for i in range(n_rows)]
    item_ids = [f"ITEM-{5000 + i}" for i in range(n_rows)]
    tx_dates = [f"2024-0{random.randint(1, 9):01d}-{random.randint(10, 28):02d}" for _ in range(n_rows)]
    numeric_vals = [round(float(random.uniform(10.0, 500.0)), 2) for _ in range(n_rows)]

    df3_src = pd.DataFrame({
        "id": item_ids,          # Matches target 'code' by value
        "name": companies,       # Matches target 'label' by value
        "date": tx_dates,        # Matches target 'date'
        "val": numeric_vals,     # Matches target 'value'
        "code": states,          # Matches target 'id' by value (state codes)
    })
    df3_tgt = pd.DataFrame({
        "code": item_ids,        # Contains ITEM-xxxx (matches source id)
        "id": states,            # Contains CA, NY (matches source code)
        "value": numeric_vals,   # Matches source val
        "date": tx_dates,        # Matches source date
        "label": companies,      # Matches source name
    })
    gt3 = [
        ("id", "code"),
        ("name", "label"),
        ("date", "date"),
        ("val", "value"),
        ("code", "id"),
    ]
    pairs.append({
        "pair_id": "schema_pair_3",
        "description": "Hard: Generic names where name signal misleads and values must decide",
        "df_source": df3_src,
        "df_target": df3_tgt,
        "ground_truth": gt3,
    })

    # =========================================================================
    # Pair 4: Hard - Identical names with conflicting semantics/types
    # =========================================================================
    statuses_str = random.choices(["ACTIVE", "PENDING", "INACTIVE", "SUSPENDED"], k=n_rows)
    http_statuses_int = random.choices([200, 201, 400, 404, 500], k=n_rows)
    exam_scores = [round(float(random.uniform(65.0, 99.0)), 1) for _ in range(n_rows)]
    age_ints = [float(random.randint(18, 75)) for _ in range(n_rows)]
    birth_dates = [f"19{random.randint(50, 99):02d}-0{random.randint(1, 9):01d}-15" for _ in range(n_rows)]
    zip_codes = [f"{random.randint(10000, 99999)}" for _ in range(n_rows)]
    user_notes = [f"Customer feedback ticket reference #{100 + i}" for i in range(n_rows)]

    df4_src = pd.DataFrame({
        "status": statuses_str,      # String status (NOT matching int status)
        "score": exam_scores,        # True match with score
        "age": age_ints,             # Int age (NOT matching date age)
        "zip": zip_codes,            # Matches postal_code
        "notes": user_notes,         # Matches commentary
    })
    df4_tgt = pd.DataFrame({
        "status": http_statuses_int, # Int HTTP code
        "score": exam_scores,        # True match
        "age": birth_dates,          # Date strings
        "postal_code": zip_codes,    # True match
        "commentary": user_notes,    # True match
    })
    gt4 = [
        ("score", "score"),
        ("zip", "postal_code"),
        ("notes", "commentary"),
    ]
    pairs.append({
        "pair_id": "schema_pair_4",
        "description": "Hard: False friend columns with identical names but conflicting types/distributions",
        "df_source": df4_src,
        "df_target": df4_tgt,
        "ground_truth": gt4,
    })

    # =========================================================================
    # Pair 5: Real-World Enterprise - Asymmetric columns, null-heavy fields
    # =========================================================================
    emp_ids = [f"EMP-{8000 + i}" for i in range(n_rows)]
    full_names = [f"Employee {i}" for i in range(n_rows)]
    work_emails = [f"staff.{i}@corporate-hq.com" for i in range(n_rows)]
    depts = random.choices(["Engineering", "Finance", "Legal", "Operations", "Sales"], k=n_rows)
    hire_dates = [f"201{random.randint(2, 9)}-0{random.randint(1, 9):01d}-01" for _ in range(n_rows)]
    salaries = [float(random.randint(55000, 160000)) for _ in range(n_rows)]
    ssns = [f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for _ in range(n_rows)]

    # Source has 80% null fax number and a temporary notes column that has no target counterpart
    fax_numbers = [f"+1-800-FAX-{i}" if random.random() < 0.2 else None for i in range(n_rows)]
    temp_notes = [f"Internal audit note {i}" for i in range(n_rows)]

    # Target has office phone, emergency contact (unmatched), and timestamp (unmatched)
    office_phones = [f"+1-202-555-01{i:02d}" for i in range(n_rows)]
    emergency_contacts = [f"Contact Person {i}" for i in range(n_rows)]
    audit_timestamps = [f"2026-03-01T12:{i % 60:02d}:00Z" for i in range(n_rows)]

    df5_src = pd.DataFrame({
        "emp_id": emp_ids,
        "full_name": full_names,
        "work_email": work_emails,
        "dept_name": depts,
        "hire_date": hire_dates,
        "salary": salaries,
        "tax_id": ssns,
        "fax_number": fax_numbers,
        "temp_notes": temp_notes,
    })
    # Target reordered and has unmatched columns
    df5_tgt = pd.DataFrame({
        "ssn": ssns,
        "employee_code": emp_ids,
        "staff_name": full_names,
        "department": depts,
        "compensation": salaries,
        "start_date": hire_dates,
        "office_phone": office_phones,
        "corporate_email": work_emails,
        "emergency_contact": emergency_contacts,
        "audit_timestamp": audit_timestamps,
    })
    gt5 = [
        ("emp_id", "employee_code"),
        ("full_name", "staff_name"),
        ("work_email", "corporate_email"),
        ("dept_name", "department"),
        ("hire_date", "start_date"),
        ("salary", "compensation"),
        ("tax_id", "ssn"),
    ]
    pairs.append({
        "pair_id": "schema_pair_5",
        "description": "Real-World: Asymmetric schemas, reordered columns, nulls, and unmatched fields",
        "df_source": df5_src,
        "df_target": df5_tgt,
        "ground_truth": gt5,
    })

    return pairs


def save_datasets() -> None:
    """Save datasets into data/test_schemas/ and legacy data/synthetic/."""
    pairs = generate_all_datasets()
    TEST_SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs, start=1):
        pair_id = f"schema_pair_{i}"
        pair_dir = TEST_SCHEMAS_DIR / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        df_src: pd.DataFrame = pair["df_source"]
        df_tgt: pd.DataFrame = pair["df_target"]
        gt_list: List[Tuple[str, str]] = pair["ground_truth"]

        # 1. Save CSVs in test_schemas
        src_path = pair_dir / "source.csv"
        tgt_path = pair_dir / "target.csv"
        gt_path = pair_dir / "ground_truth.csv"

        df_src.to_csv(src_path, index=False)
        df_tgt.to_csv(tgt_path, index=False)

        # Save ground truth in clean (source_column, target_column) format
        gt_df = pd.DataFrame(gt_list, columns=["source_column", "target_column"])
        gt_df.to_csv(gt_path, index=False)

        # 2. Save JSON in data/synthetic for benchmark harness
        src_profiles = build_pair_profile(df_src, "Source")
        tgt_profiles = build_pair_profile(df_tgt, "Target")

        json_gt = [(f"Source.{src_col}", f"Target.{tgt_col}") for src_col, tgt_col in gt_list]
        synthetic_payload = {
            "pair_id": f"pair_{i}",
            "description": pair["description"],
            "profiles_a": src_profiles,
            "profiles_b": tgt_profiles,
            "ground_truth": json_gt,
        }

        json_path = SYNTHETIC_DIR / f"pair_{i}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(synthetic_payload, f, indent=2)

        print(f"Generated {pair_id}: {df_src.shape[1]}x{df_tgt.shape[1]} cols, {len(gt_list)} ground-truth matches.")


if __name__ == "__main__":
    save_datasets()
