"""
Column name normalization.

Converts database column names such as:
    customer_id
    CustomerID
    cust_no
    DOB
into meaningful normalized tokens.
"""

import re


ABBREVIATIONS = {
    "cust": ["customer"],
    "customer": ["customer"],
    "id": ["identifier"],
    "no": ["number"],
    "num": ["number"],
    "nbr": ["number"],
    "dob": ["date", "birth"],
    "addr": ["address"],
    "qty": ["quantity"],
    "amt": ["amount"],
    "yr": ["year"],
    "yrs": ["years"],
    "mth": ["month"],
    "mo": ["month"],
    "tel": ["telephone"],
    "ph": ["phone"],
    "sts": ["status"],
    "stat": ["status"],
    "ord": ["order"],
    "prod": ["product"],
    "acct": ["account"],
    "acc": ["account"],
    "desc": ["description"],
    "dt": ["date"],
    "ts": ["timestamp"],
    "fname": ["first", "name"],
    "lname": ["last", "name"]
}


def tokenize_column_name(name: str):
    """
    Convert different naming styles into tokens.
    """

    if not isinstance(name, str):
        raise TypeError("Column name must be a string")

    name = name.strip()

    # Replace separators with spaces
    name = re.sub(r"[_\-\.\s]+", " ", name)

    # Split camelCase:
    # customerId -> customer Id
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)

    # Split acronym followed by normal word:
    # CUSTOMERID -> CUSTOMER ID
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)

    # Convert to lowercase
    name = name.lower()

    # Remove unwanted characters
    name = re.sub(r"[^a-z0-9 ]", " ", name)

    # Split into tokens
    tokens = name.split()

    return tokens


def expand_abbreviation(token: str):
    """
    Expand one abbreviation.
    """

    return ABBREVIATIONS.get(token, [token])


def normalize_column_name(name: str):
    """
    Full normalization pipeline.
    """

    tokens = tokenize_column_name(name)

    expanded_tokens = []

    for token in tokens:
        expanded_tokens.extend(expand_abbreviation(token))

    # Remove duplicate tokens while preserving order
    result = []
    seen = set()

    for token in expanded_tokens:
        if token not in seen:
            result.append(token)
            seen.add(token)

    return result


if __name__ == "__main__":

    test_names = [
        "customer_id",
        "CustomerID",
        "CUSTOMER_ID",
        "cust_no",
        "order_qty",
        "addr_line_1",
        "dob",
        "phone_number",
        "productID",
        "order_date",
        "accountNumber"
    ]

    print("=" * 60)
    print("COLUMN NAME NORMALIZATION")
    print("=" * 60)

    for name in test_names:
        result = normalize_column_name(name)
        print(f"{name:20s} -> {result}")