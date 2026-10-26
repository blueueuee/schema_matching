"""Column name normalization and tokenization.

Splits identifiers (snake_case, camelCase, PascalCase, kebab, dotted, digits, acronyms),
expands abbreviations, strips noise tokens, and returns clean normalized token lists.
"""

from typing import List, Set
import re

ABBREVIATIONS = {
    "cust": ["customer"],
    "customer": ["customer"],
    "client": ["client"],
    "id": ["identifier"],
    "identifier": ["identifier"],
    "no": ["number"],
    "num": ["number"],
    "nbr": ["number"],
    "dob": ["date", "birth"],
    "addr": ["address"],
    "address": ["address"],
    "qty": ["quantity"],
    "amt": ["amount"],
    "yr": ["year"],
    "yrs": ["year"],
    "mth": ["month"],
    "mo": ["month"],
    "tel": ["telephone"],
    "ph": ["phone"],
    "phone": ["phone"],
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
    "lname": ["last", "name"],
    "mname": ["middle", "name"],
    "emp": ["employee"],
    "dept": ["department"],
    "pct": ["percent"],
    "bal": ["balance"],
    "cat": ["category"],
    "loc": ["location"],
    "src": ["source"],
    "tgt": ["target"],
    "curr": ["currency"],
    "cd": ["code"],
    "ref": ["reference"],
    "txn": ["transaction"],
    "tx": ["transaction"],
    "org": ["organization"],
    "cnt": ["country"],
    "cntry": ["country"],
    "zip": ["postal", "code"],
    "postal": ["postal", "code"],
    "avg": ["average"],
    "min": ["minimum"],
    "max": ["maximum"],
    "std": ["standard"],
    "val": ["value"],
    "attr": ["attribute"],
    "reg": ["registration"],
    "grp": ["group"],
    "pos": ["position"],
    "idx": ["index"],
    "seq": ["sequence"],
    "msg": ["message"],
    "cfg": ["config"],
}

# Generic noise prefixes/suffixes that add zero semantic value when combined with other words
STOP_TOKENS: Set[str] = {"tbl", "col", "fld", "column", "field", "table"}


def tokenize_column_name(name: str) -> List[str]:
    """Tokenize a column name across naming conventions into individual words."""
    if not isinstance(name, str):
        raise TypeError("Column name must be a string")

    name = name.strip()
    if not name:
        return []

    # If qualified identifier (e.g. DB1.Users.user_id), take the leaf column name
    if "." in name and not name.startswith("."):
        name = name.split(".")[-1]

    # Split kebab, underscores, slashes, and periods into spaces
    name = re.sub(r"[_\-\./\\]+", " ", name)

    # Split acronyms preceding title-case word: "CustID" -> "Cust ID", "NASAProject" -> "NASA Project"
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)

    # Split camelCase and digits: "customerId" -> "customer Id", "order2Date" -> "order 2 Date"
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    name = re.sub(r"([a-zA-Z])([0-9]+)", r"\1 \2", name)
    name = re.sub(r"([0-9]+)([a-zA-Z])", r"\1 \2", name)

    # Lowercase and clean non-alphanumeric
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", " ", name)

    tokens = [t for t in name.split() if t]
    return tokens


def expand_abbreviation(token: str) -> List[str]:
    """Expand a single token using the abbreviation dictionary."""
    return ABBREVIATIONS.get(token, [token])


def normalize_column_name(name: str) -> List[str]:
    """Full normalization pipeline with token splitting, expansion, and stop-token stripping."""
    tokens = tokenize_column_name(name)
    if not tokens:
        return []

    expanded: List[str] = []
    for token in tokens:
        expanded.extend(expand_abbreviation(token))

    # If more than one token, drop noise tokens like 'col' or 'tbl'
    if len(expanded) > 1:
        filtered = [t for t in expanded if t not in STOP_TOKENS]
        if filtered:
            expanded = filtered

    # Deduplicate tokens while preserving left-to-right order
    result: List[str] = []
    seen: Set[str] = set()
    for token in expanded:
        if token not in seen:
            result.append(token)
            seen.add(token)

    return result