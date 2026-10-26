"""Unit tests for column name normalization."""

from person_a.normalize_names import normalize_column_name


def test_customer_id():
    assert normalize_column_name("customer_id") == ["customer", "identifier"]


def test_customer_id_camel_case():
    assert normalize_column_name("CustomerID") == ["customer", "identifier"]


def test_customer_abbreviation():
    assert normalize_column_name("cust_id") == ["customer", "identifier"]


def test_order_quantity():
    assert normalize_column_name("order_qty") == ["order", "quantity"]


def test_date_of_birth():
    assert normalize_column_name("dob") == ["date", "birth"]


def test_product_id():
    assert normalize_column_name("productID") == ["product", "identifier"]


def test_pascal_case_and_acronyms():
    assert normalize_column_name("CustID") == ["customer", "identifier"]
    assert normalize_column_name("EmpDeptCode") == ["employee", "department", "code"]


def test_kebab_and_dotted():
    assert normalize_column_name("user-email-addr") == ["user", "email", "address"]
    assert normalize_column_name("db.orders.order_placed_dt") == ["order", "placed", "date"]


def test_stop_tokens():
    assert normalize_column_name("col_first_name") == ["first", "name"]
    assert normalize_column_name("tbl_customer_id") == ["customer", "identifier"]