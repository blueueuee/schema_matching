from normalize_names import normalize_column_name


def test_customer_id():
    assert normalize_column_name("customer_id") == [
        "customer",
        "identifier"
    ]


def test_customer_id_camel_case():
    assert normalize_column_name("CustomerID") == [
        "customer",
        "identifier"
    ]


def test_customer_abbreviation():
    assert normalize_column_name("cust_id") == [
        "customer",
        "identifier"
    ]


def test_order_quantity():
    assert normalize_column_name("order_qty") == [
        "order",
        "quantity"
    ]


def test_date_of_birth():
    assert normalize_column_name("dob") == [
        "date",
        "birth"
    ]


def test_product_id():
    assert normalize_column_name("productID") == [
        "product",
        "identifier"
    ]


if __name__ == "__main__":
    print("Running normalization tests...")

    test_customer_id()
    test_customer_id_camel_case()
    test_customer_abbreviation()
    test_order_quantity()
    test_date_of_birth()
    test_product_id()

    print("✓ All normalization tests passed")