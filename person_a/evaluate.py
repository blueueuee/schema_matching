def evaluate_matches(
    predicted,
    ground_truth
):

    predicted = set(predicted)
    ground_truth = set(ground_truth)

    true_positive = len(
        predicted & ground_truth
    )

    false_positive = len(
        predicted - ground_truth
    )

    false_negative = len(
        ground_truth - predicted
    )

    precision = (
        true_positive
        /
        (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0
    )

    recall = (
        true_positive
        /
        (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative
    }


if __name__ == "__main__":

    ground_truth = {
        ("customer_id", "cust_identifier"),
        ("customer_name", "full_customer_name"),
        ("dob", "date_of_birth"),
        ("phone_no", "telephone_number"),
        ("order_amount", "total_amt")
    }

    predicted = {
        ("customer_id", "cust_identifier"),
        ("customer_name", "full_customer_name"),
        ("dob", "date_of_birth"),
        ("phone_no", "telephone_number"),
        ("order_amount", "total_amt")
    }

    results = evaluate_matches(
        predicted,
        ground_truth
    )

    print(
        f"Precision: {results['precision']:.2%}"
    )

    print(
        f"Recall: {results['recall']:.2%}"
    )

    print(
        f"F1: {results['f1']:.2%}"
    )