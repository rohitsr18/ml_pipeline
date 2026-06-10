from sklearn.metrics import accuracy_score, classification_report

ACCURACY_THRESHOLD = 0.9


def evaluate_model(model, data):
    """Evaluate the model and decide if it's good enough to deploy"""

    y_pred = model.predict(data["X_test"])
    accuracy = accuracy_score(data["y_test"], y_pred)

    report = classification_report(
        data["y_test"], y_pred, target_names=["setosa", "versicolor", "virginica"]
    )

    print(f"\n{'='*40}")
    print(f" Accuracy: {accuracy:.4f}")
    print(f"{'='*40}")
    print(report)

    passed = accuracy >= ACCURACY_THRESHOLD
    if passed:
        print(f"[Eval] PASSED ({accuracy:.4f}>={ACCURACY_THRESHOLD})")
    else:
        print(f"[Eval] FAILED ({accuracy:.4f}<{ACCURACY_THRESHOLD})")

    return {"accuracy": accuracy, "passed": passed}


if __name__ == "__main__":
    from data_pipeline import load_data, preprocess
    from train_pipeline import train_model

    df = load_data()
    data = preprocess(df)
    model = train_model(data)
    evaluate_model(model, data)
