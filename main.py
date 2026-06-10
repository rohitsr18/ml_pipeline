from data_pipeline import load_data, preprocess
from train_pipeline import train_model, save_model
from evaluate import evaluate_model


def run_pipeline():
    """Run the full ML pipeline: load-> preprocess->train->evaluate->save"""

    print("\n=== Starting ML Pipeline===\n")

    # Step 1: Load and Preprocess the data
    df = load_data()
    data = preprocess(df)

    # Step 2: Train the model
    model = train_model(data)

    # Step 3: Evaluate the model
    results = evaluate_model(model, data)

    # Step 4: Save the model if it passed the evaluation
    if results["passed"]:
        save_model(model, data["scaler"], data["feature_names"])
        print("\n=== Pipeline Completed Successfully ===\n")
    else:
        print("\n=== Pipeline Failed: Model did not meet accuracy threshold ===\n")


if __name__ == "__main__":
    run_pipeline()
