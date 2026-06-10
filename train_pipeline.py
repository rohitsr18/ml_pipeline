import joblib
from sklearn.ensemble import RandomForestClassifier
from config import MODEL_PATH, RANDOM_STATE

def train_model(data):
    """Train a Random Forest Classifier"""

    model= RandomForestClassifier(
        n_estimators =100,
        max_depth =5,
        random_state = RANDOM_STATE
    )
    model.fit(data["X_train"], data["y_train"])
    print(f"[Train] Trained :{type(model).__name__}")
    return model

def save_model(model, scaler, feature_names):
    """Save model, scaler, metadata to disk"""

    artifact= {
        "model": model,
        "scaler":scaler,
        "feature_names": feature_names,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"[Train] Saved model to {MODEL_PATH}")

if __name__ =="__main__":
    from data_pipeline import load_data, preprocess

    df = load_data()
    data = preprocess(df)
    model = train_model(data)
    save_model(model, data["scaler"], data["feature_names"])
    print("[Train] Done ✓")