import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE

def load_data():
    """Load the raw data into a DataFrame"""

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns= iris.feature_names)
    df["target"]= iris.target
    df.to_csv(DATA_DIR / "raw_data.csv", index=False)
    print(f"[Data] Loaded {len(df)} rows")
    return df

def preprocess(df):
    """Split and Scale the data"""

    X= df.drop("target", axis=1)
    y= df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
     X,y,test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[Data] Train:{len(X_train)} | Test:{len(X_test)}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }

if __name__ == "__main__":
    df = load_data()
    data= preprocess(df)
    print("[Data] Done ✓")
