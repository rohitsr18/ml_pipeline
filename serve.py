import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from config import MODEL_PATH

app = FastAPI(title="ML Pipeline API", version="1.0")


# Define ehat input looks like
class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Load the model at startup
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(
        [
            [
                request.sepal_length,
                request.sepal_width,
                request.petal_length,
                request.petal_width,
            ]
        ]
    )
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return {
        "prediction": int(prediction),
        "class_name": CLASS_NAMES[prediction],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
