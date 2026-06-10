# ML Pipeline — End-to-End Deployment

A simple machine learning pipeline that trains a model on the Iris dataset and serves predictions via a REST API. Deployed using Docker and Kubernetes on GCP.

## What it does

1. Loads the Iris dataset and splits it into train/test sets
2. Scales features using StandardScaler
3. Trains a Random Forest classifier
4. Evaluates accuracy — only saves the model if it passes a quality threshold
5. Serves predictions through a FastAPI endpoint

## Project structure

```
ml_pipeline/
├── config.py            # Paths and settings
├── data_pipeline.py     # Data loading and preprocessing
├── train_pipeline.py    # Model training and saving
├── evaluate.py          # Model evaluation and quality gate
├── serve.py             # FastAPI prediction server
├── main.py              # Runs the full pipeline end-to-end
├── Dockerfile
├── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── data/
└── models/
```

## Running locally

```bash
pip install -r requirements.txt
python main.py           # trains and evaluates the model
python serve.py          # starts the API on port 8000
```

## Testing the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Returns something like:
```json
{"prediction": 0, "class_name": "setosa"}
```

## Docker

```bash
docker build -t ml-pipeline:1.0 .
docker run -p 8000:8000 ml-pipeline:1.0
```

## Deploying to GCP (GKE)

Push the image to Artifact Registry:

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-pipeline-repo/ml-pipeline:1.0
```

Create a cluster and deploy:

```bash
gcloud container clusters create ml-pipeline-cluster --zone=us-central1-a --num-nodes=2 --machine-type=e2-small
gcloud container clusters get-credentials ml-pipeline-cluster --zone=us-central1-a
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Get the external IP:

```bash
kubectl get svc ml-pipeline-service
```

## Cleanup

To avoid charges when you're done:

```bash
gcloud container clusters delete ml-pipeline-cluster --zone=us-central1-a
gcloud artifacts repositories delete ml-pipeline-repo --location=us-central1
```

## Notes

- The model is baked into the Docker image at build time (the Dockerfile runs `python main.py`)
- The quality gate in `evaluate.py` requires >= 90% accuracy before saving
- The `/health` endpoint is used by Kubernetes liveness probes to restart unhealthy pods
