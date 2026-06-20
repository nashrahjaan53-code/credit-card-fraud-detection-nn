# Credit Card Fraud Detection — MLOps Edition

A production-grade fraud detection system built with PyTorch, wrapped in a full MLOps
pipeline: experiment tracking, CI/CD quality gates, containerised serving, and live
drift monitoring.

## Architecture

```
Raw data (DVC)
     │
     ▼
Data validation (Great Expectations)
     │
     ▼
Model training (PyTorch + MLflow tracking)
     │
  Quality gate — AUC ≥ 0.95
     │
     ▼
MLflow Model Registry  ──►  Staging  ──►  Production
                                │
                         Smoke tests pass
                                │
                                ▼
                     FastAPI serving endpoint
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              Prometheus               Evidently AI
              + Grafana             (drift detection)
              (live metrics)        (weekly batch job)
                                            │
                              drift > 20%? ─┘
                                            │
                                            ▼
                                   Retraining triggered
```

## Key MLOps features

| Feature | Tool | Why it matters |
|---|---|---|
| Data versioning | DVC + S3 | Reproducible experiments |
| Data validation | Great Expectations | Catches schema drift before training |
| Experiment tracking | MLflow | Every run logged — params, metrics, artifacts |
| Model registry | MLflow Registry | Staging → Production promotion gate |
| CI/CD | GitHub Actions | Auto-trains on push, blocks bad models |
| Quality gate | AUC ≥ 0.95 check | Never ships a degraded model |
| Containerisation | Docker + ECR | Consistent deployments |
| Serving | FastAPI + uvicorn | Sub-10ms p99 latency |
| Monitoring | Prometheus + Grafana | Real-time prediction metrics |
| Drift detection | Evidently AI | Weekly batch job, auto-triggers retrain |

## Model performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.978 |
| AUPRC | 0.891 |
| F1 Score | 0.884 |
| Precision | 0.901 |
| Recall | 0.867 |

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
— 284,807 transactions, 0.17% fraud rate (severely imbalanced, handled with class weights).

## CI/CD pipeline

Every push to `main` runs:

```
lint → unit tests → data validation → train → quality gate → docker build → deploy staging → smoke tests → deploy production (manual approve)
```

The quality gate at step 5 blocks the pipeline if AUC drops below 0.95 — the model
never reaches production without meeting the bar.

## Running locally

```bash
# Clone and install
git clone https://github.com/nashrahjaan53-code/credit-card-fraud-detection-nn
cd credit-card-fraud-detection-nn
pip install -r requirements.txt

# Pull data
dvc pull

# Train (logs to MLflow)
python src/train.py

# View experiments
mlflow ui  # open http://localhost:5000

# Start full stack (API + MLflow + Prometheus + Grafana)
docker-compose up

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
                     0.14, -0.33, -0.17, -0.45, -0.06, -0.22, 0.00, 0.00,
                     0.06, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07,
                    -0.41, 0.06, 0.13, -0.03, 1.79],
       "transaction_id": "txn-001"}'
```

## Project structure

```
├── src/
│   ├── train.py          # MLflow-tracked training script
│   └── serve.py          # FastAPI inference endpoint
├── scripts/
│   ├── validate_data.py  # Great Expectations gate
│   ├── promote_model.py  # MLflow registry promotion
│   └── smoke_test.py     # Post-deploy validation
├── monitoring/
│   ├── drift_detector.py # Evidently drift job
│   └── prometheus.yml    # Metrics scrape config
├── .github/workflows/
│   └── ml-pipeline.yml   # Full CI/CD pipeline
├── Dockerfile
├── docker-compose.yml    # Local dev stack
└── dvc.yaml              # Data pipeline definition
```

## Monitoring dashboards

Grafana (`:3000`) tracks:
- Prediction volume per minute
- Fraud rate trend
- Risk score distribution
- p50 / p95 / p99 prediction latency
- Feature drift score (weekly)