<div align="center">

# 🛡️ FraudOps AI

### Enterprise Credit Card Fraud Detection with End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?style=for-the-badge&logo=pytorch)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Model_API-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-black?style=for-the-badge&logo=githubactions)

An enterprise-grade fraud detection platform that combines **Deep Learning**, **MLOps**, and **real-time monitoring** to detect fraudulent credit card transactions. The project demonstrates the complete machine learning lifecycle—from data versioning and experiment tracking to automated deployment, drift monitoring, and continuous retraining.

</div>

---

# 📌 Project Overview

Building an accurate fraud detection model is only the beginning. Production machine learning systems must continuously monitor data quality, track experiments, validate incoming data, detect concept drift, and safely deploy new model versions.

FraudOps AI implements an end-to-end MLOps pipeline that automates the complete lifecycle of a fraud detection system while ensuring reproducibility, reliability, and scalability.

---

# ✨ Key Features

- 💳 Credit Card Fraud Detection
- 🤖 PyTorch Deep Learning Model
- 📊 MLflow Experiment Tracking
- 📦 Model Registry & Versioning
- 📁 Data Version Control (DVC)
- ✅ Great Expectations Data Validation
- ⚡ FastAPI Inference API
- 🐳 Docker Containerization
- 🔄 GitHub Actions CI/CD
- 📈 Prometheus Metrics
- 📉 Grafana Dashboards
- 🚨 Evidently AI Drift Detection
- 🔁 Automatic Model Retraining

---

# 🏗️ System Architecture

```text
                  📂 Raw Transaction Dataset
                              │
                              ▼
           ┌──────────────────────────────┐
           │ Data Versioning (DVC + S3)   │
           └──────────────────────────────┘
                              │
                              ▼
      ┌─────────────────────────────────────────┐
      │ Great Expectations Data Validation      │
      └─────────────────────────────────────────┘
                              │
                              ▼
       ┌────────────────────────────────────┐
       │ PyTorch Model Training + MLflow    │
       └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────┐
         │ Model Evaluation (AUC ≥ 0.95)  │
         └────────────────────────────────┘
                              │
                              ▼
          ┌───────────────────────────────┐
          │ MLflow Model Registry         │
          └───────────────────────────────┘
                              │
                Staging ─────────► Production
                              │
                              ▼
        ┌────────────────────────────────────┐
        │ FastAPI Prediction Service         │
        └────────────────────────────────────┘
                    │                  │
                    ▼                  ▼
         Prometheus Metrics     Evidently AI
         + Grafana Dashboard    Drift Detection
                    │                  │
                    └──────────┬───────┘
                               ▼
                  Automatic Model Retraining
```

---

# 📂 Project Structure

```text
FraudOps-AI/
│
├── data/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── api/
│   └── main.py
│
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   └── evidently/
│
├── mlruns/
├── docker/
├── .github/workflows/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚙️ Technology Stack

### Programming

- Python

### Machine Learning

- PyTorch
- Scikit-learn

### MLOps

- MLflow
- DVC
- Great Expectations

### API

- FastAPI

### Monitoring

- Prometheus
- Grafana
- Evidently AI

### DevOps

- Docker
- GitHub Actions

---

# 📈 Pipeline Components

| Component | Technology | Purpose |
|------------|------------|---------|
| Data Versioning | DVC | Dataset reproducibility |
| Validation | Great Expectations | Data quality assurance |
| Training | PyTorch | Fraud prediction |
| Experiment Tracking | MLflow | Compare model runs |
| Model Registry | MLflow Registry | Version management |
| Deployment | FastAPI | Online inference |
| Monitoring | Prometheus & Grafana | Production metrics |
| Drift Detection | Evidently AI | Detect data distribution changes |
| CI/CD | GitHub Actions | Automated testing & deployment |

---

# 🚀 Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train.py
```

### Start the API

```bash
uvicorn api.main:app --reload
```

### Launch Monitoring

```bash
docker compose up
```

---

# 📊 Production Metrics

- ✅ AUC-ROC > 0.95
- 📈 Precision
- 📉 Recall
- 🎯 F1 Score
- 🚨 Fraud Detection Rate
- 📊 Feature Drift Score
- ⚡ API Latency
- 💾 Model Version History

---

# 🌍 Real-World Applications

- Banking & Financial Services
- Digital Payment Platforms
- FinTech Solutions
- Online Transaction Monitoring
- Risk Management Systems
- Enterprise Fraud Prevention

---

# 🔮 Future Improvements

- Kubernetes deployment
- Real-time Kafka streaming
- Feature Store integration
- SHAP explainability
- Online learning pipeline
- Auto-scaling inference service
- Multi-model ensemble serving

---

# 📜 License

This project is licensed under the MIT License.
