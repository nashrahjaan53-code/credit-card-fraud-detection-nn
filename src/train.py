import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import json
import os
import logging

from src.model import FraudClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Training logic ─────────────────────────────────────────────────────────────
def train(
    data_path: str = "data/creditcard.csv",
    n_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    experiment_name: str = "fraud-detection",
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log all hyperparameters
        params = dict(n_epochs=n_epochs, batch_size=batch_size, lr=lr)
        mlflow.log_params(params)
        mlflow.set_tags({
            "model_type": "PyTorch-NN",
            "dataset": "creditcard-kaggle",
            "git_commit": os.getenv("GIT_SHA", "local"),
        })

        # ── Data ──────────────────────────────────────────────────────────────
        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path)

        # Drop 'Time' (irrelevant) to match data_loader.py and serve.py (29 features)
        X = df.drop(columns=["Class", "Time"]).values
        y = df["Class"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Save scaler as artifact
        import joblib
        import tempfile
        import pathlib
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(scaler, "artifacts/scaler.pkl")
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = pathlib.Path(tmp) / "scaler.pkl"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessor")

        # Log dataset stats
        mlflow.log_metrics({
            "train_size": len(X_train),
            "test_size":  len(X_test),
            "fraud_rate_pct": round(y.mean() * 100, 4),
        })

        # ── Class weights (handles imbalance) ─────────────────────────────────
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        pos_weight = torch.tensor([weights[1] / weights[0]], dtype=torch.float32)

        # ── Dataloaders ────────────────────────────────────────────────────────
        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_te = torch.tensor(X_test,  dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
        )

        # ── Model ──────────────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on %s", device)
        model = FraudClassifier(X_train.shape[1]).to(device)

        # BCEWithLogitsLoss combines Sigmoid + BCELoss for numerical stability
        # pos_weight correctly handles class imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

        # ── Training loop ──────────────────────────────────────────────────────
        best_auc = 0.0
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                logits = model(X_te.to(device)).squeeze(1)
                preds = torch.sigmoid(logits).cpu().numpy()

            auc  = roc_auc_score(y_test, preds)
            auprc = average_precision_score(y_test, preds)
            preds_bin = (preds >= 0.5).astype(int)
            f1  = f1_score(y_test, preds_bin)
            prec = precision_score(y_test, preds_bin)
            rec  = recall_score(y_test, preds_bin)

            scheduler.step(epoch_loss)

            mlflow.log_metrics({
                "train_loss": round(epoch_loss / len(train_loader), 4),
                "roc_auc":    round(auc,  4),
                "auprc":      round(auprc, 4),
                "f1":         round(f1,   4),
                "precision":  round(prec, 4),
                "recall":     round(rec,  4),
            }, step=epoch)

            logger.info("Epoch %d/%d — AUC: %.4f  F1: %.4f", epoch+1, n_epochs, auc, f1)

            # Save best model checkpoint
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "best_model.pt")

        # ── Log final model ────────────────────────────────────────────────────
        model.load_state_dict(torch.load("best_model.pt", weights_only=True))
        mlflow.pytorch.log_model(model, "model", serialization_format="pickle")

        # Log final summary metrics
        final_metrics = {
            "best_roc_auc": round(best_auc, 4),
            "final_f1":     round(f1, 4),
            "final_recall": round(rec, 4),
            "run_id":       run.info.run_id,
        }
        mlflow.log_metrics({
            "best_roc_auc": round(best_auc, 4),
            "final_f1":     round(f1, 4),
            "final_recall": round(rec, 4),
        })

        # Save metrics as JSON artifact for CI quality gate
        with open("metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        mlflow.log_artifact("metrics.json")

        logger.info("Run ID: %s  |  Best AUC: %.4f", run.info.run_id, best_auc)
        return run.info.run_id, final_metrics


if __name__ == "__main__":
    train()