import pandas as pd
import json
import logging
import os
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DRIFT_THRESHOLD   = float(os.getenv("DRIFT_THRESHOLD", "0.2"))   # share of drifted features
REPORT_OUTPUT_DIR = os.getenv("REPORT_DIR", "monitoring/reports")


def load_reference_data(path: str = "data/reference.csv") -> pd.DataFrame:
    """Training data slice used as the reference distribution."""
    return pd.read_csv(path)


def load_production_data(days: int = 1) -> pd.DataFrame:
    """
    Load recent production traffic from your feature store / DB.
    Replace this stub with your actual data source.
    """
    # Stub: in production, query your database or feature store
    # e.g. pd.read_sql("SELECT * FROM predictions WHERE ts > NOW() - INTERVAL '1 day'", conn)
    logger.warning("Using stub production data — replace with real source")
    import numpy as np
    ref = load_reference_data()
    noise = pd.DataFrame(
        ref.values + 0.5 * ref.values.std() * np.random.randn(*ref.shape),
        columns=ref.columns,
    )
    return noise.sample(min(5000, len(noise)))


def run_drift_report(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    """Run Evidently drift report and return summary dict."""
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ])
    # report.run() mutates the report in-place and returns None in Evidently v0.4+
    report.run(reference_data=reference, current_data=production)

    # Save HTML report — call on the report object itself
    html_path = f"{REPORT_OUTPUT_DIR}/drift_{timestamp}.html"
    report.save_html(html_path)
    logger.info("Drift report saved to %s", html_path)

    # Extract summary — use as_dict() (not .dict()) for Evidently v0.4+
    result = report.as_dict()
    drift_value = result["metrics"][0]["value"]

    share_drifted = drift_value["share"]
    n_drifted = drift_value["count"]
    drift_detected = bool(share_drifted >= DRIFT_THRESHOLD)

    summary = {
        "timestamp": timestamp,
        "dataset_drift_detected": drift_detected,
        "share_drifted_features": round(share_drifted, 3),
        "n_drifted_features": int(n_drifted),
        "report_path": html_path,
    }

    # Save JSON summary
    json_path = f"{REPORT_OUTPUT_DIR}/drift_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def should_retrain(summary: dict) -> bool:
    """Return True if drift exceeds threshold."""
    return summary["share_drifted_features"] > DRIFT_THRESHOLD


def trigger_retraining():
    """
    Hook to trigger your retraining pipeline.
    Options: call Airflow REST API, push to a queue, or run train.py directly.
    """
    logger.info("Drift threshold exceeded — triggering retraining pipeline")

    airflow_url = os.getenv("AIRFLOW_API_URL")
    if airflow_url:
        import requests
        resp = requests.post(
            f"{airflow_url}/api/v1/dags/fraud_detection_pipeline/dagRuns",
            json={"conf": {"trigger_reason": "drift_detected"}},
            auth=(os.getenv("AIRFLOW_USER", "admin"), os.getenv("AIRFLOW_PASS", "")),
        )
        logger.info("Airflow DAG triggered: %s", resp.status_code)
    else:
        logger.info("AIRFLOW_API_URL not set — log drift only (set to auto-retrain)")


def main():
    logger.info("Starting drift detection job")
    reference  = load_reference_data()
    production = load_production_data(days=1)

    summary = run_drift_report(reference, production)

    logger.info(
        "Drift result: detected=%s  share=%.1f%%  drifted_features=%d",
        summary["dataset_drift_detected"],
        summary["share_drifted_features"] * 100,
        summary["n_drifted_features"],
    )

    if should_retrain(summary):
        trigger_retraining()
    else:
        logger.info("No significant drift detected — no retraining needed")

    return summary


if __name__ == "__main__":
    main()