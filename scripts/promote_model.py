import argparse
import mlflow
from mlflow.tracking import MlflowClient

def main():
    parser = argparse.ArgumentParser(description="Promote MLflow model to a stage")
    parser.add_argument("--run-id", required=True, help="MLflow Run ID")
    parser.add_argument("--stage", required=True, choices=["Staging", "Production", "Archived"], help="Target stage")
    args = parser.parse_args()

    model_name = "fraud-detection"
    model_uri = f"runs:/{args.run_id}/model"
    
    print(f"Registering model from run {args.run_id} under name '{model_name}'...")
    model_version = mlflow.register_model(model_uri, model_name)
    
    client = MlflowClient()
    print(f"Transitioning model version {model_version.version} to stage '{args.stage}'...")
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=args.stage,
        archive_existing_versions=True
    )
    print("Model promotion completed successfully!")

if __name__ == "__main__":
    main()
