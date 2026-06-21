import argparse
import requests
import sys

def main():
    parser = argparse.ArgumentParser(description="Smoke test deployment")
    parser.add_argument("--url", required=True, help="API base URL")
    parser.add_argument("--threshold", type=float, default=0.95, help="Unused in this stub, kept for signature match")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"Starting smoke tests against API: {base_url}")

    # 1. Test /health endpoint
    health_url = f"{base_url}/health"
    try:
        print(f"GET {health_url}")
        resp = requests.get(health_url, timeout=10)
        print(f"Response ({resp.status_code}): {resp.text}")
        if resp.status_code != 200:
            print("Smoke test failed: /health returned non-200 status code")
            sys.exit(1)
    except Exception as e:
        print(f"Smoke test failed: Could not connect to /health: {e}")
        sys.exit(1)

    # 2. Test /predict endpoint
    predict_url = f"{base_url}/predict"
    # Create 29 features: V1-V28 PCA features + Amount (Time is dropped)
    dummy_features = [0.0] * 29
    payload = {
        "features": dummy_features,
        "transaction_id": "smoke-test-123"
    }

    try:
        print(f"POST {predict_url} with features {dummy_features[:5]}...")
        resp = requests.post(predict_url, json=payload, timeout=10)
        print(f"Response ({resp.status_code}): {resp.text}")
        if resp.status_code != 200:
            print("Smoke test failed: /predict returned non-200 status code")
            sys.exit(1)
        
        data = resp.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        print("Smoke tests passed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"Smoke test failed: Could not connect to /predict: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
