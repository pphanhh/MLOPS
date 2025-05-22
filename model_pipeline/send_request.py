import mlflow
import requests
import json
from mlflow.tracking import MlflowClient
import os

# Thiết lập tracking URI nếu cần
tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
mlflow.set_tracking_uri(f"file://{tracking_path}")

def get_champion_model_info(prefix="sentiment_"):
    client = MlflowClient()
    for rm in client.search_registered_models():
        if rm.name.startswith(prefix):
            for v in client.search_model_versions(f"name='{rm.name}'"):
                if v.current_stage == "Production" or v.tags.get("champion") == "True":
                    return rm.name, v.version, v.current_stage
    return None, None, None

def send_request():
    # === Lấy thông tin mô hình champion ===
    model_name, model_version, model_stage = get_champion_model_info()
    if not model_name:
        print(" No champion model found.")
        return

    # === Dữ liệu test mẫu ===
    input_text = "Donald Trump is the 45th president of the United States."
    payload = {
        "dataframe_records": [
            {"text": input_text}
        ]
    }

    url = "http://localhost:5001/invocations"
    headers = {"Content-Type": "application/json"}

    # === Gửi request đến endpoint đã serve ===
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            prediction = response.json()
            print(f"Prediction: {prediction}")
        else:
            print(f"Failed: {response.status_code} | {response.text}")
            prediction = None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        prediction = None

    # === Ghi log vào MLflow ===
    mlflow.set_experiment("sentiment-analysis")
    with mlflow.start_run(run_name="serve_request"):
        mlflow.log_param("input_text", input_text)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_stage", model_stage)
        if prediction is not None:
            mlflow.log_param("prediction", prediction[0] if isinstance(prediction, list) else str(prediction))

if __name__ == "__main__":
    send_request()
