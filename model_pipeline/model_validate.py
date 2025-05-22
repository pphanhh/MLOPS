import pandas as pd
import mlflow
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Thiết lập tracking URI trỏ đến thư mục mlruns trong MLOPS
tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
mlflow.set_tracking_uri(f"file://{tracking_path}")

def validate_model(model_uri, test_data_path):
    test_df = pd.read_csv(test_data_path)

    # Kiểm tra cột đầu vào
    if 'text' not in test_df.columns or 'sentiment_num' not in test_df.columns:
        raise ValueError("CSV phải có cột 'text' và 'sentiment_num'")

    X_test = test_df[['text']]
    y_test = test_df['sentiment_num']

    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
    rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    print(f"   Evaluation for {model_uri}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print("")

if __name__ == "__main__":
    # Đọc latest_runs.json từ thư mục gốc (MLOPS)
    latest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../latest_runs.json"))
    if not os.path.exists(latest_path):
        raise FileNotFoundError(f"Không tìm thấy file: {latest_path}")

    with open(latest_path, "r") as f:
        latest_runs = json.load(f)

    for model_name, run_id in latest_runs.items():
        print(f"Validating model: {model_name}")
        test_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test_data.csv"))
        validate_model(f"runs:/{run_id}/model", test_data_path)

