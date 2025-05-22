import os
import glob
import time
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from datetime import datetime
import mlflow
import requests
import json
FASTAPI_URL = "http://localhost:5001"

def wait_for_fastapi(timeout=30):
    """Chờ FastAPI server sẵn sàng trước khi gọi API"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{FASTAPI_URL}/health")
            if r.status_code == 200:
                print("FastAPI server is ready.")
                return
        except requests.exceptions.ConnectionError:
            pass
        print("Waiting for FastAPI...")
        time.sleep(3)
    raise RuntimeError("FastAPI server not ready after waiting.")

# def get_champion_model_uri(prefix="sentiment_"):
#     """
#     Tìm model đang ở stage 'Production' trên MLflow Registry
#     """
#     client = MlflowClient()
#     for rm in client.search_registered_models():
#         if not rm.name.startswith(prefix):
#             continue
#         for mv in client.search_model_versions(f"name='{rm.name}'"):
#             if mv.current_stage == "Production":
#                 print(f"Found production model: {rm.name}, version: {mv.version}")
#                 return f"models:/{rm.name}/Production"
#     raise RuntimeError("Không tìm thấy model nào ở stage Production.")

def find_latest_processed_file(processed_dir="/mnt/d/python/MLOps/clone/MLOPS/data/processed"):
    """Lấy file .csv mới nhất trong thư mục processed/"""
    pattern = os.path.join(processed_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file processed nào trong {processed_dir}")
    return max(files, key=os.path.getmtime)

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def predict_batch(batch_data):
    """Gửi batch dữ liệu đến endpoint /predict và lấy kết quả"""
    payload = {"instances": [{"text": str(text)} for text in batch_data]}
    response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Prediction failed: {response.text}")
    return response.json()["predictions"]

def predict_in_batches(df, batch_size=20):
    """Chia dữ liệu thành batch và dự đoán"""
    predictions = []
    for i in range(0, len(df), batch_size):
        batch = df['cleaned_text'][i:i + batch_size].tolist()
        print(f"Predicting batch {i // batch_size + 1} ({len(batch)} samples)...")
        try:
            batch_preds = predict_batch(batch)
            predictions.extend(batch_preds)
        except Exception as e:
            print(f"Error predicting batch {i // batch_size + 1}: {str(e)}")
            # Gán giá trị mặc định (hoặc xử lý lỗi tùy ý)
            predictions.extend([None] * len(batch))
    return predictions

def main():
    # 1. Đợi MLflow server sẵn sàng
    wait_for_fastapi()

    # # 2. Lấy model URI
    # model_uri = get_champion_model_uri(prefix="sentiment_")
    # print(f"Loading champion model from '{model_uri}'")
    # model = mlflow.pyfunc.load_model(model_uri)

    # 3. Đọc file processed mới nhất
    input_file = find_latest_processed_file("/mnt/d/python/MLOps/clone/MLOPS/data/processed")
    print(f"Reading processed data from '{input_file}'")
    df = pd.read_csv(input_file)

    # 4. Chuẩn bị input cho model
    model_input = df[['cleaned_text']].rename(columns={'cleaned_text': 'text'})

    # 5. Chạy inference
    print("Running inference...")
    preds = predict_in_batches(df, batch_size=20)
    df['sentiment'] = preds

    # 6. Ghi kết quả ra thư mục labeled/
    out_dir = "/mnt/d/python/MLOps/clone/MLOPS/data/labeled"
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_file = os.path.join(out_dir, f"predicted_twitter_{date_str}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved predictions to '{out_file}'")

if __name__ == "__main__":
    main()
