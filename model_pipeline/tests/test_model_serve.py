# MLOPS/model_pipeline/tests/test_model_serve.py
import sys
import os
from fastapi.testclient import TestClient
import pytest # For fixtures and advanced testing features
import json 
import mlflow
# Add the MLOPS root to sys.path to allow importing from model_pipeline
# This allows 'from model_pipeline.model_serve import app' to work when tests are run from MLOPS root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the FastAPI app instance from your model_serve.py
# This assumes 'app' is defined at the global scope in model_serve.py
from model_pipeline.model_serve import app, PredictionRequest, PredictionInstance

client = TestClient(app)

# --- Mocking (Example - you'll need to adapt this heavily) ---
# For robust tests, you'd mock:
# 1. os.getenv to control environment variables.
# 2. mlflow.set_tracking_uri, mlflow.pyfunc.load_model, client.search_registered_models, etc.
# 3. open() for reading/writing latest_runs.json, current_champion.json.

@pytest.fixture(autouse=True) # This fixture will run for every test
def mock_env_vars_and_files(monkeypatch, tmp_path):
    """Mocks environment variables and file system operations for tests."""
    # Mock environment variables
    monkeypatch.setenv("MLFLOW_TRACKING_PATH_IN_SERVING", str(tmp_path / "mlruns"))
    monkeypatch.setenv("ARTIFACTS_MOUNT_PATH_IN_SERVING", str(tmp_path / "artifacts_mount"))

    # Create dummy directories that would be mounted
    (tmp_path / "mlruns").mkdir(exist_ok=True)
    (tmp_path / "artifacts_mount").mkdir(exist_ok=True)

    # Create dummy latest_runs.json (example content)
    dummy_latest_runs = {
        "TestModelA": "run_id_A",
        "TestModelB": "run_id_B"
    }
    with open(tmp_path / "artifacts_mount" / "latest_runs.json", "w") as f:
        json.dump(dummy_latest_runs, f)

    # Create dummy current_champion.json (or leave it absent to test that path)
    # dummy_champion = {"name": "InitialChampion", "version": "1", "uri": "models:/InitialChampion/1"}
    # with open(tmp_path / "artifacts_mount" / "current_champion.json", "w") as f:
    #     json.dump(dummy_champion, f)

    # Mock MLflow client methods (very simplified examples)
    class MockMlflowRun:
        def __init__(self, f1_score):
            self.data = type('obj', (object,), {'metrics': {'f1_score': f1_score}})

    class MockModelVersion:
        def __init__(self, name, version, stage, run_id):
            self.name = name
            self.version = version
            self.current_stage = stage
            self.run_id = run_id

    class MockRegisteredModel:
         def __init__(self, name):
             self.name = name

    def mock_get_run(run_id):
        if run_id == "run_id_A": return MockMlflowRun(0.85)
        if run_id == "run_id_B": return MockMlflowRun(0.90)
        if run_id == "run_id_prod_champ": return MockMlflowRun(0.80) # Example existing champion
        return MockMlflowRun(0.0) # Default

    def mock_search_model_versions(filter_string):
        model_name = filter_string.split("'")[1]
        if model_name == "TestModelB": # Challenger
            return [MockModelVersion(model_name, "1", "None", "run_id_B")]
        if model_name == "ExistingProdChampion": # Existing champion
             return [MockModelVersion(model_name, "2", "Production", "run_id_prod_champ")]
        return []

    def mock_search_registered_models():
         return [MockRegisteredModel("ExistingProdChampion"), MockRegisteredModel("TestModelB")]


    def mock_transition_model_version_stage(*args, **kwargs):
        print(f"MOCK: Transitioning model {kwargs.get('name')} v{kwargs.get('version')} to {kwargs.get('stage')}")
        pass # Placeholder

    def mock_set_model_version_tag(*args, **kwargs):
        print(f"MOCK: Setting tag for model {args[0]} v{args[1]}")
        pass

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "get_run", mock_get_run)
    monkeypatch.setattr(mlflow.tracking.MlflowClient, "search_model_versions", mock_search_model_versions)
    monkeypatch.setattr(mlflow.tracking.MlflowClient, "search_registered_models", mock_search_registered_models)
    monkeypatch.setattr(mlflow.tracking.MlflowClient, "transition_model_version_stage", mock_transition_model_version_stage)
    monkeypatch.setattr(mlflow.tracking.MlflowClient, "set_model_version_tag", mock_set_model_version_tag)


    # Mock mlflow.pyfunc.load_model
    class MockPyfuncModel:
        def predict(self, data):
            # Return a list of predictions, one for each input instance
            return [f"predicted_{item['text']}" for item in data]

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda model_uri: MockPyfuncModel())

    # This ensures that the startup event which loads the model runs with mocks
    # However, TestClient usually handles app startup. If issues, may need to manually trigger.
    # For TestClient, app startup usually happens implicitly on first request or can be managed with lifespan context.

def test_health_endpoint_after_startup_with_mock_model():
    # The startup event should have run due to TestClient instantiation or first call
    # Ensure the app's global 'loaded_model' is set by the mocked load_model
    # This requires careful handling of the app lifecycle in tests.
    # A simple way to ensure startup runs for TestClient is to use a context manager if app uses lifespan
    with TestClient(app) as tc: # This ensures lifespan events are run
         response = tc.get("/health")
         assert response.status_code == 200
         json_response = response.json()
         assert json_response.get("status") == "healthy"
         # Based on mock_search_model_versions and promote_if_better logic, TestModelB should be champion
         assert json_response.get("model_name") == "TestModelB" # Or whatever the logic dictates with mocks


def test_predict_endpoint_with_mock_model():
    with TestClient(app) as tc:
         payload_dict = {"instances": [{"text": "good movie"}, {"text": "bad film"}]}
         response = tc.post("/predict", json=payload_dict)
         assert response.status_code == 200
         json_response = response.json()
         assert "predictions" in json_response
         assert len(json_response["predictions"]) == 2
         assert json_response["predictions"][0] == "predicted_good movie"
         assert json_response["predictions"][1] == "predicted_bad film"

def test_model_info_endpoint():
     with TestClient(app) as tc:
         response = tc.get("/model_info")
         assert response.status_code == 200
         json_response = response.json()
         assert json_response.get("model_name") is not None # Check if a model name is reported
         # Further assertions based on what promote_if_better + mocks would set