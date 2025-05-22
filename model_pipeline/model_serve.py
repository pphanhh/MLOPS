# MLOPS/model_pipeline/model_serve.py
import os
import json
import mlflow
import mlflow.pyfunc # Explicitly import if not already
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # Ensure BaseModel is imported
from typing import List, Any, Dict # Import Any and Dict
import uvicorn
# import pandas as pd # Uncomment if you use pandas for DataFrame conversion
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Not used in serving part

import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variable Configuration ---
MLFLOW_TRACKING_PATH_IN_CONTAINER = os.getenv("MLFLOW_TRACKING_PATH_IN_SERVING", "/app/mlruns")
ARTIFACTS_MOUNT_PATH_IN_CONTAINER = os.getenv("ARTIFACTS_MOUNT_PATH_IN_SERVING", "/app/artifacts_mount")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME_SERVING", "sentiment-analysis") # Make experiment name configurable

logger.info(f"Setting MLflow tracking URI to: file://{MLFLOW_TRACKING_PATH_IN_CONTAINER}")
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_PATH_IN_CONTAINER}")
if MLFLOW_EXPERIMENT_NAME:
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")
    except Exception as e:
        logger.error(f"Could not set MLflow experiment '{MLFLOW_EXPERIMENT_NAME}': {e}")


client = MlflowClient()

# --- Pydantic Models for API ---
class PredictionInstance(BaseModel):
    text: str

class PredictionRequest(BaseModel):
    instances: List[PredictionInstance]

class PredictionResponse(BaseModel):
    predictions: List[Any]

# --- Global variable for the loaded model and its info ---
loaded_model: mlflow.pyfunc.PyFuncModel = None # Type hint for clarity
current_model_info: Dict[str, Any] = {"name": None, "version": None, "uri": None}


# --- Helper Functions for File Operations ---
def get_json_from_mounted_file(filename: str) -> Dict:
    path = os.path.join(ARTIFACTS_MOUNT_PATH_IN_CONTAINER, filename)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {path}")
            return {}
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return {}
    else:
        logger.warning(f"File not found: {path}")
    return {}

def write_json_to_mounted_file(filename: str, data: Dict):
    path = os.path.join(ARTIFACTS_MOUNT_PATH_IN_CONTAINER, filename)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure directory exists
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully wrote to {path}")
    except Exception as e:
        logger.error(f"Error writing to file {path}: {e}")

# --- Model Promotion Logic (Adapted from your code) ---
def get_current_champion_from_registry():
    """Finds a model in 'Production' stage in the MLflow Model Registry."""
    for rm in client.search_registered_models():
        # Optional: Filter by model name prefix if needed
        # if not rm.name.startswith("your_prefix_"): continue
        for v in client.search_model_versions(f"name='{rm.name}'"):
            if v.current_stage == "Production":
                try:
                    run = client.get_run(v.run_id)
                    f1 = run.data.metrics.get("f1_score", -1.0)
                    logger.info(f"Found Production model in registry: {rm.name} version {v.version} (F1: {f1:.4f})")
                    return rm.name, v.version, v.run_id, f1
                except Exception as e:
                    logger.error(f"Error fetching run {v.run_id} for Production model {rm.name} v{v.version}: {e}")
                    # Still return model info, F1 might be unavailable
                    return rm.name, v.version, v.run_id, -1.0
    logger.warning("No model found in 'Production' stage in MLflow Registry.")
    return None, None, None, -1.0

def find_best_challenger_from_latest_runs(latest_runs_data: Dict):
    """Finds the best model from latest_runs_data based on f1_score."""
    if not latest_runs_data:
        logger.warning("latest_runs_data is empty.")
        return None, None, -1.0

    best_model_name = None
    best_run_id = None
    best_f1_score = -1.0

    for model_name, run_id in latest_runs_data.items():
        try:
            run = client.get_run(run_id)
            f1 = run.data.metrics.get("f1_score", -1.0)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_model_name = model_name
                best_run_id = run_id
        except Exception as e:
            logger.error(f"Error fetching run {run_id} for model {model_name} from latest_runs: {e}")
            continue
    if best_model_name:
        logger.info(f"Best challenger from latest_runs.json: {best_model_name} (Run ID: {best_run_id}, F1: {best_f1_score:.4f})")
    else:
        logger.warning("No valid challenger found in latest_runs.json.")
    return best_model_name, best_run_id, best_f1_score

def promote_if_better():
    """Compares challenger with current champion and promotes if better. Updates current_model_info."""
    global current_model_info # To store the info of the model to be loaded

    latest_runs = get_json_from_mounted_file("latest_runs.json")
    if not latest_runs:
        logger.warning("⚠️ No latest_runs.json found or it's empty. Cannot determine challenger.")
        # Proceed to load current champion or do nothing if none
    
    challenger_name, challenger_run_id, challenger_f1 = find_best_challenger_from_latest_runs(latest_runs)

    current_champion_name, current_champion_version, current_champion_run_id, current_champion_f1 = get_current_champion_from_registry()

    logger.info(f"Current Champion in Registry: {current_champion_name} v{current_champion_version} (F1: {current_champion_f1:.4f})")
    if challenger_name:
        logger.info(f"Best Challenger from latest_runs: {challenger_name} (F1: {challenger_f1:.4f})")

    model_to_load_name = current_champion_name
    model_to_load_version = current_champion_version

    if challenger_name and challenger_run_id and (challenger_f1 > current_champion_f1):
        logger.info(f"Challenger {challenger_name} (F1: {challenger_f1:.4f}) is better than current champion {current_champion_name} (F1: {current_champion_f1:.4f}). Promoting...")
        try:
            versions = client.search_model_versions(f"name='{challenger_name}'")
            promoted_version_str = None
            for v in versions:
                if v.run_id == challenger_run_id:
                    client.transition_model_version_stage(
                        name=challenger_name,
                        version=v.version, # version is already a string from API
                        stage="Production",
                        archive_existing_versions=True
                    )
                    # Optional: Add a tag for traceability
                    client.set_model_version_tag(challenger_name, v.version, "promotion_reason", f"Auto-promoted via API: F1 {challenger_f1:.4f}")
                    logger.info(f"Promoted {challenger_name} version {v.version} to Production.")
                    model_to_load_name = challenger_name
                    model_to_load_version = v.version
                    promoted_version_str = v.version
                    break
            if not promoted_version_str:
                 logger.error(f"Could not find registered model version for challenger {challenger_name} with run ID {challenger_run_id}.")
        except Exception as e:
            logger.error(f"Error promoting challenger {challenger_name}: {e}. Will attempt to load current champion.")
            # Fallback to current champion if promotion fails
            model_to_load_name = current_champion_name
            model_to_load_version = current_champion_version
    elif challenger_name and not current_champion_name:
        logger.info(f"No current champion in Production. Promoting first valid challenger: {challenger_name} (F1: {challenger_f1:.4f}).")
        try:
            versions = client.search_model_versions(f"name='{challenger_name}'")
            promoted_version_str = None
            for v in versions:
                if v.run_id == challenger_run_id: # Ensure it's the best challenger
                    client.transition_model_version_stage(
                        name=challenger_name, version=v.version, stage="Production", archive_existing_versions=True
                    )
                    client.set_model_version_tag(challenger_name, v.version, "promotion_reason", f"Initial promotion: F1 {challenger_f1:.4f}")
                    logger.info(f"Promoted {challenger_name} version {v.version} to Production as initial champion.")
                    model_to_load_name = challenger_name
                    model_to_load_version = v.version
                    promoted_version_str = v.version
                    break
            if not promoted_version_str:
                logger.error(f"Could not find registered model version for initial challenger {challenger_name} with run ID {challenger_run_id}.")
        except Exception as e:
            logger.error(f"Error promoting initial challenger {challenger_name}: {e}")


    else:
        logger.info("Challenger is not better or no valid challenger. Keeping/loading current champion from registry if available.")

    if model_to_load_name and model_to_load_version:
        current_model_info = {
            "name": model_to_load_name,
            "version": model_to_load_version,
            "uri": f"models:/{model_to_load_name}/Production" # Always load from Production stage
        }
        write_json_to_mounted_file("current_champion.json", current_model_info)
        logger.info(f"Target model to load: {current_model_info['name']} v{current_model_info['version']}")
    else:
        logger.warning("No model determined to load after promotion logic. `current_model_info` not updated.")
        # Clear current_champion.json if no model is to be loaded
        write_json_to_mounted_file("current_champion.json", {})


# --- FastAPI Application Setup ---
app = FastAPI(
    title="Sentiment Analysis Model Server",
    description="Serves an MLflow model for sentiment analysis based on champion/challenger logic.",
    version="1.1.0" # Example version
)

@app.on_event("startup")
async def startup_event_tasks():
    global loaded_model, current_model_info
    logger.info("Application startup sequence initiated...")

    promote_if_better() # This updates current_model_info internally and writes to current_champion.json

    # After promote_if_better, current_model_info should reflect the model to be loaded.
    # If not, try to load from a previously saved current_champion.json or fresh from registry.
    if not current_model_info.get("uri"):
        logger.info("`current_model_info` not set by promotion logic. Attempting to load from `current_champion.json` or registry.")
        saved_champion_info = get_json_from_mounted_file("current_champion.json")
        if saved_champion_info and saved_champion_info.get("uri"):
            current_model_info = saved_champion_info
            logger.info(f"Loaded champion info from current_champion.json: {current_model_info}")
        else:
            logger.info("current_champion.json not found or invalid. Querying registry for Production model.")
            name, version, _, _ = get_current_champion_from_registry()
            if name and version:
                current_model_info = {"name": name, "version": version, "uri": f"models:/{name}/Production"}
                logger.info(f"Found Production model in registry: {current_model_info}")
            else:
                logger.error("No model URI found in current_champion.json or in Production stage of registry.")

    if current_model_info.get("uri"):
        model_uri_to_load = current_model_info["uri"]
        logger.info(f"Attempting to load model from final URI: {model_uri_to_load}")
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri_to_load)
            logger.info(f"Successfully loaded model: {current_model_info.get('name')} version {current_model_info.get('version')}")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load model from {model_uri_to_load}: {e}", exc_info=True)
            loaded_model = None # Ensure model is None if loading fails
    else:
        logger.error("CRITICAL: No model URI determined at startup. Model cannot be loaded.")


@app.get("/health", summary="Check API and model status")
async def health():
    if loaded_model and current_model_info.get("name"):
        return {
            "status": "healthy",
            "message": "API is operational and model is loaded.",
            "loaded_model_name": current_model_info.get("name"),
            "loaded_model_version": current_model_info.get("version"),
            "loaded_model_uri": current_model_info.get("uri")
        }
    else:
        return {
            "status": "unhealthy",
            "message": "API is operational, but the ML model is not loaded or identified.",
            "loaded_model_name": None,
            "loaded_model_version": None,
            "loaded_model_uri": None
        }

@app.post("/predict", response_model=PredictionResponse, summary="Get sentiment predictions")
async def make_prediction(request: PredictionRequest):
    if not loaded_model:
        logger.error("Prediction attempt failed: Model not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available. Please check health or try again later.")
    try:
        # Assuming MLflow pyfunc model expects a list of dicts or pandas DataFrame
        # Your current code uses list of dicts, which is fine for many models.
        input_data = [instance.dict() for instance in request.instances]
        
        # If your model specifically requires a pandas DataFrame:
        # import pandas as pd
        # input_df = pd.DataFrame(input_data)
        # raw_predictions = loaded_model.predict(input_df)
        
        logger.info(f"Received {len(input_data)} instances for prediction.")
        raw_predictions = loaded_model.predict(input_data)

        # Ensure predictions are JSON serializable (e.g., convert numpy arrays to lists)
        if hasattr(raw_predictions, 'tolist'):
            predictions_list = raw_predictions.tolist()
        else:
            predictions_list = list(raw_predictions)
        
        logger.info(f"Successfully made {len(predictions_list)} predictions.")
        return PredictionResponse(predictions=predictions_list)

    except Exception as e:
        logger.error(f"Prediction processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.get("/model_info", summary="Get information about the currently configured model for loading")
async def get_configured_model_info():
    # This reflects the model that *should* be loaded based on current_champion.json or promotion logic
    # It might differ from /health if loading failed for some reason.
    info_from_file = get_json_from_mounted_file("current_champion.json")
    if info_from_file and info_from_file.get("name"):
        return {
            "source": "current_champion.json",
            "model_name": info_from_file.get("name"),
            "model_version": info_from_file.get("version"),
            "model_uri": info_from_file.get("uri")
        }
    elif current_model_info.get("name"): # Fallback to in-memory if file is missing post-startup
         return {
            "source": "in-memory (post-promotion logic)",
            "model_name": current_model_info.get("name"),
            "model_version": current_model_info.get("version"),
            "model_uri": current_model_info.get("uri")
        }
    else:
        return {"message": "No champion model information configured or found."}


# To run this file directly for local testing (uvicorn in CMD is preferred for Docker)
if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly for local development...")
    uvicorn.run(
        "model_serve:app", # Assuming the file is named model_serve.py
        host="0.0.0.0",
        port=5001,
        reload=True # Enable reload for local development
    )