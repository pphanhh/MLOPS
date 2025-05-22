from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.db import provide_session
from airflow.utils.state import State
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowSkipException, AirflowFailException
from sqlalchemy import desc
import requests
import time
# Import necessary Airflow models
from airflow.models.taskinstance import TaskInstance 
from airflow.models.dagrun import DagRun           


# Default args for all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=3),
}

# Function to check if it's the training day (every 15 days)
@provide_session
def should_train_model(execution_date, dag, session=None):

    dag_id = dag.dag_id
    task_id = "train_model" # The task id you are checking history for

    last_runs = (
        session.query(TaskInstance)
        .join(DagRun, TaskInstance.run_id == DagRun.run_id) # Ensure DagRun and TaskInstance match
        .filter(
            TaskInstance.dag_id == dag_id,
            TaskInstance.task_id == task_id,
            TaskInstance.state == State.SUCCESS,
            DagRun.execution_date < execution_date, # Look for runs BEFORE the current one
            DagRun.state == State.SUCCESS # Optional but good practice: ensure the full DAG run was successful
        )
        .order_by(desc(DagRun.execution_date)) # Order by DAG run execution date
        .limit(1)
        .all()
    )

    if not last_runs:
        print("First training run or no previous successful training tasks found.")
        return 'train_model'  # Return task ID to execute

    # Check days since the last training
    last_run_date = last_runs[0].execution_date.date()
    today = execution_date.date() # Use the execution_date passed to the function
    days_since = (today - last_run_date).days

    if days_since < 15:
        print(f"Only {days_since} days since last training — skipping.")
        return 'skip_training'  # Return task ID to skip to
    
    print(f"{days_since} days passed — running training.")
    return 'train_model'  # Return task ID to execute

def wait_for_mlflow(timeout=90):
    import time
    import requests
    from airflow.exceptions import AirflowFailException

    url = "http://localhost:5000/api/2.0/mlflow/experiments/list"
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("MLflow server is up and responding.")
                return
        except Exception as e:
            print(f"Still waiting: {e}")
        print("Waiting for MLflow...")
        time.sleep(3)
    
    raise AirflowFailException("MLflow server not ready after waiting.")

# Define the DAG
with DAG(
    dag_id='twitter_sentiment_analysis_pipeline',
    default_args=default_args,
    description='End-to-end Twitter sentiment pipeline with MLflow champion model support',
    start_date= datetime(2025, 5, 7),
    schedule_interval='0 8 * * *',
    catchup=False,
    tags=['mlops', 'sentiment', 'twitter'], # Added tags for better organization
) as dag:

    # start_mlflow_server = BashOperator(
    # task_id='start_mlflow_server',
    # bash_command="""
    # if ! lsof -i :5000 > /dev/null; then
    #     nohup mlflow server \
    #         --backend-store-uri sqlite:///mlflow.db \
    #         --default-artifact-root ./mlruns \
    #         --host 0.0.0.0 \
    #         --port 5000 \
    #         > mlflow_server.log 2>&1 &
    #     echo "MLflow server started."
    # else
    #     echo "MLflow server already running on port 5000."
    # fi
    # """,
    # )

    # wait_mlflow_ready = PythonOperator(
    #     task_id='wait_for_mlflow',
    #     python_callable=wait_for_mlflow
    # )

    # Step 1: Crawl raw Twitter data
    crawl_data = BashOperator(
        task_id='crawl_data',
        bash_command='/home/tpa/venvs/mlops310/bin/python /mnt/d/python/MLOps/clone/MLOPS/data/crawl.py'
    )

    # Step 2: Preprocess tweets
    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='/home/tpa/venvs/mlops310/bin/python /mnt/d/python/MLOps/clone/MLOPS/data/preprocessing.py'
    )

    # Step 3: Check if it's training day or not
    check_training = BranchPythonOperator(
        task_id='check_training_condition',
        python_callable=should_train_model,
    )

    # Step 4: Train the model (only runs on training days)
    train_model = BashOperator(
        task_id='train_model',
        bash_command='python /mnt/d/MLOps2/model_pipeline/model_training.py',
    )

    model_deploy = BashOperator(
        task_id='model_deploy',
        bash_command='/home/tpa/venvs/mlops310/bin/python /mnt/d/python/MLOps/clone/MLOPS/data/model_deploy.py',
        # trigger_rule=TriggerRule.ALL_DONE,
    )

    # Step 6: Validate the new model (only on training days)
    model_validate = BashOperator(
        task_id='model_validate',
        bash_command='/home/tpa/venvs/mlops310/bin/python /mnt/d/python/MLOps/clone/MLOPS/data/model_validate.py',
        # trigger_rule=TriggerRule.ALL_DONE, 
    )

    # Step 7: Serve the new model (only on training days)
    # model_serve = BashOperator(
    #     task_id='model_serve',
    #     bash_command='python /mnt/d/MLOps2/model_pipeline/model_serve.py',
    #     trigger_rule=TriggerRule.ALL_DONE,
    # )
    model_serve = BashOperator(
    task_id='model_serve',
    bash_command='docker compose -f /mnt/d/python/clone/MLOPS/dockerfiles/docker-compose.yml up -d',
    trigger_rule=TriggerRule.ALL_DONE,
)

    # Step 8: Predict using the current model (for both training and non-training days)
    predict_data = BashOperator(
        task_id='predict_data',
        bash_command='/home/tpa/venvs/mlops310/bin/python /mnt/d/python/MLOps/clone/MLOPS/model_pipeline/predict.py',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,

    )

    # Step 9: Validate the data (after prediction)
    validate_data = BashOperator(
        task_id='validate_data',
        bash_command='/home/tpa/venvs/mlops310/bin/python /mnt/d/python/MLOps/clone/MLOPS/data/validate.py',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,

    )

    # Step 10: Ingest data into PostgreSQL
    ingest_data = BashOperator(
        task_id='ingest_data',
        bash_command='/home/tpa/venvs/mlops310/bin/python /mnt/d/python/MLOps/clone/MLOPS/data/ingest.py',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,

    )

    # Step 11: Skip training and just do validation and ingestion (if not training day)
    skip_training = EmptyOperator(
        task_id='skip_training'
    )

    # Define dependencies
    crawl_data >> preprocess_data >> check_training
    
    # If it's training day, train, deploy, validate, and serve the model
    check_training >> [train_model, skip_training]
    
    # Path taken if training occurs
    train_model >> model_deploy >> model_validate >> model_serve >> predict_data

    # Path taken if training is skipped
    skip_training >> model_serve >> predict_data

    # Tasks after the branch merges
    predict_data >> validate_data >> ingest_data

