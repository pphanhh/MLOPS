from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

# Default args for all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Function to choose labeling method at runtime
def choose_labeling_method(**context):
    # Uses dag_run.conf parameter 'label_method', defaults to 'model'
    method = context.get('dag_run').conf.get('label_method', 'model')
    if method == 'llm':
        return 'label_data_llm'
    else:
        return 'predict_data'

with DAG(
    dag_id='data_pipeline',
    default_args=default_args,
    description='End-to-end Twitter sentiment pipeline with MLflow champion model support',
    start_date=datetime.now(),
    schedule='@daily',
    catchup=False,
) as dag:

    # Step 1: Crawl raw Twitter data
    crawl_data = BashOperator(
        task_id='crawl_data',
        bash_command='python3 /mnt/d/MLOps2/data/crawl.py'
    )

    # Step 2: Preprocess tweets
    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='python3 /mnt/d/MLOps2/data/preprocessing.py'
    )

    # Step 3: Labelling data by inference

    #Predict using champion model from MLflow
    predict_data = BashOperator(
        task_id='predict_data',
        bash_command='python3 /mnt/d/MLOps2/model_pipeline/predict.py'
    )

    # Step 4: Validate data
    validate_data = BashOperator(
        task_id='validate_data',
        bash_command='python3 /mnt/d/MLOps2/data/validate.py'
    )

    # Step 5: Ingest into PostgreSQL
    ingest_data = BashOperator(
        task_id='ingest_data',
        bash_command='python3 /mnt/d/MLOps2/data/ingest.py'
    )

    # # Step 6: Train model on new labeled dataset
    # train_model = BashOperator(
    #     task_id='train_model',
    #     bash_command='python /path/to/train_model.py'
    # )

    # # Step 7: Update dashboard/reporting
    # update_dashboard = BashOperator(
    #     task_id='update_dashboard',
    #     bash_command='python /path/to/update_dashboard.py'
    # )

    # Define dependencies
    crawl_data >> preprocess_data >> predict_data >> validate_data >> ingest_data 
