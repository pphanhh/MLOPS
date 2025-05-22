from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowSkipException
from airflow.models import TaskInstance, DagRun
from airflow.utils.db import provide_session
from airflow.utils.state import State
from airflow.utils.trigger_rule import TriggerRule
from sqlalchemy import desc

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 5),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@provide_session
def should_train_model(execution_date, dag, session=None):
    dag_id = dag.dag_id
    task_id = "model_training"

    last_runs = (
        session.query(TaskInstance)
        .join(DagRun, DagRun.dag_id == TaskInstance.dag_id)
        .filter(
            TaskInstance.dag_id == dag_id,
            TaskInstance.task_id == task_id,
            TaskInstance.state == State.SUCCESS,
            DagRun.execution_date < execution_date,
            DagRun.run_id == TaskInstance.run_id,
        )
        .order_by(desc(DagRun.execution_date))
        .limit(1)
        .all()
    )

    if not last_runs:
        print("✅ First training run.")
        return True

    last_run_date = last_runs[0].execution_date.date()
    today = execution_date.date()
    days_since = (today - last_run_date).days

    if days_since < 15:
        print(f"⏩ Only {days_since} days since last training — skipping.")
        raise AirflowSkipException("Not time yet.")
    print("✅ 15 days passed — running training.")
    return True

dag = DAG(
    dag_id='model_pipeline',
    default_args=default_args,
    schedule_interval='0 8 * * *',
    catchup=False,
    description='Run model training every 15 days, but validate/deploy/predict daily',
)

# PYTHON = '/home/tpa/venvs/mlops310/bin/python'

# --- Tasks ---
check_training = PythonOperator(
    task_id='check_training_condition',
    python_callable=should_train_model,
    provide_context=True,
    dag=dag,
)

model_training = BashOperator(
    task_id='model_training',
    bash_command=f'python /mnt/d/MLOps2/model_pipeline/model_training.py',
    dag=dag,
)

model_deploy = BashOperator(
    task_id='model_deploy',
    bash_command=f'python /mnt/d/MLOps2/model_pipeline/model_deploy.py',
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

model_validate = BashOperator(
    task_id='model_validate',
    bash_command=f'python /mnt/d/MLOps2/model_pipeline/model_validate.py',
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

model_serve = BashOperator(
    task_id='model_serve',
    bash_command=f'python /mnt/d/MLOps2/model_pipeline/model_serve.py',
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

send_request = BashOperator(
    task_id='send_request',
    bash_command=f'python /mnt/d/MLOps2/model_pipeline/send_request.py',
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# --- DAG chaining ---
check_training >> model_training
model_training >> model_deploy >> model_validate >> model_serve >> send_request
check_training >> model_deploy  # skip training vẫn chạy các task sau
