from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

from prepare_data.extraction import load_raw_data_from_s3
from prepare_data.transform_general import transform
from mlop.transform_ml import ml_prepare
from mlop.ml_train_eval import train_models
from mlop.mlflow_config import setup_mlflow, copy_mlruns_to_s3
from deploy import deploy_models

default_args = {
    'owner': 'User',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'etl_ml_pipeline',
    default_args=default_args,
    description='ETL and ML pipeline',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['etl', 'ml'],
)

# Define the tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=load_raw_data_from_s3,
    provide_context=True,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform,
    provide_context=True,
    dag=dag,
)

prepare_ml_task = PythonOperator(
    task_id='prepare_ml_data',
    python_callable=ml_prepare,
    provide_context=True,
    dag=dag,
)

setup_mlflow_task = PythonOperator(
    task_id='setup_mlflow',
    python_callable=setup_mlflow,
    provide_context=True,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id='train_ml_models',
    python_callable=train_models,
    provide_context=True,
    dag=dag,
)

copy_mlruns_task = PythonOperator(
    task_id='copy_mlruns',
    python_callable=copy_mlruns_to_s3,
    provide_context=True,
    dag=dag,
)


# Set up task dependencies
extract_task >> transform_task >> prepare_ml_task >> setup_mlflow_task >> train_models_task >> copy_mlruns_task
