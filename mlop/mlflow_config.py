import os
import time
import mlflow
from mlflow.tracking import MlflowClient
import psutil
from configs.cloud_config import S3_CONFIG, AWS_CREDENTIALS
import boto3
from botocore.exceptions import NoCredentialsError
from configs.utils import log_current_time


def setup_mlflow():
    # Use a local directory for the tracking server
    current_dir = os.getcwd()
    local_mlruns_dir = os.path.join(current_dir, 'mlruns')
    os.makedirs(local_mlruns_dir, exist_ok=True)
    
    # Set up the tracking URI to use the local file system
    tracking = local_mlruns_dir
    mlflow.set_tracking_uri(tracking)
    
    # Set up S3 for artifact storage
    artifact = local_mlruns_dir
    
    # Set up the experiment
    experiment_name = "MS_Prediction_Models"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment:
        print(f"Using existing experiment '{experiment_name}' (ID: {experiment.experiment_id})")
    else:
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact)
        print(f"Created new experiment '{experiment_name}' (ID: {experiment_id})")
    
    print(f"MLflow tracking: {tracking}")
    print(f"MLflow artifact: {artifact}")
    
    log_current_time()
    
    return tracking, artifact

def copy_mlruns_to_s3(parquet_s3_dir):
        
    # Configure MLflow to use S3 for artifact storage via environment variables
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"https://s3.{AWS_CREDENTIALS['region_name']}.amazonaws.com"
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_CREDENTIALS['aws_access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_CREDENTIALS['aws_secret_access_key']
    os.environ['AWS_DEFAULT_REGION'] = AWS_CREDENTIALS['region_name']
    
    s3_bucket = S3_CONFIG['bucket_name']

    current_dir = os.getcwd()
    local_mlruns_dir = os.path.join(current_dir, 'mlruns')
    s3_prefix = parquet_s3_dir.replace('s3a://', 's3://')
    s3_mlruns_prefix = f"{s3_prefix}/mlruns"


    s3 = boto3.client('s3',
                      aws_access_key_id=AWS_CREDENTIALS['aws_access_key_id'],
                      aws_secret_access_key=AWS_CREDENTIALS['aws_secret_access_key'],
                      region_name=AWS_CREDENTIALS['region_name'])

    for root, dirs, files in os.walk(local_mlruns_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_mlruns_dir)
            s3_path = f"{s3_mlruns_prefix}/{relative_path}"
            cleaned_path = s3_path.replace('s3://msnflcognbucket/', '')
            s3.upload_file(local_path, s3_bucket, cleaned_path)
        
    print('For UI: mlflow ui --host 0.0.0.0')
    
    print(f"Successfully copied mlruns to {s3_mlruns_prefix}")
    
    log_current_time()

    return s3_mlruns_prefix


        
