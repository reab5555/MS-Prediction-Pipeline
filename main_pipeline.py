from prepare_data.extraction import load_raw_data_from_s3
from prepare_data.transform_general import transform
from mlop.transform_ml import ml_prepare
from mlop.ml_train_eval import train_models
from mlop.mlflow_config import setup_mlflow, copy_mlruns_to_s3
from deploy import deploy_models
from configs.utils import log_current_time
import os

def main():
    log_current_time()
    print("Starting ETL process...")
    
    # Step 1: Load the raw data from S3
    raw_df, file_name = load_raw_data_from_s3()
    
    # Step 2: Transform the data
    ml_ready_table_path, parquet_s3_dir = transform(raw_df, file_name)
    
    # Step 3: Prepare data for ML
    df_clean, pipe_stages = ml_prepare(ml_ready_table_path)
    
    print("ETL process completed.")
    
    # Setup MLflow
    artifact_uri = setup_mlflow()
    
    # Step 4: Train models
    print("Starting model training...")
    train_models(df_clean, pipe_stages, artifact_uri)
    
    print("Model training completed.")

    # Step 5: Copy mlruns folder to S3
    print("\nCopying mlruns folder to S3...")
    copy_mlruns_to_s3(parquet_s3_dir)
    
    # Step 6: Deploying Models
    print("\nDeploying Models...")
    deploy_models()
    
if __name__ == "__main__":
    main()