import os
import json
import time
from datetime import datetime
import logging
import threading
import mlflow
from mlflow.tracking import MlflowClient
import gradio as gr
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, OneHotEncoder, StringIndexer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, when, count, mean, isnan, isnull
from pyspark.sql.types import DoubleType
from flask import Flask, request, jsonify
from configs.cloud_config import AWS_CREDENTIALS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
response_model = None
response_input_cols = None
response_target_col = None
spark = None

def initialize_spark(AWS_CREDENTIALS):
    global spark
    if spark is None:
        spark = SparkSession.builder \
            .appName("ML Deployment") \
            .config("spark.jars.packages", 
                    "org.apache.hadoop:hadoop-aws:3.3.4,"
                    "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
                    "org.apache.hadoop:hadoop-common:3.3.4") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
            .config("spark.hadoop.fs.s3a.access.key", AWS_CREDENTIALS['aws_access_key_id']) \
            .config("spark.hadoop.fs.s3a.secret.key", AWS_CREDENTIALS['aws_secret_access_key']) \
            .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_CREDENTIALS['region_name']}.amazonaws.com") \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .getOrCreate()
    return spark

def initialize_mlflow():
    current_dir = os.getcwd()
    local_mlruns_dir = os.path.join(current_dir, 'mlruns')
    mlflow.set_tracking_uri(f"file:{local_mlruns_dir}")
    logging.info(f"Set MLflow tracking URI to: {mlflow.get_tracking_uri()}")

def load_model_local(model_name_pattern):
    global spark
    try:
        if spark is None:
            spark = initialize_spark(AWS_CREDENTIALS)
        
        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name("MS_Prediction_Models")
        if experiment is None:
            raise Exception("Experiment 'MS_Prediction_Models' not found")
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        latest_run = None
        for run in runs:
            artifacts = client.list_artifacts(run.info.run_id)
            if any(artifact.path == model_name_pattern for artifact in artifacts):
                if latest_run is None or run.info.start_time > latest_run.info.start_time:
                    latest_run = run
        
        if latest_run is None:
            raise FileNotFoundError(f"No run found with model: {model_name_pattern}")
        
        model_uri = f"runs:/{latest_run.info.run_id}/{model_name_pattern}"
        logging.info(f"Loading model from URI: {model_uri}")
        
        # Use mlflow.spark.load_model with the active Spark session
        loaded_model = mlflow.spark.load_model(model_uri, dfs_tmpdir=None)
        
        feature_columns = latest_run.data.params.get('feature_columns', '').split(',')
        target_column = latest_run.data.params.get('target_column', '')
        
        logging.info(f"Loaded model with features: {feature_columns} and target: {target_column}")
        
        return loaded_model, feature_columns, target_column, latest_run.info.run_id
    except Exception as e:
        logging.error(f"Error loading model from local path: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None

def load_models():
    global response_model, response_input_cols, response_target_col
    try:
        response_model, response_input_cols, response_target_col, run_id = load_model_local('Best_response_model')
        if response_model is None:
            raise Exception("Failed to load response model from local path")
        response_model.run_id = run_id  # Add run_id as an attribute to the model
        logging.info("Loaded response model from local path")
    except Exception as e:
        logging.error(f"Error loading response model: {str(e)}")
        raise

def predict_response(data):
    predictions = response_model.transform(data)
    result = predictions.select(
        when(col("prediction") == 1, "Yes")
        .otherwise("No")
        .alias("response")
    ).collect()[0]["response"]
    return result

# Flask app
app_response = Flask("Response_Prediction")

@app_response.route('/predict', methods=['POST'])
def api_predict_response():
    data = request.json
    df = spark.createDataFrame([data])
    prediction = predict_response(df)
    return jsonify({"response_prediction": prediction})  # This will be "Yes" or "No"

def get_or_create_spark_session():
    global spark
    if 'spark' not in globals() or spark is None:
        spark = initialize_spark(AWS_CREDENTIALS)
    return spark

def gradio_predict_response(age, gender, current_edss, lesion_new, wb_new, ms_type):
    spark = get_or_create_spark_session()
    input_dict = {
        "age": float(age),
        "gender": gender,
        "current_edss": float(current_edss),
        "lesion_new": float(lesion_new),
        "wb_new": float(wb_new),
        "ms_type": ms_type
    }
    df = spark.createDataFrame([input_dict])
    try:
        result = predict_response(df)
        return result  # This will now be "Yes" or "No"
    except Exception as e:
        logging.error(f"Error in response prediction: {str(e)}")
        return f"Error in prediction: {str(e)}"
        
def create_response_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Response Prediction")
        gr.Markdown("Enter patient data to predict treatment response")
        
        with gr.Row():
            age = gr.Number(label="Age")
            gender = gr.Radio(["F", "M"], label="Gender")
            current_edss = gr.Number(label="EDSS")
            lesion_new = gr.Number(label="Lesion")
            wb_new = gr.Number(label="WB")
            ms_type = gr.Dropdown(["RR", "APMS", "PPMS"], label="MS Type")
        
        output = gr.Textbox(label="Treatment Response")
        
        predict_btn = gr.Button("Predict")
        predict_btn.click(
            fn=gradio_predict_response,
            inputs=[age, gender, current_edss, lesion_new, wb_new, ms_type],
            outputs=output
        )

    return interface
    
def run_flask_app(app, port):
    app.run(debug=True, use_reloader=False, port=port)

def run_gradio_interface(interface, port):
    interface.launch(server_name='0.0.0.0', server_port=port, share=True)

def run_inference_apps():
    # Load models
    load_models()

    # Create Gradio interfaces
    response_interface = create_response_interface()
    
    # Close any existing Gradio instances
    gr.close_all()

    # Run Flask apps
    threading.Thread(target=run_flask_app, args=(app_response, 5002)).start()

    # Launch Gradio apps in separate threads
    threading.Thread(target=run_gradio_interface, args=(response_interface, 5001)).start()

    # Wait for all services to start
    time.sleep(5)

    # Keep the main thread alive
    while True:
        time.sleep(1)

def deploy_models():
    logging.info("Deploying models...")
    initialize_spark(AWS_CREDENTIALS)
    run_inference_apps()
    logging.info("Models deployed successfully.")

if __name__ == "__main__":
    initialize_mlflow()
    deploy_models()