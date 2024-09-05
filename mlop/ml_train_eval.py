import os
import json
import time
from datetime import datetime
import mlflow
import mlflow.spark
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from tqdm import tqdm

from configs.cloud_config import AWS_CREDENTIALS
from configs.utils import log_current_time
from mlop.mlflow_config import setup_mlflow, copy_mlruns_to_s3

def initialize_spark(AWS_CREDENTIALS):
    return SparkSession.builder \
        .appName("ML Train") \
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
        
def load_model_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def get_model_instance(model_name):
    model_map = {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "GBTClassifier": GBTClassifier
    }
    model_class = model_map[model_name]
    return model_class(featuresCol="features", labelCol="response")

def get_classification_models(config):
    return [(name, get_model_instance(name)) for name in config['classification_models']]

def get_param_grid(model, config):
    model_name = model.__class__.__name__
    if model_name in config['hyperparameters']:
        param_grid = ParamGridBuilder()
        for param, values in config['hyperparameters'][model_name].items():
            param_grid.addGrid(getattr(model, param), values)
        return param_grid.build()
    else:
        return ParamGridBuilder().build()
        
def get_feature_importance(model, feature_cols):
    if hasattr(model, 'featureImportances'):
        importances = model.featureImportances.toArray()
    elif hasattr(model, 'stages') and hasattr(model.stages[-1], 'featureImportances'):
        importances = model.stages[-1].featureImportances.toArray()
    else:
        print("Model doesn't have feature importances")
        return None
    
    # Create a list of tuples (feature, importance)
    feature_importance = list(zip(feature_cols, importances))
    # Sort the features by importance in descending order
    sorted_feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    return sorted_feature_importance

def train_and_evaluate(df_clean, pipe_stages, models, config):
    overall_best_model = None
    overall_best_metrics = None
    overall_best_metric = float('-inf')
    overall_best_name = None
    overall_best_params = None
    overall_best_feature_importance = None

    target = 'response'
    features = ['age_scaled', 'lesion_new_scaled', 'wb_new_scaled', 'current_edss_scaled', 'gender_index', 'ms_type_vec']
    
    with mlflow.start_run(run_name=f"Run_{target}_{int(time.time())}"):

        for name, model in tqdm(models, desc="Training models", unit="model"):
            print(f"\nTraining model: {name}")
            
            # Create the full pipeline
            pipeline = Pipeline(stages=pipe_stages + [model])

            param_grid = get_param_grid(model, config)

            evaluator = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction", metricName="f1")
            
            cv = CrossValidator(estimator=pipeline,
                                      estimatorParamMaps=param_grid,
                                      evaluator=evaluator,
                                      numFolds=5)

            cv_model = cv.fit(df_clean)
            cv_model.getNumFolds()
                
            best_pipeline_model = cv_model.bestModel
            
            val_predictions = best_pipeline_model.transform(df_clean)
                
            metrics = {
                    "F1": evaluator.evaluate(val_predictions),
                    "Precision": evaluator.setMetricName("weightedPrecision").evaluate(val_predictions),
                    "Recall": evaluator.setMetricName("weightedRecall").evaluate(val_predictions)
                }
                
            comparison_metric = metrics["F1"]
                
            if comparison_metric > overall_best_metric:
                    overall_best_metric = comparison_metric
                    overall_best_model = best_pipeline_model
                    overall_best_metrics = metrics
                    overall_best_name = name
                    overall_best_params = best_pipeline_model.stages[-1].extractParamMap()
                    overall_best_feature_importance = get_feature_importance(best_pipeline_model, features)
                
            print(f"Model: {name}, Metrics: {metrics}")
            print(overall_best_feature_importance)

        if overall_best_model is None:
            print("No successful model was trained. Please check the data and model configurations.")
            return None, None, None, None, None
    
        # Log and register the model
        model_log_name = "Best_response_model"
        print(f'Logging {model_log_name}...')
        mlflow.spark.log_model(overall_best_model, model_log_name)
        mlflow.log_param("best_algorithm", overall_best_name) 
        for param, value in overall_best_params.items():
            mlflow.log_param(f"best_{param.name}", value)
        for metric_name, value in overall_best_metrics.items():
            mlflow.log_metric(f"best_{metric_name}", value)
        if overall_best_feature_importance:
            mlflow.log_dict(dict(overall_best_feature_importance), f"feature_importance.json")
        mlflow.log_param("target_column", target)
        mlflow.log_param("feature_columns", ','.join(features))
            
        # Register the best model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_log_name}"
        registered_model_name = f"{model_log_name}_{overall_best_name}"
        mlflow.register_model(model_uri, registered_model_name)
        print(f"Registered model '{registered_model_name}' with URI: {model_uri}")
        
    log_current_time()

    return overall_best_model, overall_best_metrics, overall_best_name, overall_best_feature_importance, overall_best_params


def train_models(df_clean, pipe_stages, artifact_uri, config_path='mlop/ml_tuning.json'):
    spark = initialize_spark(AWS_CREDENTIALS)

    config = load_model_config(config_path)

    # Response Prediction (Classification)
    print("\nTraining classification models for response prediction...")
    overall_best_model, overall_best_metrics, overall_best_name, overall_best_feature_importance, overall_best_params = train_and_evaluate(
        df_clean,
        pipe_stages,
        get_classification_models(config), 
        config=config
    )
    
    spark.stop()
    
    log_current_time()
