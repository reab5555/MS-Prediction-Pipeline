from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, expr, count, lower
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import MinMaxScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array as ml_vector_to_array
from configs.cloud_config import AWS_CREDENTIALS
from configs.utils import log_current_time
import numpy as np
from datetime import datetime
import json
import os
    
def ml_prepare(ml_ready_table_path):
    ml_ready_table_path = ml_ready_table_path.replace('s3://', 's3a://')
    
    # Initialize Spark session with updated configurations
    spark = SparkSession.builder \
        .appName("ML") \
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

    
    # Read the Parquet file
    df = spark.read.parquet(ml_ready_table_path)
    
    # Remove rows with NA values in any column
    df_clean_1 = df.dropna()
    
    df_clean_2 = df_clean_1.drop('patient_id')
    
    # Filter rows to keep only specified `ms_type` values
    allowed_ms_types = ['RR', 'APMS', 'PPMS']
    df_clean_3 = df_clean_2.filter(col('ms_type').isin(allowed_ms_types))

    # Count the occurrences of each ms_type
    ms_type_counts = df_clean_3.groupBy('ms_type').agg(count('*').alias('count')).collect()
    
    # Keep only ms_types that are actually present in the data
    present_ms_types = [row['ms_type'] for row in ms_type_counts if row['count'] > 10]
    
    # Filter the dataframe again to keep only the present ms_types
    df_clean_4 = df_clean_3.filter(col('ms_type').isin(present_ms_types))

    # Verify the filtering
    for row in df_clean_4.groupBy('ms_type').agg(count('*').alias('count')).collect():
        print(f"{row['ms_type']}: {row['count']}")
    
    # Function to remove outliers using IQR method
    def remove_outliers(df, columns):
        for column in columns:
            # Compute Q1, Q3, and IQR
            quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
            Q1 = quantiles[0]
            Q3 = quantiles[1]
            IQR = Q3 - Q1
            
            # Define lower and upper bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out the outliers
            df = df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))
        
        log_current_time()

        return df
    
    # Remove outliers from specified columns
    columns_for_outlier_removal = ['age', 'lesion_new', 'wb_new']
    df_clean_5 = remove_outliers(df_clean_4, columns_for_outlier_removal)
    
    
    # Convert 'yes' and 'no' to 1 and 0, and ensure the column is of IntegerType
    df_clean = df_clean_5.withColumn('response',
                                     when(lower(col('response')) == 'yes', 1)
                                     .when(lower(col('response')) == 'no', 0)
                                     .otherwise(None)
                                     .cast(IntegerType()))

    pipe_stages = []
    
    # Gender encoding (F=0, M=1)
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index", stringOrderType="alphabetDesc")
    pipe_stages += [gender_indexer]
    
    # One-hot encoding for ms_type
    string_indexer = StringIndexer(inputCol='ms_type', outputCol='ms_type_index')
    pipe_stages += [string_indexer]
    
    encoder = OneHotEncoder(inputCols=['ms_type_index'], outputCols=['ms_type_vec'], dropLast=False)
    pipe_stages += [encoder]
        
    # Combine numeric columns into a single vector column
    numeric_cols = ['age', 'lesion_new', 'wb_new', 'current_edss']
    encoded_cols = ["gender_index", "ms_type_vec"]
    assembler = VectorAssembler(inputCols=numeric_cols + encoded_cols, outputCol="features")
    pipe_stages.append(assembler)
    
    # Min-Max scaling for numeric features
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    pipe_stages.append(scaler)
    
    log_current_time()

    return df_clean, pipe_stages