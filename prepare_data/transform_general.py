import pandas as pd
import numpy as np
from datetime import datetime
from configs.cloud_config import S3_CONFIG, AWS_CREDENTIALS
from configs.utils import log_current_time
import awswrangler as wr
import boto3
import re
import json

def load_column_types(file_path='prepare_data/schemas/parquet_schema.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

def normalize_text(text):
    if isinstance(text, str):
        return text.encode('utf-8', errors='ignore').decode('ascii', errors='ignore')
    return text

def remove_symbols(text):
    if isinstance(text, str):
        return text.replace(';', '').replace(',', '').replace('@', '')  # add more symbols as needed
    return text

def is_valid_entry(x):
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x != '?' and x.strip() != ''
    return True

def normalize_column_name(name):
    # Convert to lowercase and replace spaces and special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading and trailing underscores
    return name.strip('_')

def safe_convert(value, dtype):
    try:
        if dtype == 'int64':
            return pd.to_numeric(value, errors='coerce').astype('Int64')
        elif dtype == 'float64':
            return pd.to_numeric(value, errors='coerce')
        elif dtype == 'datetime64[ns]':
            return pd.to_datetime(value, errors='coerce')
        elif dtype == 'bool':
            if isinstance(value, str):
                return value.lower() in ['true', 'yes', '1']
            return bool(value)
        elif dtype.startswith('string') or dtype.startswith('object'):
            return str(value) if pd.notna(value) else None
        else:
            return pd.to_numeric(value, errors='coerce')
    except:
        return None

def transform(raw_df, parquet_prefix, column_types_file='prepare_data/schemas/parquet_schema.json'):
    """
    Transform the DataFrame using pandas and save it as a Parquet file to S3.
    """
    log_current_time()

    # Load column types from JSON file
    column_types = load_column_types(column_types_file)

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = raw_df.copy()

    # Remove 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]

    # Rename columns
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Data cleaning and transformation steps
    columns_to_check = ['patient_id', 'age', 'gender', 'responder', 'ms_type', 'current_edss', 'current_treat', 'report', 'wb_new', 'lesion_new']
    df = df.dropna(subset=columns_to_check)

    # Filter rows where 'patient_id' is either an integer or a string that represents a digit
    df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['patient_id'])

    # Remove rows with '0' or 0 in 'current_treat' column
    df = df[df['current_treat'] != '0']

    # Normalize text encoding and handle non-English characters
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].apply(normalize_text)

    # Remove symbols from string columns
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].apply(remove_symbols)

    # Remove rows with '?' or empty/whitespace-only strings in any column
    df = df[df.apply(lambda row: row.map(is_valid_entry).all(), axis=1)]

    # Gender normalization
    df['gender'] = df['gender'].map({'M': 'M', 'F': 'F'})

    # Add 'Treatment Response' column
    df['response'] = df['responder'].map({'R': 'yes'}).fillna('no')

    # Ensure correct data types and set non-conforming cells to None
    for col, dtype in column_types.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_convert(x, dtype))

    # Create a folder with the current date time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{parquet_prefix}_ver_{current_time}"

    # Load table schemas from the JSON file
    with open('prepare_data/schemas/tables_schema.json', 'r') as file:
        table_schemas = json.load(file)

    # Configure AWS credentials for awswrangler
    boto3.setup_default_session(**AWS_CREDENTIALS)

    # Create separate DataFrames for each table and save as Parquet files
    for table_name, schema in table_schemas.items():
        table_columns = list(schema.keys())
        table_df = df[table_columns].copy()

        # Ensure correct data types
        for col, dtype in schema.items():
            if dtype == 'INTEGER':
                table_df[col] = table_df[col].apply(lambda x: safe_convert(x, 'int64'))
            elif dtype == 'FLOAT':
                table_df[col] = table_df[col].apply(lambda x: safe_convert(x, 'float64'))
            elif dtype == 'DATE':
                table_df[col] = table_df[col].apply(lambda x: safe_convert(x, 'datetime64[ns]'))
            elif dtype.startswith('VARCHAR'):
                table_df[col] = table_df[col].apply(lambda x: safe_convert(x, 'string'))
            elif dtype == 'BOOLEAN':
                table_df[col] = table_df[col].apply(lambda x: safe_convert(x, 'bool'))

        # Save the table DataFrame as a Parquet file to S3
        parquet_key = f"{folder_name}/{table_name}.parquet"
        s3_path = f"s3://{S3_CONFIG['bucket_name']}/db/{parquet_key}"
        parquet_s3_dir =  f"s3://{S3_CONFIG['bucket_name']}/db/{folder_name}"

        wr.s3.to_parquet(
            df=table_df,
            path=s3_path,
            index=False
        )

        print(f"Table {table_name} saved to S3 as {s3_path}")

        # Print the number of nulls in each column for this table
        null_counts = table_df.isnull().sum()
        print(f"\nNull counts in {table_name} table:")
        print(null_counts)
    
    log_current_time()
    
    # Returning the last table path, which is the ML features table
    return s3_path, parquet_s3_dir