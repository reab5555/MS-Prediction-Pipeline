import os
import pandas as pd
import io
from configs.cloud_config import get_s3_client, S3_CONFIG
from configs.utils import log_current_time


def load_raw_data_from_s3():
    s3_client = get_s3_client()
    file_key = 'MRI_NFL_cogn_Aug1-2024.xlsx'

    try:
        response = s3_client.get_object(Bucket=S3_CONFIG['bucket_name'], Key=file_key)
        excel_content = response['Body'].read()

        df = pd.read_excel(io.BytesIO(excel_content), dtype=str)
        
        log_current_time()

        return df, os.path.splitext(file_key)[0]
        
    except Exception as e:
        print(f"An error occurred while loading data from S3: {e}")
        return None, None