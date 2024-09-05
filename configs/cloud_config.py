import os
import boto3

# AWS Configuration
AWS_CREDENTIALS = {
    'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
    'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
    'region_name': os.environ.get('AWS_REGION', 'eu-central-1')
}

# S3 Configuration
S3_CONFIG = {
    'bucket_name': os.environ.get('S3_BUCKET_NAME', 'msnflcognbucket')
}

def get_s3_client():
    return boto3.client('s3', **AWS_CREDENTIALS)