# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /workspace

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install AWS CLI
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws

# Copy the main script
COPY main_pipeline.py .

# Copy deploy script
COPY deploy.py .

# Copy Airflow script
COPY airflow_pipe_run.py .

# Copy other necessary files
COPY prepare_data/extraction.py prepare_data/
COPY prepare_data/transform_general.py prepare_data/
COPY prepare_data/schemas/parquet_schema.json prepare_data/schemas/
COPY prepare_data/schemas/tables_schema.json prepare_data/schemas/
COPY mlop/transform_ml.py mlop/
COPY mlop/ml_train_eval.py mlop/
COPY mlop/mlflow_config.py mlop/
COPY mlop/ml_tuning.json mlop/
COPY configs/utils.py configs/
COPY configs/cloud_config.py configs/

# Command to run the main script
CMD ["python", "main_pipeline.py"]
