# Makefile for setting up Airflow in a specific environment and Docker operations

# Variables
ENV_DIR = env
AIRFLOW_DIR = airflow_home
REQUIREMENTS = requirements.txt
SCRIPT = main_pipeline.py
DAGS_DIR = $(AIRFLOW_DIR)/dags
IMAGE_NAME = ms_ml_pipe_run
TAG = latest
DOCKER_FILE = Dockerfile
CONTAINER_NAME = ml-pipeline-container
ADMIN_USERNAME = admin
ADMIN_PASSWORD = admin
ADMIN_EMAIL = admin@example.com

# Default command
all: setup_airflow move_files build run_webserver run_scheduler

# Create a Python environment in the `env` directory and install Airflow and dependencies
setup_airflow:
	python3 -m venv $(ENV_DIR)
	$(ENV_DIR)/bin/pip install --no-cache-dir -r $(REQUIREMENTS)

# Move the installed Airflow folder to the same directory as the Makefile
move_files: setup_airflow
	mkdir -p $(AIRFLOW_DIR)
	cp -r $(ENV_DIR)/lib/python*/site-packages/airflow $(AIRFLOW_DIR)/
	mkdir -p $(DAGS_DIR)
	cp $(SCRIPT) $(AIRFLOW_DIR)/
	cp $(SCRIPT) $(DAGS_DIR)/

# Initialize the Airflow database
init_db:
	export AIRFLOW_HOME=$(PWD)/$(AIRFLOW_DIR) && \
	$(ENV_DIR)/bin/airflow db init

# Build the Docker image
build: move_files init_db
	docker build -t $(IMAGE_NAME):$(TAG) -f $(DOCKER_FILE) .

# Run the Docker container
run:
	docker run --name $(CONTAINER_NAME) -p 8080:8080 $(IMAGE_NAME):$(TAG)

# Run the Docker container interactively
run-interactive:
	docker run -it --name $(CONTAINER_NAME) $(IMAGE_NAME):$(TAG) /bin/bash

# Start the Airflow webserver
run_webserver:
	export AIRFLOW_HOME=$(PWD)/$(AIRFLOW_DIR) && \
	$(ENV_DIR)/bin/airflow webserver --port 8080 &

# Start the Airflow scheduler
run_scheduler:
	export AIRFLOW_HOME=$(PWD)/$(AIRFLOW_DIR) && \
	$(ENV_DIR)/bin/airflow scheduler &

# Create the Airflow admin user with a random password
create_user:
	export AIRFLOW_HOME=$(PWD)/$(AIRFLOW_DIR) && \
	$(ENV_DIR)/bin/airflow users create \
		--username $(ADMIN_USERNAME) \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email $(ADMIN_EMAIL) \
		--password $(ADMIN_PASSWORD) && \
	echo "Admin user '$(ADMIN_USERNAME)' created with password: $(ADMIN_PASSWORD)"

# Stop the Docker container
stop:
	docker stop $(CONTAINER_NAME)

# Remove the Docker container
remove:
	docker rm $(CONTAINER_NAME)

# Remove the Docker image
remove-image:
	docker rmi $(IMAGE_NAME):$(TAG)

# Clean up: stop and remove container, and remove image
clean: stop remove remove-image
	rm -rf $(ENV_DIR) $(AIRFLOW_DIR)

# Push the image to a registry (replace with your registry details)
push:
	docker push $(IMAGE_NAME):$(TAG)

# Pull the image from a registry
pull:
	docker pull $(IMAGE_NAME):$(TAG)

# Display the list of running containers
ps:
	docker ps

# Display the list of all containers (including stopped ones)
ps-all:
	docker ps -a

# Display the list of images
images:
	docker images

# Help command to display available commands
help:
	@echo "Available commands:"
	@echo "  make              - Set up Airflow, build the Docker image, and run the container"
	@echo "  make setup_airflow - Set up Airflow and install dependencies"
	@echo "  make move_files    - Move necessary files to the Airflow directory"
	@echo "  make init_db       - Initialize the Airflow database"
	@echo "  make build         - Build the Docker image"
	@echo "  make run           - Run the Docker container"
	@echo "  make run-interactive - Run the Docker container interactively"
	@echo "  make run_webserver - Initialize the Airflow database and start the Airflow webserver"
	@echo "  make run_scheduler - Start the Airflow scheduler"
	@echo "  make create_user   - Create the Airflow admin user with generated credentials"
	@echo "  make stop          - Stop the Docker container"
	@echo "  make remove        - Remove the Docker container"
	@echo "  make remove-image  - Remove the Docker image"
	@echo "  make clean         - Clean up: stop and remove container, remove image, and clean directories"
	@echo "  make push          - Push the image to a registry"
	@echo "  make pull          - Pull the image from a registry"
	@echo "  make ps            - Display running containers"
	@echo "  make ps-all        - Display all containers"
	@echo "  make images        - Display Docker images"

.PHONY: all setup_airflow move_files init_db build run run-interactive run_webserver run_scheduler create_user stop remove remove-image clean push pull ps ps-all images help
