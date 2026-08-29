# Databricks notebook source
# DBTITLE 1,Setup - Get Job Parameters
# Get parameters passed from the job
dbutils.widgets.text("model_version", "3", "Model Version")
dbutils.widgets.text("endpoint_name", "example-titanic-serving-stg", "Staging Endpoint Name")

model_version = dbutils.widgets.get("model_version")
endpoint_name = dbutils.widgets.get("endpoint_name")

print(f"Deploying model version: {model_version}")
print(f"Target endpoint: {endpoint_name}")

# COMMAND ----------

# DBTITLE 1,Import Libraries
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.deployments import get_deploy_client

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
deploy_client = get_deploy_client("databricks")

# COMMAND ----------

# DBTITLE 1,Get Model from Registry
# Get the model from Unity Catalog
CATALOG_NAME = "ml_catalog"
SCHEMA_NAME = "titanic_schema"
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.example_titanic_model"

try:
    model_version_details = client.get_model_version(name=model_name, version=model_version)
    print(f"✅ Found model: {model_name} version {model_version}")
    print(f"Status: {model_version_details.status}")
except Exception as e:
    print(f"Error retrieving model: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,Deploy to Staging Endpoint
# Configure the staging endpoint
endpoint_config = {
    "served_entities": [
        {
            "name": f"titanic-model-v{model_version}",
            "entity_name": model_name,
            "entity_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": f"titanic-model-v{model_version}",
                "traffic_percentage": 100
            }
        ]
    }
}

# Deploy or update the endpoint
try:
    existing_endpoint = deploy_client.get_endpoint(endpoint_name)
    print(f"Updating existing endpoint: {endpoint_name}")
    deploy_client.update_endpoint(
        endpoint=endpoint_name,
        config=endpoint_config
    )
    print(f"✅ Staging endpoint updated successfully")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"Creating new endpoint: {endpoint_name}")
        deploy_client.create_endpoint(
            name=endpoint_name,
            config=endpoint_config
        )
        print(f"Staging endpoint created successfully")
    else:
        print(f"An error occurred: {e}")
        raise

# COMMAND ----------

# DBTITLE 1,Wait for Endpoint Ready
import time

print("Waiting for endpoint to be ready...")
max_wait_time = 600  # 10 minutes
start_time = time.time()

while True:
    endpoint = deploy_client.get_endpoint(endpoint_name)
    state = endpoint.get('state', {}).get('config_update', 'UNKNOWN')
    
    print(f"Endpoint state: {state}")
    
    if state == 'UPDATE_SUCCEEDED' or state == 'NOT_UPDATING':
        print("Endpoint is ready!")
        break
    elif state in ['UPDATE_FAILED', 'UPDATE_CANCELED']:
        print(f"Endpoint update failed with state: {state}")
        raise Exception(f"Endpoint deployment failed: {state}")
    
    if time.time() - start_time > max_wait_time:
        print("imeout waiting for endpoint to be ready")
        raise Exception("Endpoint deployment timeout")
    
    time.sleep(30)

# COMMAND ----------

# DBTITLE 1,Update Model Alias
# Set the staging alias on the model
client.set_registered_model_alias(
    name=model_name,
    alias="staging",
    version=model_version
)

print(f"✅ Model alias 'staging' set to version {model_version}")
print(f"\n🎯 Deployment to staging complete!")
print(f"Endpoint: {endpoint_name}")
print(f"Model: {model_name} v{model_version}")

# COMMAND ----------

