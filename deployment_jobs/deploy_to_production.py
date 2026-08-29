# Databricks notebook source
# DBTITLE 1,Setup - Get Job Parameters
# Get parameters passed from the job
dbutils.widgets.text("model_version", "3", "Model Version")
dbutils.widgets.text("endpoint_name", "example-titanic-serving-prd", "Production Endpoint Name")

model_version = dbutils.widgets.get("model_version")
endpoint_name = dbutils.widgets.get("endpoint_name")

print(f"🚀 Deploying model version {model_version} to PRODUCTION")
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

# DBTITLE 1,Verify Model from Staging
# Verify the model exists and has the staging alias
CATALOG_NAME = "ml_catalog"
SCHEMA_NAME = "titanic_schema"
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.example_titanic_model"

try:
    # Get model version details
    model_version_details = client.get_model_version(name=model_name, version=model_version)
    print(f"Found model: {model_name} version {model_version}")
    print(f"Status: {model_version_details.status}")
    
    # Verify it has staging alias
    staging_model = client.get_model_version_by_alias(name=model_name, alias="staging")
    if staging_model.version == model_version:
        print(f"Model version {model_version} is aliased as 'staging'")
    else:
        print(f"Warning: Staging alias points to version {staging_model.version}, not {model_version}")
except Exception as e:
    print(f"Error retrieving model: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,Deploy to Production Endpoint
# Configure the production endpoint
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
    print(f"Updating existing production endpoint: {endpoint_name}")
    deploy_client.update_endpoint(
        endpoint=endpoint_name,
        config=endpoint_config
    )
    print(f"✅ Production endpoint updated successfully")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"Creating new production endpoint: {endpoint_name}")
        deploy_client.create_endpoint(
            name=endpoint_name,
            config=endpoint_config
        )
        print(f"Production endpoint created successfully")
    else:
        print(f"Error deploying to production: {e}")
        raise

# COMMAND ----------

# DBTITLE 1,Wait for Endpoint Ready
import time

print("\nWaiting for production endpoint to be ready...")
max_wait_time = 600  # 10 minutes
start_time = time.time()

while True:
    endpoint = deploy_client.get_endpoint(endpoint_name)
    state = endpoint.get('state', {}).get('config_update', 'UNKNOWN')
    
    print(f"Endpoint state: {state}")
    
    if state == 'UPDATE_SUCCEEDED' or state == 'NOT_UPDATING':
        print("Production endpoint is ready!")
        break
    elif state in ['UPDATE_FAILED', 'UPDATE_CANCELED']:
        print(f"Endpoint update failed with state: {state}")
        raise Exception(f"Production endpoint deployment failed: {state}")
    
    if time.time() - start_time > max_wait_time:
        print("Timeout waiting for endpoint to be ready")
        raise Exception("Production endpoint deployment timeout")
    
    time.sleep(30)

# COMMAND ----------

# DBTITLE 1,Update Model Alias to Production
# Set the production alias on the model
client.set_registered_model_alias(
    name=model_name,
    alias="production",
    version=model_version
)

print(f"\n✅ Model alias 'production' set to version {model_version}")

# COMMAND ----------

# DBTITLE 1,Deployment Summary
print("\n" + "="*70)
print("🎉 PRODUCTION DEPLOYMENT COMPLETE")
print("="*70)
print(f"\n✅ Model: {model_name}")
print(f"✅ Version: {model_version}")
print(f"✅ Endpoint: {endpoint_name}")
print(f"✅ Alias: production")
print(f"\n🔗 Endpoint URL:")
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
print(f"   https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations")
print("\n📊 Next Steps:")
print("   1. Monitor endpoint metrics in the Serving UI")
print("   2. Run smoke tests against production")
print("   3. Monitor application logs for errors")
print("   4. Set up alerts for latency/error spikes")
print("="*70)

# COMMAND ----------

