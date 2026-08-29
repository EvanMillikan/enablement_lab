# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Overview - Batch Inference
# MAGIC %md
# MAGIC # Overview
# MAGIC
# MAGIC In this notebook, we will go through how to create a Databricks jobs to run batch inference deployment. 
# MAGIC
# MAGIC Batch inference is used when you don't need a real-time inference only need to run inference in a scheduled timer (Daily, weekly, etc.) 
# MAGIC
# MAGIC ## Assumption
# MAGIC
# MAGIC This notebook assume that you have followed the proper approval workflow to get your model pre-approved for production.
# MAGIC
# MAGIC The difference from real-time deployment workflow is that you can skip the serving part. But you still need to make sure that your model passes the launch review meeting.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup - Import Libraries
import mlflow
from mlflow.tracking.client import MlflowClient
import pandas as pd
from pyspark.sql import functions as F
from datetime import datetime

# Set MLflow registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()


# COMMAND ----------

# DBTITLE 1,Configuration
# Unity Catalog configuration
CATALOG_NAME = "ml_catalog"
SCHEMA_NAME = "titanic_schema"

TEST_TABLE_NAME = "test" # Load from test data; In real-life usecase, this would be your base table (i.e. user base) 
INFERENCE_RESULT_TABLE_NAME = "titanic_prediction" # output table 

# Model configuration
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.example_titanic_model"
model_alias = "production"

print(f"Source Table: {CATALOG_NAME}.{SCHEMA_NAME}.{TEST_TABLE_NAME}")
print(f"Output Table: {CATALOG_NAME}.{SCHEMA_NAME}.{INFERENCE_RESULT_TABLE_NAME}")
print(f"Model: {model_name}@{model_alias}")

# COMMAND ----------

# DBTITLE 1,Load Test Data
# Load test data from Unity Catalog
test_data = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{TEST_TABLE_NAME}").toPandas()

# COMMAND ----------

# DBTITLE 1,Load Production Model
# Get the production model version
try:
    registered_model = client.get_model_version_by_alias(
        name=model_name,
        alias=model_alias
    )
    
    print(f"Loaded model from production")
    print(f"Model Name: {registered_model.name}")
    print(f"Model Version: {registered_model.version}")
    print(f"Description: {registered_model.description}")
    
    # Load the model
    model_uri = f"models:/{model_name}@{model_alias}" # You can use this URI to load a model that you aliased
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    print(f"\nModel loaded successfully")
    print(f"Model URI: {model_uri}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# COMMAND ----------

# DBTITLE 1,Prepare Features for Inference
# Prepare Feature for Inference
features = ["Fare", "Age", "Pclass", "SibSp", "Parch", "Sex"] 
X_test = test_data[features]

# COMMAND ----------

# DBTITLE 1,Run Batch Inference
# Run Prediction
predictions = loaded_model.predict(X_test)

# COMMAND ----------

# Create results DataFrame
test_data['prediction'] = predictions

# COMMAND ----------

# DBTITLE 1,Prepare Results DataFrame
# Prepare the Results Dataframe
# We only want to keep the Primary Key + Predictions + Metadata(s)
results_pd = test_data[['PassengerId']].copy()
results_pd['prediction'] = predictions
results_pd['model_version'] = registered_model.version
results_pd['model_name'] = model_name
results_pd['inference_timestamp'] = datetime.now()

# COMMAND ----------

# DBTITLE 1,Save Results to Unity Catalog
# Convert to Spark DataFrame
results_spark = spark.createDataFrame(results_pd)

# Save to Unity Catalog table
target_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.{INFERENCE_RESULT_TABLE_NAME}"

print(f"Saving results to {target_table}...")

# Write to partitioned table - So we can use the same output table, but new data will be written to a new partition
results_spark.write.mode("append").partitionBy("inference_timestamp").saveAsTable(target_table)

print(f"Results saved successfully to {target_table}")
print(f"Total rows written: {results_spark.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Schedule your batch job
# MAGIC
# MAGIC On top right, click on schedule -> add schedule -> configure schedule settings -> create job

# COMMAND ----------

