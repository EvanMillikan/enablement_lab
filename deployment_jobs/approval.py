# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Manual Approval Gate
# MAGIC %md
# MAGIC # 🛡️ Manual Approval Required
# MAGIC
# MAGIC This task requires **manual approval** before proceeding to production deployment.
# MAGIC
# MAGIC ## Review Process
# MAGIC 1. Review the test results below
# MAGIC 2. Verify the staging endpoint is performing as expected
# MAGIC 3. Check for any anomalies or concerns
# MAGIC 4. If approved, manually trigger the production deployment job
# MAGIC
# MAGIC ## Decision Criteria
# MAGIC - ✅ All quality gates passed
# MAGIC - ✅ P99 latency meets requirements
# MAGIC - ✅ Success rate meets requirements
# MAGIC - ✅ No critical errors in staging
# MAGIC - ✅ Stakeholder approval obtained

# COMMAND ----------

# DBTITLE 1,Retrieve Test Results
import json

# Retrieve test results from the previous task
try:
    test_results_json = dbutils.jobs.taskValues.get(
        taskKey="run_validation_tests",
        key="test_results",
        debugValue="{}"  # Default if running outside job context
    )
    test_results = json.loads(test_results_json)
    
    print("Test results retrieved successfully")
except Exception as e:
    print(f"Running in debug mode (not part of job run)")
    # Mock data for testing the notebook independently
    test_results = {
        "endpoint_name": "example-titanic-serving-stg",
        "num_requests": 5000,
        "num_workers": 25,
        "total_time_sec": 45.2,
        "throughput_rps": 110.6,
        "success_rate": 1.0,
        "p99_latency_ms": 245.3,
        "p99_threshold_ms": 250.0,
        "quality_gates_passed": True
    }

# COMMAND ----------

# DBTITLE 1,Display Test Results Summary
print("="*70)
print("STAGING VALIDATION TEST RESULTS")
print("="*70)
print(f"\n🎯 Endpoint: {test_results.get('endpoint_name', 'N/A')}")
print(f"\n📊 Load Test Configuration:")
print(f"  • Total Requests: {test_results.get('num_requests', 'N/A'):,}")
print(f"  • Concurrent Workers: {test_results.get('num_workers', 'N/A')}")
print(f"  • Total Time: {test_results.get('total_time_sec', 0):.1f} seconds")
print(f"\n⚡ Performance Metrics:")
print(f"  • Throughput: {test_results.get('throughput_rps', 0):.1f} requests/second")
print(f"  • Success Rate: {test_results.get('success_rate', 0)*100:.1f}%")
print(f"  • P99 Latency: {test_results.get('p99_latency_ms', 0):.1f}ms")
print(f"  • P99 Threshold: {test_results.get('p99_threshold_ms', 0):.0f}ms")

if test_results.get('quality_gates_passed', False):
    print(f"\nStatus: All quality gates PASSED")
else:
    print(f"\nStatus: Quality gates FAILED")

print("="*70)

# COMMAND ----------

# DBTITLE 1,Approval Instructions
# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC ### To Approve and Deploy to Production:
# MAGIC
# MAGIC 1. **Review** the test results above carefully
# MAGIC 2. **Verify** staging endpoint performance in the UI
# MAGIC 3. **Get stakeholder approval** if required
# MAGIC 4. **Manually trigger** the production deployment job:
# MAGIC    - Go to the Jobs UI
# MAGIC    - Find the `Titanic_Production_Deployment` job
# MAGIC    - Click "Run Now"
# MAGIC    - Pass the same `model_version` parameter
# MAGIC
# MAGIC ### To Reject:
# MAGIC
# MAGIC - Simply do not trigger the production deployment job
# MAGIC - Investigate issues and retry staging deployment with fixes

# COMMAND ----------

# DBTITLE 1,Notification Helper
# Optional: Send notification to stakeholders
# This is a placeholder - integrate with your notification system

print("\n📧 Approval Required Notification")
print("Send this information to your deployment approvers:")
print(f"\nSubject: Approval Required - Model v{dbutils.widgets.get('model_version')} Staging Validation Complete")
print(f"\nStaging validation completed successfully.")
print(f"Review results and approve for production deployment.")
print(f"\nTest Summary:")
print(f"  - Success Rate: {test_results.get('success_rate', 0)*100:.1f}%")
print(f"  - P99 Latency: {test_results.get('p99_latency_ms', 0):.1f}ms")
print(f"  - Throughput: {test_results.get('throughput_rps', 0):.1f} req/s")
print(f"\nAction: Manually change approval_tag_name to 'Approved' to continue deployment to production.")

# COMMAND ----------

# DBTITLE 1,Waiting for Manual Approval
from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

# by default, the approval tag name here is populated with the approval task name
tag_name = dbutils.widgets.get("approval_tag_name")

# fetch the model version's UC tags
tags = client.get_model_version(model_name, model_version).tags

# check if any tag matches the approval tag name
if not any(tag == tag_name for tag in tags.keys()):
  raise Exception("Model version not approved for deployment")
else:
  # if tag is found, check if it is approved
  if tags.get(tag_name).lower() == "approved":
    print("Model version approved for deployment")
  else:
    raise Exception("Model version not approved for deployment")

# COMMAND ----------

