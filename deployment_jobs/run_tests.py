# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup - Get Job Parameters
# Get parameters passed from the job
dbutils.widgets.text("endpoint_name", "example-titanic-serving-stg", "Endpoint Name")
dbutils.widgets.text("num_requests", "5000", "Number of Requests")
dbutils.widgets.text("num_workers", "25", "Concurrent Workers")
dbutils.widgets.text("p99_threshold_ms", "250", "P99 Latency Threshold (ms)")

endpoint_name = dbutils.widgets.get("endpoint_name")
num_requests = int(dbutils.widgets.get("num_requests"))
num_workers = int(dbutils.widgets.get("num_workers"))
p99_threshold_ms = float(dbutils.widgets.get("p99_threshold_ms"))

print(f"Testing endpoint: {endpoint_name}")
print(f"Load test: {num_requests} requests with {num_workers} workers")
print(f"P99 threshold: {p99_threshold_ms}ms")

# COMMAND ----------

# DBTITLE 1,Import Libraries
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Prepare Test Data
CATALOG_NAME = "ml_catalog"
SCHEMA_NAME = "titanic_schema"
TRAIN_TABLE_NAME = "train"

train = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.{TRAIN_TABLE_NAME}").toPandas()
features = ["Fare", "Age", "Pclass", "SibSp", "Parch", "Sex"] 
X = train[features]
y = train["Survived"]

X = X.head(1)

payload = {"dataframe_split": X.to_dict(orient="split")}
payload_json = json.dumps(payload, allow_nan=True)

print("Test payload prepared")

# COMMAND ----------

# DBTITLE 1,Setup Endpoint Connection
# Get authentication token
token = dbutils.secrets.get(scope='my-secrets', key='databricks-token')

# Construct endpoint URL
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
url = f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

print(f"Endpoint URL: {url}")

# COMMAND ----------

# DBTITLE 1,Load Test Functions
def send_request(request_id):
    """Send a single request and record metrics"""
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, data=payload_json, timeout=30)
        latency = time.time() - start_time
        
        return {
            'request_id': request_id,
            'status_code': response.status_code,
            'latency': latency,
            'success': response.status_code == 200,
            'response': response.json() if response.status_code == 200 else None,
            'error': None
        }
    except Exception as e:
        latency = time.time() - start_time
        return {
            'request_id': request_id,
            'status_code': None,
            'latency': latency,
            'success': False,
            'response': None,
            'error': str(e)
        }

def load_test(num_requests, num_workers):
    """Run load test with concurrent requests"""
    print(f"Starting load test: {num_requests} requests with {num_workers} concurrent workers...\n")
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 1000 == 0:
                print(f"Completed {len(results)}/{num_requests} requests")
    
    total_time = time.time() - start_time
    
    return results, total_time

# COMMAND ----------

# DBTITLE 1,Run Load Test
# Execute the load test
results, total_time = load_test(num_requests, num_workers)

# Calculate metrics
latencies = [r['latency'] for r in results]
successes = [r for r in results if r['success']]
failures = [r for r in results if not r['success']]

print("\n" + "="*60)
print("LOAD TEST RESULTS")
print("="*60)
print(f"Total requests: {num_requests}")
print(f"Concurrent workers: {num_workers}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Throughput: {num_requests/total_time:.2f} requests/second")
print(f"\nSuccess rate: {len(successes)/num_requests*100:.2f}% ({len(successes)}/{num_requests})")
print(f"Failed requests: {len(failures)}")
print(f"\nLatency statistics (seconds):")
print(f"  Min: {np.min(latencies):.3f}")
print(f"  Max: {np.max(latencies):.3f}")
print(f"  Mean: {np.mean(latencies):.3f}")
print(f"  Median: {np.median(latencies):.3f}")
print(f"  P95: {np.percentile(latencies, 95):.3f}")
print(f"  P99: {np.percentile(latencies, 99):.3f}")

if failures:
    print(f"\nError samples:")
    for f in failures[:5]:
        print(f"  Request {f['request_id']}: {f['error'] or f'HTTP {f['status_code']}'}") 

# COMMAND ----------

# DBTITLE 1,Quality Gate Validation
# Validate against thresholds
p99_latency = np.percentile(latencies, 99)
success_rate = len(successes) / num_requests

P99_THRESHOLD_SEC = p99_threshold_ms / 1000  # Convert to seconds
SUCCESS_RATE_THRESHOLD = 0.95

print("\n" + "="*60)
print("QUALITY GATE VALIDATION")
print("="*60)

# Check P99 latency
p99_passed = p99_latency <= P99_THRESHOLD_SEC
print(f"{'PASS' if p99_passed else 'FAIL'} | P99 Latency: {p99_latency*1000:.1f}ms (Required: <= {p99_threshold_ms}ms)")

# Check success rate
success_rate_passed = success_rate >= SUCCESS_RATE_THRESHOLD
print(f"{'PASS' if success_rate_passed else 'FAIL'} | Success Rate: {success_rate*100:.1f}% (Required: >= {SUCCESS_RATE_THRESHOLD*100:.0f}%)")

print("="*60)

# Fail the job if quality gates don't pass
if not (p99_passed and success_rate_passed):
    error_msg = "Quality gate validation failed. Production deployment blocked."
    print(f"\n{error_msg}")
    raise Exception(error_msg)
else:
    print("\nAll quality gates passed! Ready for production deployment.")

# COMMAND ----------

# DBTITLE 1,Save Test Results
# Save test results to dbutils for the approval step
test_summary = {
    "endpoint_name": endpoint_name,
    "num_requests": num_requests,
    "num_workers": num_workers,
    "total_time_sec": total_time,
    "throughput_rps": num_requests / total_time,
    "success_rate": success_rate,
    "p99_latency_ms": p99_latency * 1000,
    "p99_threshold_ms": p99_threshold_ms,
    "quality_gates_passed": True
}

# Store as job task value for downstream tasks to access
dbutils.jobs.taskValues.set(key="test_results", value=json.dumps(test_summary))

print("\nTest results saved for approval review")

# COMMAND ----------

