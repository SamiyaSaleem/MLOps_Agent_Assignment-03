import os

import mlflow
from mlflow.tracking import MlflowClient

from agent.tools.mlflow_tools import find_best_run

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = "BreastCancerClassifier"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

result = find_best_run(
    experiment_name="MLOps_Assignment_03",
    metric="f1_score",
)

if result["status"] != "success" or not result["best_run"]:
    raise RuntimeError("No suitable completed training run found")

best = result["best_run"]

run_id = best["run_id"]
model_uri = best["model_uri"]

if not model_uri:
    raise RuntimeError("Selected run has no recorded model URI")

if best["dataset_version"] != "dataset-v2":
    raise RuntimeError("Unexpected dataset version")

print("Selected model:", best["run_name"])
print("Run ID:", run_id)
print("Model URI:", model_uri)
print("F1 score:", best["metrics"]["f1_score"])

# Avoid creating duplicate versions when rerunning this script.
existing = client.search_model_versions(
    filter_string=f"name = '{MODEL_NAME}'"
)

matching = [
    version for version in existing
    if version.run_id == run_id
]

if matching:
    version = matching[0]
    print("Model already registered.")
else:
    version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        tags={
            "dataset_version": best["dataset_version"],
            "evaluation_metric": "f1_score",
            "approval_status": "candidate",
        },
    )

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="candidate",
    version=version.version,
)

print("\nMODEL REGISTRATION SUCCESSFUL")
print("Name:", MODEL_NAME)
print("Version:", version.version)
print("Alias: candidate")
print("Source Run:", run_id)
