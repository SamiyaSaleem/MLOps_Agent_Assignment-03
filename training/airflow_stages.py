import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agent.tools.dvc_tools import get_current_dataset_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/train.csv"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

EXPERIMENT_NAME = "MLOps_Assignment_03"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def get_run_directory(run_key):
    """Store artifacts separately for every Airflow DAG run."""

    if not run_key.isalnum() or len(run_key) > 64:
        raise ValueError("Invalid pipeline run identifier")

    directory = PROJECT_ROOT / ".pipeline_runs" / run_key
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def save_json(path, data):
    """Save structured execution evidence."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path):
    """Load execution evidence from an earlier stage."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_provenance():
    """Retrieve verified DVC dataset provenance."""

    result = get_current_dataset_version()

    if result["status"] != "success":
        error_message = result.get(
            "error",
            "Unknown DVC error",
        )

        logger.error(
            "DVC provenance lookup failed: %s",
            error_message,
        )

        raise RuntimeError(
            f"DVC provenance lookup failed: {error_message}"
        )

    if not result["workspace_clean"]:
        raise RuntimeError(
            "DVC dataset has uncommitted changes"
        )

    versions = result["version_tags"]

    if not versions:
        raise RuntimeError(
            "Dataset has no recorded DVC version"
        )

    version = sorted(
        versions,
        key=lambda value: int(value.split("-v")[-1]),
    )[-1]

    dataset_commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            f"refs/tags/{version}",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        timeout=15,
    ).strip()

    return {
        "dataset_version": version,
        "dataset_commit": dataset_commit,
        "dataset_hash": result["dvc_hash"],
    }

def validate_dataset(run_key):
    """Stage 1: Validate the DVC-tracked dataset."""

    logger.info("STAGE 1: VALIDATE DATASET")

    if not DATA_PATH.exists():
        raise FileNotFoundError("Training dataset is missing")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty")

    if "target" not in df.columns:
        raise ValueError("Target column is missing")

    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values")

    if df["target"].nunique() != 2:
        raise ValueError("Expected binary classification labels")

    if len(df) < 100:
        raise ValueError("Insufficient training records")

    if len(df.columns) != 31:
        raise ValueError("Unexpected dataset schema")

    provenance = get_provenance()

    result = {
        "status": "passed",
        "rows": len(df),
        "columns": len(df.columns),
        "dataset_provenance": provenance,
    }

    directory = get_run_directory(run_key)

    save_json(directory / "validation.json", result)

    logger.info("Dataset validation successful")
    logger.info("Dataset rows: %s", len(df))
    logger.info("DVC version: %s", provenance["dataset_version"])

    return result


def preprocess_data(run_key):
    """Stage 2: Create reproducible train/test datasets."""

    logger.info("STAGE 2: PREPROCESS DATA")

    directory = get_run_directory(run_key)

    validation = load_json(directory / "validation.json")

    if validation["status"] != "passed":
        raise RuntimeError("Dataset validation did not pass")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Preserve feature names and prevent test-set leakage.
    # Feature scaling happens inside the training pipeline,
    # where StandardScaler is fitted only on training data.

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        directory / "processed_data.joblib",
    )

    result = {
        "status": "passed",
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }

    save_json(directory / "preprocessing.json", result)

    logger.info("Training records: %s", len(X_train))
    logger.info("Testing records: %s", len(X_test))

    return result


def train_model(run_key):
    """Stage 3: Train one classification model."""

    logger.info("STAGE 3: TRAIN MODEL")

    directory = get_run_directory(run_key)

    processed = joblib.load(directory / "processed_data.joblib")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
    )

    model.fit(
        processed["X_train"],
        processed["y_train"],
    )

    joblib.dump(model, directory / "model.joblib")

    result = {
        "status": "trained",
        "model_type": "LogisticRegression",
        "C": 1.0,
        "random_state": 42,
    }

    save_json(directory / "training.json", result)

    logger.info("Model training completed")

    return result


def evaluate_and_log(run_key):
    """Stage 4: Evaluate the model and record an MLflow run."""

    logger.info("STAGE 4: EVALUATE AND LOG TO MLFLOW")

    directory = get_run_directory(run_key)

    processed = joblib.load(directory / "processed_data.joblib")
    model = joblib.load(directory / "model.joblib")

    X_test = processed["X_test"]
    y_test = processed["y_test"]

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
    }

    provenance = load_json(
        directory / "validation.json"
    )["dataset_provenance"]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"Airflow_Training_{run_key}"
    ) as run:

        mlflow.set_tags({
            "pipeline": "airflow",
            "airflow_run_key": run_key,
            "dataset_version": provenance["dataset_version"],
            "dataset_git_commit": provenance["dataset_commit"],
            "dvc_hash": provenance["dataset_hash"],
            "model_name": "Airflow_Logistic_Regression",
        })

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "C": 1.0,
            "random_state": 42,
            "test_size": 0.2,
            "training_samples": len(processed["X_train"]),
            "test_samples": len(X_test),
        })

        mlflow.log_metrics(metrics)

        mlflow.log_dict(
            {
                "metrics": metrics,
                "confusion_matrix": confusion_matrix(
                    y_test, predictions
                ).tolist(),
                "dataset_provenance": provenance,
            },
            "evaluation/results.json",
        )

        signature = infer_signature(X_test, predictions)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test.head(2),
            serialization_format="skops",
        )

        mlflow.set_tag(
            "logged_model_uri",
            model_info.model_uri,
        )

        mlflow_run_id = run.info.run_id

    result = {
        "status": "evaluated",
        "metrics": metrics,
        "mlflow_run_id": mlflow_run_id,
        "model_uri": model_info.model_uri,
        "dataset_provenance": provenance,
    }

    save_json(directory / "evaluation.json", result)

    logger.info("MLflow run ID: %s", mlflow_run_id)
    logger.info("F1 score: %.4f", metrics["f1_score"])

    return result


def model_quality_gate(run_key):
    """Stage 5: Reject models below the required F1 threshold."""

    logger.info("STAGE 5: MODEL QUALITY GATE")

    directory = get_run_directory(run_key)

    evaluation = load_json(directory / "evaluation.json")

    f1 = evaluation["metrics"]["f1_score"]

    threshold = float(
        os.getenv("MODEL_MIN_F1", "0.90")
    )

    # Validate the threshold configuration.
    if not 0 <= threshold <= 1:
        raise ValueError("F1 threshold must be between 0 and 1")

    if f1 < threshold:

        result = {
            "status": "failed",
            "f1_score": f1,
            "threshold": threshold,
            "mlflow_run_id": evaluation["mlflow_run_id"],
        }

        save_json(directory / "quality_gate.json", result)

        logger.error(
            "Quality gate FAILED: F1 %.4f < %.4f",
            f1,
            threshold,
        )

        raise RuntimeError(
            f"Model quality gate failed: "
            f"F1={f1:.4f}, required={threshold:.4f}"
        )

    result = {
        "status": "passed",
        "f1_score": f1,
        "threshold": threshold,
        "mlflow_run_id": evaluation["mlflow_run_id"],
        "promotion": "not_performed",
    }

    save_json(directory / "quality_gate.json", result)

    logger.info(
        "Quality gate PASSED: F1 %.4f >= %.4f",
        f1,
        threshold,
    )

    logger.info(
        "Model passed evaluation but was NOT "
        "automatically promoted to production."
    )

    return result


STAGES = {
    "validate": validate_dataset,
    "preprocess": preprocess_data,
    "train": train_model,
    "evaluate": evaluate_and_log,
    "quality_gate": model_quality_gate,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=STAGES.keys(),
    )

    parser.add_argument(
        "--run-key",
        required=True,
    )

    args = parser.parse_args()

    result = STAGES[args.stage](args.run_key)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
