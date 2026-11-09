import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

EXPERIMENT_NAME = "MLOps_Assignment_03"


def git_command(*args):
    """Run a fixed Git operation in our project."""

    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        timeout=15,
    ).strip()


def get_dataset_provenance():
    """Read the current dataset's verified DVC provenance."""

    from agent.tools.dvc_tools import (
        get_current_dataset_version,
    )

    result = get_current_dataset_version()

    if result["status"] != "success":
        raise RuntimeError(
            "Cannot retrieve dataset provenance"
        )

    if not result["workspace_clean"]:
        raise RuntimeError(
            "Dataset has uncommitted changes. "
            "Restore or track the dataset before training."
        )

    versions = result["version_tags"]

    if not versions:
        raise RuntimeError(
            "Dataset does not match a recorded version."
        )

    # Our latest dataset version.
    version = versions[-1]

    # Identify the commit that recorded the dataset.
    dataset_commit = git_command(
        "rev-parse",
        f"refs/tags/{version}",
    )

    return {
        "dataset_version": version,
        "dataset_commit": dataset_commit,
        "training_code_commit": result["git_commit"],
        "dataset_hash": result["dvc_hash"],
    }


def load_data():
    """Load the DVC-tracked dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty")

    if "target" not in df.columns:
        raise ValueError("Target column is missing")

    X = df.drop(columns=["target"])
    y = df["target"]

    if X.isnull().values.any():
        raise ValueError("Dataset contains missing features")

    if y.isnull().values.any():
        raise ValueError("Dataset contains missing labels")

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def get_models():
    """Return five reproducible model configurations."""

    return {
        "Logistic_Regression_C1": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
            ),
        ),

        "Logistic_Regression_C01": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                max_iter=2000,
                random_state=42,
            ),
        ),

        "Random_Forest_100": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),

        "Random_Forest_200": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
        ),

        "Decision_Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
        ),
    }


def train_models():
    """Train, evaluate, and log five MLflow runs."""

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    provenance = get_dataset_provenance()

    X_train, X_test, y_train, y_test = load_data()

    models = get_models()

    results = []

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        with mlflow.start_run(run_name=model_name) as run:

            mlflow.set_tags({
                "model_name": model_name,
                "dataset_version": provenance["dataset_version"],
                "dataset_git_commit": provenance["dataset_commit"],
                "training_code_commit": provenance["training_code_commit"],
                "dvc_hash": provenance["dataset_hash"],
                "pipeline_stage": "training",
                "evaluation_split": "holdout",
            })

            # Record model hyperparameters.
            mlflow.log_param(
                "model_type",
                type(model).__name__,
            )

            mlflow.log_param(
                "dataset_version",
                provenance["dataset_version"],
            )

            mlflow.log_param(
                "test_size",
                0.2,
            )

            mlflow.log_param(
                "random_state",
                42,
            )

            mlflow.log_param(
                "training_samples",
                len(X_train),
            )

            mlflow.log_param(
                "test_samples",
                len(X_test),
            )

            for param_name, value in model.get_params(
                deep=False
            ).items():
                mlflow.log_param(
                    str(param_name),
                    str(value),
                )

            # Train the model.
            model.fit(X_train, y_train)

            # Generate predictions.
            predictions = model.predict(X_test)

            # Calculate actual evaluation metrics.
            metrics = {
                "accuracy": accuracy_score(
                    y_test, predictions
                ),
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
            }

            mlflow.log_metrics(metrics)

            # Save evaluation evidence.
            mlflow.log_dict(
                {
                    "confusion_matrix": confusion_matrix(
                        y_test,
                        predictions,
                    ).tolist(),
                    "metrics": metrics,
                    "dataset_provenance": provenance,
                },
                "evaluation/results.json",
            )

            # Record the expected model input/output format.
            signature = infer_signature(
                X_test,
                predictions,
            )

            # MLflow 3 uses 'name' for logged models.
            model_info = mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    signature=signature,
    input_example=X_test.head(2),
    serialization_format="skops",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ],
)

            mlflow.set_tag(
                "logged_model_uri",
                model_info.model_uri,
            )

            results.append({
                "model": model_name,
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                **metrics,
            })

            print("Run ID:", run.info.run_id)

            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

    print("\nAll training runs completed!")

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print(
        results_df[
            ["model", "accuracy", "precision", "recall", "f1_score"]
        ].to_string(index=False)
    )

    best = results_df.sort_values(
        "f1_score",
        ascending=False,
    ).iloc[0]

    print("\nBest candidate based on F1 score:")
    print(best["model"])
    print(f"F1 Score: {best['f1_score']:.4f}")
    print(f"Run ID: {best['run_id']}")

    print(
        "\nNote: This is an evaluation candidate, "
        "not an automatically approved production model."
    )


if __name__ == "__main__":
    train_models()
