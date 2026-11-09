import os
import re

from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "MLOps_Assignment_03"

ALLOWED_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1_score",
}

RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def get_client():
    """Connect to the configured MLflow tracking server."""
    uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
    return MlflowClient(tracking_uri=uri)


def error_response(operation, exc):
    """Return an error without exposing internal connection details."""
    return {
        "status": "error",
        "operation": operation,
        "error": "MLflow operation failed or service unavailable",
        "error_type": type(exc).__name__,
    }


def serialize_run(run):
    """Convert an MLflow run into structured agent-readable data."""
    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "status": run.info.status,
        "start_time": run.info.start_time,
        "metrics": dict(run.data.metrics),
        "parameters": dict(run.data.params),
        "dataset_version": run.data.tags.get("dataset_version"),
        "dataset_git_commit": run.data.tags.get(
            "dataset_git_commit"
        ),
        "dvc_hash": run.data.tags.get("dvc_hash"),
        "model_uri": run.data.tags.get("logged_model_uri"),
    }


def list_experiments():
    """List available MLflow experiments."""
    try:
        client = get_client()
        experiments = client.search_experiments()

        return {
            "status": "success",
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                }
                for exp in experiments
            ],
        }

    except Exception as exc:
        return error_response("list_experiments", exc)


def list_training_runs(experiment_name=EXPERIMENT_NAME, limit=5):
    """Retrieve recent runs from an experiment."""
    if not isinstance(limit, int) or not 1 <= limit <= 100:
        return {
            "status": "error",
            "error": "limit must be between 1 and 100",
        }

    try:
        client = get_client()
        experiment = client.get_experiment_by_name(
            experiment_name
        )

        if experiment is None:
            return {
                "status": "error",
                "error": "Experiment not found",
            }

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"],
        )

        return {
            "status": "success",
            "experiment": experiment_name,
            "runs": [serialize_run(run) for run in runs],
        }

    except Exception as exc:
        return error_response("list_training_runs", exc)


def get_run_details(run_id):
    """Retrieve one run's recorded details and artifacts."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        return {
            "status": "error",
            "error": "Invalid run ID",
        }

    try:
        client = get_client()
        run = client.get_run(run_id)

        artifacts = client.list_artifacts(run_id)

        details = serialize_run(run)

        details["tags"] = dict(run.data.tags)

        details["artifacts"] = [
            {
                "path": artifact.path,
                "is_dir": artifact.is_dir,
            }
            for artifact in artifacts
        ]

        return {
            "status": "success",
            "run": details,
            "note": (
                "MLflow 3 logged models may be stored separately "
                "from run artifacts. See model_uri."
            ),
        }

    except Exception as exc:
        return error_response("get_run_details", exc)


def find_best_run(
    experiment_name=EXPERIMENT_NAME,
    metric="f1_score",
    limit=100,
):
    """Identify the highest-scoring completed run."""
    if metric not in ALLOWED_METRICS:
        return {
            "status": "error",
            "error": "Unsupported evaluation metric",
        }

    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        return {
            "status": "error",
            "error": "Invalid run limit",
        }

    try:
        client = get_client()
        experiment = client.get_experiment_by_name(
            experiment_name
        )

        if experiment is None:
            return {
                "status": "error",
                "error": "Experiment not found",
            }

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=limit,
            order_by=["start_time DESC"],
        )

        candidates = [
            run for run in runs
            if metric in run.data.metrics
        ]

        if not candidates:
            return {
                "status": "success",
                "best_run": None,
                "reason": "No completed runs contain this metric",
            }

        best = max(
            candidates,
            key=lambda run: run.data.metrics[metric],
        )

        return {
            "status": "success",
            "selection_metric": metric,
            "runs_evaluated": len(candidates),
            "best_run": serialize_run(best),
            "recommendation": (
                "Best candidate among evaluated runs by the "
                "specified metric. Production readiness requires "
                "additional validation and approval."
            ),
            "limitation": (
                "Only the most recent runs within the requested "
                "limit were considered."
            ),
        }

    except Exception as exc:
        return error_response("find_best_run", exc)


def compare_experiments(experiment_names, metric="f1_score"):
    """Compare completed runs across named experiments."""
    if (
        not isinstance(experiment_names, list)
        or not 1 <= len(experiment_names) <= 10
        or any(not isinstance(name, str) for name in experiment_names)
    ):
        return {
            "status": "error",
            "error": "Provide a list of 1 to 10 experiment names",
        }

    if metric not in ALLOWED_METRICS:
        return {
            "status": "error",
            "error": "Unsupported evaluation metric",
        }

    try:
        client = get_client()
        comparison = []

        for name in experiment_names:
            exp = client.get_experiment_by_name(name)

            if exp is None:
                comparison.append({
                    "experiment": name,
                    "error": "Experiment not found",
                })
                continue

            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                max_results=100,
                order_by=["start_time DESC"],
            )

            valid_runs = [
                run for run in runs
                if metric in run.data.metrics
            ]

            best = (
                max(
                    valid_runs,
                    key=lambda run: run.data.metrics[metric],
                )
                if valid_runs else None
            )

            comparison.append({
                "experiment": name,
                "completed_runs_evaluated": len(valid_runs),
                "best_run": serialize_run(best) if best else None,
            })

        return {
            "status": "success",
            "metric": metric,
            "comparison": comparison,
        }

    except Exception as exc:
        return error_response("compare_experiments", exc)


def get_model_registry_info(model_name=None):
    """Inspect model registry entries and versions."""
    try:
        client = get_client()

        if model_name is not None:
            if not isinstance(model_name, str) or not model_name:
                return {
                    "status": "error",
                    "error": "Invalid model name",
                }

            models = [client.get_registered_model(model_name)]
        else:
            models = client.search_registered_models(
                max_results=100
            )

        results = []

        for model in models:
            results.append({
                "name": model.name,
                "description": model.description,
                "aliases": dict(model.aliases or {}),
                "versions": [
                    {
                        "version": version.version,
                        "run_id": version.run_id,
                        "source": version.source,
                        "status": version.status,
                    }
                    for version in client.search_model_versions(
    filter_string=f"name = '{model.name}'"
)
                ],
            })

        return {
            "status": "success",
            "registered_models": results,
        }

    except Exception as exc:
        return error_response("get_model_registry_info", exc)
