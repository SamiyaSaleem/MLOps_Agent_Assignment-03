import os
import re
from urllib.parse import quote

import requests


BASE_URL = os.getenv(
    "AIRFLOW_API_URL",
    "http://127.0.0.1:8080",
).rstrip("/")

APPROVED_DAG = "mlops_training_pipeline"

TASK_IDS = {
    "validate_dataset",
    "preprocess_data",
    "train_model",
    "evaluate_and_log",
    "model_quality_gate",
}

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:+-]{1,200}$")


def error_response(operation, exc):
    """Return a safe error without credentials or internal details."""
    result = {
        "status": "error",
        "operation": operation,
        "error": "Airflow request failed",
        "error_type": type(exc).__name__,
    }

    if isinstance(exc, requests.HTTPError):
        result["http_status"] = exc.response.status_code

    return result


def get_token():
    """Authenticate using configured credentials."""
    username = os.getenv("AIRFLOW_USERNAME")
    password = os.getenv("AIRFLOW_PASSWORD")

    if not username or not password:
        raise RuntimeError("Airflow credentials are not configured")

    response = requests.post(
        f"{BASE_URL}/auth/token",
        json={
            "username": username,
            "password": password,
        },
        timeout=15,
    )
    response.raise_for_status()

    return response.json()["access_token"]


def api_request(method, path, **kwargs):
    """Make an authenticated request to the Airflow 3 API."""

    token = get_token()

    response = requests.request(
        method,
        f"{BASE_URL}/api/v2/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        timeout=20,
        **kwargs,
    )

    response.raise_for_status()
    return response.json()


def validate_run_id(run_id):
    return (
        isinstance(run_id, str)
        and RUN_ID_PATTERN.fullmatch(run_id) is not None
    )


def list_dags():
    """Retrieve registered Airflow DAGs."""

    try:
        data = api_request("GET", "dags")

        return {
            "status": "success",
            "dags": [
                {
                    "dag_id": dag["dag_id"],
                    "is_paused": dag.get("is_paused"),
                }
                for dag in data.get("dags", [])
            ],
        }

    except Exception as exc:
        return error_response("list_dags", exc)


def get_latest_pipeline_status():
    """Retrieve the latest training pipeline run."""

    try:
        data = api_request(
            "GET",
            f"dags/{APPROVED_DAG}/dagRuns",
            params={
                "limit": 1,
                "order_by": "-run_after",
            },
        )

        runs = data.get("dag_runs", [])

        if not runs:
            return {
                "status": "success",
                "latest_run": None,
                "message": "No training runs found",
            }

        run = runs[0]

        return {
            "status": "success",
            "latest_run": {
                "dag_id": run.get("dag_id"),
                "run_id": run.get("dag_run_id"),
                "state": run.get("state"),
                "start_date": run.get("start_date"),
                "end_date": run.get("end_date"),
            },
        }

    except Exception as exc:
        return error_response("get_latest_pipeline_status", exc)


def get_task_statuses(run_id):
    """Retrieve the task statuses of a specific run."""

    if not validate_run_id(run_id):
        return {
            "status": "error",
            "error": "Invalid DAG run ID",
        }

    try:
        encoded_id = quote(run_id, safe="")

        data = api_request(
            "GET",
            f"dags/{APPROVED_DAG}/dagRuns/"
            f"{encoded_id}/taskInstances",
        )

        tasks = data.get("task_instances", [])

        return {
            "status": "success",
            "run_id": run_id,
            "tasks": [
                {
                    "task_id": task.get("task_id"),
                    "state": task.get("state"),
                    "try_number": task.get("try_number"),
                    "start_date": task.get("start_date"),
                    "end_date": task.get("end_date"),
                }
                for task in tasks
            ],
        }

    except Exception as exc:
        return error_response("get_task_statuses", exc)


def get_failed_tasks(run_id):
    """Find failed and upstream-failed tasks."""

    result = get_task_statuses(run_id)

    if result["status"] != "success":
        return result

    failed = [
        task
        for task in result["tasks"]
        if task["state"] in {"failed", "upstream_failed"}
    ]

    return {
        "status": "success",
        "run_id": run_id,
        "failed_tasks": failed,
    }


def get_task_diagnostics(run_id, task_id, try_number=1):
    """Identify known failures using sanitized Airflow log evidence."""

    if not validate_run_id(run_id):
        return {
            "status": "error",
            "error": "Invalid DAG run ID",
        }

    if task_id not in TASK_IDS:
        return {
            "status": "error",
            "error": "Unapproved task ID",
        }

    if type(try_number) is not int or not 1 <= try_number <= 20:
        return {
            "status": "error",
            "error": "Invalid attempt number",
        }

    try:
        encoded_id = quote(run_id, safe="")

        data = api_request(
            "GET",
            f"dags/{APPROVED_DAG}/dagRuns/"
            f"{encoded_id}/taskInstances/"
            f"{task_id}/logs/{try_number}",
            params={"full_content": True},
        )

        entries = data.get("content", [])

        if isinstance(entries, str):
            log_text = entries[:100000]

        elif isinstance(entries, list):
            log_text = "\n".join(
                str(entry.get("event", ""))
                if isinstance(entry, dict)
                else str(entry)
                for entry in entries
            )[:100000]

        else:
            return {
                "status": "error",
                "error": "Unsupported Airflow log format",
            }

        failure_patterns = [
            (
                "IndentationError",
                "Python indentation or syntax problem",
                "Correct the Python indentation and rerun.",
            ),
            (
                "SyntaxError",
                "Python syntax problem",
                "Correct the Python syntax and rerun.",
            ),
            (
                "ModuleNotFoundError",
                "Missing Python dependency or module",
                "Check the Python environment and dependencies.",
            ),
            (
                "FileNotFoundError",
                "Required file is missing",
                "Verify required files and configured paths.",
            ),
            (
                "ConnectionError",
                "External service connection failure",
                "Check the external service and connection settings.",
            ),
            (
                "TimeoutExpired",
                "Execution timeout",
                "Investigate the slow operation and timeout configuration.",
            ),
            (
                "quality gate failed",
                "Model quality threshold failure",
                "Inspect evaluation metrics and the configured threshold.",
            ),
        ]

        root_cause = "Unknown"
        recommendation = (
            "Inspect the task logs manually for further evidence."
        )

        for pattern, cause, action in failure_patterns:
            if pattern.lower() in log_text.lower():
                root_cause = cause
                recommendation = action
                break

        diagnostic_count = sum(
            log_text.lower().count(word)
            for word in ("error", "failed", "traceback")
        )

        return {
            "status": "success",
            "run_id": run_id,
            "task_id": task_id,
            "try_number": try_number,
            "root_cause": root_cause,
            "recommendation": recommendation,
            "failure_indicators_found": diagnostic_count,
            "diagnosis_confidence": (
                "pattern_match"
                if root_cause != "Unknown"
                else "insufficient_evidence"
            ),
            "note": (
                "Diagnosis is based on recognized patterns "
                "in actual Airflow task logs. "
                "Raw logs are not exposed."
            ),
        }

    except Exception as exc:
        return error_response("get_task_diagnostics", exc)

def trigger_training_pipeline(confirmed=False):
    """Trigger only the approved training DAG after confirmation."""

    if confirmed is not True:
        return {
            "status": "confirmation_required",
            "message": (
                "Explicit user confirmation is required "
                "before starting a training workflow."
            ),
        }

    try:
        data = api_request(
            "POST",
            f"dags/{APPROVED_DAG}/dagRuns",
            json={
                "logical_date": None,
                "conf": {},
            },
        )

        return {
            "status": "success",
            "dag_id": data.get("dag_id"),
            "run_id": data.get("dag_run_id"),
            "state": data.get("state"),
            "message": "Training pipeline triggered",
        }

    except Exception as exc:
        return error_response("trigger_training_pipeline", exc)
