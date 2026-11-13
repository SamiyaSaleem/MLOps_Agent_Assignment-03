import requests

from agent.tools import airflow_tools


TEST_RUN_ID = "manual__2026-11-09T12:32:47.155372+00:00"


def test_list_dags(monkeypatch):
    def fake_api(method, path, **kwargs):
        assert method == "GET"
        assert path == "dags"

        return {
            "dags": [
                {
                    "dag_id": "mlops_training_pipeline",
                    "is_paused": False,
                }
            ]
        }

    monkeypatch.setattr(
        airflow_tools,
        "api_request",
        fake_api,
    )

    result = airflow_tools.list_dags()

    assert result["status"] == "success"
    assert result["dags"][0]["dag_id"] == "mlops_training_pipeline"


def test_latest_pipeline_status(monkeypatch):
    def fake_api(method, path, **kwargs):
        return {
            "dag_runs": [
                {
                    "dag_id": "mlops_training_pipeline",
                    "dag_run_id": TEST_RUN_ID,
                    "state": "success",
                    "start_date": "2026-11-09T12:32:48Z",
                    "end_date": "2026-11-09T12:33:04Z",
                }
            ]
        }

    monkeypatch.setattr(
        airflow_tools,
        "api_request",
        fake_api,
    )

    result = airflow_tools.get_latest_pipeline_status()

    assert result["status"] == "success"
    assert result["latest_run"]["state"] == "success"


def test_no_pipeline_runs(monkeypatch):
    monkeypatch.setattr(
        airflow_tools,
        "api_request",
        lambda *args, **kwargs: {"dag_runs": []},
    )

    result = airflow_tools.get_latest_pipeline_status()

    assert result["status"] == "success"
    assert result["latest_run"] is None


def test_task_statuses(monkeypatch):
    def fake_api(method, path, **kwargs):
        return {
            "task_instances": [
                {
                    "task_id": "validate_dataset",
                    "state": "success",
                    "try_number": 1,
                },
                {
                    "task_id": "train_model",
                    "state": "success",
                    "try_number": 1,
                },
            ]
        }

    monkeypatch.setattr(
        airflow_tools,
        "api_request",
        fake_api,
    )

    result = airflow_tools.get_task_statuses(TEST_RUN_ID)

    assert result["status"] == "success"
    assert len(result["tasks"]) == 2
    assert all(
        task["state"] == "success"
        for task in result["tasks"]
    )


def test_failed_tasks(monkeypatch):
    def fake_api(method, path, **kwargs):
        return {
            "task_instances": [
                {
                    "task_id": "validate_dataset",
                    "state": "failed",
                },
                {
                    "task_id": "preprocess_data",
                    "state": "upstream_failed",
                },
                {
                    "task_id": "train_model",
                    "state": "upstream_failed",
                },
            ]
        }

    monkeypatch.setattr(
        airflow_tools,
        "api_request",
        fake_api,
    )

    result = airflow_tools.get_failed_tasks(TEST_RUN_ID)

    assert result["status"] == "success"
    assert len(result["failed_tasks"]) == 3
    assert result["failed_tasks"][0]["state"] == "failed"


def test_invalid_run_id():
    result = airflow_tools.get_task_statuses(
        "invalid/run/id"
    )

    assert result["status"] == "error"
    assert "Invalid" in result["error"]


def test_invalid_task_id():
    result = airflow_tools.get_task_diagnostics(
        TEST_RUN_ID,
        "unauthorized_task",
    )

    assert result["status"] == "error"
    assert result["error"] == "Unapproved task ID"


def test_trigger_requires_confirmation(monkeypatch):
    calls = []

    def fake_api(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("API must not be called")

    monkeypatch.setattr(
        airflow_tools,
        "api_request",
        fake_api,
    )

    result = airflow_tools.trigger_training_pipeline()

    assert result["status"] == "confirmation_required"
    assert calls == []


def test_airflow_service_failure(monkeypatch):
    def fake_api(*args, **kwargs):
        raise requests.ConnectionError(
            "Service unavailable"
        )

    monkeypatch.setattr(
        airflow_tools,
        "api_request",
        fake_api,
    )

    result = airflow_tools.list_dags()

    assert result["status"] == "error"
    assert result["error_type"] == "ConnectionError"
    assert "dags" not in result
