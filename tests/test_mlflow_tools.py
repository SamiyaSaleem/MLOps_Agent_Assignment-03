from types import SimpleNamespace
from agent.tools import mlflow_tools

def make_run(
    run_id="a" * 32,
    f1=0.95,
    status="FINISHED",
    model_name="Test_Model",
):
    """Create a fake MLflow run for automated testing."""

    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            status=status,
            start_time=1000,
        ),
        data=SimpleNamespace(
            metrics={
                "accuracy": 0.94,
                "precision": 0.95,
                "recall": 0.95,
                "f1_score": f1,
            },
            params={"C": "1.0"},
            tags={
                "mlflow.runName": model_name,
                "dataset_version": "dataset-v2",
                "dataset_git_commit": "98dd5e9",
                "dvc_hash": "test-hash",
                "logged_model_uri": "models:/test-model",
            },
        ),
    )


def fake_experiment():
    return SimpleNamespace(
        experiment_id="1",
        name="MLOps_Assignment_03",
    )


def test_list_experiments(monkeypatch):

    class FakeClient:
        def search_experiments(self):
            return [fake_experiment()]

    monkeypatch.setattr(
        mlflow_tools,
        "get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.list_experiments()

    assert result["status"] == "success"
    assert len(result["experiments"]) == 1
    assert result["experiments"][0]["name"] == "MLOps_Assignment_03"


def test_list_training_runs(monkeypatch):

    class FakeClient:
        def get_experiment_by_name(self, name):
            return fake_experiment()

        def search_runs(self, **kwargs):
            return [
                make_run(),
                make_run(
                    run_id="b" * 32,
                    f1=0.90,
                ),
            ]

    monkeypatch.setattr(
        mlflow_tools,
        "get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.list_training_runs()

    assert result["status"] == "success"
    assert len(result["runs"]) == 2


def test_find_best_run(monkeypatch):

    class FakeClient:
        def get_experiment_by_name(self, name):
            return fake_experiment()

        def search_runs(self, **kwargs):
            return [
                make_run(
                    run_id="a" * 32,
                    f1=0.90,
                ),
                make_run(
                    run_id="b" * 32,
                    f1=0.98,
                ),
            ]

    monkeypatch.setattr(
        mlflow_tools,
        "get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.find_best_run()

    assert result["status"] == "success"
    assert result["best_run"]["metrics"]["f1_score"] == 0.98
    assert result["best_run"]["run_id"] == "b" * 32


def test_invalid_metric():

    result = mlflow_tools.find_best_run(
        metric="invalid_metric",
    )

    assert result["status"] == "error"
    assert "Unsupported" in result["error"]


def test_invalid_run_id():

    result = mlflow_tools.get_run_details(
        "invalid-run-id",
    )

    assert result["status"] == "error"
    assert "Invalid run ID" in result["error"]


def test_service_failure(monkeypatch):

    def failing_client():
        raise ConnectionError(
            "MLflow server unavailable"
        )

    monkeypatch.setattr(
        mlflow_tools,
        "get_client",
        failing_client,
    )

    result = mlflow_tools.list_experiments()

    assert result["status"] == "error"
    assert result["error_type"] == "ConnectionError"


def test_no_fabricated_results(monkeypatch):

    def failing_client():
        raise ConnectionError(
            "MLflow server unavailable"
        )

    monkeypatch.setattr(
        mlflow_tools,
        "get_client",
        failing_client,
    )

    result = mlflow_tools.find_best_run()

    assert result["status"] == "error"
    assert "best_run" not in result


def test_compare_experiments(monkeypatch):

    class FakeClient:
        def get_experiment_by_name(self, name):
            return fake_experiment()

        def search_runs(self, **kwargs):
            return [
                make_run(
                    run_id="a" * 32,
                    f1=0.90,
                ),
                make_run(
                    run_id="b" * 32,
                    f1=0.95,
                ),
            ]

    monkeypatch.setattr(
        mlflow_tools,
        "get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.compare_experiments(
        ["MLOps_Assignment_03"],
    )

    assert result["status"] == "success"
    assert len(result["comparison"]) == 1

    best = result["comparison"][0]["best_run"]

    assert best["metrics"]["f1_score"] == 0.95


def test_registry_query(monkeypatch):

    class FakeClient:

        def search_registered_models(self, max_results):
            return [
                SimpleNamespace(
                    name="BreastCancerClassifier",
                    description="Test model",
                    aliases={"candidate": "1"},
                    latest_versions=[],
                )
            ]

        def search_model_versions(self, filter_string):
            return [
                SimpleNamespace(
                    version="1",
                    run_id="a" * 32,
                    source="models:/test",
                    status="READY",
                )
            ]

    monkeypatch.setattr(
        mlflow_tools,
        "get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.get_model_registry_info()

    assert result["status"] == "success"

    registered_models = result["registered_models"]

    assert len(registered_models) == 1

    model = registered_models[0]

    assert model["name"] == "BreastCancerClassifier"
    assert model["aliases"]["candidate"] == "1"

    assert len(model["versions"]) == 1
    assert model["versions"][0]["version"] == "1"
    assert model["versions"][0]["status"] == "READY"
