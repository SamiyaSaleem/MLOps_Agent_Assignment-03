from agent import main as agent_cli


def test_rejected_kubernetes_scaling(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "builtins.input",
        lambda _: "NO",
    )

    monkeypatch.setattr(
        agent_cli,
        "scale_inference_api",
        lambda **kwargs: calls.append(kwargs),
    )

    result = agent_cli.execute_scaling(3)

    assert result == 1
    assert calls == []


def test_approved_kubernetes_scaling(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "builtins.input",
        lambda _: "CONFIRM SCALE 3",
    )

    def fake_scaling(**kwargs):
        calls.append(kwargs)
        return {
            "status": "success",
            "requested_replicas": 3,
        }

    monkeypatch.setattr(
        agent_cli,
        "scale_inference_api",
        fake_scaling,
    )

    monkeypatch.setattr(
        agent_cli,
        "get_deployment_health",
        lambda: {
            "status": "success",
            "desired_replicas": 3,
            "ready_replicas": 3,
            "healthy": True,
        },
    )

    result = agent_cli.execute_scaling(3)

    assert result == 0
    assert calls == [
        {"replicas": 3, "confirmed": True}
    ]


def test_invalid_replica_count(monkeypatch):
    calls = []

    monkeypatch.setattr(
        agent_cli,
        "scale_inference_api",
        lambda **kwargs: calls.append(kwargs),
    )

    result = agent_cli.execute_scaling(5)

    assert result == 1
    assert calls == []


def test_rejected_training_trigger(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "builtins.input",
        lambda _: "NO",
    )

    monkeypatch.setattr(
        agent_cli,
        "trigger_training_pipeline",
        lambda **kwargs: calls.append(kwargs),
    )

    result = agent_cli.execute_training()

    assert result == 1
    assert calls == []


def test_approved_training_trigger(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "builtins.input",
        lambda _: "CONFIRM TRAINING",
    )

    def fake_training(**kwargs):
        calls.append(kwargs)
        return {
            "status": "success",
            "dag_id": "mlops_training_pipeline",
            "state": "queued",
        }

    monkeypatch.setattr(
        agent_cli,
        "trigger_training_pipeline",
        fake_training,
    )

    result = agent_cli.execute_training()

    assert result == 0
    assert calls == [
        {"confirmed": True}
    ]
