import json
import subprocess

from agent.tools import kubernetes_tools as kt


def test_deployment_healthy(monkeypatch):
    def fake_get_json(arguments):
        return {
            "status": "success",
            "data": {
                "spec": {"replicas": 2},
                "status": {
                    "readyReplicas": 2,
                    "availableReplicas": 2,
                },
            },
        }

    monkeypatch.setattr(kt, "get_json_resource", fake_get_json)

    result = kt.get_deployment_health()

    assert result["status"] == "success"
    assert result["healthy"] is True
    assert result["ready_replicas"] == 2


def test_deployment_unhealthy(monkeypatch):
    monkeypatch.setattr(
        kt,
        "get_json_resource",
        lambda arguments: {
            "status": "success",
            "data": {
                "spec": {"replicas": 2},
                "status": {
                    "readyReplicas": 1,
                    "availableReplicas": 1,
                },
            },
        },
    )

    result = kt.get_deployment_health()

    assert result["status"] == "success"
    assert result["healthy"] is False
    assert result["ready_replicas"] == 1


def test_service_endpoints(monkeypatch):
    monkeypatch.setattr(
        kt,
        "get_json_resource",
        lambda arguments: {
            "status": "success",
            "data": {
                "items": [
                    {
                        "endpoints": [
                            {
                                "addresses": ["10.244.0.5"],
                                "conditions": {"ready": True},
                            },
                            {
                                "addresses": ["10.244.0.6"],
                                "conditions": {"ready": True},
                            },
                        ]
                    }
                ]
            },
        },
    )

    result = kt.get_service_endpoints()

    assert result["status"] == "success"
    assert result["endpoint_count"] == 2
    assert all(
        endpoint["ready"]
        for endpoint in result["endpoints"]
    )


def test_list_resources(monkeypatch):
    def fake_get_json(arguments):
        resource = arguments[1]

        return {
            "status": "success",
            "data": {
                "items": [
                    {
                        "metadata": {
                            "name": f"example-{resource}"
                        },
                        "status": {"phase": "Running"},
                    }
                ]
            },
        }

    monkeypatch.setattr(kt, "get_json_resource", fake_get_json)

    result = kt.list_kubernetes_resources()

    assert result["status"] == "success"
    assert result["namespace"] == "mlops-agent"
    assert len(result["resources"]["pods"]) == 1


def test_scaling_requires_confirmation(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        raise AssertionError("kubectl must not execute")

    monkeypatch.setattr(kt, "run_kubectl", fake_run)

    result = kt.scale_inference_api(replicas=3)

    assert result["status"] == "confirmation_required"
    assert calls == []


def test_reject_invalid_replica_count():
    for replicas in (-1, 0, 4, 100, "3", True):
        result = kt.scale_inference_api(
            replicas=replicas,
            confirmed=True,
        )

        assert result["status"] == "error"


def test_approved_scaling_command_mocked(monkeypatch):
    commands = []

    def fake_run(arguments, timeout=20):
        commands.append(arguments)
        return {
            "status": "success",
            "output": "deployment scaled",
        }

    monkeypatch.setattr(kt, "run_kubectl", fake_run)

    result = kt.scale_inference_api(
        replicas=3,
        confirmed=True,
    )

    assert result["status"] == "success"
    assert commands == [
        [
            "scale",
            "deployment",
            "mlops-inference",
            "--replicas=3",
        ]
    ]


def test_reject_unapproved_pod():
    result = kt.get_pod_logs(
        "kube-system-pod",
    )

    assert result["status"] == "error"
    assert result["error"] == "Unapproved Pod name"


def test_reject_invalid_log_limit():
    result = kt.get_pod_logs(
        "mlops-inference-78c8874554-7mdr8",
        tail_lines=1000,
    )

    assert result["status"] == "error"


def test_pod_label_verification(monkeypatch):
    monkeypatch.setattr(
        kt,
        "get_json_resource",
        lambda arguments: {
            "status": "success",
            "data": {
                "metadata": {
                    "labels": {"app": "unapproved-app"}
                }
            },
        },
    )

    result = kt.get_pod_logs(
        "mlops-inference-78c8874554-7mdr8"
    )

    assert result["status"] == "error"
    assert "approved application" in result["error"]


def test_pod_log_diagnosis(monkeypatch):
    monkeypatch.setattr(
        kt,
        "get_json_resource",
        lambda arguments: {
            "status": "success",
            "data": {
                "metadata": {
                    "labels": {"app": "mlops-inference"}
                }
            },
        },
    )

    monkeypatch.setattr(
        kt,
        "run_kubectl",
        lambda arguments, timeout=20: {
            "status": "success",
            "output": "ERROR: Model loading failed",
        },
    )

    result = kt.get_pod_logs(
        "mlops-inference-78c8874554-7mdr8"
    )

    assert result["status"] == "success"
    assert "model_loading_failure" in (
        result["diagnostic_categories"]
    )


def test_invalid_kubernetes_json(monkeypatch):
    monkeypatch.setattr(
        kt,
        "run_kubectl",
        lambda arguments, timeout=20: {
            "status": "success",
            "output": "not-valid-json",
        },
    )

    result = kt.get_json_resource(["get", "pods"])

    assert result["status"] == "error"
    assert result["error"] == "Invalid Kubernetes JSON response"


def test_kubectl_timeout(monkeypatch):
    def fake_subprocess(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="kubectl",
            timeout=20,
        )

    monkeypatch.setattr(
        kt.subprocess,
        "run",
        fake_subprocess,
    )

    result = kt.run_kubectl(["get", "pods"])

    assert result["status"] == "error"
    assert "timed out" in result["error"]


def test_kubectl_context_and_namespace(monkeypatch):
    captured = []

    def fake_subprocess(command, **kwargs):
        captured.append(command)

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="{}",
            stderr="",
        )

    monkeypatch.setattr(
        kt.subprocess,
        "run",
        fake_subprocess,
    )

    result = kt.run_kubectl(["get", "pods"])

    assert result["status"] == "success"

    command = captured[0]

    assert "--context" in command
    assert "kind-mlops-agent" in command
    assert "--namespace" in command
    assert "mlops-agent" in command
