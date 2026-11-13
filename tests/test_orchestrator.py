import pytest

from agent import orchestrator


def test_approved_read_only_tool():
    result = orchestrator.execute_tool(
        "get_deployment_health",
        {},
    )

    assert result["status"] in {"success", "error"}


def test_reject_unknown_tool():
    result = orchestrator.execute_tool(
        "delete_kubernetes_cluster",
        {},
    )

    assert result["status"] == "error"
    assert result["error"] == "Tool is not approved"


def test_reject_scaling_tool():
    result = orchestrator.execute_tool(
        "scale_inference_api",
        {"replicas": 3, "confirmed": True},
    )

    assert result["status"] == "error"


def test_reject_airflow_trigger():
    result = orchestrator.execute_tool(
        "trigger_training_pipeline",
        {"confirmed": True},
    )

    assert result["status"] == "error"


def test_reject_tool_arguments():
    result = orchestrator.execute_tool(
        "get_deployment_health",
        {"namespace": "kube-system"},
    )

    assert result["status"] == "error"


def test_invalid_empty_question():
    result = orchestrator.ask_agent("")

    assert result.status == "error"
    assert result.error == "invalid_question"


def test_question_too_long():
    result = orchestrator.ask_agent("A" * 2001)

    assert result.status == "error"
    assert result.error == "question_too_long"


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = orchestrator.ask_agent(
        "Check Kubernetes health"
    )

    assert result.status == "error"
    assert result.error == "missing_api_key"


def test_tool_failure_handling(monkeypatch):
    def failing_tool():
        raise RuntimeError("Simulated tool failure")

    monkeypatch.setitem(
        orchestrator.READ_ONLY_TOOLS,
        "test_failure",
        failing_tool,
    )

    result = orchestrator.execute_tool(
        "test_failure",
        {},
    )

    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"


def test_unexpected_tool_response(monkeypatch):
    monkeypatch.setitem(
        orchestrator.READ_ONLY_TOOLS,
        "test_invalid_response",
        lambda: "invalid response",
    )

    result = orchestrator.execute_tool(
        "test_invalid_response",
        {},
    )

    assert result["status"] == "error"
    assert result["error"] == "Unexpected tool response"


def test_tool_definitions_are_read_only():
    definitions = orchestrator.build_tool_definitions()

    tool_names = {
        item["function"]["name"]
        for item in definitions
    }

    assert "scale_inference_api" not in tool_names
    assert "trigger_training_pipeline" not in tool_names
    assert "get_deployment_health" in tool_names
