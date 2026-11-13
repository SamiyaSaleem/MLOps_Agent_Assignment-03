import pytest    
from types import SimpleNamespace
from agent import orchestrator

def test_approved_read_only_tool(monkeypatch):
    calls = []

    def fake_health_tool():
        calls.append("executed")

        return {
            "status": "success",
            "deployment": "mlops-inference",
            "desired_replicas": 2,
            "ready_replicas": 2,
            "available_replicas": 2,
            "healthy": True,
        }

    monkeypatch.setitem(
        orchestrator.READ_ONLY_TOOLS,
        "get_deployment_health",
        fake_health_tool,
    )

    result = orchestrator.execute_tool(
        "get_deployment_health",
        {},
    )

    assert result["status"] == "success"
    assert result["healthy"] is True
    assert calls == ["executed"]
    
    
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
    


def test_llm_selects_and_executes_approved_tool(monkeypatch):
    """
    Simulate an LLM requesting Kubernetes deployment health.
    No real Groq API or Kubernetes cluster is required.
    """

    monkeypatch.setenv(
        "GROQ_API_KEY",
        "dummy-test-key",
    )

    calls = []

    def fake_health_tool():
        calls.append("get_deployment_health")

        return {
            "status": "success",
            "desired_replicas": 2,
            "ready_replicas": 2,
            "healthy": True,
        }

    monkeypatch.setitem(
        orchestrator.READ_ONLY_TOOLS,
        "get_deployment_health",
        fake_health_tool,
    )

    tool_request = SimpleNamespace(
        id="call_001",
        function=SimpleNamespace(
            name="get_deployment_health",
            arguments="{}",
        ),
    )

    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[tool_request],
                )
            )
        ]
    )

    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Both Kubernetes replicas are healthy.",
                    tool_calls=None,
                )
            )
        ]
    )

    responses = iter([
        first_response,
        final_response,
    ])

    class FakeCompletions:
        def create(self, **kwargs):
            return next(responses)

    class FakeChat:
        completions = FakeCompletions()

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr(
        orchestrator,
        "Groq",
        FakeGroq,
    )

    result = orchestrator.ask_agent(
        "Are my Kubernetes replicas healthy?"
    )

    assert result.status == "success"

    assert result.tools_used == [
        "get_deployment_health"
    ]

    assert calls == [
        "get_deployment_health"
    ]

    assert (
        "Both Kubernetes replicas are healthy"
        in result.answer
    )

    assert result.evidence[0].status == "success"


def test_llm_cannot_execute_unauthorized_tool(monkeypatch):
    """Simulate an LLM requesting a forbidden scaling operation."""

    result = orchestrator.execute_tool(
        "scale_inference_api",
        {
            "replicas": 3,
            "confirmed": True,
        },
    )

    assert result["status"] == "error"
    assert result["error"] == "Tool is not approved"
