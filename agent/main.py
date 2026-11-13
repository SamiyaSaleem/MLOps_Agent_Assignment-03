import argparse
import json
from agent.orchestrator import ask_agent
from agent.tools.kubernetes_tools import (
    scale_inference_api,
    get_deployment_health,
)
from agent.tools.airflow_tools import trigger_training_pipeline


def request_confirmation(action, confirmation_phrase):
    print("\nACTION REQUIRES APPROVAL")
    print("=" * 50)
    print(action)
    print("Type the following phrase to approve:")
    print(confirmation_phrase)

    try:
        response = input("\nConfirmation: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAction cancelled.")
        return False

    if response != confirmation_phrase:
        print("Confirmation rejected. No action executed.")
        return False

    return True


def execute_scaling(replicas):
    if type(replicas) is not int or not 1 <= replicas <= 3:
        print("Replica count must be between 1 and 3.")
        return 1

    approved = request_confirmation(
        f"Scale Kubernetes inference deployment to {replicas} replicas.",
        f"CONFIRM SCALE {replicas}",
    )

    if not approved:
        return 1

    result = scale_inference_api(
        replicas=replicas,
        confirmed=True,
    )

    print("\nKubernetes Scaling Result:")
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") != "success":
        return 1

    print("\nChecking deployment status...")

    health = get_deployment_health()

    print(json.dumps(health, indent=2, default=str))

    print(
        "\nScaling request accepted. "
        "Replicas may take time to become ready."
    )

    return 0


def execute_training():
    approved = request_confirmation(
        "Trigger the approved Airflow training pipeline.",
        "CONFIRM TRAINING",
    )

    if not approved:
        return 1

    result = trigger_training_pipeline(
        confirmed=True,
    )

    print("\nAirflow Training Result:")
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") != "success":
        return 1

    print(
        "\nTraining request accepted. "
        "Check Airflow for the final DAG run status."
    )

    return 0


def display_agent_response(question):
    result = ask_agent(question)

    print("\nMLOps AI AGENT")
    print("=" * 50)

    print(result.answer)

    print("\nTools Used:")
    print(result.tools_used)

    return 0 if result.status == "success" else 1


def interactive_session():
    print("=" * 50)
    print("Intelligent MLOps AI Agent")
    print("DVC | MLflow | Airflow | Kubernetes")
    print("=" * 50)

    print("Type exit to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        display_agent_response(question)


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent MLOps AI Agent"
    )

    actions = parser.add_mutually_exclusive_group()

    actions.add_argument(
        "--question",
        type=str,
        help="Ask the MLOps agent a question",
    )

    actions.add_argument(
        "--scale-replicas",
        type=int,
        help="Request Kubernetes inference scaling",
    )

    actions.add_argument(
        "--trigger-training",
        action="store_true",
        help="Request an Airflow training pipeline run",
    )

    arguments = parser.parse_args()

    if arguments.question is not None:
        raise SystemExit(
            display_agent_response(arguments.question)
        )

    if arguments.scale_replicas is not None:
        raise SystemExit(
            execute_scaling(arguments.scale_replicas)
        )

    if arguments.trigger_training:
        raise SystemExit(
            execute_training()
        )

    interactive_session()


if __name__ == "__main__":
    main()
