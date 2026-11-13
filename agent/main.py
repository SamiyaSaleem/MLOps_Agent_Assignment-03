import argparse

from agent.orchestrator import ask_agent


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent MLOps AI Agent"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Ask the MLOps agent a question",
    )

    args = parser.parse_args()

    if args.question:
        result = ask_agent(args.question)

        print("\nMLOps AI AGENT")
        print("=" * 50)

        print(result.answer)

        print("\nTools Used:")
        print(result.tools_used)

        if result.status != "success":
            raise SystemExit(1)

        return

    print("=" * 50)
    print("Intelligent MLOps AI Agent")
    print("DVC | MLflow | Airflow | Kubernetes")
    print("=" * 50)

    print("Type 'exit' to quit.\n")

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

        result = ask_agent(question)

        print("\nAgent:")
        print(result.answer)

        print("\nTools Used:", result.tools_used)
        print("-" * 50)


if __name__ == "__main__":
    main()
