import argparse
from agent.orchestrator import ask_agent

def main():
    p = argparse.ArgumentParser(
        description="Intelligent MLOps AI Agent"
    )

    p.add_argument(
        "--question",
        type=str,
        help="Ask the MLOps agent a question",
    )

    arg = p.parse_args()

    if arg.question:
        r = ask_agent(arg.question)

        print("\nMLOps AI AGENT")
        print("=" * 50)

        print(r.answer)

        print("\nTools Used:")
        print(r.tools_used)

        if r.status != "success":
            raise SystemExit(1)

        return

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

        r = ask_agent(question)

        print("\nAgent:")
        print(r.answer)

        print("\nTools Used:", r.tools_used)
        print("-" * 50)


if __name__ == "__main__":
    main()
