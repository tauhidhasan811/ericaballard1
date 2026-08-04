"""Quick local REPL for testing Polo without the FastAPI layer / widget.

Run: uv run python scripts/chat_cli.py
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import polo_graph


def main() -> None:
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    print("Polo is ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        result = polo_graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        reply = next(
            (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
            "",
        )
        print(f"Polo: {reply}\n")
        print(f"[profile so far: {result.get('dog_profile', {})}]\n")


if __name__ == "__main__":
    main()
