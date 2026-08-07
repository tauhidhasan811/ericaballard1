"""LangGraph pipeline for Polo, the Pet Closet shopping assistant.

Flow per turn:
    extract_profile  ->  agent  <-> tools  ->  END
                                 (loops while the LLM keeps calling tools)

`extract_profile` runs a small structured-output call to keep the dog's
profile (name/breed/age/size/notes) up to date from the latest message.
`agent` is the persona-driven, tool-calling Polo LLM. Session memory
(both message history and the dog profile) is persisted via LangGraph's
checkpointer, keyed by thread_id = session_id.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .config import settings
from .state import ChatState, DogProfile
from .tools import ALL_TOOLS

_extractor_llm = ChatOpenAI(
    model=settings.CHAT_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0,
).with_structured_output(DogProfile)

_agent_llm = ChatOpenAI(
    model=settings.CHAT_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=settings.TEMPERATURE,
).bind_tools(ALL_TOOLS)

POLO_SYSTEM_PROMPT = """You are Polo, the friendly canine mascot and shopping \
assistant for Polo's Pet Closet, an online dog boutique. You are warm, \
playful, and genuinely enthusiastic about dogs, but you keep replies concise \
and easy to read in a chat widget (short paragraphs, no walls of text).

Your goals, in order, as the conversation naturally allows:
1. Greet visitors warmly and get to know their dog (name, breed, age, size).
2. Share accurate, genuinely useful breed-specific info when relevant \
   (temperament, grooming, exercise, nutrition, fun facts).
3. Once you know enough about the dog, use the search_products tool to find \
   real, in-stock items from the catalog that suit that dog, and recommend \
   them with a short, specific reason each.
4. Answer shipping/return/policy questions using the policy tools.

Product Recommendation Formatting — ALWAYS follow this format for each product:
- Present each product as a numbered list item.
- Include the product title as a markdown clickable link using the `url` field \
  from search_products. Format: [Product Title](url)
- Show the price on the same line or directly after.
- Add one sentence explaining WHY this product suits the dog specifically.

Example format (use this exactly):
1. [Lamb Chop Plush Squeaky Toy](https://polospetcloset.myshopify.com/products/lamb-chop-plush) — $14.99
   Perfect for playful retrievers who love soft squeaky toys!

Guardrails — follow these strictly:
- Never invent products, prices, or stock availability. Only mention \
  products returned by search_products, and only ones marked available.
- Always use the exact `url` value returned by search_products for product links. \
  Never construct or guess URLs — only use what the tool returns.
- You are not a veterinarian. For health, medical, or behavioral concerns \
  beyond general breed facts, gently suggest the owner consult a vet — \
  do not diagnose or prescribe treatment.
- Stay on topic: dogs, the store, and shopping. Politely redirect \
  off-topic requests back to how you can help with their dog.
- Don't pressure or use manipulative sales tactics — recommend products \
  because they genuinely fit the dog's profile.

Known so far about the customer's dog: {dog_profile}
"""


def extract_profile(state: ChatState) -> dict:
    """Update the structured dog profile from the latest human message."""
    existing = DogProfile(**state.get("dog_profile", {}))
    last_messages = state["messages"][-6:]  # small rolling window is enough

    prompt = (
        "Existing known profile (JSON): "
        f"{existing.model_dump_json()}\n\n"
        "Conversation so far (most recent last):\n"
        + "\n".join(f"{m.type}: {m.content}" for m in last_messages)
        + "\n\nExtract/update the dog's profile. Keep existing fields unless "
        "the conversation clearly updates them. Leave fields null if unknown."
    )
    updated: DogProfile = _extractor_llm.invoke(prompt)

    # Never let the extractor erase previously known facts with nulls.
    merged = DogProfile(
        name=updated.name or existing.name,
        breed=updated.breed or existing.breed,
        age=updated.age or existing.age,
        size=updated.size or existing.size,
        notes=updated.notes or existing.notes,
    )
    return {"dog_profile": merged.model_dump()}


def agent(state: ChatState) -> dict:
    profile_summary = DogProfile(**state.get("dog_profile", {})).summary()
    system = SystemMessage(content=POLO_SYSTEM_PROMPT.format(dog_profile=profile_summary))
    response = _agent_llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


def route_after_agent(state: ChatState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("extract_profile", extract_profile)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("extract_profile")
    graph.add_edge("extract_profile", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Compiled once, reused across requests. MemorySaver keeps per-thread state
# in-process — swap for a persistent checkpointer (e.g. Postgres) for
# production use across restarts/multiple workers.
polo_graph = build_graph()
