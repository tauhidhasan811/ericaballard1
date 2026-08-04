"""FastAPI layer exposing Polo as a /chat endpoint for the storefront widget."""
from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from .config import settings
from .graph import polo_graph
from .products import catalog

app = FastAPI(title="Polo Shopping Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the storefront domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    dog_profile: dict


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    result = polo_graph.invoke({"messages": [HumanMessage(content=req.message)]}, config)

    reply_text = next(
        (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
        "",
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply_text,
        dog_profile=result.get("dog_profile", {}),
    )


@app.post("/admin/refresh-catalog")
def refresh_catalog() -> dict:
    """Force-refresh the cached product catalog ahead of the TTL."""
    catalog.force_refresh()
    return {"status": "ok", "product_count": len(catalog.all())}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "shop": settings.SHOP_DOMAIN}
