# Polo — AI Shopping Assistant (Polo's Pet Closet)

LangGraph + FastAPI backend for Polo, a branded conversational shopping
assistant. Products are pulled live from Shopify's **public**
`https://{shop}/products.json` storefront feed — no Admin API access needed.

## Architecture

```
app/
  config.py     # env-driven settings
  products.py   # fetch + normalize + TTL-cache + search Shopify's products.json
  tools.py      # LangChain tools: search_products, get_shipping_policy, get_return_policy
  state.py      # LangGraph state schema + DogProfile model
  graph.py       # LangGraph pipeline: extract_profile -> agent <-> tools
  main.py       # FastAPI app exposing POST /chat
scripts/
  chat_cli.py   # local REPL for testing without the HTTP layer
```

### Conversation flow

Each turn:
1. **extract_profile** — a small structured-output LLM call updates the
   known `DogProfile` (name, breed, age, size, notes) from the latest
   message, without erasing previously known facts.
2. **agent** — the persona-driven Polo LLM (system prompt includes the
   current dog profile) decides whether to respond directly or call a tool.
3. **tools** — `search_products` hits the live/cached Shopify catalog;
   `get_shipping_policy` / `get_return_policy` answer FAQs. Loops back to
   the agent until it produces a final reply.

Session state (message history + dog profile) is persisted per
`session_id` via LangGraph's checkpointer (in-memory `MemorySaver` here —
swap for a Postgres/Redis checkpointer for multi-worker production use).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd polo-assistant
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
uv sync
```

## Run the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi! I have a 2 year old chihuahua named Biscuit."}'
```

Response includes a `session_id` — pass it back on subsequent requests to
keep the same conversation/profile:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id from previous response>", "message": "What clothes would suit her?"}'
```

## Run the local CLI (no HTTP layer)

```bash
uv run python scripts/chat_cli.py
```

## Force a catalog refresh

The catalog is cached for `PRODUCT_CACHE_TTL_MINUTES` (default 30). To
refresh immediately (e.g. after a product launch):

```bash
curl -X POST http://localhost:8000/admin/refresh-catalog
```

## Notes / next steps

- **Matching quality**: `search_products` currently does weighted keyword
  matching over title/description/product_type/tags (most products in this
  store have no tags). For better semantic matching ("something for a dog
  that overheats easily"), swap in embeddings (e.g. `OpenAIEmbeddings` +
  a vector store) over the same normalized product records.
- **Pagination**: `/products.json` is fetched in pages of 250 until empty;
  raise the safety cap in `products.py` if the catalog ever exceeds ~10k
  items.
- **Guardrails**: the system prompt in `graph.py` forbids inventing
  products/prices and defers medical questions to a vet. Tighten further
  (e.g. add an output moderation node) before production if needed.
- **Widget integration**: point the Shopify theme's chat widget at
  `POST /chat`, storing `session_id` in the browser session/localStorage
  so the conversation persists across page loads.
