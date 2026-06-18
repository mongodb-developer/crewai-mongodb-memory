# crewai-mongodb-memory — MongoDB Atlas memory backend for CrewAI

> ⚠️ **ALPHA — NOT AN OFFICIAL MONGODB PRODUCT.** This integration is in **Alpha** and is **not** a supported or official MongoDB product. **Use at your own risk.**

MongoDB Atlas integration for **CrewAI** providing the **Memory Store (MS)** capability:
`MongoDBStorageBackend`, a full implementation of CrewAI's Unified Memory `StorageBackend`
protocol that makes Atlas the long-term memory layer for CrewAI agents and crews — including
semantic recall via **Atlas Vector Search**.

- **appName:** `devrel-integ-crewai-python`
- **Embeddings:** Voyage AI 3.5 (`voyage-3.5`, 1024-dim)
- **Extension point:** [`crewai.memory.storage.backend.StorageBackend`](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/memory/storage/backend.py)
- **Plan:** see [`PLAN.md`](./PLAN.md) — the per-integration 7-phase plan.
- **Schema:** see [`EDD.md`](./EDD.md) — the MongoDB data model (entities, indexes, diagram).

## Capabilities

- Drop-in `StorageBackend` for `crewai.memory.unified_memory.Memory(storage=...)` —
  implements all 8 protocol methods (`save`, `search`, `delete`, `update`, `get_record`,
  `list_records`, `get_scope_info`, `list_scopes`).
- **Semantic recall** via Atlas `$vectorSearch`, prefiltered by hierarchical `scope`,
  `categories`, and arbitrary `metadata` — the surface the upstream `RedisStorageBackend`
  (PR [#5919](https://github.com/crewAIInc/crewAI/pull/5919)) explicitly leaves unimplemented.
- Hierarchical scopes (`/crew/team/user`) with prefix queries over descendants.
- **Durable short-term conversation memory** via `ConversationMemory` — persists each chat
  turn to Atlas and replays the last N turns into the agent's context. CrewAI builds a fresh
  Crew per `kickoff()` with no shared state, so this keeps a multi-turn thread coherent *and*
  lets it survive a process restart. Stored in its **own `conversations` collection** (kept
  out of the vector index), **without embeddings** (replay is recency/order based, not
  semantic), using the MongoDB **bucket pattern** (an array of turns per document) for cheap
  append + range reads.

- Own-the-client design: appName + driver-info handshake always present, non-overridable.


## Architecture Overview

The backend stores one MongoDB document per `MemoryRecord` (keyed by the record `id`) and
maps the protocol onto MongoDB:

- Collection `memories` — one document per memory record.
- Indexes — `scope`, `categories`, `created_at`, plus an Atlas Vector Search index over
  `embedding` (`numDimensions: 1024`, cosine) with `filter` paths `scope_ancestors` +
  `categories`.
- Queries — `replace_one` upserts for writes; `$vectorSearch` for semantic `search()`.
- Scope-prefix filtering inside `$vectorSearch` uses a precomputed `scope_ancestors` array
  (vector-search filters don't support `$regex`).

See [`EDD.md`](./EDD.md) for the full schema contract.

## Prerequisites

- Python 3.10+
- A MongoDB connection: local `mongodb://localhost:27017` works for CRUD/scope operations;
  **MongoDB Atlas** is required for `search()` (Vector Search).
- `VOYAGE_API_KEY` if you embed query/document text with `voyage-3.5` (the demos do).
- `GEMINI_API_KEY` for the agentic demo (`demo/agent_demo.py`).

## Quick Start

```bash
# 1. Install the package
pip install crewai-mongodb-memory          # or, from this repo: pip install -e ".[dev]"

# 2. Use it as a CrewAI memory backend
python - <<'PY'
from crewai.memory.unified_memory import Memory
from crewai_mongodb_memory import MongoDBStorageBackend

backend = MongoDBStorageBackend("mongodb+srv://…")   # owns its own client
memory = Memory(storage=backend)                      # drop-in Unified Memory backend
PY

# 3. Run the demos (see demo/requirements.txt)
pip install -r demo/requirements.txt
export ATLAS_URI="<your Atlas connection string>"
export VOYAGE_API_KEY="<your Voyage key>"
export GEMINI_API_KEY="<your Gemini key>"   # agent demo only
python demo/memory_demo.py    # vector recall over the canonical corpus
python demo/agent_demo.py     # Gemini agent with long-term preference memory
python demo/cli_demo.py       # interactive REPL: chat + /remember /recall /scope

# 4. Run the acceptance tests (offline via mongomock; Atlas test auto-skips without creds)
pytest -q
```

Expected: `memory_demo.py` prints scope info + top-k semantic matches; `agent_demo.py`
shows a brand-new crew recalling preferences stored in a previous session; the test suite
reports 9 passed (or 8 passed + 1 skipped without Atlas creds).

## Environment variables

| Name | Required | Example | Description |
|---|---|---|---|
| `ATLAS_URI` | for `search()` / demos | `mongodb+srv://…` | Atlas connection string |
| `VOYAGE_API_KEY` | when embedding text | `pa-…` | Voyage AI key for `voyage-3.5` |
| `GEMINI_API_KEY` | agent demo only | `AIza…` | Powers the Gemini model via CrewAI |

## Project structure

```
src/crewai_mongodb_memory/   # MongoDBStorageBackend + Voyage embedding helper
demo/                 # memory_demo.py, agent_demo.py, requirements.txt
tests/                # acceptance tests (CRUD/scope/vector + appName + driver-info)
EDD.md                # MongoDB schema contract (entities, indexes, Mermaid diagram)
AGENTS.md             # guide for AI coding agents
PLAN.md               # the per-integration 7-phase plan
```

## Why MongoDB?

- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/) — semantic recall over agent memory, natively.
- [Atlas Search (full-text)](https://www.mongodb.com/docs/atlas/atlas-search/)
- One operational database for memory + application data — no extra vector store to run.

## Additional resources

- Outreach: [`outreach/blog.md`](./outreach/blog.md), [`outreach/social.md`](./outreach/social.md)
- Upstream protocol: [`StorageBackend`](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/memory/storage/backend.py) · precedent PR [#5919](https://github.com/crewAIInc/crewAI/pull/5919)
- Package: https://pypi.org/project/crewai-mongodb-memory/ (pending publish)

## Status

See [`PLAN.md`](./PLAN.md) for the current phase and `memory-bank/progress.md` for the board.
