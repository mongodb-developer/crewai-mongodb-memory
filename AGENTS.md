# AGENTS.md — guide for AI coding agents

A structured guide for AI agents working in `crewai-mongodb-memory`: how to build and test, where
key files live, and the MongoDB-specific rules to follow.

## What this is

`MongoDBStorageBackend` — a full implementation of CrewAI's Unified Memory
[`StorageBackend`](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/memory/storage/backend.py)
protocol, backed by MongoDB Atlas (with `$vectorSearch` for semantic recall). It plugs into
`crewai.memory.unified_memory.Memory(storage=...)`.

## Build and test commands

```bash
# Install (editable) + dev deps
pip install -e ".[dev]"

# Run the acceptance test suite (offline via mongomock; Atlas vector test auto-skips)
pytest -q

# Run the full suite incl. the live Atlas $vectorSearch test
export ATLAS_URI="mongodb+srv://…"
export VOYAGE_API_KEY="pa-…"
pytest -q

# Run the demos
pip install -r demo/requirements.txt
python demo/memory_demo.py     # corpus load + scope inspection + vector recall
python demo/agent_demo.py      # Gemini agent w/ long-term preference memory (needs GEMINI_API_KEY)
```

## Project structure

- `src/crewai_mongodb_memory/backend.py` — `MongoDBStorageBackend` (the 8 protocol methods).
- `src/crewai_mongodb_memory/embeddings.py` — Voyage 3.5 helper (`embed_text`).
- `demo/memory_demo.py`, `demo/agent_demo.py`, `demo/requirements.txt`.
- `tests/test_acceptance.py` — one test per acceptance criterion (CRUD/scope/vector +
  appName + driver-info + protocol membership).
- `EDD.md` — the MongoDB data model (source of truth for schema).
- `PLAN.md` — the 7-phase integration plan.

## Environment variables and configuration

| Name | Required | Description |
|---|---|---|
| `ATLAS_URI` | for `search()` / demos | Atlas connection string |
| `VOYAGE_API_KEY` | when embedding text | Voyage AI key for `voyage-3.5` |
| `GEMINI_API_KEY` | agent demo only | Powers the Gemini model via CrewAI |

## Conventions (do not break)

- The package **owns its `MongoClient`** — built from a connection string, never passed in.
- `appName` is fixed to `devrel-integ-crewai-python` and the driver-info handshake
  (`crewai-mongodb-memory`) is attached; both are **non-overridable** (caller `appname`/`appName`/
  `driver` kwargs are stripped).
- Embeddings use **Voyage AI 3.5** (`voyage-3.5`, 1024-dim). The backend is embedding
  source-agnostic at the protocol boundary: `search()` receives a `query_embedding` vector
  directly — no silent fallback.
- `embedding` is `exclude=True` on CrewAI's `MemoryRecord`; persist it explicitly (`_to_doc`).
- Scope-prefix filtering inside `$vectorSearch` uses the precomputed `scope_ancestors`
  array, because vector-search filters do not support `$regex`. Keep `scope_ancestors` in
  sync with `scope` on every write.

## MongoDB Skills

Use the official MongoDB agent skills from https://github.com/mongodb/agent-skills
whenever the task is MongoDB-specific and a matching skill exists.

## When To Use EDD.md

Use [EDD.md](./EDD.md) as the source of truth for the MongoDB data model in this repository.

Consult [EDD.md](./EDD.md) before making changes that touch:

- MongoDB collections, document structure, or field names
- Code paths that read or write database records (`backend.py`)
- The `StorageBackend` protocol method ↔ MongoDB operation mapping
- Index definitions (ordinary, multikey, Atlas Vector Search)
- Schema documentation, Mermaid diagrams, or entity modeling discussions
