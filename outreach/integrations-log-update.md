# Integrations Log — Update (2026-06-07)

## What shipped

### CrewAI — MongoDB-backed agent memory + Atlas Vector Search recall

- **Package:** `crewai-mongodb-memory` v0.1.0
- **PyPI:** `crewai-mongodb-memory` *(publish pending)*
- **GitHub:** `mongodb-developer/crewai-mongodb-memory` *(repo pending)* — code in `integrations/crewai-ms/`
- **Upstream anchor:** [crewAIInc/crewAI#5919](https://github.com/crewAIInc/crewAI/pull/5919) (open `RedisStorageBackend` PR on the same protocol)
- **Extension point:** [`crewai.memory.storage.backend.StorageBackend`](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/memory/storage/backend.py) (CrewAI Unified Memory)

Built `MongoDBStorageBackend` as a **full implementation** of CrewAI's Unified Memory
`StorageBackend` protocol — making MongoDB Atlas the durable, long-term memory layer for
CrewAI agents and crews. It implements all eight protocol methods (`save`, `search`,
`delete`, `update`, `get_record`, `list_records`, `get_scope_info`, `list_scopes`) and adds
semantic recall via **Atlas Vector Search** (`$vectorSearch`) prefiltered by hierarchical
scope, categories, and metadata. It plugs in directly: `Memory(storage=MongoDBStorageBackend(uri))`.

Acceptance tests are green (9/9 — 8 offline via mongomock + 1 live Atlas `$vectorSearch`),
and the live Gemini + Atlas demo verified **long-term preference memory across sessions**:
a CrewAI agent stores user preferences in Session 1, and a brand-new crew with no shared
context recalls them in Session 2 to make a grounded recommendation.

**Why it matters / the wedge:** the in-flight upstream `RedisStorageBackend` (PR #5919)
implements the same protocol but explicitly leaves `search()`/listing unsupported. MongoDB
ships the **full** protocol including native vector search — a structural advantage no other
backend offers today. CrewAI is the highest-reach Tier-1 target (52K★ / 18.5K dependents).

**Conventions baked in:** owned `MongoClient`; appName `devrel-integ-crewai-python`;
driver-info handshake `crewai-mongodb-memory` (both non-overridable); Voyage AI 3.5 (1024-dim) embeddings.

**Status:** Build ✅ · Tests ✅ · Demos ✅ · Crawlability docs ✅ · Publish (PyPI) ⏳ · Upstream PR ⏳
**Wrike:** project *CrewAI — MS (MongoDBStorageBackend)* under Quarterly Roadmap → [open](https://www.wrike.com/open.htm?id=4478663876) (Plan & Research ✅, Code ✅, Publish active, Upstream PR active)
