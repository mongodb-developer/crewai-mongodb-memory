"""Demo: MongoDB Atlas memory backend for CrewAI — cross-session recall.

Loads the canonical team-member corpus (``data/embeddings.json``, Voyage 3.5 vectors)
into a :class:`MongoDBStorageBackend`, then proves the full ``StorageBackend`` surface:

1. ``save`` the corpus as ``MemoryRecord`` documents under a hierarchical scope.
2. Scope inspection (``list_scopes`` / ``get_scope_info`` / ``list_records``).
3. Atlas ``$vectorSearch`` recall via ``search()`` — the surface the upstream Redis
   backend (PR #5919) explicitly does NOT support.

Run:
    export ATLAS_URI="mongodb+srv://..."
    export VOYAGE_API_KEY="..."
    python demo/memory_demo.py

Without ATLAS_URI the script runs the CRUD/scope portion against a local MongoDB (or
prints guidance). The vector recall step requires Atlas.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from crewai_mongodb_memory import MemoryRecord, MongoDBStorageBackend, embed_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "data" / "embeddings.json"
SCOPE = "/crew/staffing/team"


def main() -> None:
    uri = os.environ.get("ATLAS_URI")
    if not uri:
        print("Set ATLAS_URI (and VOYAGE_API_KEY) to run the full vector demo.")
        return

    backend = MongoDBStorageBackend(uri, database_name="crewai_memory_demo")
    print(f"appName attributed as: {backend.client.options.pool_options.metadata}")

    # 1. Seed the corpus as MemoryRecords ------------------------------------
    docs = json.loads(DATASET.read_text())
    backend.col.delete_many({"scope": {"$regex": "^/crew/staffing"}})
    records = [
        MemoryRecord(
            content=f"{d['name']} — {d['title']} ({d.get('department', '?')})",
            scope=SCOPE,
            categories=[d.get("department", "unknown")],
            metadata={"name": d["name"], "title": d["title"]},
            embedding=d["embedding"],
        )
        for d in docs
    ]
    backend.save(records)
    print(f"Saved {len(records)} memory records under {SCOPE}")

    # 2. Scope inspection -----------------------------------------------------
    print("Child scopes of /crew:", backend.list_scopes("/crew"))
    info = backend.get_scope_info("/crew/staffing")
    print(
        f"Scope /crew/staffing → {info.record_count} records, "
        f"categories={info.categories}"
    )

    # 3. Atlas vector recall via search() ------------------------------------
    if not os.environ.get("VOYAGE_API_KEY"):
        print("Set VOYAGE_API_KEY to run vector recall.")
        backend.close()
        return

    print("Building Atlas Vector Search index (idempotent)...")
    backend.ensure_vector_index(wait=True)

    query = "who can lead a React frontend project?"
    qv = embed_text(query, input_type="query")
    hits: list = []
    for _ in range(20):
        hits = backend.search(qv, scope_prefix=SCOPE, limit=5)
        if hits:
            break
        time.sleep(2)

    print(f"\nQuery: {query!r}")
    for rec, score in hits:
        print(f"  ({score:.3f}) {rec.metadata.get('name')} — {rec.metadata.get('title')}")

    backend.close()


if __name__ == "__main__":
    main()
