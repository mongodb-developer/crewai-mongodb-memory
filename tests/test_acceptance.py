"""Acceptance suite for crewai-mongodb-memory.

One test per Phase-2 acceptance criterion. Non-search tests run on mongomock;
`test_search_vector` requires Atlas + a Voyage key and is skipped otherwise.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import pytest

from crewai_mongodb_memory import MemoryRecord, MongoDBStorageBackend
from crewai_mongodb_memory.backend import APP_NAME, DRIVER_NAME

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "data" / "embeddings.json"
VECTOR_DB = "test_crewai_vector_db"


def _rec(content: str, scope: str = "/", **kw) -> MemoryRecord:
    return MemoryRecord(content=content, scope=scope, **kw)


# 1. save + get_record round-trip (incl. embedding) ---------------------------
def test_save_and_get_record(mock_backend: MongoDBStorageBackend):
    rec = _rec("Alex leads the React team", scope="/crew/eng", embedding=[0.1] * 8)
    mock_backend.save([rec])
    got = mock_backend.get_record(rec.id)
    assert got is not None
    assert got.content == "Alex leads the React team"
    assert got.scope == "/crew/eng"
    assert got.embedding == [0.1] * 8


# 2. list_records newest-first + limit/offset ---------------------------------
def test_list_records_order_and_paging(mock_backend: MongoDBStorageBackend):
    base = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
    for i in range(5):
        mock_backend.save(
            [_rec(f"msg {i}", scope="/crew/a", created_at=base + _dt.timedelta(minutes=i))]
        )
    recs = mock_backend.list_records(scope_prefix="/crew/a", limit=3)
    assert [r.content for r in recs] == ["msg 4", "msg 3", "msg 2"]
    page2 = mock_backend.list_records(scope_prefix="/crew/a", limit=3, offset=3)
    assert [r.content for r in page2] == ["msg 1", "msg 0"]


# 3. update + delete by id ----------------------------------------------------
def test_update_and_delete_by_id(mock_backend: MongoDBStorageBackend):
    rec = _rec("original", scope="/crew/a")
    mock_backend.save([rec])
    rec.content = "updated"
    mock_backend.update(rec)
    assert mock_backend.get_record(rec.id).content == "updated"
    assert mock_backend.delete(record_ids=[rec.id]) == 1
    assert mock_backend.get_record(rec.id) is None


# 4. scoped / categorical / time-based delete ---------------------------------
def test_scoped_deletes(mock_backend: MongoDBStorageBackend):
    old = _dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc)
    mock_backend.save([_rec("a", scope="/crew/x", categories=["t1"])])
    mock_backend.save([_rec("b", scope="/crew/x/child", categories=["t2"])])
    mock_backend.save([_rec("c", scope="/crew/y", created_at=old)])

    # category delete
    assert mock_backend.delete(categories=["t2"]) == 1
    # older_than delete
    assert mock_backend.delete(older_than=_dt.datetime(2021, 1, 1, tzinfo=_dt.timezone.utc)) == 1
    # scope-prefix delete (matches /crew/x and descendants — only /crew/x left)
    assert mock_backend.delete(scope_prefix="/crew/x") == 1
    # empty criteria is a safe no-op
    assert mock_backend.delete() == 0


# 5. scope info + list_scopes -------------------------------------------------
def test_scope_info_and_list_scopes(mock_backend: MongoDBStorageBackend):
    mock_backend.save([_rec("a", scope="/crew/eng", categories=["code"])])
    mock_backend.save([_rec("b", scope="/crew/eng/frontend", categories=["ui"])])
    mock_backend.save([_rec("c", scope="/crew/sales")])

    info = mock_backend.get_scope_info("/crew/eng")
    assert info.record_count == 2  # /crew/eng + /crew/eng/frontend
    assert set(info.categories) == {"code", "ui"}
    assert "/crew/eng/frontend" in info.child_scopes

    assert mock_backend.list_scopes("/crew") == ["/crew/eng", "/crew/sales"]
    assert mock_backend.list_scopes("/") == ["/crew"]


# 6. search (Atlas only) ------------------------------------------------------
@pytest.mark.skipif(
    not (os.environ.get("ATLAS_URI") and os.environ.get("VOYAGE_API_KEY")),
    reason="search needs ATLAS_URI + VOYAGE_API_KEY",
)
def test_search_vector():
    import time

    from crewai_mongodb_memory import embed_text

    backend = MongoDBStorageBackend(os.environ["ATLAS_URI"], database_name=VECTOR_DB)
    docs = json.loads(DATASET.read_text())
    backend.col.delete_many({"scope": "/corpus/team"})
    records = [
        MemoryRecord(
            content=f"{d['name']} — {d['title']}",
            scope="/corpus/team",
            categories=[d.get("department", "unknown")],
            metadata={"name": d["name"]},
            embedding=d["embedding"],
        )
        for d in docs
    ]
    backend.save(records)
    assert backend.ensure_vector_index(wait=True), "vector index never became queryable"

    qv = embed_text("who can lead a React frontend project?", input_type="query")
    hits: list = []
    for _ in range(20):
        hits = backend.search(qv, scope_prefix="/corpus/team", limit=5)
        if hits:
            break
        time.sleep(2)
    assert hits, "no vector search results"
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    backend.close()


# 7. appName + driver-info present --------------------------------------------
def test_appname_and_driver_present():
    import mongomock
    from unittest.mock import patch

    assert APP_NAME == "devrel-integ-crewai-python"
    assert DRIVER_NAME == "crewai-mongodb-memory"

    captured = {}
    real_init = mongomock.MongoClient.__init__

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        kwargs.pop("appname", None)
        kwargs.pop("driver", None)
        real_init(self, *args, **kwargs)

    with patch("crewai_mongodb_memory.backend.MongoClient", mongomock.MongoClient):
        with patch.object(mongomock.MongoClient, "__init__", spy_init):
            MongoDBStorageBackend("mongodb://localhost:27017", database_name="x")

    assert captured.get("appname") == APP_NAME
    assert captured.get("driver") is not None
    assert captured["driver"].name == DRIVER_NAME


# 8a. tracking not overridable ------------------------------------------------
def test_tracking_not_overridable():
    import mongomock
    from unittest.mock import patch

    captured = {}
    real_init = mongomock.MongoClient.__init__

    def spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        kwargs.pop("appname", None)
        kwargs.pop("driver", None)
        real_init(self, *args, **kwargs)

    with patch("crewai_mongodb_memory.backend.MongoClient", mongomock.MongoClient):
        with patch.object(mongomock.MongoClient, "__init__", spy_init):
            MongoDBStorageBackend(
                "mongodb://localhost:27017",
                database_name="x",
                appname="evil-app",
                appName="evil-app",
                driver="not-a-driver",
            )

    assert captured.get("appname") == APP_NAME
    assert captured["driver"].name == DRIVER_NAME


# 8b. satisfies the StorageBackend protocol -----------------------------------
def test_satisfies_storage_backend_protocol(mock_backend: MongoDBStorageBackend):
    """If CrewAI is installed, assert runtime_checkable protocol membership;
    otherwise assert the full method surface is present (duck-typed)."""
    try:
        from crewai.memory.storage.backend import StorageBackend

        assert isinstance(mock_backend, StorageBackend)
    except Exception:
        for method in (
            "save",
            "search",
            "delete",
            "update",
            "get_record",
            "list_records",
            "get_scope_info",
            "list_scopes",
        ):
            assert callable(getattr(mock_backend, method))
