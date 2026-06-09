"""MongoDB Atlas-backed ``StorageBackend`` for CrewAI's Unified Memory System.

CrewAI's ``crewai.memory.unified_memory.Memory`` accepts any object implementing the
``StorageBackend`` protocol (``crewai.memory.storage.backend.StorageBackend``):

    save / search / delete / update / get_record / list_records /
    get_scope_info / list_scopes

This module ships :class:`MongoDBStorageBackend`, a **full** implementation of that
protocol — including vector ``search()`` via Atlas ``$vectorSearch`` (the open upstream
``RedisStorageBackend`` PR #5919 explicitly leaves search/listing unsupported). Each
:class:`MemoryRecord` is stored as one document keyed by its ``id``.

Conventions applied here (100 Integs — baked in, non-overridable):
- ``appName = devrel-integ-crewai-python`` so server telemetry attributes traffic.
- ``driver_info`` handshake metadata identifies the ``crewai-mongodb-memory`` library
  (distinct from appName; see the ``add-client-metadata`` convention).
- The backend **always constructs and owns its own** ``MongoClient`` from a connection
  string, so these are guaranteed present on every connection with no caller opt-out.

The ``MemoryRecord`` / ``ScopeInfo`` types are imported from CrewAI when installed; a
lightweight local mirror is used as a fallback so the backend (and its tests) can run
without the full CrewAI dependency tree.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import TYPE_CHECKING, Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.driver_info import DriverInfo

from .embeddings import VOYAGE_DIM

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database

# -- CrewAI types (with a lightweight fallback for standalone use/tests) -------
try:  # pragma: no cover - exercised when CrewAI is installed
    from crewai.memory.types import MemoryRecord, ScopeInfo

    _HAS_CREWAI = True
except Exception:  # pragma: no cover - fallback mirror of the upstream models
    from uuid import uuid4

    from pydantic import BaseModel, Field

    _HAS_CREWAI = False

    class MemoryRecord(BaseModel):  # type: ignore[no-redef]
        """Local mirror of ``crewai.memory.types.MemoryRecord`` (fallback only)."""

        id: str = Field(default_factory=lambda: str(uuid4()))
        content: str = ""
        scope: str = "/"
        categories: list[str] = Field(default_factory=list)
        metadata: dict[str, Any] = Field(default_factory=dict)
        importance: float = 0.5
        created_at: _dt.datetime = Field(default_factory=_dt.datetime.utcnow)
        last_accessed: _dt.datetime = Field(default_factory=_dt.datetime.utcnow)
        embedding: list[float] | None = Field(default=None, exclude=True, repr=False)
        source: str | None = None
        private: bool = False

    class ScopeInfo(BaseModel):  # type: ignore[no-redef]
        """Local mirror of ``crewai.memory.types.ScopeInfo`` (fallback only)."""

        path: str
        record_count: int = 0
        categories: list[str] = Field(default_factory=list)
        oldest_record: _dt.datetime | None = None
        newest_record: _dt.datetime | None = None
        child_scopes: list[str] = Field(default_factory=list)


APP_NAME = "devrel-integ-crewai-python"
"""MongoDB connection appName for server-side attribution (100 Integs convention)."""

DRIVER_NAME = "crewai-mongodb-memory"
"""driver_info name attached to the MongoDB handshake (distinct from appName)."""

# Resolve the package version lazily so driver_info reports the installed version.
try:  # pragma: no cover - trivial
    from . import __version__ as _PKG_VERSION
except Exception:  # pragma: no cover
    _PKG_VERSION = "0.0.0"


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _scope_regex(prefix: str) -> dict[str, Any]:
    """Anchored regex matching ``prefix`` and any descendant scope path."""
    anchored = re.escape(prefix.rstrip("/"))
    # Match the scope itself or any child path (prefix followed by "/...").
    return {"$regex": f"^{anchored}(/|$)"}


def _scope_ancestors(scope: str) -> list[str]:
    """Return ``scope`` plus every ancestor prefix.

    e.g. ``/crew/staffing/team`` → ``["/crew", "/crew/staffing", "/crew/staffing/team"]``.

    Stored on each document so Atlas ``$vectorSearch`` can prefilter by scope prefix
    using array-equality (``{"scope_ancestors": prefix}``) — ``$vectorSearch`` filters
    do not support ``$regex``, so a precomputed ancestor list is the supported pattern.
    """
    normalized = (scope or "/").rstrip("/")
    if not normalized:
        return ["/"]
    parts = normalized.split("/")[1:]  # drop leading empty segment
    ancestors: list[str] = []
    acc = ""
    for part in parts:
        acc = f"{acc}/{part}"
        ancestors.append(acc)
    return ancestors



class MongoDBStorageBackend:
    """MongoDB / Atlas implementation of CrewAI's ``StorageBackend`` protocol.

    One document per :class:`MemoryRecord`, keyed by the record ``id`` (``_id``). Vector
    ``search()`` uses Atlas ``$vectorSearch`` prefiltered by scope/categories/metadata.

    The backend **always constructs and owns its own** :class:`MongoClient` from the
    supplied ``connection_string``. The 100 Integs telemetry conventions are baked in and
    cannot be bypassed: ``appName`` and ``driver_info`` are always set, and any
    ``appname``/``appName``/``driver`` value a caller passes via ``client_kwargs`` is
    ignored in favor of the convention values.

    Args:
        connection_string: MongoDB / Atlas connection URI. **Required.**
        database_name: Database to use. Default ``"crewai_memory"``.
        collection_name: Collection for memory records. Default ``"memories"``.
        vector_search_index: Name of the Atlas Vector Search index on ``embedding``.
        **client_kwargs: Extra args forwarded to :class:`MongoClient` (e.g. ``tls=True``).
            ``appname``/``appName``/``driver`` are reserved (convention values win).
    """

    def __init__(
        self,
        connection_string: str,
        *,
        database_name: str = "crewai_memory",
        collection_name: str = "memories",
        vector_search_index: str = "idx_crewai_memory",
        **client_kwargs: Any,
    ) -> None:
        if not connection_string:
            raise ValueError("connection_string is required")

        self.vector_search_index = vector_search_index

        driver_info = DriverInfo(name=DRIVER_NAME, version=_PKG_VERSION)

        # The integration owns the client. appName + driver_info are mandatory and
        # non-overridable: strip any caller-supplied values so tracking is always present.
        client_kwargs.pop("appname", None)
        client_kwargs.pop("appName", None)
        client_kwargs.pop("driver", None)

        self.client = MongoClient(
            connection_string,
            appname=APP_NAME,
            driver=driver_info,
            **client_kwargs,
        )

        self.db: Database = self.client[database_name]
        self.col: Collection = self.db[collection_name]

        self._ensure_indexes()

    # -- setup -----------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create the supporting btree indexes (scope / categories / created_at)."""
        self.col.create_index([("scope", ASCENDING)], name="scope_1")
        self.col.create_index([("categories", ASCENDING)], name="categories_1")
        self.col.create_index([("created_at", DESCENDING)], name="created_at_-1")

    def ensure_vector_index(
        self,
        *,
        wait: bool = True,
        timeout: int = 180,
    ) -> bool:
        """Create the Atlas Vector Search index on ``embedding`` if missing.

        Declares ``scope`` and ``categories`` as ``filter`` paths so ``$vectorSearch``
        can prefilter. Requires an Atlas cluster (unsupported on local MongoDB). Returns
        True once the index is queryable. No-op-safe if it already exists.
        """
        import time

        from pymongo.operations import SearchIndexModel

        existing = {idx["name"] for idx in self.col.list_search_indexes()}
        if self.vector_search_index not in existing:
            model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": VOYAGE_DIM,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "scope_ancestors"},
                        {"type": "filter", "path": "categories"},

                    ]
                },
                name=self.vector_search_index,
                type="vectorSearch",
            )
            self.col.create_search_index(model)

        if not wait:
            return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            for idx in self.col.list_search_indexes():
                if idx["name"] == self.vector_search_index and idx.get("queryable"):
                    return True
            time.sleep(3)
        return False

    # -- (de)serialization -----------------------------------------------------

    @staticmethod
    def _to_doc(record: MemoryRecord) -> dict[str, Any]:
        """Serialize a ``MemoryRecord`` to a MongoDB document.

        ``embedding`` is ``exclude=True`` on the model, so we persist it explicitly.
        """
        doc = record.model_dump()
        doc["_id"] = doc.pop("id")
        doc["embedding"] = record.embedding
        # Precomputed ancestor list enables Atlas $vectorSearch scope-prefix filtering
        # (vector-search filters do not support $regex).
        doc["scope_ancestors"] = _scope_ancestors(record.scope)
        return doc


    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> MemoryRecord:
        """Deserialize a MongoDB document back into a ``MemoryRecord``."""
        data = dict(doc)
        data["id"] = data.pop("_id")
        embedding = data.pop("embedding", None)
        record = MemoryRecord(**data)
        record.embedding = embedding
        return record

    # -- StorageBackend protocol ----------------------------------------------

    def save(self, records: list[MemoryRecord]) -> None:
        """Upsert a batch of memory records (idempotent by ``id``)."""
        for record in records:
            self.col.replace_one({"_id": record.id}, self._to_doc(record), upsert=True)


    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Vector similarity search via Atlas ``$vectorSearch``, prefiltered.

        The ``filter`` clause restricts candidates *before* the ANN comparison: by
        ``scope_prefix`` (anchored regex over descendants), ``categories`` (any-of), and
        any ``metadata_filter`` (dotted paths). Results below ``min_score`` are dropped.
        Returns ``[(MemoryRecord, score), ...]`` ordered by relevance.

        Requires an Atlas cluster with the vector index built (see
        :meth:`ensure_vector_index`).
        """
        vs_filter: dict[str, Any] = {}
        if scope_prefix:
            # $vectorSearch filters don't support $regex; match the precomputed
            # ancestor array by equality (prefix appears in every descendant's list).
            vs_filter["scope_ancestors"] = scope_prefix.rstrip("/") or "/"
        if categories:

            vs_filter["categories"] = {"$in": categories}
        if metadata_filter:
            for key, value in metadata_filter.items():
                vs_filter[f"metadata.{key}"] = value

        vector_stage: dict[str, Any] = {
            "index": self.vector_search_index,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": max(limit * 20, 100),
            "limit": limit,
        }
        if vs_filter:
            vector_stage["filter"] = vs_filter

        pipeline = [
            {"$vectorSearch": vector_stage},
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]
        if min_score > 0.0:
            pipeline.append({"$match": {"_score": {"$gte": min_score}}})

        results: list[tuple[MemoryRecord, float]] = []
        for doc in self.col.aggregate(pipeline):
            score = float(doc.pop("_score", 0.0))
            results.append((self._from_doc(doc), score))
        return results

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: _dt.datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete records matching the given criteria. Returns the number deleted.

        Criteria are ANDed. With no criteria, this is a no-op (returns 0) to avoid
        accidentally wiping the whole collection.
        """
        query: dict[str, Any] = {}
        if record_ids is not None:
            query["_id"] = {"$in": record_ids}
        if scope_prefix:
            query["scope"] = _scope_regex(scope_prefix)
        if categories:
            query["categories"] = {"$in": categories}
        if older_than is not None:
            query["created_at"] = {"$lt": older_than}
        if metadata_filter:
            for key, value in metadata_filter.items():
                query[f"metadata.{key}"] = value

        if not query:
            return 0
        return self.col.delete_many(query).deleted_count

    def update(self, record: MemoryRecord) -> None:
        """Replace an existing record with the same ``id`` (upsert)."""
        self.col.replace_one({"_id": record.id}, self._to_doc(record), upsert=True)

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """Return a single record by ``id``, or ``None`` if not found."""
        doc = self.col.find_one({"_id": record_id})
        return self._from_doc(doc) if doc else None

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List records (newest first) optionally filtered by scope prefix."""
        query: dict[str, Any] = {}
        if scope_prefix:
            query["scope"] = _scope_regex(scope_prefix)
        cursor = (
            self.col.find(query)
            .sort([("created_at", DESCENDING), ("_id", DESCENDING)])
            .skip(offset)
            .limit(limit)
        )
        return [self._from_doc(doc) for doc in cursor]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        """Aggregate count, categories, date range, and child scopes for ``scope``."""
        query = {"scope": _scope_regex(scope)}
        docs = list(self.col.find(query, {"embedding": 0}))

        categories: set[str] = set()
        child_scopes: set[str] = set()
        oldest: _dt.datetime | None = None
        newest: _dt.datetime | None = None
        normalized = scope.rstrip("/") or "/"

        for doc in docs:
            for cat in doc.get("categories", []):
                categories.add(cat)
            created = doc.get("created_at")
            if created is not None:
                if oldest is None or created < oldest:
                    oldest = created
                if newest is None or created > newest:
                    newest = created
            doc_scope = (doc.get("scope") or "/").rstrip("/") or "/"
            if doc_scope != normalized and doc_scope.startswith(
                normalized if normalized != "/" else "/"
            ):
                remainder = doc_scope[len(normalized):].lstrip("/")
                if remainder:
                    child = remainder.split("/", 1)[0]
                    base = "" if normalized == "/" else normalized
                    child_scopes.add(f"{base}/{child}")

        return ScopeInfo(
            path=scope,
            record_count=len(docs),
            categories=sorted(categories),
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=sorted(child_scopes),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        """List immediate child scope paths under ``parent``."""
        normalized = parent.rstrip("/") or "/"
        scopes: set[str] = set()
        for scope in self.col.distinct("scope"):
            s = (scope or "/").rstrip("/") or "/"
            if normalized == "/":
                if s != "/":
                    first = s.lstrip("/").split("/", 1)[0]
                    scopes.add(f"/{first}")
            elif s != normalized and s.startswith(normalized + "/"):
                remainder = s[len(normalized):].lstrip("/")
                if remainder:
                    child = remainder.split("/", 1)[0]
                    scopes.add(f"{normalized}/{child}")
        return sorted(scopes)

    # -- maintenance -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying client (the backend always owns it)."""
        self.client.close()


__all__ = [
    "MongoDBStorageBackend",
    "MemoryRecord",
    "ScopeInfo",
    "APP_NAME",
    "DRIVER_NAME",
    "VOYAGE_DIM",
]
