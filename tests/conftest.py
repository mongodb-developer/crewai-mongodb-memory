"""Shared pytest fixtures for the crewai-mongodb-memory acceptance suite.

Non-search tests run on ``mongomock`` (no infra needed). Because the backend always
owns its own client, the offline fixture patches ``MongoClient`` in the backend module
with a mongomock-backed shim that tolerates (and records) the ``appname``/``driver``
kwargs the backend always sets. The Atlas ``search`` test needs a real cluster + Voyage
key; it is skipped automatically when ATLAS_URI / VOYAGE_API_KEY are not set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / "demo" / ".env")
except ImportError:
    pass


def make_mongomock_factory(captured: dict | None = None):
    """Return a ``MongoClient`` replacement backed by mongomock.

    mongomock's client rejects the real driver's ``appname``/``driver`` kwargs, so we
    strip them before delegating — optionally recording them in ``captured`` so tests
    can assert the convention values were passed.
    """
    import mongomock

    def factory(*args, **kwargs):
        if captured is not None:
            captured.clear()
            captured.update(kwargs)
            if args:
                captured["_uri"] = args[0]
        kwargs.pop("appname", None)
        kwargs.pop("appName", None)
        kwargs.pop("driver", None)
        return mongomock.MongoClient(*args, **kwargs)

    return factory


@pytest.fixture()
def mock_backend(monkeypatch):
    """A MongoDBStorageBackend backed by mongomock (works offline)."""
    import crewai_mongodb_memory.backend as backend_mod
    from crewai_mongodb_memory import MongoDBStorageBackend

    monkeypatch.setattr(backend_mod, "MongoClient", make_mongomock_factory())
    backend = MongoDBStorageBackend("mongodb://localhost:27017", database_name="test_db")
    yield backend


@pytest.fixture()
def atlas_uri() -> str | None:
    return os.environ.get("ATLAS_URI")


@pytest.fixture()
def has_voyage() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY"))
