"""``crewai-mongodb-memory`` — MongoDB Atlas memory backend for CrewAI.

Exposes :class:`MongoDBStorageBackend`, a full implementation of CrewAI's
``StorageBackend`` protocol (Unified Memory System), including vector ``search()`` via
Atlas ``$vectorSearch``.

Usage::

    from crewai.memory.unified_memory import Memory
    from crewai_mongodb_memory import MongoDBStorageBackend

    backend = MongoDBStorageBackend("mongodb+srv://...")
    memory = Memory(storage=backend)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .backend import (
    APP_NAME,
    DRIVER_NAME,
    MemoryRecord,
    MongoDBStorageBackend,
    ScopeInfo,
)
from .conversation import ASSISTANT, USER, ConversationMemory, Turn
from .embeddings import VOYAGE_DIM, VOYAGE_MODEL, embed_text

__all__ = [
    "MongoDBStorageBackend",
    "MemoryRecord",
    "ScopeInfo",
    "ConversationMemory",
    "Turn",
    "USER",
    "ASSISTANT",

    "embed_text",
    "APP_NAME",
    "DRIVER_NAME",
    "VOYAGE_DIM",
    "VOYAGE_MODEL",
    "__version__",
]

