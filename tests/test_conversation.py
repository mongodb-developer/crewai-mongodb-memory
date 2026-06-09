"""Tests for crewai_mongodb_memory.ConversationMemory (short-term chat transcript).

Conversation memory is stored in its **own** ``conversations`` collection using the
bucket pattern, with **no embeddings** (replay is recency/order based). All of these run
on mongomock — no Atlas or Voyage key needed.
"""

from __future__ import annotations

from crewai_mongodb_memory import (
    ASSISTANT,
    USER,
    ConversationMemory,
    MongoDBStorageBackend,
)


# 1. turns are stored in order and rendered chronologically -------------------
def test_add_and_order(mock_backend: MongoDBStorageBackend):
    conv = ConversationMemory(mock_backend, session_id="s1")
    conv.add_turn(USER, "hello")
    conv.add_turn(ASSISTANT, "hi there")
    conv.add_turn(USER, "what's the weather?")

    turns = conv.turns()
    assert [t.content for t in turns] == ["hello", "hi there", "what's the weather?"]
    assert [t.turn for t in turns] == [1, 2, 3]
    assert [t.role for t in turns] == [USER, ASSISTANT, USER]


# 2. add_exchange stores a user+assistant pair --------------------------------
def test_add_exchange(mock_backend: MongoDBStorageBackend):
    conv = ConversationMemory(mock_backend, session_id="s1")
    conv.add_exchange("ping", "pong")
    turns = conv.turns()
    assert [(t.role, t.content) for t in turns] == [
        (USER, "ping"),
        (ASSISTANT, "pong"),
    ]


# 3. recent_turns + history_text honor the limit ------------------------------
def test_recent_and_history_text(mock_backend: MongoDBStorageBackend):
    conv = ConversationMemory(mock_backend, session_id="s1", max_turns=2)
    for i in range(4):
        conv.add_turn(USER if i % 2 == 0 else ASSISTANT, f"m{i}")

    recent = conv.recent_turns()  # default max_turns=2
    assert [t.content for t in recent] == ["m2", "m3"]

    text = conv.history_text()
    assert text == "User: m2\nAssistant: m3"

    # explicit limit overrides the default
    assert conv.history_text(limit=1) == "Assistant: m3"
    # empty session renders empty string
    assert ConversationMemory(mock_backend, session_id="empty").history_text() == ""


# 4. sessions are isolated -----------------------------------------------------
def test_session_isolation(mock_backend: MongoDBStorageBackend):
    a = ConversationMemory(mock_backend, session_id="alice")
    b = ConversationMemory(mock_backend, session_id="bob")
    a.add_turn(USER, "for alice")
    b.add_turn(USER, "for bob")

    assert [t.content for t in a.turns()] == ["for alice"]
    assert [t.content for t in b.turns()] == ["for bob"]
    assert a.count() == 1 and b.count() == 1


# 5. clear wipes only the session's turns -------------------------------------
def test_clear(mock_backend: MongoDBStorageBackend):
    a = ConversationMemory(mock_backend, session_id="alice")
    b = ConversationMemory(mock_backend, session_id="bob")
    a.add_exchange("hi", "hello")
    b.add_turn(USER, "still here")

    removed = a.clear()
    assert removed == 2
    assert a.turns() == []
    assert [t.content for t in b.turns()] == ["still here"]


# 6. bucket pattern: turns roll over across buckets, order preserved ----------
def test_bucketing_rolls_over(mock_backend: MongoDBStorageBackend):
    conv = ConversationMemory(mock_backend, session_id="s1", bucket_size=3)
    for i in range(7):
        conv.add_turn(USER, f"m{i}")

    # 7 turns at bucket_size=3 → 3 buckets (3 + 3 + 1).
    buckets = list(conv.col.find({"session_id": "s1"}).sort("bucket_seq", 1))
    assert [b["turn_count"] for b in buckets] == [3, 3, 1]
    assert [b["bucket_seq"] for b in buckets] == [0, 1, 2]
    assert [b["start_turn"] for b in buckets] == [1, 4, 7]
    assert [b["end_turn"] for b in buckets] == [3, 6, 7]

    # Full chronological order is preserved across buckets.
    assert [t.content for t in conv.turns()] == [f"m{i}" for i in range(7)]
    assert conv.count() == 7

    # recent_turns spanning a bucket boundary still returns chronological order.
    assert [t.content for t in conv.recent_turns(4)] == ["m3", "m4", "m5", "m6"]


# 7. turns are stored WITHOUT embeddings --------------------------------------
def test_no_embeddings_stored(mock_backend: MongoDBStorageBackend):
    conv = ConversationMemory(mock_backend, session_id="s1")
    conv.add_turn(USER, "no vectors here")
    bucket = conv.col.find_one({"session_id": "s1"})
    assert bucket is not None
    entry = bucket["turns"][0]
    assert "embedding" not in entry
    assert set(entry.keys()) == {"turn", "role", "content", "ts"}


# 8. conversations live in their OWN collection (not `memories`) --------------
def test_separate_collection(mock_backend: MongoDBStorageBackend):
    conv = ConversationMemory(mock_backend, session_id="s1")
    conv.add_turn(USER, "hi")
    # Default conversation collection name.
    assert conv.col.name == "conversations"
    # The long-term memories collection is untouched by conversation writes.
    assert mock_backend.col.name == "memories"
    assert mock_backend.col.count_documents({}) == 0
