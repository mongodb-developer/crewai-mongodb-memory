"""MongoDB Atlas-backed short-term **conversation memory** for CrewAI agents.

Where :class:`~crewai_mongodb_memory.backend.MongoDBStorageBackend` provides *long-term*,
semantically-searchable memory (CrewAI's ``StorageBackend`` protocol), this module adds
the complementary *short-term* piece: a durable, ordered chat transcript.

Design choices (deliberately different from long-term memory):

- **Separate collection** (``conversations``, not ``memories``). Conversation turns have a
  different lifecycle and access pattern than long-term facts; keeping them apart keeps the
  vector index and scope aggregations on ``memories`` clean.
- **No embeddings.** Short-term replay is purely recency/order based — we always want "the
  last N turns", never "the most *similar* turn". Embedding every turn would burn Voyage
  calls and storage on vectors we never query. (If you want semantic recall over old chat,
  promote a summary into long-term ``memories`` instead.)
- **Bucket pattern.** Turns are append-heavy and read in ranges, so we use the MongoDB
  *bucket* pattern: one document holds an array of up to ``bucket_size`` turns. New turns
  ``$push`` onto the current (newest, not-yet-full) bucket; a new bucket rolls over when it
  fills. This keeps document counts low and range reads cheap vs. one-doc-per-turn.

CrewAI builds a fresh Crew per ``kickoff()`` with no shared in-process state, so a
multi-turn conversation loses its thread between turns. Replaying recent turns from Atlas
restores it — and because it's persisted, the thread also survives a process restart.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymongo.collection import Collection

    from .backend import MongoDBStorageBackend

USER = "user"
ASSISTANT = "assistant"


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


@dataclass(frozen=True)
class Turn:
    """One conversation turn (a single user or assistant message)."""

    turn: int
    role: str
    content: str
    ts: _dt.datetime | None = None


class ConversationMemory:
    """Durable, ordered chat transcript stored in MongoDB Atlas (bucketed).

    Reuses the connection owned by a :class:`MongoDBStorageBackend` (so the appName +
    driver-info handshake is preserved) but stores turns in its **own collection** using
    the bucket pattern — no embeddings.

    Args:
        backend: A :class:`MongoDBStorageBackend`; its ``MongoClient``/database are reused.
        session_id: Logical conversation id. Buckets are partitioned by this value.
        collection_name: Collection for conversation buckets. Default ``"conversations"``.
        bucket_size: Max turns stored per bucket document. Default ``50``.
        max_turns: Default number of recent turns returned by :meth:`recent_turns` /
            :meth:`history_text`.
    """

    def __init__(
        self,
        backend: "MongoDBStorageBackend",
        *,
        session_id: str = "default",
        collection_name: str = "conversations",
        bucket_size: int = 50,
        max_turns: int = 6,
    ) -> None:
        self.session_id = session_id
        self.bucket_size = bucket_size
        self.max_turns = max_turns
        self.col: "Collection" = backend.db[collection_name]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        # One bucket lookup pattern: by session, ordered by bucket sequence.
        self.col.create_index(
            [("session_id", 1), ("bucket_seq", 1)],
            name="session_bucket",
        )

    # -- writes ----------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> Turn:
        """Append one turn (``role`` is typically ``"user"`` or ``"assistant"``).

        Pushes onto the newest non-full bucket, or rolls over into a new bucket.
        Returns the stored :class:`Turn`.
        """
        latest = self.col.find_one(
            {"session_id": self.session_id},
            sort=[("bucket_seq", -1)],
        )
        next_turn = (latest["end_turn"] + 1) if latest else 1
        entry = {"turn": next_turn, "role": role, "content": content, "ts": _now()}

        if latest is not None and latest.get("turn_count", 0) < self.bucket_size:
            # Append to the current bucket.
            self.col.update_one(
                {"_id": latest["_id"]},
                {
                    "$push": {"turns": entry},
                    "$inc": {"turn_count": 1},
                    "$set": {"end_turn": next_turn, "updated_at": entry["ts"]},
                },
            )
        else:
            # Roll over into a new bucket.
            bucket_seq = (latest["bucket_seq"] + 1) if latest else 0
            self.col.insert_one(
                {
                    "session_id": self.session_id,
                    "bucket_seq": bucket_seq,
                    "turn_count": 1,
                    "start_turn": next_turn,
                    "end_turn": next_turn,
                    "turns": [entry],
                    "created_at": entry["ts"],
                    "updated_at": entry["ts"],
                }
            )

        return Turn(turn=next_turn, role=role, content=content, ts=entry["ts"])

    def add_exchange(self, user_message: str, assistant_message: str) -> None:
        """Convenience: append a user message followed by the assistant reply."""
        self.add_turn(USER, user_message)
        self.add_turn(ASSISTANT, assistant_message)

    # -- reads -----------------------------------------------------------------

    def turns(self) -> list[Turn]:
        """All stored turns for this session, oldest-first (chronological)."""
        out: list[Turn] = []
        for bucket in self.col.find({"session_id": self.session_id}).sort("bucket_seq", 1):
            for e in bucket.get("turns", []):
                out.append(
                    Turn(turn=e["turn"], role=e["role"], content=e["content"],
                         ts=e.get("ts"))
                )
        return out

    def recent_turns(self, limit: int | None = None) -> list[Turn]:
        """The last ``limit`` turns (defaults to ``max_turns``), oldest-first.

        Reads only the buckets needed to cover ``limit`` turns (newest buckets first),
        then returns them in chronological order.
        """
        limit = self.max_turns if limit is None else limit
        if limit <= 0:
            return []
        collected: list[Turn] = []
        # Walk buckets newest-first, prepending, until we have enough turns.
        for bucket in self.col.find({"session_id": self.session_id}).sort("bucket_seq", -1):
            entries = bucket.get("turns", [])
            for e in reversed(entries):
                collected.append(
                    Turn(turn=e["turn"], role=e["role"], content=e["content"],
                         ts=e.get("ts"))
                )
            if len(collected) >= limit:
                break
        collected.reverse()  # back to chronological
        return collected[-limit:]

    def history_text(
        self,
        limit: int | None = None,
        *,
        user_label: str = "User",
        assistant_label: str = "Assistant",
    ) -> str:
        """Render recent turns as a plain-text transcript for prompt injection.

        Returns an empty string when there is no history yet.
        """
        lines: list[str] = []
        for t in self.recent_turns(limit):
            label = assistant_label if t.role == ASSISTANT else user_label
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)

    def count(self) -> int:
        """Total number of stored turns in this session (summed across buckets)."""
        pipeline: list[dict[str, Any]] = [
            {"$match": {"session_id": self.session_id}},
            {"$group": {"_id": None, "n": {"$sum": "$turn_count"}}},
        ]
        docs = list(self.col.aggregate(pipeline))
        return int(docs[0]["n"]) if docs else 0

    # -- maintenance -----------------------------------------------------------

    def clear(self) -> int:
        """Delete all buckets (and thus all turns) for this session.

        Returns the number of *turns* removed (not buckets).
        """
        removed = self.count()
        self.col.delete_many({"session_id": self.session_id})
        return removed


__all__ = ["ConversationMemory", "Turn", "USER", "ASSISTANT"]
