"""Agentic demo: a CrewAI (Gemini) agent with MongoDB Atlas long-term memory.

Shows the ``MongoDBStorageBackend`` powering durable, cross-session agent memory through
two CrewAI tools the Gemini agent calls on its own:

- ``remember_preference`` — store a durable user preference as a ``MemoryRecord``.
- ``recall_preferences``  — Atlas ``$vectorSearch`` over stored preferences.

The point: memory lives in **MongoDB Atlas**, not the crew's in-process context — so a
**brand-new Crew** in Session 2 (no shared state) answers correctly by recalling what
Session 1 taught. This is the surface the upstream Redis backend (PR #5919) leaves
unimplemented; MongoDB does it natively via vector search.

Setup (demo/.env is auto-loaded):
    ATLAS_URI=mongodb+srv://...
    VOYAGE_API_KEY=...     # bring-your-own 1024-dim Voyage 3.5 vectors
    GEMINI_API_KEY=...     # powers the Gemini model (CrewAI via litellm)

Install:
    pip install -e ".[dev]" crewai voyageai

Run:
    python demo/agent_demo.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# litellm (CrewAI's LLM layer) reads GEMINI_API_KEY / GOOGLE_API_KEY.
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Keep the demo non-interactive and quiet (no tracing prompt at shutdown).
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


warnings.filterwarnings("ignore")

from crewai import Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

from crewai_mongodb_memory import MemoryRecord, MongoDBStorageBackend, embed_text  # noqa: E402

DEMO_DB = "crewai_mem_agent_demo"
SCOPE = "/users/alex/preferences"
MODEL = os.environ.get("GEMINI_MODEL", "gemini/gemini-2.5-flash")

# Module-level backend handle so the function tools can reach it.
_BACKEND: MongoDBStorageBackend | None = None


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


@tool("remember_preference")
def remember_preference(fact: str) -> str:
    """Store a durable user preference/fact in MongoDB Atlas long-term memory."""
    assert _BACKEND is not None
    rec = MemoryRecord(
        content=fact,
        scope=SCOPE,
        categories=["preference"],
        embedding=embed_text(fact, input_type="document"),
    )
    _BACKEND.save([rec])
    return f"Stored preference: {fact}"


@tool("recall_preferences")
def recall_preferences(query: str) -> str:
    """Retrieve relevant user preferences from MongoDB Atlas via $vectorSearch."""
    assert _BACKEND is not None
    qv = embed_text(query, input_type="query")
    hits = _BACKEND.search(qv, scope_prefix=SCOPE, limit=3)
    if not hits:
        return "No relevant preferences found."
    return "\n".join(f"- {rec.content} (score={score:.3f})" for rec, score in hits)


def build_agent() -> Agent:
    return Agent(
        role="Personal Concierge",
        goal="Help the user using durable long-term memory stored in MongoDB Atlas.",
        backstory=(
            "You remember user preferences across sessions. When the user shares a "
            "durable preference, call remember_preference. When a request may depend on "
            "what you know about them, call recall_preferences first and use the results."
        ),
        tools=[remember_preference, recall_preferences],
        llm=MODEL,
        verbose=True,
    )


def run_task(description: str, expected_output: str) -> str:
    """Run a single-task Crew (a fresh Crew each call = a fresh 'session')."""
    agent = build_agent()
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff())


def main() -> None:
    global _BACKEND

    uri = os.environ.get("ATLAS_URI")
    if not all([uri, os.environ.get("GEMINI_API_KEY"), os.environ.get("VOYAGE_API_KEY")]):
        print("This demo needs ATLAS_URI + GEMINI_API_KEY + VOYAGE_API_KEY (see demo/.env).")
        sys.exit(1)

    print(f"=== CrewAI × MongoDB Atlas long-term memory (agentic demo, model={MODEL}) ===")
    _BACKEND = MongoDBStorageBackend(uri, database_name=DEMO_DB)
    _BACKEND.delete(scope_prefix=SCOPE)  # clean slate for a repeatable demo

    print("Ensuring Atlas Vector Search index (first build can take ~1 min)...")
    if not _BACKEND.ensure_vector_index(wait=True):
        print("Vector index did not become queryable in time.")
        sys.exit(1)
    print("Index queryable.")

    # --- Session 1: the agent learns + stores preferences ---
    banner("SESSION 1 — agent stores durable preferences in Atlas (via remember_preference)")
    out1 = run_task(
        description=(
            "The user says: 'I'm vegetarian, I avoid dairy, and I always prefer window "
            "seats on flights.' Store each durable preference using remember_preference."
        ),
        expected_output="A short confirmation of what was stored.",
    )
    print(f"\nAgent (session 1): {out1}")

    # Atlas indexing is async — wait until the stored facts are searchable.
    print("\nWaiting for Atlas to index the new memories...")
    for _ in range(20):
        if _BACKEND.search(embed_text("food", input_type="query"), scope_prefix=SCOPE, limit=1):
            break
        time.sleep(2)
    print("Memories searchable.")

    # --- Session 2: a brand-new Crew (no shared state) recalls from Atlas ---
    banner("SESSION 2 — fresh Crew (no shared context) answers via recall_preferences")
    out2 = run_task(
        description=(
            "The user is booking a long flight and pre-ordering an in-flight meal. "
            "Use recall_preferences to look up what you know about them, then recommend "
            "a seat and a meal that fit their preferences."
        ),
        expected_output="A seat + meal recommendation grounded in recalled preferences.",
    )
    print(f"\nAgent (session 2, brand-new crew): {out2}")

    banner("Proof: direct $vectorSearch recall over stored preferences")
    hits = _BACKEND.search(
        embed_text("dietary and seating preferences", input_type="query"),
        scope_prefix=SCOPE,
        limit=5,
    )
    for rec, score in hits:
        print(f"  • [{score:.3f}] {rec.content}")

    _BACKEND.delete(scope_prefix=SCOPE)
    _BACKEND.close()
    print("\nDemo complete — Gemini stored + recalled preferences via MongoDB Atlas Vector Search.")


if __name__ == "__main__":
    main()
