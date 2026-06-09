"""Interactive CLI: chat with a CrewAI agent backed by MongoDB Atlas long-term memory.

A beautiful, REPL-style terminal demo for ``crewai-mongodb-memory``. You talk to a CrewAI
"Personal Concierge" agent whose durable memory lives in **MongoDB Atlas** (not the
crew's in-process context). Teach it preferences, then watch a *brand-new* Crew recall
them across turns via Atlas ``$vectorSearch``.

Short-term conversation memory **also** lives in MongoDB Atlas via the package's
``ConversationMemory`` helper: each turn is persisted and the last few turns are replayed
into the agent's context, so the chat thread stays coherent across turns *and* survives a
CLI restart. The agent can also search the web through a Composio MCP tool for current,
real-time facts.

Everything is rendered with `rich`: a branded banner, spinners while the agent thinks,
color-coded panels for stored / recalled memories, and score bars for vector hits.

Slash commands:
    /remember <fact>   store a durable preference  -> MongoDB Atlas
    /recall <query>    direct $vectorSearch recall  (with relevance score bars)
    /scope             show the current scope: record count + categories
    /list              list stored memory records
    /history           show stored conversation turns (MongoDB Atlas)
    /clear             wipe both demo scopes (preferences + conversation)
    /help              show the command list
    /exit              quit

Anything else you type is sent to the Gemini agent, which decides on its own whether to
call ``remember_preference`` / ``recall_preferences``, or search the web — showcasing
tool autonomy on top of Atlas-backed long-term + short-term memory.


Setup (demo/.env is auto-loaded):
    ATLAS_URI=mongodb+srv://...
    VOYAGE_API_KEY=...     # bring-your-own 1024-dim Voyage 3.5 vectors
    GEMINI_API_KEY=...     # powers the Gemini model (CrewAI via litellm)

Install:
    pip install -r demo/requirements.txt        # includes rich

Run:
    python demo/cli_demo.py
"""

from __future__ import annotations

import logging
import os
import sys

import warnings
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# litellm (CrewAI's LLM layer) reads GEMINI_API_KEY / GOOGLE_API_KEY. Use a single
# canonical var so the google-genai client doesn't warn that both are set.
if os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# Quiet the "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" notice and similar noise.
logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)

# Keep the demo non-interactive and quiet (no tracing prompt at shutdown).
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

warnings.filterwarnings("ignore")

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print(
        "This demo needs `rich`. Install demo deps:\n"
        "    pip install -r demo/requirements.txt\n"
        "  (or just: pip install rich)"
    )
    sys.exit(1)

from crewai_mongodb_memory import (  # noqa: E402
    APP_NAME,
    ASSISTANT,
    USER,
    ConversationMemory,
    MemoryRecord,
    MongoDBStorageBackend,
    embed_text,
)

DEMO_DB = "crewai_mem_cli_demo"
SCOPE = "/users/alex/preferences"
# Short-term conversation memory lives in MongoDB Atlas too — handled by the package's
# ConversationMemory helper (its own scope, same backend/collection) so chat context
# survives across turns *and* CLI restarts.
CONV_SESSION = "alex"
HISTORY_TURNS = 6  # how many recent turns to replay into the agent's context
MODEL = os.environ.get("GEMINI_MODEL", "gemini/gemini-3.1-flash-lite")


# MongoDB brand greens for a polished, on-brand terminal look.
GREEN = "#00ED64"
DARK = "#001E2B"

console = Console()

# Module-level handles so the function tools / commands can reach them.
_BACKEND: MongoDBStorageBackend | None = None
_CONV: ConversationMemory | None = None




# --------------------------------------------------------------------------- UI


def banner() -> None:
    title = Text()
    title.append("CrewAI", style=f"bold {GREEN}")
    title.append("  ×  ", style="bold white")
    title.append("MongoDB Atlas", style="bold #13AA52")
    subtitle = Text(
        "Interactive long-term memory  ·  semantic recall via $vectorSearch",
        style="italic grey70",
    )
    body = Group(Align.center(title), Align.center(subtitle))
    console.print(Panel(body, border_style=GREEN, padding=(1, 4)))


def help_panel() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style=f"bold {GREEN}")
    table.add_column(style="grey85")
    rows = [
        ("/remember <fact>", "store a durable preference  →  MongoDB Atlas"),
        ("/recall <query>", "direct $vectorSearch recall (with score bars)"),
        ("/scope", "current scope: record count + categories"),
        ("/list", "list stored memory records"),
        ("/history", "show stored conversation turns (MongoDB Atlas)"),
        ("/clear", "wipe both scopes (preferences + conversation)"),
        ("/help", "show this command list"),
        ("/exit", "quit"),
        ("<anything else>", "chat (remembers/recalls + can web-search via Composio)"),
    ]

    for cmd, desc in rows:
        table.add_row(cmd, desc)
    console.print(Panel(table, title="Commands", border_style="grey50", padding=(1, 1)))


def score_bar(score: float, width: int = 24) -> Text:
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    bar = Text()
    bar.append("█" * filled, style=GREEN)
    bar.append("░" * (width - filled), style="grey37")
    bar.append(f"  {score:.3f}", style="grey70")
    return bar


def show_hits(query: str, hits: list[tuple[MemoryRecord, float]]) -> None:
    if not hits:
        console.print(
            Panel(
                Text("No relevant memories found.", style="grey70"),
                title=f"recall · {query!r}",
                border_style="grey50",
            )
        )
        return
    table = Table(show_header=True, header_style=f"bold {GREEN}", box=None, expand=True)
    table.add_column("memory", style="white", ratio=2)
    table.add_column("relevance", justify="left", ratio=1)
    for rec, score in hits:
        table.add_row(rec.content, score_bar(score))
    console.print(
        Panel(table, title=f"🔵 recall · {query!r}", border_style="#13AA52", padding=(1, 1))
    )


def show_scope() -> None:
    assert _BACKEND is not None
    info = _BACKEND.get_scope_info(SCOPE)
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold grey85")
    table.add_column(style="white")
    table.add_row("scope", SCOPE)
    table.add_row("records", str(info.record_count))
    table.add_row("categories", ", ".join(info.categories) or "—")
    table.add_row("database", DEMO_DB)
    table.add_row("appName", APP_NAME)
    console.print(Panel(table, title="📂 scope", border_style="grey50", padding=(1, 1)))


def show_list() -> None:
    assert _BACKEND is not None
    records = _BACKEND.list_records(scope_prefix=SCOPE, limit=50)
    if not records:
        console.print(Panel(Text("No memories stored yet.", style="grey70"),
                            border_style="grey50"))
        return
    table = Table(show_header=True, header_style=f"bold {GREEN}", box=None, expand=True)
    table.add_column("#", justify="right", style="grey50", width=3)
    table.add_column("memory", style="white")
    table.add_column("categories", style="grey70")
    for i, rec in enumerate(records, 1):
        table.add_row(str(i), rec.content, ", ".join(rec.categories) or "—")
    console.print(Panel(table, title="🗂  stored memories", border_style="grey50",
                        padding=(1, 1)))


# ------------------------------------------------------------------ memory ops


def do_remember(fact: str) -> None:
    assert _BACKEND is not None
    rec = MemoryRecord(
        content=fact,
        scope=SCOPE,
        categories=["preference"],
        embedding=embed_text(fact, input_type="document"),
    )
    with console.status("[grey70]Embedding + writing to MongoDB Atlas…", spinner="dots"):
        _BACKEND.save([rec])
    console.print(
        Panel(Text(fact, style="white"), title="🟢 stored in Atlas",
              border_style=GREEN, padding=(0, 2))
    )


def do_recall(query: str) -> None:
    assert _BACKEND is not None
    with console.status("[grey70]Running $vectorSearch on Atlas…", spinner="dots"):
        qv = embed_text(query, input_type="query")
        hits = _BACKEND.search(qv, scope_prefix=SCOPE, limit=5)
    show_hits(query, hits)


# ------------------------------------------------------- short-term conversation
# Short-term chat memory is provided by the package's ``ConversationMemory`` (durable in
# MongoDB Atlas). The demo just renders it; all storage logic lives in the library.


def show_history() -> None:
    assert _CONV is not None
    turns = _CONV.turns()
    if not turns:
        console.print(Panel(Text("No conversation turns stored yet.", style="grey70"),
                            border_style="grey50"))
        return
    table = Table(show_header=True, header_style=f"bold {GREEN}", box=None, expand=True)
    table.add_column("#", justify="right", style="grey50", width=3)
    table.add_column("role", style="grey70", width=10)
    table.add_column("message", style="white")
    for t in turns:
        table.add_row(str(t.turn), t.role, t.content)
    console.print(Panel(table, title="💬 conversation memory (MongoDB Atlas · bucketed)",
                        border_style="#13AA52", padding=(1, 1)))




# -------------------------------------------------------------------- agent ops



def _build_tools():
    from crewai.tools import tool

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

    return [remember_preference, recall_preferences]


def chat(message: str) -> None:
    """Send a turn to a fresh CrewAI Crew, replaying conversation history from Atlas.

    Each turn spins up a brand-new Crew (no in-process shared state). Short-term
    context is restored by replaying the last :data:`HISTORY_TURNS` turns — loaded
    straight from MongoDB Atlas via :class:`ConversationMemory` — into the task
    description, so the agent follows the thread (and the thread survives a CLI
    restart). The user message and the agent's reply are then persisted back to Atlas.
    """
    assert _CONV is not None
    from crewai import Agent, Crew, Process, Task


    agent = Agent(
        role="Personal Concierge",
        goal="Help the user using durable long-term memory stored in MongoDB Atlas.",
        backstory=(
            "You remember user preferences across sessions. When the user shares a "
            "durable preference, call remember_preference. When a request may depend on "
            "what you know about them, call recall_preferences first and use the results. "
            "You can also search the web for current, real-time information (dates, news, "
            "facts) via your web search tool — use it instead of guessing, and never make "
            "up a current date."
        ),
        tools=_build_tools(),
        llm=MODEL,
        verbose=True,
    )


    history = _CONV.history_text(HISTORY_TURNS)
    if history:
        description = (
            "Conversation so far (oldest first):\n"
            f"{history}\n\n"
            f"User's new message: {message}\n\n"
            "Reply to the new message, taking the prior conversation into account."
        )
    else:
        description = message

    task = Task(
        description=description,
        expected_output="A concise, helpful reply grounded in the user's stored memory.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    with console.status("[grey70]Concierge is thinking…", spinner="dots"):
        reply = str(crew.kickoff())

    # Persist this turn (user + assistant) to MongoDB Atlas short-term memory.
    _CONV.add_exchange(message, reply)


    console.print(
        Panel(Text(reply, style="white"), title="🤖 Concierge",
              border_style=GREEN, padding=(1, 2))
    )



# --------------------------------------------------------------------- runtime


def setup_backend(uri: str, *, need_index: bool) -> None:
    global _BACKEND, _CONV
    with console.status("[grey70]Connecting to MongoDB Atlas…", spinner="dots"):
        _BACKEND = MongoDBStorageBackend(uri, database_name=DEMO_DB)
        _CONV = ConversationMemory(
            _BACKEND,
            session_id=CONV_SESSION,
            max_turns=HISTORY_TURNS,
        )

    if need_index:

        with console.status(
            "[grey70]Ensuring Atlas Vector Search index (first build ~1 min)…",
            spinner="dots",
        ):
            ok = _BACKEND.ensure_vector_index(wait=True)
        if not ok:
            console.print("[yellow]Vector index not queryable yet — recall may be empty "
                          "for a moment.[/]")
        else:
            console.print(f"[{GREEN}]✓ Atlas Vector Search index ready.[/]")


def main() -> None:
    banner()

    uri = os.environ.get("ATLAS_URI")
    has_voyage = bool(os.environ.get("VOYAGE_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))

    if not uri:
        console.print("[red]ATLAS_URI is required (see demo/.env).[/]")
        sys.exit(1)
    if not has_voyage:
        console.print("[red]VOYAGE_API_KEY is required for embeddings (see demo/.env).[/]")
        sys.exit(1)
    if not has_gemini:
        console.print(
            "[yellow]No GEMINI_API_KEY — free-form chat is disabled. "
            "Slash commands (/remember, /recall, …) still work.[/]"
        )

    setup_backend(uri, need_index=True)
    help_panel()
    show_scope()

    try:
        while True:
            try:
                line = Prompt.ask(f"\n[bold {GREEN}]›[/]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue

            if line in ("/exit", "/quit"):
                break
            if line == "/help":
                help_panel()
            elif line == "/scope":
                show_scope()
            elif line == "/list":
                show_list()
            elif line == "/history":
                show_history()
            elif line == "/clear":
                assert _BACKEND is not None and _CONV is not None
                n = _BACKEND.delete(scope_prefix=SCOPE)
                m = _CONV.clear()
                console.print(
                    f"[grey70]Cleared {n} preference record(s) and {m} conversation "
                    f"turn(s).[/]"
                )

            elif line.startswith("/remember"):
                fact = line[len("/remember"):].strip()
                if fact:
                    do_remember(fact)
                else:
                    console.print("[yellow]Usage: /remember <fact>[/]")
            elif line.startswith("/recall"):
                query = line[len("/recall"):].strip()
                if query:
                    do_recall(query)
                else:
                    console.print("[yellow]Usage: /recall <query>[/]")
            elif line.startswith("/"):
                console.print(f"[yellow]Unknown command: {line}. Try /help.[/]")
            else:
                if not has_gemini:
                    console.print(
                        "[yellow]Free-form chat needs GEMINI_API_KEY. "
                        "Use /remember and /recall instead.[/]"
                    )
                    continue
                chat(line)
    finally:
        if _BACKEND is not None:
            _BACKEND.close()
        console.print(f"\n[{GREEN}]Bye — memories persist in MongoDB Atlas.[/]")


if __name__ == "__main__":
    main()
