"""Voyage AI 3.5 embedding helper.

Per the 100 Integs conventions, all embedding paths use Voyage AI 3.5
(``voyage-3.5``, 1024 dimensions). The API key is read from the ``VOYAGE_API_KEY``
environment variable and is never hardcoded.

Note: at the CrewAI ``StorageBackend`` boundary, ``search()`` already receives a
``query_embedding`` produced by CrewAI's own embedder, so the backend itself is
embedding source-agnostic. This helper exists for demos and for callers who want a
convenience text-embedding path. There is **no silent fallback** — if you ask to embed
text, ``voyageai`` must be installed and ``VOYAGE_API_KEY`` set.
"""

from __future__ import annotations

from typing import Literal

VOYAGE_MODEL = "voyage-3.5"
VOYAGE_DIM = 1024

InputType = Literal["query", "document"]


def embed_text(text: str, *, input_type: InputType = "document") -> list[float]:
    """Embed a single string with Voyage AI 3.5 and return a 1024-dim vector.

    Args:
        text: The text to embed.
        input_type: ``"query"`` for query-time embeddings, ``"document"`` for stored
            documents. Improves retrieval quality when the SDK exposes it.

    Returns:
        A list of 1024 floats.

    Raises:
        RuntimeError: if the ``voyageai`` package is not installed.
    """
    try:
        import voyageai
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "voyageai is required for embedding. Install with `pip install voyageai` "
            "and set VOYAGE_API_KEY."
        ) from exc

    client = voyageai.Client()  # reads VOYAGE_API_KEY from the environment
    result = client.embed([text], model=VOYAGE_MODEL, input_type=input_type)
    return result.embeddings[0]


__all__ = ["embed_text", "VOYAGE_MODEL", "VOYAGE_DIM"]
