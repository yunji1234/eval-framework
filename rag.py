"""
rag.py — Retrieval-Augmented Generation using ChromaDB and Claude.

Copied from rag-system/rag.py for standalone use in eval-framework.
"""

import logging
import os
import sys
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from config import COLLECTION_NAME, DEFAULT_DB_PATH, EMBEDDING_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
N_RESULTS    = 8
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS   = 1024

SYSTEM_PROMPT = (
    "You are a knowledgeable HR assistant for Archery Canada. "
    "Answer employee questions accurately and directly, using only the provided policy context.\n\n"
    "Rules:\n"
    "1. If the context contains the answer, state it clearly. "
    "Never say information is unavailable when it appears in the retrieved sections.\n"
    "2. If a policy lists only the permitted uses for something (e.g. sick leave), "
    "treat any unlisted use as not permitted and say so directly "
    "(e.g. 'No, that is not a permitted use under this policy').\n"
    "3. Only say the information is not covered if it genuinely does not appear "
    "anywhere in the provided context."
)


# ── Lazy singletons ───────────────────────────────────────────────────────────
_embedding_function: "embedding_functions.SentenceTransformerEmbeddingFunction | None" = None
_anthropic_client: "anthropic.Anthropic | None" = None


def _get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    global _embedding_function
    if _embedding_function is None:
        logger.debug("Loading embedding model '%s' (first call only)…", EMBEDDING_MODEL)
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_function


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def get_collection(db_path: str = DEFAULT_DB_PATH) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=db_path)
    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found in '{db_path}'. "
            "Run 'python ingest.py' to build the vector store first."
        ) from exc


def _fetch_chunk_by_index(
    collection: chromadb.Collection, source: str, chunk_index: int
) -> dict | None:
    """Fetch a single chunk by source filename and chunk_index, or return None."""
    result = collection.get(
        where={"$and": [
            {"source":      {"$eq": source}},
            {"chunk_index": {"$eq": chunk_index}},
        ]},
        include=["documents", "metadatas"],
    )
    if not result["documents"]:
        return None
    return {
        "text":        result["documents"][0],
        "source":      result["metadatas"][0]["source"],
        "chunk_index": result["metadatas"][0]["chunk_index"],
        "distance":    None,  # not a vector-search result
    }


def search(query: str, n_results: int = N_RESULTS, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    if n_results < 1:
        raise ValueError(f"n_results must be >= 1, got {n_results}")

    collection = get_collection(db_path)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":        results["documents"][0][i],
            "source":      results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance":    results["distances"][0][i],
        })

    # For every retrieved chunk, also pull in the neighbouring chunks so that
    # content split across a boundary is never silently truncated.  The previous
    # chunk catches answers that start just before the matched text (e.g. the
    # duration of a probation period whose outcomes are in the next chunk); the
    # next chunk catches answers that run past the end of the matched text.
    seen = {(c["source"], c["chunk_index"]) for c in chunks}
    extras: list[dict] = []
    for chunk in chunks:
        for offset in (-2, -1, +1, +2):
            neighbour = _fetch_chunk_by_index(
                collection, chunk["source"], chunk["chunk_index"] + offset
            )
            if neighbour and (neighbour["source"], neighbour["chunk_index"]) not in seen:
                extras.append(neighbour)
                seen.add((neighbour["source"], neighbour["chunk_index"]))

    chunks.extend(extras)
    # Present chunks in document order so the model reads coherent passages.
    chunks.sort(key=lambda c: (c["source"], c["chunk_index"]))
    return chunks


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return (
            "No relevant HR policy context was found for the following question. "
            "If you can answer from general knowledge, do so, but clearly state "
            "that no policy document was available.\n\n"
            f"Employee Question: {question}"
        )

    context_sections = []
    for i, chunk in enumerate(chunks, 1):
        context_sections.append(
            f"--- Source {i}: {chunk['source']} (chunk {chunk['chunk_index']}) ---\n"
            f"{chunk['text']}"
        )

    context_block = "\n\n".join(context_sections)

    return (
        "Use the following HR policy excerpts to answer the employee's question.\n\n"
        f"{context_block}\n\n"
        f"Employee Question: {question}"
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ask(question: str, n_results: int = N_RESULTS, db_path: str = DEFAULT_DB_PATH) -> str:
    client = _get_anthropic_client()
    chunks = search(question, n_results=n_results, db_path=db_path)
    prompt = build_prompt(question, chunks)

    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise EnvironmentError(
            "Anthropic API key is invalid or revoked. "
            "Check ANTHROPIC_API_KEY in your .env file."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError(
            "Anthropic rate limit reached. Wait a moment and try again."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError(
            "Could not reach the Anthropic API. Check your internet connection."
        ) from exc
    except anthropic.APIError as exc:
        raise RuntimeError(f"Anthropic API error: {exc}") from exc

    if not message.content:
        raise RuntimeError(
            "Claude returned an empty response (no content blocks). "
            f"stop_reason={message.stop_reason!r}"
        )

    return message.content[0].text
