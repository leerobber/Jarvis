"""
Long-term memory — ChromaDB vector store.
Persists experiences, facts, and reflections across sessions.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings

from jarvis.config import config

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Persistent semantic memory backed by ChromaDB.
    Uses Ollama embeddings via a custom embedding function.
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=config.MEMORY_CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="jarvis_memory",
            metadata={"hnsw:space": "cosine"},
        )
        # Import here to avoid circular at module load time
        from jarvis.core.ollama_client import OllamaClient
        self._ollama = OllamaClient()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        memory_type: str = "experience",
        metadata: Optional[dict] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Embed and persist a piece of information. Returns the memory ID."""
        embedding = self._ollama.embed(text)
        if not embedding:
            logger.warning("Empty embedding — skipping long-term storage.")
            return ""

        mid = memory_id or str(uuid.uuid4())
        meta = {
            "type": memory_type,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        self._collection.upsert(
            ids=[mid],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
        logger.debug(f"Stored memory [{memory_type}] id={mid}")
        return mid

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        """Return the top-N most semantically similar memories."""
        embedding = self._ollama.embed(query)
        if not embedding:
            return []

        where = {"type": memory_type} if memory_type else None
        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, self._collection.count() or 1),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

        memories = []
        for i, doc in enumerate(results["documents"][0]):
            memories.append({
                "id": results["ids"][0][i],
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "relevance": 1 - results["distances"][0][i],
            })
        return memories

    def retrieve_text(self, query: str, n_results: int = 5) -> str:
        """Convenience — returns memories as a formatted string."""
        memories = self.retrieve(query, n_results)
        if not memories:
            return ""
        lines = []
        for m in memories:
            tag = m["metadata"].get("type", "memory")
            lines.append(f"[{tag}] {m['text']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats / management
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def clear_all(self) -> None:
        self._client.delete_collection("jarvis_memory")
        self._collection = self._client.get_or_create_collection(
            name="jarvis_memory",
            metadata={"hnsw:space": "cosine"},
        )
