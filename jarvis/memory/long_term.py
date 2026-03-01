"""
Long-term memory — ChromaDB vector store.
Persists experiences, facts, and reflections across sessions.

Includes a consolidation pass that clusters similar memories and
summarises them to keep the store lean and relevant.
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
        min_relevance: float = 0.0,
    ) -> list[dict]:
        """Return the top-N most semantically similar memories.

        Args:
            min_relevance: Minimum relevance score a memory must have to be
                           included.  Relevance is computed as
                           ``1 - cosine_distance``, where cosine_distance is
                           in [0, 2], so relevance is in [-1, 1].  Positive
                           values indicate meaningful similarity; a threshold
                           around 0.3–0.5 is typical.  Defaults to 0.0.
        """
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
            relevance = 1 - results["distances"][0][i]
            if relevance < min_relevance:
                continue
            memories.append({
                "id": results["ids"][0][i],
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "relevance": relevance,
            })
        return memories

    def retrieve_text(self, query: str, n_results: int = 5, min_relevance: float = 0.0) -> str:
        """Convenience — returns memories as a formatted string."""
        memories = self.retrieve(query, n_results, min_relevance=min_relevance)
        if not memories:
            return ""
        lines = []
        for m in memories:
            tag = m["metadata"].get("type", "memory")
            lines.append(f"[{tag}] {m['text']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Consolidation — merge near-duplicate memories to keep the store lean
    # ------------------------------------------------------------------

    def consolidate(self, llm, similarity_threshold: float = 0.92) -> int:
        """
        Find clusters of very similar memories and merge each cluster into a
        single consolidated summary memory.
        Returns the net number of memories removed.

        Uses llm.fast_generate() for summarisation.
        """
        total = self._collection.count()
        if total < 10:
            return 0

        logger.info(f"Running memory consolidation on {total} memories…")

        try:
            all_data = self._collection.get(include=["documents", "metadatas", "embeddings"])
        except Exception as e:
            logger.error(f"Consolidation fetch failed: {e}")
            return 0

        ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]
        embeddings = all_data["embeddings"]

        if not embeddings:
            return 0

        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not available — skipping consolidation.")
            return 0

        emb_matrix = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_norm = emb_matrix / norms

        visited: set[int] = set()
        clusters: list[list[int]] = []

        for i in range(len(ids)):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            sims = emb_norm @ emb_norm[i]
            for j in range(i + 1, len(ids)):
                if j not in visited and float(sims[j]) >= similarity_threshold:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)

        if not clusters:
            logger.info("Consolidation: no clusters found.")
            return 0

        removed = 0
        for cluster in clusters:
            texts = [docs[i] for i in cluster]
            combined = "\n---\n".join(texts)
            prompt = (
                "Summarise the following related memory entries into a single concise memory. "
                "Preserve all important facts. Reply with ONLY the summary text.\n\n"
                f"{combined[:2000]}"
            )
            try:
                summary = llm.fast_generate(prompt, temperature=0.1).strip()
            except Exception:
                continue

            mem_type = metas[cluster[0]].get("type", "consolidated")
            del_ids = [ids[i] for i in cluster]
            self._collection.delete(ids=del_ids)
            self.store(
                summary,
                memory_type="consolidated",
                metadata={"source_count": len(cluster), "original_type": mem_type},
            )
            removed += len(cluster) - 1

        logger.info(f"Consolidation complete — removed {removed} redundant memories.")
        return removed

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
