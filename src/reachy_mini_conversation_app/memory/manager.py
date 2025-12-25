"""Memory manager for high-level memory operations with OpenAI integration."""

import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from reachy_mini_conversation_app.memory.database import (
    MemoryDatabase,
    Memory,
    KnownPerson,
    LearnedFact,
)


logger = logging.getLogger(__name__)


class MemoryManager:
    """High-level memory manager with async operations and OpenAI integration.

    Provides semantic search, memory preloading, and integration with the
    conversation system.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Initialise the memory manager.

        Args:
            db_path: Path to the DuckDB file. If None, uses default.
            openai_api_key: OpenAI API key for embeddings. If None, uses env var.
        """
        self.db = MemoryDatabase(db_path)
        self._openai_client: Optional[AsyncOpenAI] = None
        self._openai_api_key = openai_api_key
        self._ready = False
        self._preloaded = False

    @property
    def is_ready(self) -> bool:
        """Check if the memory manager is ready."""
        return self._ready

    async def initialize(self, openai_api_key: Optional[str] = None) -> None:
        """Async initialisation of the memory manager.

        Args:
            openai_api_key: OpenAI API key. Uses stored key or env var if None.
        """
        if openai_api_key:
            self._openai_api_key = openai_api_key

        if self._openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=self._openai_api_key)

        self._ready = True
        logger.info("MemoryManager initialised and ready")

    def close(self) -> None:
        """Close the memory manager and database."""
        self.db.close()
        self._ready = False

    # ---------- Memory Preloading ----------

    async def preload_memories_from_files(self, personality_dir: Path) -> int:
        """Preload memories from personality files.

        Loads memories from:
        - main-user: Basic user info
        - main-user-memories: Detailed preferences
        - robot-personality: Core personality (stored as system memory)

        Args:
            personality_dir: Path to the personality-and-user directory.

        Returns:
            Number of memories loaded.
        """
        if self._preloaded:
            logger.info("Memories already preloaded, skipping")
            return 0

        count = 0

        # Check if we already have preloaded memories
        existing = self.db.get_memories_by_source("preloaded")
        if existing:
            logger.info(f"Found {len(existing)} existing preloaded memories")
            self._preloaded = True
            return len(existing)

        # Load main-user file
        main_user_file = personality_dir / "main-user"
        if main_user_file.exists():
            content = main_user_file.read_text(encoding="utf-8").strip()
            # Split into paragraphs and store each as a memory
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for para in paragraphs:
                if para and not para.startswith("#"):
                    self.db.add_memory(
                        content=para,
                        source="preloaded",
                        metadata={"file": "main-user", "type": "user_info"},
                    )
                    count += 1
            logger.info(f"Loaded {len(paragraphs)} memories from main-user")

        # Load main-user-memories file
        memories_file = personality_dir / "main-user-memories"
        if memories_file.exists():
            content = memories_file.read_text(encoding="utf-8").strip()
            # Each line that's not empty is a memory
            lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
            for line in lines:
                if line and not line.startswith("#"):
                    self.db.add_memory(
                        content=line,
                        source="preloaded",
                        metadata={"file": "main-user-memories", "type": "preference"},
                    )
                    count += 1
            logger.info(f"Loaded {len(lines)} memories from main-user-memories")

        self._preloaded = True
        logger.info(f"Total preloaded memories: {count}")
        return count

    # ---------- Memory Operations ----------

    async def add_memory(
        self,
        content: str,
        source: str = "voice_command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a new memory.

        Args:
            content: The memory content.
            source: Source of the memory.
            metadata: Optional metadata.

        Returns:
            The ID of the inserted memory.
        """
        # Generate embedding if OpenAI client is available
        embedding = None
        if self._openai_client:
            try:
                embedding = await self._generate_embedding(content)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        return self.db.add_memory(
            content=content,
            source=source,
            metadata=metadata,
            embedding=embedding,
        )

    def get_relevant_memories(
        self,
        context: Optional[str] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Get memories relevant to the current context.

        Uses keyword matching for relevance scoring. Preloaded memories
        (from personality files) are boosted in relevance.

        TODO(Phase 3): Add vector similarity search using stored embeddings
        for semantic matching beyond keyword overlap.

        Args:
            context: Current conversation context for relevance.
            limit: Maximum number of memories to return.

        Returns:
            List of relevant memories, scored by relevance.
        """
        # Get a larger pool of memories to search through for relevance scoring
        # Using 100 ensures we don't miss important older memories
        all_memories = self.db.get_all_memories(limit=100)

        if not context:
            return all_memories[:limit]

        # Simple relevance scoring based on keyword overlap
        context_words = set(context.lower().split())
        scored_memories = []

        for memory in all_memories:
            memory_words = set(memory.content.lower().split())
            overlap = len(context_words & memory_words)
            # Boost preloaded memories
            boost = 1.5 if memory.source == "preloaded" else 1.0
            memory.relevance_score = overlap * boost
            scored_memories.append(memory)

        # Sort by relevance score, then by timestamp
        scored_memories.sort(
            key=lambda m: (m.relevance_score, m.timestamp),
            reverse=True,
        )

        return scored_memories[:limit]

    async def search_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by keyword.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching memories.
        """
        return self.db.search_memories(query, limit)

    async def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory.

        Args:
            memory_id: The memory ID.

        Returns:
            True if deleted.
        """
        return self.db.delete_memory(memory_id)

    async def update_memory(self, memory_id: int, content: str) -> bool:
        """Update a memory.

        Args:
            memory_id: The memory ID.
            content: New content.

        Returns:
            True if updated.
        """
        return self.db.update_memory(memory_id, content)

    def get_all_memories(self, limit: int = 100) -> List[Memory]:
        """Get all memories.

        Args:
            limit: Maximum number to return.

        Returns:
            List of memories.
        """
        return self.db.get_all_memories(limit)

    # ---------- Person Operations ----------

    async def add_person(
        self,
        name: str,
        photo_base64: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a new known person.

        Args:
            name: Person's name.
            photo_base64: Base64 encoded photo.
            metadata: Optional metadata.

        Returns:
            The person's ID.
        """
        return self.db.add_person(name, photo_base64, metadata)

    def get_all_persons(self) -> List[KnownPerson]:
        """Get all known persons.

        Returns:
            List of KnownPerson objects.
        """
        return self.db.get_all_persons()

    def get_person_by_id(self, person_id: int) -> Optional[KnownPerson]:
        """Get a person by ID.

        Args:
            person_id: The person's ID.

        Returns:
            KnownPerson if found.
        """
        return self.db.get_person_by_id(person_id)

    def get_person_by_name(self, name: str) -> Optional[KnownPerson]:
        """Get a person by name.

        Args:
            name: The person's name.

        Returns:
            KnownPerson if found.
        """
        return self.db.get_person_by_name(name)

    async def delete_person(self, person_id: int) -> bool:
        """Delete a person.

        Args:
            person_id: The person's ID.

        Returns:
            True if deleted.
        """
        return self.db.delete_person(person_id)

    async def update_person_photo(self, person_id: int, photo_base64: str) -> bool:
        """Update a person's photo.

        Args:
            person_id: The person's ID.
            photo_base64: New base64 encoded photo.

        Returns:
            True if updated.
        """
        return self.db.update_person_photo(person_id, photo_base64)

    def set_current_person(self, person_id: Optional[int]) -> None:
        """Set the current person for the session.

        Args:
            person_id: The person's ID, or None to clear.
        """
        self.db.set_current_person(person_id)

    def get_current_person(self) -> Optional[KnownPerson]:
        """Get the current person for the session.

        Returns:
            KnownPerson if set.
        """
        return self.db.get_current_person()

    # ---------- Learning Operations ----------

    async def add_learned_fact(
        self,
        fact: str,
        confidence: float,
        context: str = "",
    ) -> int:
        """Add an autonomously learned fact.

        Args:
            fact: The learned fact.
            confidence: Confidence score (0.0-1.0).
            context: Learning context.

        Returns:
            The fact's ID.
        """
        return self.db.add_learned_fact(fact, confidence, context)

    def get_pending_facts(self, limit: int = 50) -> List[LearnedFact]:
        """Get facts pending approval.

        Args:
            limit: Maximum number to return.

        Returns:
            List of pending facts.
        """
        return self.db.get_pending_facts(limit)

    async def approve_fact(self, fact_id: int) -> bool:
        """Approve a learned fact.

        Args:
            fact_id: The fact's ID.

        Returns:
            True if approved.
        """
        return self.db.approve_fact(fact_id)

    async def reject_fact(self, fact_id: int) -> bool:
        """Reject (delete) a learned fact.

        Args:
            fact_id: The fact's ID.

        Returns:
            True if deleted.
        """
        return self.db.delete_fact(fact_id)

    def get_approved_facts(self, limit: int = 100) -> List[LearnedFact]:
        """Get approved facts.

        Args:
            limit: Maximum number to return.

        Returns:
            List of approved facts.
        """
        return self.db.get_approved_facts(limit)

    # ---------- Memory Formatting ----------

    def format_memories_for_prompt(
        self,
        memories: List[Memory],
        max_tokens: int = 2000,
    ) -> str:
        """Format memories for injection into a prompt.

        Args:
            memories: List of memories to format.
            max_tokens: Approximate token limit (rough estimate).

        Returns:
            Formatted string for prompt injection.
        """
        if not memories:
            return ""

        lines = []
        char_count = 0
        max_chars = max_tokens * 4  # Rough estimate: 1 token ~= 4 chars

        for memory in memories:
            line = f"- {memory.content}"
            if char_count + len(line) > max_chars:
                break
            lines.append(line)
            char_count += len(line)

        return "\n".join(lines)

    def format_person_context(self, person: Optional[KnownPerson] = None) -> str:
        """Format the current person context for prompt injection.

        Args:
            person: The person to format, or None to use current.

        Returns:
            Formatted string for prompt injection.
        """
        if person is None:
            person = self.get_current_person()

        if person is None:
            return ""

        return f"[The person currently talking to you is {person.name}]"

    def get_full_context_for_prompt(self, conversation_context: str = "") -> str:
        """Get the full memory context for prompt injection.

        Combines:
        - Current person context
        - Relevant memories (scored by context relevance)
        - Approved learned facts

        Args:
            conversation_context: Current conversation for relevance scoring.

        Returns:
            Formatted context string.
        """
        parts = []

        # Person context
        person_ctx = self.format_person_context()
        if person_ctx:
            parts.append(person_ctx)

        # Get relevant memories using context for relevance scoring
        try:
            memories = self.get_relevant_memories(
                context=conversation_context if conversation_context else None,
                limit=15,
            )
            if memories:
                memory_text = self.format_memories_for_prompt(memories)
                if memory_text:
                    parts.append(f"\n## Things you know about Pedro:\n{memory_text}")
        except Exception as e:
            logger.warning(f"Failed to get memories for prompt: {e}")

        # Get approved facts
        try:
            facts = self.db.get_approved_facts(limit=10)
            if facts:
                fact_lines = [f"- {f.fact}" for f in facts]
                parts.append(f"\n## Things you've learned:\n" + "\n".join(fact_lines))
        except Exception as e:
            logger.warning(f"Failed to get facts for prompt: {e}")

        return "\n".join(parts)

    # ---------- Embedding Operations ----------
    # TODO(Phase 3): Implement vector similarity search using these embeddings.
    # Currently embeddings are stored but not used for retrieval.
    # Phase 3 will add cosine similarity search for semantic memory matching.

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for text using OpenAI.

        Embeddings are stored in the database for future semantic search.
        TODO(Phase 3): Implement vector similarity retrieval.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            Exception: If embedding generation fails.
        """
        if not self._openai_client:
            raise RuntimeError("OpenAI client not initialised")

        response = await self._openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )

        return response.data[0].embedding

    # ---------- Stats ----------

    def get_stats(self) -> Dict[str, int]:
        """Get memory system statistics.

        Returns:
            Dictionary with counts.
        """
        return {
            "memories": self.db.get_memory_count(),
            "persons": self.db.get_person_count(),
            "pending_facts": len(self.db.get_pending_facts()),
            "approved_facts": len(self.db.get_approved_facts()),
        }

    def backup(self, backup_path: Optional[str] = None) -> str:
        """Create a database backup.

        Args:
            backup_path: Path for backup. If None, auto-generates.

        Returns:
            Path to the backup file.
        """
        return self.db.backup(backup_path)
