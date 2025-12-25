"""DuckDB database initialization and schema management for the memory system."""

import os
import json
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import duckdb


logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Represents a stored memory."""

    id: int
    content: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0


@dataclass
class KnownPerson:
    """Represents a known person for face recognition."""

    id: int
    name: str
    photo_base64: str
    face_embedding: Optional[List[float]] = None
    last_seen: Optional[datetime] = None
    recognition_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnedFact:
    """Represents an autonomously learned fact."""

    id: int
    fact: str
    confidence: float
    context: str
    timestamp: datetime
    approved: bool = False


@dataclass
class ConversationTranscript:
    """Represents a stored conversation transcript."""

    id: int
    session_id: str
    person_id: Optional[int]
    person_name: Optional[str]
    transcript: str
    audio_path: Optional[str]
    duration_seconds: float
    started_at: datetime
    ended_at: datetime
    speaker_diarization: Optional[List[Dict[str, Any]]] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WakeEvent:
    """Represents a wake word detection event."""

    id: int
    timestamp: datetime
    confidence: float
    model_name: Optional[str] = None
    triggered_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryDatabase:
    """DuckDB database manager for the memory system.

    Handles database initialization, schema management, and low-level operations.
    Thread-safe with connection per thread pattern.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialise the memory database.

        Args:
            db_path: Path to the DuckDB file. If None, uses default location.
        """
        if db_path is None:
            # Default to ~/.reachy_mini/memories.duckdb
            db_dir = Path.home() / ".reachy_mini"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "memories.duckdb")

        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialise schema
        self._init_schema()
        logger.info(f"Memory database initialised at {self.db_path}")

    @property
    def _conn(self) -> duckdb.DuckDBPyConnection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = duckdb.connect(self.db_path)
        return self._local.conn

    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._conn

        # User memories table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(50) DEFAULT 'manual',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                embedding_json TEXT
            )
        """)

        # Known persons table for face recognition
        conn.execute("""
            CREATE TABLE IF NOT EXISTS known_persons (
                id INTEGER PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                photo_base64 TEXT NOT NULL,
                face_embedding_json TEXT,
                last_seen TIMESTAMP,
                recognition_count INTEGER DEFAULT 0,
                metadata JSON
            )
        """)

        # Autonomously learned facts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_facts (
                id INTEGER PRIMARY KEY,
                fact TEXT NOT NULL,
                confidence FLOAT DEFAULT 0.0,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved BOOLEAN DEFAULT FALSE
            )
        """)

        # Face recognition cache
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recognition_cache (
                frame_hash VARCHAR(64) PRIMARY KEY,
                person_id INTEGER,
                confidence FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Current user session tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS current_session (
                id INTEGER PRIMARY KEY DEFAULT 1,
                current_person_id INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Conversation transcripts table for full conversation storage
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_transcripts (
                id INTEGER PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL UNIQUE,
                person_id INTEGER,
                person_name VARCHAR(255),
                transcript TEXT NOT NULL,
                audio_path TEXT,
                duration_seconds FLOAT DEFAULT 0.0,
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP NOT NULL,
                speaker_diarization JSON,
                word_timestamps JSON,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Wake word detection events for analytics and debugging
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wake_events (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence FLOAT NOT NULL,
                model_name VARCHAR(100),
                triggered_session_id VARCHAR(64),
                metadata JSON
            )
        """)

        # Schema version tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)

        # Insert schema version if not exists
        result = conn.execute("SELECT version FROM schema_version").fetchone()
        if result is None:
            conn.execute(f"INSERT INTO schema_version VALUES ({self.SCHEMA_VERSION})")

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON user_memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_source ON user_memories(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_persons_name ON known_persons(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_approved ON learned_facts(approved)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON recognition_cache(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_session ON conversation_transcripts(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_person ON conversation_transcripts(person_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_started ON conversation_transcripts(started_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_ended ON conversation_transcripts(ended_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_wake_events_timestamp ON wake_events(timestamp)")

        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def backup(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the database.

        Args:
            backup_path: Path for the backup file. If None, creates next to original.

        Returns:
            Path to the backup file.
        """
        if backup_path is None:
            backup_path = f"{self.db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self._lock:
            shutil.copy2(self.db_path, backup_path)

        logger.info(f"Database backed up to {backup_path}")
        return backup_path

    # ---------- Memory Operations ----------

    def add_memory(
        self,
        content: str,
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """Add a new memory to the database.

        Args:
            content: The memory content.
            source: Source of the memory (e.g., 'voice_command', 'preloaded', 'manual').
            metadata: Optional metadata dictionary.
            embedding: Optional embedding vector for semantic search.

        Returns:
            The ID of the inserted memory.
        """
        with self._lock:
            embedding_json = json.dumps(embedding) if embedding else None
            metadata_json = json.dumps(metadata) if metadata else None

            result = self._conn.execute(
                """
                INSERT INTO user_memories (content, source, metadata, embedding_json)
                VALUES (?, ?, ?, ?)
                RETURNING id
                """,
                [content, source, metadata_json, embedding_json],
            ).fetchone()

            self._conn.commit()
            return result[0] if result else -1

    def get_all_memories(self, limit: int = 100, offset: int = 0) -> List[Memory]:
        """Get all memories with pagination.

        Args:
            limit: Maximum number of memories to return.
            offset: Number of memories to skip.

        Returns:
            List of Memory objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, content, source, timestamp, metadata
            FROM user_memories
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            [limit, offset],
        ).fetchall()

        memories = []
        for row in rows:
            metadata = json.loads(row[4]) if row[4] else {}
            memories.append(
                Memory(
                    id=row[0],
                    content=row[1],
                    source=row[2],
                    timestamp=row[3],
                    metadata=metadata,
                )
            )
        return memories

    def search_memories(self, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by keyword.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching Memory objects.
        """
        # Simple case-insensitive substring search
        rows = self._conn.execute(
            """
            SELECT id, content, source, timestamp, metadata
            FROM user_memories
            WHERE LOWER(content) LIKE LOWER(?)
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [f"%{query}%", limit],
        ).fetchall()

        memories = []
        for row in rows:
            metadata = json.loads(row[4]) if row[4] else {}
            memories.append(
                Memory(
                    id=row[0],
                    content=row[1],
                    source=row[2],
                    timestamp=row[3],
                    metadata=metadata,
                )
            )
        return memories

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            result = self._conn.execute(
                "DELETE FROM user_memories WHERE id = ? RETURNING id",
                [memory_id],
            ).fetchone()
            self._conn.commit()
            return result is not None

    def update_memory(self, memory_id: int, content: str) -> bool:
        """Update a memory's content.

        Args:
            memory_id: The ID of the memory to update.
            content: New content.

        Returns:
            True if updated, False if not found.
        """
        with self._lock:
            result = self._conn.execute(
                "UPDATE user_memories SET content = ? WHERE id = ? RETURNING id",
                [content, memory_id],
            ).fetchone()
            self._conn.commit()
            return result is not None

    def get_memories_by_source(self, source: str, limit: int = 100) -> List[Memory]:
        """Get all memories from a specific source.

        Args:
            source: The source to filter by.
            limit: Maximum number of memories to return.

        Returns:
            List of Memory objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, content, source, timestamp, metadata
            FROM user_memories
            WHERE source = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [source, limit],
        ).fetchall()

        memories = []
        for row in rows:
            metadata = json.loads(row[4]) if row[4] else {}
            memories.append(
                Memory(
                    id=row[0],
                    content=row[1],
                    source=row[2],
                    timestamp=row[3],
                    metadata=metadata,
                )
            )
        return memories

    # ---------- Known Persons Operations ----------

    def add_person(
        self,
        name: str,
        photo_base64: str,
        metadata: Optional[Dict[str, Any]] = None,
        face_embedding: Optional[List[float]] = None,
    ) -> int:
        """Add a new known person.

        Args:
            name: Person's name.
            photo_base64: Base64 encoded photo.
            metadata: Optional metadata.
            face_embedding: Optional face embedding vector.

        Returns:
            The ID of the inserted person.
        """
        with self._lock:
            metadata_json = json.dumps(metadata) if metadata else None
            embedding_json = json.dumps(face_embedding) if face_embedding else None

            result = self._conn.execute(
                """
                INSERT INTO known_persons (name, photo_base64, metadata, face_embedding_json, last_seen)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                [name, photo_base64, metadata_json, embedding_json],
            ).fetchone()

            self._conn.commit()
            return result[0] if result else -1

    def get_all_persons(self) -> List[KnownPerson]:
        """Get all known persons.

        Returns:
            List of KnownPerson objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, name, photo_base64, face_embedding_json, last_seen, recognition_count, metadata
            FROM known_persons
            ORDER BY name
            """
        ).fetchall()

        persons = []
        for row in rows:
            metadata = json.loads(row[6]) if row[6] else {}
            face_embedding = json.loads(row[3]) if row[3] else None
            persons.append(
                KnownPerson(
                    id=row[0],
                    name=row[1],
                    photo_base64=row[2],
                    face_embedding=face_embedding,
                    last_seen=row[4],
                    recognition_count=row[5],
                    metadata=metadata,
                )
            )
        return persons

    def get_person_by_id(self, person_id: int) -> Optional[KnownPerson]:
        """Get a person by ID.

        Args:
            person_id: The person's ID.

        Returns:
            KnownPerson if found, None otherwise.
        """
        row = self._conn.execute(
            """
            SELECT id, name, photo_base64, face_embedding_json, last_seen, recognition_count, metadata
            FROM known_persons
            WHERE id = ?
            """,
            [person_id],
        ).fetchone()

        if row is None:
            return None

        metadata = json.loads(row[6]) if row[6] else {}
        face_embedding = json.loads(row[3]) if row[3] else None
        return KnownPerson(
            id=row[0],
            name=row[1],
            photo_base64=row[2],
            face_embedding=face_embedding,
            last_seen=row[4],
            recognition_count=row[5],
            metadata=metadata,
        )

    def get_person_by_name(self, name: str) -> Optional[KnownPerson]:
        """Get a person by name.

        Args:
            name: The person's name.

        Returns:
            KnownPerson if found, None otherwise.
        """
        row = self._conn.execute(
            """
            SELECT id, name, photo_base64, face_embedding_json, last_seen, recognition_count, metadata
            FROM known_persons
            WHERE LOWER(name) = LOWER(?)
            """,
            [name],
        ).fetchone()

        if row is None:
            return None

        metadata = json.loads(row[6]) if row[6] else {}
        face_embedding = json.loads(row[3]) if row[3] else None
        return KnownPerson(
            id=row[0],
            name=row[1],
            photo_base64=row[2],
            face_embedding=face_embedding,
            last_seen=row[4],
            recognition_count=row[5],
            metadata=metadata,
        )

    def update_person_last_seen(self, person_id: int) -> None:
        """Update a person's last seen timestamp and increment recognition count.

        Args:
            person_id: The person's ID.
        """
        with self._lock:
            self._conn.execute(
                """
                UPDATE known_persons
                SET last_seen = CURRENT_TIMESTAMP, recognition_count = recognition_count + 1
                WHERE id = ?
                """,
                [person_id],
            )
            self._conn.commit()

    def delete_person(self, person_id: int) -> bool:
        """Delete a person by ID.

        Args:
            person_id: The person's ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            result = self._conn.execute(
                "DELETE FROM known_persons WHERE id = ? RETURNING id",
                [person_id],
            ).fetchone()
            self._conn.commit()
            return result is not None

    def update_person_photo(self, person_id: int, photo_base64: str) -> bool:
        """Update a person's photo.

        Args:
            person_id: The person's ID.
            photo_base64: New base64 encoded photo.

        Returns:
            True if updated, False if not found.
        """
        with self._lock:
            result = self._conn.execute(
                "UPDATE known_persons SET photo_base64 = ? WHERE id = ? RETURNING id",
                [photo_base64, person_id],
            ).fetchone()
            self._conn.commit()
            return result is not None

    def update_person_embedding(self, person_id: int, face_embedding: List[float]) -> bool:
        """Update a person's face embedding.

        Args:
            person_id: The person's ID.
            face_embedding: New face embedding vector.

        Returns:
            True if updated, False if not found.
        """
        with self._lock:
            embedding_json = json.dumps(face_embedding)
            result = self._conn.execute(
                "UPDATE known_persons SET face_embedding_json = ? WHERE id = ? RETURNING id",
                [embedding_json, person_id],
            ).fetchone()
            self._conn.commit()
            return result is not None

    # ---------- Learned Facts Operations ----------

    def add_learned_fact(
        self,
        fact: str,
        confidence: float,
        context: str = "",
    ) -> int:
        """Add a new learned fact.

        Args:
            fact: The learned fact.
            confidence: Confidence score (0.0-1.0).
            context: Context in which the fact was learned.

        Returns:
            The ID of the inserted fact.
        """
        with self._lock:
            result = self._conn.execute(
                """
                INSERT INTO learned_facts (fact, confidence, context)
                VALUES (?, ?, ?)
                RETURNING id
                """,
                [fact, confidence, context],
            ).fetchone()

            self._conn.commit()
            return result[0] if result else -1

    def get_pending_facts(self, limit: int = 50) -> List[LearnedFact]:
        """Get facts pending approval.

        Args:
            limit: Maximum number of facts to return.

        Returns:
            List of LearnedFact objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, fact, confidence, context, timestamp, approved
            FROM learned_facts
            WHERE approved = FALSE
            ORDER BY confidence DESC, timestamp DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        return [
            LearnedFact(
                id=row[0],
                fact=row[1],
                confidence=row[2],
                context=row[3],
                timestamp=row[4],
                approved=row[5],
            )
            for row in rows
        ]

    def approve_fact(self, fact_id: int) -> bool:
        """Approve a learned fact.

        Args:
            fact_id: The fact's ID.

        Returns:
            True if approved, False if not found.
        """
        with self._lock:
            result = self._conn.execute(
                "UPDATE learned_facts SET approved = TRUE WHERE id = ? RETURNING id",
                [fact_id],
            ).fetchone()
            self._conn.commit()
            return result is not None

    def delete_fact(self, fact_id: int) -> bool:
        """Delete a learned fact.

        Args:
            fact_id: The fact's ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            result = self._conn.execute(
                "DELETE FROM learned_facts WHERE id = ? RETURNING id",
                [fact_id],
            ).fetchone()
            self._conn.commit()
            return result is not None

    def get_approved_facts(self, limit: int = 100) -> List[LearnedFact]:
        """Get approved facts.

        Args:
            limit: Maximum number of facts to return.

        Returns:
            List of LearnedFact objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, fact, confidence, context, timestamp, approved
            FROM learned_facts
            WHERE approved = TRUE
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        return [
            LearnedFact(
                id=row[0],
                fact=row[1],
                confidence=row[2],
                context=row[3],
                timestamp=row[4],
                approved=row[5],
            )
            for row in rows
        ]

    # ---------- Current Session Operations ----------

    def set_current_person(self, person_id: Optional[int]) -> None:
        """Set the current person for the session.

        Args:
            person_id: The person's ID, or None to clear.
        """
        with self._lock:
            # Upsert the current session
            self._conn.execute(
                """
                INSERT INTO current_session (id, current_person_id, last_updated)
                VALUES (1, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE SET
                    current_person_id = EXCLUDED.current_person_id,
                    last_updated = CURRENT_TIMESTAMP
                """,
                [person_id],
            )
            self._conn.commit()

    def get_current_person(self) -> Optional[KnownPerson]:
        """Get the current person for the session.

        Returns:
            KnownPerson if set, None otherwise.
        """
        row = self._conn.execute(
            "SELECT current_person_id FROM current_session WHERE id = 1"
        ).fetchone()

        if row is None or row[0] is None:
            return None

        return self.get_person_by_id(row[0])

    # ---------- Recognition Cache Operations ----------
    # TODO(Phase 3): These cache methods will be used when Vision API face
    # recognition is implemented. They cache recognition results to avoid
    # repeated API calls for the same person in a short time window.

    def cache_recognition(
        self,
        frame_hash: str,
        person_id: Optional[int],
        confidence: float,
    ) -> None:
        """Cache a face recognition result.

        Args:
            frame_hash: Hash of the frame.
            person_id: Recognised person ID, or None if unknown.
            confidence: Confidence score.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO recognition_cache (frame_hash, person_id, confidence)
                VALUES (?, ?, ?)
                ON CONFLICT (frame_hash) DO UPDATE SET
                    person_id = EXCLUDED.person_id,
                    confidence = EXCLUDED.confidence,
                    timestamp = CURRENT_TIMESTAMP
                """,
                [frame_hash, person_id, confidence],
            )
            self._conn.commit()

    def get_cached_recognition(self, frame_hash: str) -> Optional[tuple]:
        """Get cached recognition result.

        Args:
            frame_hash: Hash of the frame.

        Returns:
            Tuple of (person_id, confidence) if cached, None otherwise.
        """
        row = self._conn.execute(
            """
            SELECT person_id, confidence
            FROM recognition_cache
            WHERE frame_hash = ?
            AND timestamp > CURRENT_TIMESTAMP - INTERVAL 5 MINUTES
            """,
            [frame_hash],
        ).fetchone()

        return row if row else None

    def clear_old_cache(self, minutes: int = 10) -> int:
        """Clear old cache entries.

        Args:
            minutes: Age threshold in minutes.

        Returns:
            Number of entries deleted.
        """
        with self._lock:
            result = self._conn.execute(
                f"""
                DELETE FROM recognition_cache
                WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL {minutes} MINUTES
                """
            )
            self._conn.commit()
            return result.rowcount if hasattr(result, "rowcount") else 0

    def get_memory_count(self) -> int:
        """Get total number of memories.

        Returns:
            Count of memories.
        """
        result = self._conn.execute("SELECT COUNT(*) FROM user_memories").fetchone()
        return result[0] if result else 0

    def get_person_count(self) -> int:
        """Get total number of known persons.

        Returns:
            Count of persons.
        """
        result = self._conn.execute("SELECT COUNT(*) FROM known_persons").fetchone()
        return result[0] if result else 0

    # ---------- Conversation Transcript Operations ----------

    def add_transcript(
        self,
        session_id: str,
        transcript: str,
        started_at: datetime,
        ended_at: datetime,
        person_id: Optional[int] = None,
        person_name: Optional[str] = None,
        audio_path: Optional[str] = None,
        duration_seconds: float = 0.0,
        speaker_diarization: Optional[List[Dict[str, Any]]] = None,
        word_timestamps: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a new conversation transcript.

        Args:
            session_id: Unique session identifier.
            transcript: Full transcript text.
            started_at: When the conversation started.
            ended_at: When the conversation ended.
            person_id: Optional ID of the person who was speaking.
            person_name: Optional name of the person.
            audio_path: Optional path to the audio file.
            duration_seconds: Duration of the conversation.
            speaker_diarization: Speaker diarization data (speaker labels with timestamps).
            word_timestamps: Word-level timestamps from transcription.
            metadata: Optional metadata dictionary.

        Returns:
            The ID of the inserted transcript.
        """
        with self._lock:
            metadata_json = json.dumps(metadata) if metadata else None
            diarization_json = json.dumps(speaker_diarization) if speaker_diarization else None
            timestamps_json = json.dumps(word_timestamps) if word_timestamps else None

            result = self._conn.execute(
                """
                INSERT INTO conversation_transcripts
                (session_id, person_id, person_name, transcript, audio_path,
                 duration_seconds, started_at, ended_at, speaker_diarization,
                 word_timestamps, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                [session_id, person_id, person_name, transcript, audio_path,
                 duration_seconds, started_at, ended_at, diarization_json,
                 timestamps_json, metadata_json],
            ).fetchone()

            self._conn.commit()
            return result[0] if result else -1

    def get_transcript_by_session(self, session_id: str) -> Optional[ConversationTranscript]:
        """Get a transcript by session ID.

        Args:
            session_id: The session identifier.

        Returns:
            ConversationTranscript if found, None otherwise.
        """
        row = self._conn.execute(
            """
            SELECT id, session_id, person_id, person_name, transcript, audio_path,
                   duration_seconds, started_at, ended_at, speaker_diarization,
                   word_timestamps, metadata
            FROM conversation_transcripts
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()

        if row is None:
            return None

        speaker_diarization = json.loads(row[9]) if row[9] else None
        word_timestamps = json.loads(row[10]) if row[10] else None
        metadata = json.loads(row[11]) if row[11] else {}
        return ConversationTranscript(
            id=row[0],
            session_id=row[1],
            person_id=row[2],
            person_name=row[3],
            transcript=row[4],
            audio_path=row[5],
            duration_seconds=row[6],
            started_at=row[7],
            ended_at=row[8],
            speaker_diarization=speaker_diarization,
            word_timestamps=word_timestamps,
            metadata=metadata,
        )

    def get_recent_transcripts(self, limit: int = 20) -> List[ConversationTranscript]:
        """Get recent conversation transcripts.

        Args:
            limit: Maximum number of transcripts to return.

        Returns:
            List of ConversationTranscript objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, session_id, person_id, person_name, transcript, audio_path,
                   duration_seconds, started_at, ended_at, speaker_diarization,
                   word_timestamps, metadata
            FROM conversation_transcripts
            ORDER BY ended_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        transcripts = []
        for row in rows:
            speaker_diarization = json.loads(row[9]) if row[9] else None
            word_timestamps = json.loads(row[10]) if row[10] else None
            metadata = json.loads(row[11]) if row[11] else {}
            transcripts.append(
                ConversationTranscript(
                    id=row[0],
                    session_id=row[1],
                    person_id=row[2],
                    person_name=row[3],
                    transcript=row[4],
                    audio_path=row[5],
                    duration_seconds=row[6],
                    started_at=row[7],
                    ended_at=row[8],
                    speaker_diarization=speaker_diarization,
                    word_timestamps=word_timestamps,
                    metadata=metadata,
                )
            )
        return transcripts

    def search_transcripts(self, query: str, limit: int = 20) -> List[ConversationTranscript]:
        """Search transcripts by keyword.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching ConversationTranscript objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, session_id, person_id, person_name, transcript, audio_path,
                   duration_seconds, started_at, ended_at, speaker_diarization,
                   word_timestamps, metadata
            FROM conversation_transcripts
            WHERE LOWER(transcript) LIKE LOWER(?)
            ORDER BY ended_at DESC
            LIMIT ?
            """,
            [f"%{query}%", limit],
        ).fetchall()

        transcripts = []
        for row in rows:
            speaker_diarization = json.loads(row[9]) if row[9] else None
            word_timestamps = json.loads(row[10]) if row[10] else None
            metadata = json.loads(row[11]) if row[11] else {}
            transcripts.append(
                ConversationTranscript(
                    id=row[0],
                    session_id=row[1],
                    person_id=row[2],
                    person_name=row[3],
                    transcript=row[4],
                    audio_path=row[5],
                    duration_seconds=row[6],
                    started_at=row[7],
                    ended_at=row[8],
                    speaker_diarization=speaker_diarization,
                    word_timestamps=word_timestamps,
                    metadata=metadata,
                )
            )
        return transcripts

    def get_transcripts_by_person(self, person_id: int, limit: int = 50) -> List[ConversationTranscript]:
        """Get transcripts for a specific person.

        Args:
            person_id: The person's ID.
            limit: Maximum number of results.

        Returns:
            List of ConversationTranscript objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, session_id, person_id, person_name, transcript, audio_path,
                   duration_seconds, started_at, ended_at, speaker_diarization,
                   word_timestamps, metadata
            FROM conversation_transcripts
            WHERE person_id = ?
            ORDER BY ended_at DESC
            LIMIT ?
            """,
            [person_id, limit],
        ).fetchall()

        transcripts = []
        for row in rows:
            speaker_diarization = json.loads(row[9]) if row[9] else None
            word_timestamps = json.loads(row[10]) if row[10] else None
            metadata = json.loads(row[11]) if row[11] else {}
            transcripts.append(
                ConversationTranscript(
                    id=row[0],
                    session_id=row[1],
                    person_id=row[2],
                    person_name=row[3],
                    transcript=row[4],
                    audio_path=row[5],
                    duration_seconds=row[6],
                    started_at=row[7],
                    ended_at=row[8],
                    speaker_diarization=speaker_diarization,
                    word_timestamps=word_timestamps,
                    metadata=metadata,
                )
            )
        return transcripts

    def delete_transcript(self, transcript_id: int) -> bool:
        """Delete a transcript by ID.

        Args:
            transcript_id: The transcript's ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            result = self._conn.execute(
                "DELETE FROM conversation_transcripts WHERE id = ? RETURNING id",
                [transcript_id],
            ).fetchone()
            self._conn.commit()
            return result is not None

    def get_transcript_count(self) -> int:
        """Get total number of transcripts.

        Returns:
            Count of transcripts.
        """
        result = self._conn.execute("SELECT COUNT(*) FROM conversation_transcripts").fetchone()
        return result[0] if result else 0

    def get_total_conversation_duration(self) -> float:
        """Get total duration of all conversations in seconds.

        Returns:
            Total duration in seconds.
        """
        result = self._conn.execute(
            "SELECT COALESCE(SUM(duration_seconds), 0) FROM conversation_transcripts"
        ).fetchone()
        return result[0] if result else 0.0

    def get_transcript_by_id(self, transcript_id: int) -> Optional[ConversationTranscript]:
        """Get a transcript by ID.

        Args:
            transcript_id: The transcript ID.

        Returns:
            ConversationTranscript if found, None otherwise.
        """
        row = self._conn.execute(
            """
            SELECT id, session_id, person_id, person_name, transcript, audio_path,
                   duration_seconds, started_at, ended_at, speaker_diarization,
                   word_timestamps, metadata
            FROM conversation_transcripts
            WHERE id = ?
            """,
            [transcript_id],
        ).fetchone()

        if row is None:
            return None

        speaker_diarization = json.loads(row[9]) if row[9] else None
        word_timestamps = json.loads(row[10]) if row[10] else None
        metadata = json.loads(row[11]) if row[11] else {}
        return ConversationTranscript(
            id=row[0],
            session_id=row[1],
            person_id=row[2],
            person_name=row[3],
            transcript=row[4],
            audio_path=row[5],
            duration_seconds=row[6],
            started_at=row[7],
            ended_at=row[8],
            speaker_diarization=speaker_diarization,
            word_timestamps=word_timestamps,
            metadata=metadata,
        )

    def cleanup_old_transcripts(self, days: int = 90) -> tuple[int, List[str]]:
        """Delete transcripts older than specified days.

        Args:
            days: Delete transcripts older than this many days.

        Returns:
            Tuple of (count deleted, list of audio file paths that can be deleted).
        """
        # First get the audio paths for files that can be deleted
        rows = self._conn.execute(
            """
            SELECT audio_path FROM conversation_transcripts
            WHERE ended_at < CURRENT_TIMESTAMP - INTERVAL ? DAY
            AND audio_path IS NOT NULL
            """,
            [days],
        ).fetchall()

        audio_paths = [row[0] for row in rows if row[0]]

        with self._lock:
            result = self._conn.execute(
                """
                DELETE FROM conversation_transcripts
                WHERE ended_at < CURRENT_TIMESTAMP - INTERVAL ? DAY
                """,
                [days],
            )
            self._conn.commit()
            count = result.rowcount if hasattr(result, 'rowcount') else 0

        return count, audio_paths

    def get_transcripts_in_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
    ) -> List[ConversationTranscript]:
        """Get transcripts within a date range.

        Args:
            start_date: Start of the date range.
            end_date: End of the date range.
            limit: Maximum number of results.

        Returns:
            List of ConversationTranscript objects.
        """
        rows = self._conn.execute(
            """
            SELECT id, session_id, person_id, person_name, transcript, audio_path,
                   duration_seconds, started_at, ended_at, speaker_diarization,
                   word_timestamps, metadata
            FROM conversation_transcripts
            WHERE started_at >= ? AND ended_at <= ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [start_date, end_date, limit],
        ).fetchall()

        transcripts = []
        for row in rows:
            speaker_diarization = json.loads(row[9]) if row[9] else None
            word_timestamps = json.loads(row[10]) if row[10] else None
            metadata = json.loads(row[11]) if row[11] else {}
            transcripts.append(
                ConversationTranscript(
                    id=row[0],
                    session_id=row[1],
                    person_id=row[2],
                    person_name=row[3],
                    transcript=row[4],
                    audio_path=row[5],
                    duration_seconds=row[6],
                    started_at=row[7],
                    ended_at=row[8],
                    speaker_diarization=speaker_diarization,
                    word_timestamps=word_timestamps,
                    metadata=metadata,
                )
            )
        return transcripts

    # ---------- Wake Event Operations ----------

    def add_wake_event(
        self,
        confidence: float,
        model_name: Optional[str] = None,
        triggered_session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Log a wake word detection event.

        Args:
            confidence: Detection confidence score (0.0-1.0).
            model_name: Name of the wake word model that triggered.
            triggered_session_id: Session ID started by this wake event.
            metadata: Additional event metadata.

        Returns:
            ID of the created wake event.
        """
        with self._lock:
            metadata_json = json.dumps(metadata) if metadata else None
            result = self._conn.execute(
                """
                INSERT INTO wake_events (confidence, model_name, triggered_session_id, metadata)
                VALUES (?, ?, ?, ?)
                RETURNING id
                """,
                [confidence, model_name, triggered_session_id, metadata_json],
            ).fetchone()
            self._conn.commit()
            return result[0] if result else 0

    def get_recent_wake_events(self, limit: int = 50) -> List[WakeEvent]:
        """Get recent wake word detection events.

        Args:
            limit: Maximum events to return.

        Returns:
            List of wake events, most recent first.
        """
        rows = self._conn.execute(
            """
            SELECT id, timestamp, confidence, model_name, triggered_session_id, metadata
            FROM wake_events
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

        events = []
        for row in rows:
            metadata = {}
            if row[5]:
                try:
                    metadata = json.loads(row[5])
                except json.JSONDecodeError:
                    pass
            events.append(
                WakeEvent(
                    id=row[0],
                    timestamp=row[1],
                    confidence=row[2],
                    model_name=row[3],
                    triggered_session_id=row[4],
                    metadata=metadata,
                )
            )
        return events

    def get_wake_event_count(self, hours: Optional[int] = None) -> int:
        """Get count of wake events.

        Args:
            hours: If specified, only count events in the last N hours.

        Returns:
            Count of wake events.
        """
        if hours:
            result = self._conn.execute(
                """
                SELECT COUNT(*) FROM wake_events
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
                """,
                [hours],
            ).fetchone()
        else:
            result = self._conn.execute("SELECT COUNT(*) FROM wake_events").fetchone()
        return result[0] if result else 0

    def get_wake_event_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get wake event statistics for the specified period.

        Args:
            hours: Number of hours to analyze.

        Returns:
            Dictionary with statistics.
        """
        result = self._conn.execute(
            """
            SELECT
                COUNT(*) as total,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence
            FROM wake_events
            WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL ? HOUR
            """,
            [hours],
        ).fetchone()

        return {
            "period_hours": hours,
            "total_events": result[0] if result else 0,
            "avg_confidence": round(result[1], 3) if result and result[1] else 0.0,
            "min_confidence": round(result[2], 3) if result and result[2] else 0.0,
            "max_confidence": round(result[3], 3) if result and result[3] else 0.0,
        }

    def cleanup_old_wake_events(self, days: int = 30) -> int:
        """Delete wake events older than specified days.

        Args:
            days: Delete events older than this many days.

        Returns:
            Number of events deleted.
        """
        with self._lock:
            result = self._conn.execute(
                """
                DELETE FROM wake_events
                WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL ? DAY
                """,
                [days],
            )
            self._conn.commit()
            return result.rowcount if hasattr(result, 'rowcount') else 0
