"""Transcription service for storing full conversation transcripts.

Uses OpenAI gpt-4o-transcribe for high-quality transcription with
speaker diarization, storing results in DuckDB for full-text search.
"""

import asyncio
import logging
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_RECORDINGS_PATH = "~/.reachy_mini/recordings"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""

    text: str
    speaker: str = "unknown"
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0


@dataclass
class ConversationTranscript:
    """Full conversation transcript with metadata."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    audio_file_path: Optional[str] = None
    full_text: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationRecorder:
    """Records audio for later transcription."""

    def __init__(
        self,
        recordings_path: Optional[str] = None,
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> None:
        """Initialize the recorder.

        Args:
            recordings_path: Directory to store recordings.
            sample_rate: Audio sample rate.
            channels: Number of audio channels.
        """
        if recordings_path is None:
            recordings_path = DEFAULT_RECORDINGS_PATH

        self._recordings_path = Path(recordings_path).expanduser()
        self._recordings_path.mkdir(parents=True, exist_ok=True)

        self._sample_rate = sample_rate
        self._channels = channels

        self._current_session_id: Optional[str] = None
        self._audio_buffer: List[NDArray[np.int16]] = []
        self._session_start_time: Optional[datetime] = None
        self._is_recording = False

        logger.info(f"ConversationRecorder initialized: {self._recordings_path}")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def current_session_id(self) -> Optional[str]:
        """Get current recording session ID."""
        return self._current_session_id

    def start_session(self) -> str:
        """Start a new recording session.

        Returns:
            Session ID for this recording.
        """
        self._current_session_id = str(uuid.uuid4())[:8]
        self._audio_buffer = []
        self._session_start_time = datetime.now()
        self._is_recording = True

        logger.info(f"Recording session started: {self._current_session_id}")
        return self._current_session_id

    def add_audio(self, audio_data: NDArray[np.int16]) -> None:
        """Add audio data to current recording.

        Args:
            audio_data: Audio samples to record.
        """
        if not self._is_recording:
            return

        self._audio_buffer.append(audio_data.copy())

    def end_session(self) -> Optional[str]:
        """End current recording session and save audio file.

        Returns:
            Path to saved audio file, or None if no recording.
        """
        if not self._is_recording or not self._audio_buffer:
            self._is_recording = False
            return None

        # Generate filename
        timestamp = self._session_start_time.strftime("%Y%m%d-%H%M%S") if self._session_start_time else "unknown"
        filename = f"conversation-{timestamp}-{self._current_session_id}.wav"
        filepath = self._recordings_path / filename

        # Concatenate all audio
        audio_data = np.concatenate(self._audio_buffer)

        # Save as WAV file
        try:
            with wave.open(str(filepath), "wb") as wav_file:
                wav_file.setnchannels(self._channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self._sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            logger.info(f"Recording saved: {filepath} ({len(audio_data)} samples)")

        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            filepath = None

        # Reset state
        session_id = self._current_session_id
        self._current_session_id = None
        self._audio_buffer = []
        self._session_start_time = None
        self._is_recording = False

        return str(filepath) if filepath else None

    def get_session_duration(self) -> float:
        """Get duration of current session in seconds."""
        if not self._audio_buffer:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        return total_samples / self._sample_rate


class TranscriptionService:
    """Service for transcribing audio using OpenAI gpt-4o-transcribe.

    Provides async transcription with speaker diarization and
    storage in DuckDB for full-text search.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        recordings_path: Optional[str] = None,
        memory_database: Optional[Any] = None,
    ) -> None:
        """Initialize the transcription service.

        Args:
            openai_api_key: OpenAI API key. Uses env var if None.
            recordings_path: Directory for recordings.
            memory_database: MemoryDatabase instance for storing transcripts.
        """
        if recordings_path is None:
            recordings_path = DEFAULT_RECORDINGS_PATH

        self._recordings_path = Path(recordings_path).expanduser()
        self._openai_client: Optional[AsyncOpenAI] = None
        self._openai_api_key = openai_api_key
        self._memory_db = memory_database
        self._initialized = False

        # Pending transcription jobs
        self._pending_jobs: List[str] = []  # File paths waiting for transcription
        self._transcription_task: Optional[asyncio.Task[None]] = None

    async def initialize(self, openai_api_key: Optional[str] = None) -> bool:
        """Initialize the transcription service.

        Args:
            openai_api_key: OpenAI API key.

        Returns:
            True if initialization succeeded.
        """
        if not OPENAI_AVAILABLE:
            logger.warning("openai package not available - transcription disabled")
            return False

        if openai_api_key:
            self._openai_api_key = openai_api_key

        if self._openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=self._openai_api_key)
            self._initialized = True
            logger.info("TranscriptionService initialized")
            return True

        logger.warning("No OpenAI API key provided - transcription disabled")
        return False

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    async def transcribe_file(self, audio_path: str) -> Optional[ConversationTranscript]:
        """Transcribe an audio file using gpt-4o-transcribe.

        Args:
            audio_path: Path to audio file.

        Returns:
            ConversationTranscript with results, or None on failure.
        """
        if not self._initialized or self._openai_client is None:
            logger.error("TranscriptionService not initialized")
            return None

        path = Path(audio_path)
        if not path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        try:
            # Read the audio file
            with open(path, "rb") as audio_file:
                # Use OpenAI transcription API
                response = await self._openai_client.audio.transcriptions.create(
                    model=TRANSCRIPTION_MODEL,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                )

            # Parse response
            transcript = ConversationTranscript(
                session_id=path.stem.split("-")[-1] if "-" in path.stem else path.stem,
                start_time=datetime.now(),  # TODO: Get from file metadata
                audio_file_path=str(path),
                full_text=response.text,
            )

            # Extract segments if available
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    transcript.segments.append(
                        TranscriptSegment(
                            text=seg.text,
                            start_time=seg.start,
                            end_time=seg.end,
                        )
                    )

            logger.info(f"Transcribed {path.name}: {len(transcript.full_text)} chars")
            return transcript

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return None

    def queue_transcription(self, audio_path: str) -> None:
        """Queue an audio file for background transcription.

        Args:
            audio_path: Path to audio file.
        """
        self._pending_jobs.append(audio_path)
        logger.debug(f"Queued for transcription: {audio_path}")

        # Start background worker if not running
        if self._transcription_task is None or self._transcription_task.done():
            self._transcription_task = asyncio.create_task(self._process_pending_jobs())

    async def _process_pending_jobs(self) -> None:
        """Background task to process pending transcription jobs."""
        while self._pending_jobs:
            audio_path = self._pending_jobs.pop(0)
            try:
                transcript = await self.transcribe_file(audio_path)
                if transcript:
                    # Store in DuckDB
                    if self._memory_db:
                        try:
                            # Calculate duration from audio file
                            duration_seconds = 0.0
                            try:
                                with wave.open(audio_path, "rb") as wav:
                                    frames = wav.getnframes()
                                    rate = wav.getframerate()
                                    duration_seconds = frames / rate
                            except Exception:
                                pass

                            transcript_id = self._memory_db.add_transcript(
                                session_id=transcript.session_id,
                                transcript=transcript.full_text,
                                started_at=transcript.start_time,
                                ended_at=transcript.end_time or datetime.now(),
                                audio_path=transcript.audio_file_path,
                                duration_seconds=duration_seconds,
                                metadata=transcript.metadata,
                            )
                            logger.info(f"Transcript stored in database (ID: {transcript_id}): {audio_path}")
                        except Exception as e:
                            logger.error(f"Failed to store transcript in database: {e}")
                    else:
                        logger.warning("No database configured - transcript not stored")

                    logger.info(f"Transcription complete: {audio_path}")
            except Exception as e:
                logger.error(f"Error processing transcription job: {e}")

            # Small delay between jobs
            await asyncio.sleep(1.0)

    async def search_transcripts(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ConversationTranscript]:
        """Search transcripts using full-text search.

        Args:
            query: Search query.
            limit: Maximum results to return.

        Returns:
            List of matching transcripts.
        """
        # TODO: Implement DuckDB full-text search
        logger.warning("Transcript search not yet implemented")
        return []

    async def shutdown(self) -> None:
        """Shutdown the transcription service."""
        if self._transcription_task:
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass

        logger.info("TranscriptionService shut down")
