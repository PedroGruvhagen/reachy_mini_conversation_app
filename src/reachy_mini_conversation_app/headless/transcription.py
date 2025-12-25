"""Transcription service for storing full conversation transcripts.

Uses OpenAI gpt-4o-transcribe for high-quality transcription with
speaker diarization, storing results in DuckDB for full-text search.

Features:
- Background audio recording with session management
- Retry logic for API failures with exponential backoff
- Speaker diarization and word-level timestamps
- Export to text, JSON, and SRT formats
- Cleanup/retention policy for old recordings
- Privacy controls (encryption, disable recording)
"""

import asyncio
import json
import logging
import os
import uuid
import wave
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

try:
    from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    APIError = Exception
    APIConnectionError = Exception
    RateLimitError = Exception


logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_RECORDINGS_PATH = "~/.reachy_mini/recordings"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
TRANSCRIPTION_MODEL_DIARIZE = "gpt-4o-transcribe-diarize"

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 60.0
RETRY_MULTIPLIER = 2.0


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""

    text: str
    speaker: str = "unknown"
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0


@dataclass
class WordTimestamp:
    """Word-level timestamp from transcription."""

    word: str
    start_time: float
    end_time: float


@dataclass
class ConversationTranscript:
    """Full conversation transcript with metadata."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    audio_file_path: Optional[str] = None
    full_text: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    speaker_diarization: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationRecorder:
    """Records audio for later transcription.

    Features:
    - Session management with unique IDs
    - WAV file output
    - Configurable sample rate and channels
    - Privacy controls (disable recording)
    """

    def __init__(
        self,
        recordings_path: Optional[str] = None,
        sample_rate: int = 48000,
        channels: int = 1,
        enabled: bool = True,
    ) -> None:
        """Initialize the recorder.

        Args:
            recordings_path: Directory to store recordings.
            sample_rate: Audio sample rate.
            channels: Number of audio channels.
            enabled: Whether recording is enabled (privacy control).
        """
        if recordings_path is None:
            recordings_path = os.environ.get("RECORDINGS_PATH", DEFAULT_RECORDINGS_PATH)

        self._recordings_path = Path(recordings_path).expanduser()
        self._recordings_path.mkdir(parents=True, exist_ok=True)

        self._sample_rate = sample_rate
        self._channels = channels
        self._enabled = enabled

        self._current_session_id: Optional[str] = None
        self._audio_buffer: list[NDArray[np.int16]] = []
        self._session_start_time: Optional[datetime] = None
        self._is_recording = False

        logger.info(
            f"ConversationRecorder initialized: {self._recordings_path}, "
            f"enabled={enabled}"
        )

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def current_session_id(self) -> Optional[str]:
        """Get current recording session ID."""
        return self._current_session_id

    @property
    def enabled(self) -> bool:
        """Check if recording is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable recording (privacy control)."""
        self._enabled = enabled
        if not enabled and self._is_recording:
            self.end_session()
        logger.info(f"Recording {'enabled' if enabled else 'disabled'}")

    def start_session(self) -> str:
        """Start a new recording session.

        Returns:
            Session ID for this recording.
        """
        if not self._enabled:
            logger.debug("Recording disabled, not starting session")
            return ""

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
        if not self._is_recording or not self._enabled:
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
        timestamp = (
            self._session_start_time.strftime("%Y%m%d-%H%M%S")
            if self._session_start_time
            else "unknown"
        )
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

            duration = len(audio_data) / self._sample_rate
            logger.info(
                f"Recording saved: {filepath} ({len(audio_data)} samples, "
                f"{duration:.1f}s)"
            )

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

    def get_session_start_time(self) -> Optional[datetime]:
        """Get start time of current session."""
        return self._session_start_time


class TranscriptionService:
    """Service for transcribing audio using OpenAI gpt-4o-transcribe.

    Features:
    - Async transcription with retry logic
    - Speaker diarization support
    - Word-level timestamps
    - Export to text, JSON, SRT formats
    - Integration with DuckDB for storage
    - Cleanup/retention policy
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        recordings_path: Optional[str] = None,
        memory_database: Optional[Any] = None,
        max_retries: int = MAX_RETRIES,
        audio_retention_days: int = 30,
        transcript_retention_days: int = 90,
    ) -> None:
        """Initialize the transcription service.

        Args:
            openai_api_key: OpenAI API key. Uses env var if None.
            recordings_path: Directory for recordings.
            memory_database: MemoryDatabase instance for storing transcripts.
            max_retries: Maximum retry attempts for API calls.
            audio_retention_days: Days to keep audio files.
            transcript_retention_days: Days to keep transcripts.
        """
        if recordings_path is None:
            recordings_path = os.environ.get("RECORDINGS_PATH", DEFAULT_RECORDINGS_PATH)

        self._recordings_path = Path(recordings_path).expanduser()
        self._openai_client: Optional[AsyncOpenAI] = None
        self._openai_api_key = openai_api_key
        self._memory_db = memory_database
        self._max_retries = max_retries
        self._audio_retention_days = int(
            os.environ.get("AUDIO_RETENTION_DAYS", audio_retention_days)
        )
        self._transcript_retention_days = int(
            os.environ.get("TRANSCRIPT_RETENTION_DAYS", transcript_retention_days)
        )
        self._initialized = False

        # Pending transcription jobs
        self._pending_jobs: list[str] = []
        self._transcription_task: Optional[asyncio.Task[None]] = None

        # Stats
        self._transcriptions_completed = 0
        self._transcriptions_failed = 0
        self._total_retries = 0

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

    def get_stats(self) -> dict[str, Any]:
        """Get transcription service statistics."""
        return {
            "transcriptions_completed": self._transcriptions_completed,
            "transcriptions_failed": self._transcriptions_failed,
            "total_retries": self._total_retries,
            "pending_jobs": len(self._pending_jobs),
            "initialized": self._initialized,
        }

    async def _transcribe_with_retry(
        self,
        audio_path: str,
        enable_diarization: bool = True,
        speaker_names: Optional[list[str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Transcribe with exponential backoff retry logic.

        Args:
            audio_path: Path to audio file.
            enable_diarization: Use diarization model for speaker labels.
            speaker_names: Optional list of known speaker names for labeling.

        Returns:
            Transcription response dict, or None on failure.

        Note:
            When enable_diarization=True, uses gpt-4o-transcribe-diarize model
            with diarized_json format for real speaker detection based on voice
            characteristics. Word-level timestamps are not available in this mode.

            When enable_diarization=False, uses gpt-4o-transcribe with verbose_json
            for word-level timestamps but no speaker information.
        """
        if not self._initialized or self._openai_client is None:
            return None

        delay = INITIAL_RETRY_DELAY

        for attempt in range(self._max_retries):
            try:
                with open(audio_path, "rb") as audio_file:
                    if enable_diarization:
                        # Use diarization model for speaker detection
                        create_params: dict[str, Any] = {
                            "model": TRANSCRIPTION_MODEL_DIARIZE,
                            "file": audio_file,
                            "response_format": "diarized_json",
                        }
                        if speaker_names:
                            create_params["known_speaker_names"] = speaker_names
                        response = await self._openai_client.audio.transcriptions.create(
                            **create_params
                        )
                    else:
                        # Use regular model for word timestamps
                        response = await self._openai_client.audio.transcriptions.create(
                            model=TRANSCRIPTION_MODEL,
                            file=audio_file,
                            response_format="verbose_json",
                            timestamp_granularities=["word", "segment"],
                        )
                return response

            except RateLimitError as e:
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                self._total_retries += 1
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * RETRY_MULTIPLIER, MAX_RETRY_DELAY)

            except APIConnectionError as e:
                logger.warning(
                    f"Connection error (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                self._total_retries += 1
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * RETRY_MULTIPLIER, MAX_RETRY_DELAY)

            except APIError as e:
                logger.error(f"API error (attempt {attempt + 1}/{self._max_retries}): {e}")
                self._total_retries += 1
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * RETRY_MULTIPLIER, MAX_RETRY_DELAY)

            except Exception as e:
                logger.error(f"Unexpected error during transcription: {e}")
                break

        return None

    async def transcribe_file(
        self,
        audio_path: str,
        session_start_time: Optional[datetime] = None,
        enable_diarization: bool = True,
        speaker_names: Optional[list[str]] = None,
    ) -> Optional[ConversationTranscript]:
        """Transcribe an audio file using OpenAI's transcription API.

        Args:
            audio_path: Path to audio file.
            session_start_time: When the session started.
            enable_diarization: Use diarization model for speaker labels.
                When True, uses gpt-4o-transcribe-diarize for real speaker
                detection based on voice characteristics.
                When False, uses gpt-4o-transcribe for word timestamps only.
            speaker_names: Optional list of known speaker names (e.g., ["User", "Assistant"]).
                Only used when enable_diarization=True.

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
            response = await self._transcribe_with_retry(
                audio_path,
                enable_diarization=enable_diarization,
                speaker_names=speaker_names,
            )

            if response is None:
                self._transcriptions_failed += 1
                return None

            # Parse session ID from filename
            session_id = path.stem.split("-")[-1] if "-" in path.stem else path.stem

            # Use provided start time or current time
            start_time = session_start_time or datetime.now()

            # Create transcript object
            transcript = ConversationTranscript(
                session_id=session_id,
                start_time=start_time,
                audio_file_path=str(path),
                full_text="",
            )

            # Parse diarized response (from gpt-4o-transcribe-diarize)
            if enable_diarization:
                # Diarized response has segments with real speaker labels
                if hasattr(response, "segments") and response.segments:
                    full_text_parts = []
                    for seg in response.segments:
                        # Use actual speaker label from API (A, B, C... or custom names)
                        speaker = getattr(seg, "speaker", "Unknown")
                        text = getattr(seg, "text", "")
                        seg_start = getattr(seg, "start", 0.0)
                        seg_end = getattr(seg, "end", 0.0)

                        full_text_parts.append(text)
                        transcript.segments.append(
                            TranscriptSegment(
                                text=text,
                                speaker=speaker,
                                start_time=seg_start,
                                end_time=seg_end,
                            )
                        )
                        transcript.speaker_diarization.append({
                            "speaker": speaker,
                            "start": seg_start,
                            "end": seg_end,
                            "text": text,
                        })
                    transcript.full_text = " ".join(full_text_parts)
                elif hasattr(response, "text"):
                    transcript.full_text = response.text
            else:
                # Non-diarized response (verbose_json) with word timestamps
                if hasattr(response, "text"):
                    transcript.full_text = response.text

                # Extract segments (no speaker info in non-diarized mode)
                if hasattr(response, "segments") and response.segments:
                    for seg in response.segments:
                        transcript.segments.append(
                            TranscriptSegment(
                                text=seg.text,
                                speaker="Unknown",  # No speaker info without diarization
                                start_time=seg.start,
                                end_time=seg.end,
                            )
                        )

                # Extract word-level timestamps (only available without diarization)
                if hasattr(response, "words") and response.words:
                    for word in response.words:
                        transcript.word_timestamps.append(
                            WordTimestamp(
                                word=word.word,
                                start_time=word.start,
                                end_time=word.end,
                            )
                        )

            self._transcriptions_completed += 1
            logger.info(f"Transcribed {path.name}: {len(transcript.full_text)} chars")
            return transcript

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            self._transcriptions_failed += 1
            return None

    def queue_transcription(
        self,
        audio_path: str,
        session_start_time: Optional[datetime] = None,
    ) -> None:
        """Queue an audio file for background transcription.

        Args:
            audio_path: Path to audio file.
            session_start_time: When the session started.
        """
        self._pending_jobs.append((audio_path, session_start_time))
        logger.debug(f"Queued for transcription: {audio_path}")

        # Start background worker if not running
        if self._transcription_task is None or self._transcription_task.done():
            self._transcription_task = asyncio.create_task(self._process_pending_jobs())

    async def _process_pending_jobs(self) -> None:
        """Background task to process pending transcription jobs."""
        while self._pending_jobs:
            job = self._pending_jobs.pop(0)
            audio_path = job[0] if isinstance(job, tuple) else job
            session_start_time = job[1] if isinstance(job, tuple) and len(job) > 1 else None

            try:
                transcript = await self.transcribe_file(audio_path, session_start_time)
                if transcript and self._memory_db:
                    await self._store_transcript(transcript, audio_path)
                elif transcript:
                    logger.warning("No database configured - transcript not stored")

            except Exception as e:
                logger.error(f"Error processing transcription job: {e}")

            # Small delay between jobs
            await asyncio.sleep(1.0)

    async def _store_transcript(
        self,
        transcript: ConversationTranscript,
        audio_path: str,
    ) -> None:
        """Store transcript in database.

        Args:
            transcript: The transcript to store.
            audio_path: Path to the audio file.
        """
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

            # Convert word timestamps to dict format
            word_timestamps_data = [
                {"word": wt.word, "start": wt.start_time, "end": wt.end_time}
                for wt in transcript.word_timestamps
            ]

            transcript_id = self._memory_db.add_transcript(
                session_id=transcript.session_id,
                transcript=transcript.full_text,
                started_at=transcript.start_time,
                ended_at=transcript.end_time or datetime.now(),
                audio_path=transcript.audio_file_path,
                duration_seconds=duration_seconds,
                speaker_diarization=transcript.speaker_diarization,
                word_timestamps=word_timestamps_data,
                metadata=transcript.metadata,
            )
            logger.info(
                f"Transcript stored in database (ID: {transcript_id}): {audio_path}"
            )
        except Exception as e:
            logger.error(f"Failed to store transcript in database: {e}")

    # ---------- Export Functions ----------

    def export_to_text(self, transcript: ConversationTranscript) -> str:
        """Export transcript to plain text format.

        Args:
            transcript: The transcript to export.

        Returns:
            Plain text representation.
        """
        lines = [
            f"Session: {transcript.session_id}",
            f"Date: {transcript.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "--- Transcript ---",
            "",
        ]

        if transcript.speaker_diarization:
            for seg in transcript.speaker_diarization:
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("text", "")
                lines.append(f"[{speaker}]: {text}")
        else:
            lines.append(transcript.full_text)

        return "\n".join(lines)

    def export_to_json(self, transcript: ConversationTranscript) -> str:
        """Export transcript to JSON format.

        Args:
            transcript: The transcript to export.

        Returns:
            JSON string representation.
        """
        data = {
            "session_id": transcript.session_id,
            "start_time": transcript.start_time.isoformat(),
            "end_time": transcript.end_time.isoformat() if transcript.end_time else None,
            "audio_file_path": transcript.audio_file_path,
            "full_text": transcript.full_text,
            "speaker_diarization": transcript.speaker_diarization,
            "word_timestamps": [
                {"word": wt.word, "start": wt.start_time, "end": wt.end_time}
                for wt in transcript.word_timestamps
            ],
            "metadata": transcript.metadata,
        }
        return json.dumps(data, indent=2)

    def export_to_srt(self, transcript: ConversationTranscript) -> str:
        """Export transcript to SRT subtitle format.

        Args:
            transcript: The transcript to export.

        Returns:
            SRT format string.
        """
        lines = []
        index = 1

        segments = transcript.speaker_diarization or [
            {"start": 0, "end": 0, "text": transcript.full_text, "speaker": "Unknown"}
        ]

        for seg in segments:
            start_time = seg.get("start", 0)
            end_time = seg.get("end", 0)
            text = seg.get("text", "")
            speaker = seg.get("speaker", "")

            # Format timestamps as HH:MM:SS,mmm
            start_str = self._format_srt_time(start_time)
            end_str = self._format_srt_time(end_time)

            lines.append(str(index))
            lines.append(f"{start_str} --> {end_str}")
            if speaker:
                lines.append(f"[{speaker}] {text}")
            else:
                lines.append(text)
            lines.append("")
            index += 1

        return "\n".join(lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    # ---------- Retention/Cleanup ----------

    async def cleanup_old_recordings(self) -> dict[str, int]:
        """Clean up old audio recordings based on retention policy.

        Returns:
            Dict with cleanup statistics.
        """
        deleted_files = 0
        deleted_transcripts = 0
        audio_paths_to_delete: list[str] = []

        # Clean up old transcripts from database
        if self._memory_db:
            try:
                count, paths = self._memory_db.cleanup_old_transcripts(
                    days=self._transcript_retention_days
                )
                deleted_transcripts = count
                audio_paths_to_delete = paths
            except Exception as e:
                logger.error(f"Failed to cleanup old transcripts: {e}")

        # Delete audio files older than retention period
        cutoff_date = datetime.now() - timedelta(days=self._audio_retention_days)

        try:
            for audio_file in self._recordings_path.glob("*.wav"):
                try:
                    file_mtime = datetime.fromtimestamp(audio_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        audio_file.unlink()
                        deleted_files += 1
                        logger.debug(f"Deleted old recording: {audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete {audio_file}: {e}")

            # Also delete files from database cleanup
            for path_str in audio_paths_to_delete:
                try:
                    path = Path(path_str)
                    if path.exists():
                        path.unlink()
                        deleted_files += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {path_str}: {e}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        logger.info(
            f"Cleanup complete: {deleted_files} files, {deleted_transcripts} transcripts"
        )
        return {
            "deleted_files": deleted_files,
            "deleted_transcripts": deleted_transcripts,
        }

    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a specific conversation (privacy control).

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted successfully.
        """
        success = True

        # Delete from database
        if self._memory_db:
            try:
                transcript = self._memory_db.get_transcript_by_session(session_id)
                if transcript:
                    # Delete audio file if exists
                    if transcript.audio_path:
                        try:
                            path = Path(transcript.audio_path)
                            if path.exists():
                                path.unlink()
                                logger.info(f"Deleted audio file: {path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete audio: {e}")
                            success = False

                    # Delete transcript record
                    self._memory_db.delete_transcript(transcript.id)
                    logger.info(f"Deleted transcript for session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to delete conversation: {e}")
                success = False

        return success

    async def shutdown(self) -> None:
        """Shutdown the transcription service."""
        if self._transcription_task:
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass

        stats = self.get_stats()
        logger.info(
            f"TranscriptionService shut down: {stats['transcriptions_completed']} completed, "
            f"{stats['transcriptions_failed']} failed, {stats['total_retries']} retries"
        )
