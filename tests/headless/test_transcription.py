"""Unit tests for TranscriptionService and ConversationRecorder.

Tests audio recording, transcription with retry logic, export formats,
cleanup/retention policies, and privacy controls.
"""

import os
import json
import wave
import tempfile
import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import numpy as np

from reachy_mini_conversation_app.headless.transcription import (
    ConversationRecorder,
    TranscriptionService,
    ConversationTranscript,
    TranscriptSegment,
    WordTimestamp,
    MAX_RETRIES,
    INITIAL_RETRY_DELAY,
    TRANSCRIPTION_MODEL,
)


class TestConversationRecorder:
    """Test ConversationRecorder functionality."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Verify recorder creates recordings directory."""
        recordings_dir = tmp_path / "recordings"
        recorder = ConversationRecorder(recordings_path=str(recordings_dir))

        assert recordings_dir.exists()
        assert recorder.enabled is True
        assert recorder.is_recording is False

    def test_init_respects_enabled_flag(self, tmp_path: Path) -> None:
        """Verify enabled flag is respected."""
        recorder = ConversationRecorder(
            recordings_path=str(tmp_path),
            enabled=False,
        )

        assert recorder.enabled is False

    def test_start_session_when_enabled(self, tmp_path: Path) -> None:
        """Verify session starts correctly when enabled."""
        recorder = ConversationRecorder(recordings_path=str(tmp_path))

        session_id = recorder.start_session()

        assert session_id != ""
        assert len(session_id) == 8
        assert recorder.is_recording is True
        assert recorder.current_session_id == session_id

    def test_start_session_when_disabled(self, tmp_path: Path) -> None:
        """Verify session does not start when disabled."""
        recorder = ConversationRecorder(
            recordings_path=str(tmp_path),
            enabled=False,
        )

        session_id = recorder.start_session()

        assert session_id == ""
        assert recorder.is_recording is False

    def test_add_audio_when_recording(self, tmp_path: Path) -> None:
        """Verify audio is added to buffer when recording."""
        recorder = ConversationRecorder(recordings_path=str(tmp_path))
        recorder.start_session()

        audio_data = np.ones(1024, dtype=np.int16) * 1000
        recorder.add_audio(audio_data)

        assert recorder.get_session_duration() > 0

    def test_add_audio_when_not_recording(self, tmp_path: Path) -> None:
        """Verify audio is ignored when not recording."""
        recorder = ConversationRecorder(recordings_path=str(tmp_path))

        audio_data = np.ones(1024, dtype=np.int16)
        recorder.add_audio(audio_data)

        assert recorder.get_session_duration() == 0

    def test_end_session_saves_file(self, tmp_path: Path) -> None:
        """Verify end_session saves WAV file."""
        recorder = ConversationRecorder(
            recordings_path=str(tmp_path),
            sample_rate=16000,
        )
        recorder.start_session()

        # Add some audio
        audio_data = np.ones(16000, dtype=np.int16) * 1000  # 1 second
        recorder.add_audio(audio_data)

        filepath = recorder.end_session()

        assert filepath is not None
        assert Path(filepath).exists()
        assert filepath.endswith(".wav")

        # Verify WAV content
        with wave.open(filepath, "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16000
            assert wav.getnframes() == 16000

    def test_end_session_no_audio(self, tmp_path: Path) -> None:
        """Verify end_session returns None when no audio recorded."""
        recorder = ConversationRecorder(recordings_path=str(tmp_path))
        recorder.start_session()

        filepath = recorder.end_session()

        assert filepath is None
        assert recorder.is_recording is False

    def test_set_enabled_false_ends_session(self, tmp_path: Path) -> None:
        """Verify disabling recorder ends active session."""
        recorder = ConversationRecorder(recordings_path=str(tmp_path))
        recorder.start_session()
        recorder.add_audio(np.ones(1024, dtype=np.int16))

        assert recorder.is_recording is True

        recorder.set_enabled(False)

        assert recorder.is_recording is False
        assert recorder.enabled is False

    def test_session_duration_calculation(self, tmp_path: Path) -> None:
        """Verify session duration is calculated correctly."""
        recorder = ConversationRecorder(
            recordings_path=str(tmp_path),
            sample_rate=16000,
        )
        recorder.start_session()

        # Add 1 second of audio
        audio_data = np.ones(16000, dtype=np.int16)
        recorder.add_audio(audio_data)

        duration = recorder.get_session_duration()
        assert abs(duration - 1.0) < 0.01  # Should be ~1 second

    def test_session_start_time(self, tmp_path: Path) -> None:
        """Verify session start time is tracked."""
        recorder = ConversationRecorder(recordings_path=str(tmp_path))

        before = datetime.now()
        recorder.start_session()
        after = datetime.now()

        start_time = recorder.get_session_start_time()

        assert start_time is not None
        assert before <= start_time <= after


class TestTranscriptionService:
    """Test TranscriptionService functionality."""

    def test_init_default_values(self, tmp_path: Path) -> None:
        """Verify default initialization values."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        assert service.is_initialized is False
        stats = service.get_stats()
        assert stats["transcriptions_completed"] == 0
        assert stats["transcriptions_failed"] == 0
        assert stats["pending_jobs"] == 0

    @pytest.mark.asyncio
    async def test_initialize_with_api_key(self, tmp_path: Path) -> None:
        """Verify initialization with API key."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.headless.transcription.OPENAI_AVAILABLE", True):
            with patch("reachy_mini_conversation_app.headless.transcription.AsyncOpenAI"):
                result = await service.initialize(openai_api_key="test-key")

        assert result is True
        assert service.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self, tmp_path: Path) -> None:
        """Verify initialization fails without API key."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        result = await service.initialize()

        assert result is False
        assert service.is_initialized is False

    @pytest.mark.asyncio
    async def test_transcribe_file_not_initialized(self, tmp_path: Path) -> None:
        """Verify transcription fails when not initialized."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        result = await service.transcribe_file(str(tmp_path / "test.wav"))

        assert result is None

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, tmp_path: Path) -> None:
        """Verify transcription fails for non-existent file."""
        service = TranscriptionService(recordings_path=str(tmp_path))
        service._initialized = True
        service._openai_client = Mock()

        result = await service.transcribe_file(str(tmp_path / "nonexistent.wav"))

        assert result is None

    @pytest.mark.asyncio
    async def test_transcribe_with_retry_success(self, tmp_path: Path) -> None:
        """Verify successful transcription with retry logic."""
        # Create test WAV file
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.ones(16000, dtype=np.int16).tobytes())

        service = TranscriptionService(recordings_path=str(tmp_path))
        service._initialized = True

        # Mock diarized response (default mode is diarization enabled)
        mock_segment = Mock()
        mock_segment.speaker = "A"
        mock_segment.text = "Hello, this is a test transcription."
        mock_segment.start = 0.0
        mock_segment.end = 2.0

        mock_response = Mock()
        mock_response.segments = [mock_segment]
        mock_response.text = "Hello, this is a test transcription."

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        service._openai_client = mock_client

        result = await service.transcribe_file(str(wav_path))

        assert result is not None
        assert "Hello, this is a test transcription." in result.full_text
        assert service.get_stats()["transcriptions_completed"] == 1

    @pytest.mark.asyncio
    async def test_transcribe_with_retry_rate_limit(self, tmp_path: Path) -> None:
        """Verify retry logic handles rate limits."""
        # Create test WAV file
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.ones(16000, dtype=np.int16).tobytes())

        service = TranscriptionService(
            recordings_path=str(tmp_path),
            max_retries=2,
        )
        service._initialized = True

        # Mock rate limit then success
        from reachy_mini_conversation_app.headless.transcription import RateLimitError

        # Mock diarized response (default mode)
        mock_segment = Mock()
        mock_segment.speaker = "A"
        mock_segment.text = "Success after retry"
        mock_segment.start = 0.0
        mock_segment.end = 1.0

        mock_response = Mock()
        mock_response.segments = [mock_segment]
        mock_response.text = "Success after retry"

        # First call raises rate limit, second succeeds
        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=[RateLimitError("rate limited", response=Mock(), body=None), mock_response]
        )
        service._openai_client = mock_client

        result = await service.transcribe_file(str(wav_path))

        assert result is not None
        assert "Success after retry" in result.full_text
        assert service.get_stats()["total_retries"] == 1


class TestTranscriptionExports:
    """Test transcript export functionality."""

    def test_export_to_text_basic(self, tmp_path: Path) -> None:
        """Verify basic text export."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        transcript = ConversationTranscript(
            session_id="abc123",
            start_time=datetime(2024, 1, 15, 10, 30, 0),
            full_text="Hello world",
        )

        text = service.export_to_text(transcript)

        assert "Session: abc123" in text
        assert "2024-01-15 10:30:00" in text
        assert "Hello world" in text

    def test_export_to_text_with_diarization(self, tmp_path: Path) -> None:
        """Verify text export with speaker diarization."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        transcript = ConversationTranscript(
            session_id="abc123",
            start_time=datetime(2024, 1, 15, 10, 30, 0),
            full_text="Hello. Hi there.",
            speaker_diarization=[
                {"speaker": "Speaker A", "text": "Hello."},
                {"speaker": "Speaker B", "text": "Hi there."},
            ],
        )

        text = service.export_to_text(transcript)

        assert "[Speaker A]: Hello." in text
        assert "[Speaker B]: Hi there." in text

    def test_export_to_json(self, tmp_path: Path) -> None:
        """Verify JSON export."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        transcript = ConversationTranscript(
            session_id="abc123",
            start_time=datetime(2024, 1, 15, 10, 30, 0),
            end_time=datetime(2024, 1, 15, 10, 35, 0),
            full_text="Test transcript",
            word_timestamps=[
                WordTimestamp(word="Test", start_time=0.0, end_time=0.5),
                WordTimestamp(word="transcript", start_time=0.5, end_time=1.0),
            ],
        )

        json_str = service.export_to_json(transcript)
        data = json.loads(json_str)

        assert data["session_id"] == "abc123"
        assert data["full_text"] == "Test transcript"
        assert len(data["word_timestamps"]) == 2
        assert data["word_timestamps"][0]["word"] == "Test"

    def test_export_to_srt(self, tmp_path: Path) -> None:
        """Verify SRT export."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        transcript = ConversationTranscript(
            session_id="abc123",
            start_time=datetime(2024, 1, 15, 10, 30, 0),
            full_text="Hello. World.",
            speaker_diarization=[
                {"speaker": "A", "start": 0.0, "end": 1.5, "text": "Hello."},
                {"speaker": "B", "start": 2.0, "end": 3.5, "text": "World."},
            ],
        )

        srt = service.export_to_srt(transcript)

        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:01,500" in srt
        assert "[A] Hello." in srt
        assert "2\n" in srt
        assert "00:00:02,000 --> 00:00:03,500" in srt

    def test_format_srt_time(self, tmp_path: Path) -> None:
        """Verify SRT time formatting."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        assert service._format_srt_time(0) == "00:00:00,000"
        assert service._format_srt_time(1.5) == "00:00:01,500"
        assert service._format_srt_time(61.234) == "00:01:01,234"
        # Use 3662.0 to avoid floating point rounding issues
        assert service._format_srt_time(3662.0) == "01:01:02,000"


class TestTranscriptionCleanup:
    """Test cleanup and retention policies."""

    @pytest.mark.asyncio
    async def test_cleanup_old_recordings(self, tmp_path: Path) -> None:
        """Verify cleanup deletes old files."""
        recordings_dir = tmp_path / "recordings"
        recordings_dir.mkdir()

        # Create old file
        old_file = recordings_dir / "old.wav"
        old_file.touch()
        # Set modification time to 60 days ago
        old_time = datetime.now() - timedelta(days=60)
        os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))

        # Create recent file
        recent_file = recordings_dir / "recent.wav"
        recent_file.touch()

        service = TranscriptionService(
            recordings_path=str(recordings_dir),
            audio_retention_days=30,
        )

        result = await service.cleanup_old_recordings()

        assert result["deleted_files"] == 1
        assert not old_file.exists()
        assert recent_file.exists()

    @pytest.mark.asyncio
    async def test_delete_conversation(self, tmp_path: Path) -> None:
        """Verify delete_conversation removes files and database record."""
        recordings_dir = tmp_path / "recordings"
        recordings_dir.mkdir()

        # Create audio file
        audio_file = recordings_dir / "test-session.wav"
        audio_file.touch()

        # Mock database
        mock_db = Mock()
        mock_transcript = Mock()
        mock_transcript.audio_path = str(audio_file)
        mock_transcript.id = 1
        mock_db.get_transcript_by_session = Mock(return_value=mock_transcript)
        mock_db.delete_transcript = Mock()

        service = TranscriptionService(
            recordings_path=str(recordings_dir),
            memory_database=mock_db,
        )

        result = await service.delete_conversation("session-id")

        assert result is True
        assert not audio_file.exists()
        mock_db.delete_transcript.assert_called_once_with(1)


class TestDiarizationModes:
    """Test diarization vs non-diarization modes."""

    @pytest.mark.asyncio
    async def test_transcribe_with_diarization_enabled(self, tmp_path: Path) -> None:
        """Verify diarization mode uses correct model and parses speaker labels."""
        # Create test WAV file
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.ones(16000, dtype=np.int16).tobytes())

        service = TranscriptionService(recordings_path=str(tmp_path))
        service._initialized = True

        # Mock diarized response with real speaker labels from API
        mock_segment1 = Mock()
        mock_segment1.speaker = "A"
        mock_segment1.text = "Hello there."
        mock_segment1.start = 0.0
        mock_segment1.end = 1.0

        mock_segment2 = Mock()
        mock_segment2.speaker = "B"
        mock_segment2.text = "Hi, how are you?"
        mock_segment2.start = 1.0
        mock_segment2.end = 2.5

        mock_response = Mock()
        mock_response.segments = [mock_segment1, mock_segment2]
        mock_response.text = "Hello there. Hi, how are you?"

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        service._openai_client = mock_client

        result = await service.transcribe_file(
            str(wav_path),
            enable_diarization=True,
        )

        assert result is not None
        assert len(result.speaker_diarization) == 2
        assert result.speaker_diarization[0]["speaker"] == "A"
        assert result.speaker_diarization[1]["speaker"] == "B"

    @pytest.mark.asyncio
    async def test_transcribe_without_diarization(self, tmp_path: Path) -> None:
        """Verify non-diarization mode returns word timestamps."""
        # Create test WAV file
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.ones(16000, dtype=np.int16).tobytes())

        service = TranscriptionService(recordings_path=str(tmp_path))
        service._initialized = True

        # Mock verbose_json response with word timestamps
        mock_word1 = Mock(word="Hello", start=0.0, end=0.5)
        mock_word2 = Mock(word="world", start=0.5, end=1.0)

        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_response.segments = []
        mock_response.words = [mock_word1, mock_word2]

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        service._openai_client = mock_client

        result = await service.transcribe_file(
            str(wav_path),
            enable_diarization=False,
        )

        assert result is not None
        assert len(result.word_timestamps) == 2
        assert result.word_timestamps[0].word == "Hello"

    @pytest.mark.asyncio
    async def test_transcribe_with_custom_speaker_names(self, tmp_path: Path) -> None:
        """Verify custom speaker names are passed to API."""
        # Create test WAV file
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.ones(16000, dtype=np.int16).tobytes())

        service = TranscriptionService(recordings_path=str(tmp_path))
        service._initialized = True

        # Mock response with custom speaker names
        mock_segment = Mock()
        mock_segment.speaker = "User"
        mock_segment.text = "Hello."
        mock_segment.start = 0.0
        mock_segment.end = 1.0

        mock_response = Mock()
        mock_response.segments = [mock_segment]

        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
        service._openai_client = mock_client

        result = await service.transcribe_file(
            str(wav_path),
            enable_diarization=True,
            speaker_names=["User", "Assistant"],
        )

        # Verify the API was called with speaker names
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs.get("known_speaker_names") == ["User", "Assistant"]


class TestTranscriptionStats:
    """Test statistics tracking."""

    def test_stats_initial(self, tmp_path: Path) -> None:
        """Verify initial stats are zero."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        stats = service.get_stats()

        assert stats["transcriptions_completed"] == 0
        assert stats["transcriptions_failed"] == 0
        assert stats["total_retries"] == 0
        assert stats["pending_jobs"] == 0
        assert stats["initialized"] is False

    @pytest.mark.asyncio
    async def test_stats_after_failure(self, tmp_path: Path) -> None:
        """Verify stats track failures from API errors."""
        # Create test WAV file
        wav_path = tmp_path / "test.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.ones(16000, dtype=np.int16).tobytes())

        service = TranscriptionService(
            recordings_path=str(tmp_path),
            max_retries=1,
        )
        service._initialized = True

        # Mock API error that exhausts retries
        from reachy_mini_conversation_app.headless.transcription import APIError
        mock_client = AsyncMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=APIError("API error", request=Mock(), body=None)
        )
        service._openai_client = mock_client

        await service.transcribe_file(str(wav_path))

        stats = service.get_stats()
        assert stats["transcriptions_failed"] == 1


class TestTranscriptionShutdown:
    """Test shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_task(self, tmp_path: Path) -> None:
        """Verify shutdown cancels background task."""
        service = TranscriptionService(recordings_path=str(tmp_path))

        # Create a real asyncio task that we can cancel
        async def dummy_task():
            await asyncio.sleep(10)

        task = asyncio.create_task(dummy_task())
        service._transcription_task = task

        await service.shutdown()

        assert task.cancelled()


class TestWordTimestamp:
    """Test WordTimestamp dataclass."""

    def test_word_timestamp_creation(self) -> None:
        """Verify WordTimestamp creation."""
        wt = WordTimestamp(word="hello", start_time=0.5, end_time=1.0)

        assert wt.word == "hello"
        assert wt.start_time == 0.5
        assert wt.end_time == 1.0


class TestTranscriptSegment:
    """Test TranscriptSegment dataclass."""

    def test_segment_defaults(self) -> None:
        """Verify TranscriptSegment default values."""
        segment = TranscriptSegment(text="test")

        assert segment.text == "test"
        assert segment.speaker == "unknown"
        assert segment.start_time == 0.0
        assert segment.end_time == 0.0
        assert segment.confidence == 1.0

    def test_segment_with_values(self) -> None:
        """Verify TranscriptSegment with all values."""
        segment = TranscriptSegment(
            text="Hello world",
            speaker="Speaker A",
            start_time=1.5,
            end_time=3.0,
            confidence=0.95,
        )

        assert segment.text == "Hello world"
        assert segment.speaker == "Speaker A"
        assert segment.start_time == 1.5
        assert segment.end_time == 3.0
        assert segment.confidence == 0.95


class TestConversationTranscriptDataclass:
    """Test ConversationTranscript dataclass."""

    def test_transcript_defaults(self) -> None:
        """Verify ConversationTranscript default values."""
        transcript = ConversationTranscript(
            session_id="abc",
            start_time=datetime.now(),
        )

        assert transcript.session_id == "abc"
        assert transcript.end_time is None
        assert transcript.audio_file_path is None
        assert transcript.full_text == ""
        assert transcript.segments == []
        assert transcript.word_timestamps == []
        assert transcript.speaker_diarization == []
        assert transcript.metadata == {}

    def test_transcript_with_all_fields(self) -> None:
        """Verify ConversationTranscript with all fields."""
        start = datetime(2024, 1, 15, 10, 0, 0)
        end = datetime(2024, 1, 15, 10, 5, 0)

        transcript = ConversationTranscript(
            session_id="xyz789",
            start_time=start,
            end_time=end,
            audio_file_path="/path/to/audio.wav",
            full_text="Full conversation text",
            segments=[TranscriptSegment(text="Hello")],
            word_timestamps=[WordTimestamp(word="Hello", start_time=0, end_time=0.5)],
            speaker_diarization=[{"speaker": "A", "text": "Hello"}],
            metadata={"key": "value"},
        )

        assert transcript.session_id == "xyz789"
        assert transcript.end_time == end
        assert transcript.audio_file_path == "/path/to/audio.wav"
        assert len(transcript.segments) == 1
        assert len(transcript.word_timestamps) == 1
        assert len(transcript.speaker_diarization) == 1
        assert transcript.metadata["key"] == "value"
