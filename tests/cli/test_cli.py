"""Tests for lily-cli commands.

Comprehensive unit tests for the CLI interface including user, memory,
transcript management and error handling.
"""

import base64
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import pytest
import numpy as np
from typer.testing import CliRunner

from reachy_mini_conversation_app.cli.main import app
from reachy_mini_conversation_app.memory.database import (
    Memory,
    KnownPerson,
    LearnedFact,
    ConversationTranscript,
)


runner = CliRunner()


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager for testing."""
    manager = Mock()
    manager.db = Mock()
    manager.get_stats.return_value = {
        "memories": 42,
        "persons": 3,
        "pending_facts": 5,
        "approved_facts": 10,
    }
    manager.get_current_person.return_value = None
    return manager


@pytest.fixture
def mock_face_service():
    """Create a mock FaceRecognitionService for testing."""
    service = Mock()
    service.is_initialized = True
    service.detect_faces.return_value = [(100, 100, 200, 200)]  # One face detected
    service.extract_embedding.return_value = np.random.randn(512).astype(np.float32)
    service.find_matching_person.return_value = None
    return service


@pytest.fixture
def sample_person():
    """Create a sample KnownPerson for testing."""
    return KnownPerson(
        id=1,
        name="TestUser",
        photo_base64="aGVsbG8=",  # base64 of "hello"
        face_embedding=[0.1] * 512,
        last_seen=datetime.now(),
        recognition_count=5,
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_memory():
    """Create a sample Memory for testing."""
    return Memory(
        id=1,
        content="This is a test memory about robots",
        source="cli",
        timestamp=datetime.now(),
        metadata={"added_via": "test"},
        relevance_score=1.0,
    )


@pytest.fixture
def sample_fact():
    """Create a sample LearnedFact for testing."""
    return LearnedFact(
        id=1,
        fact="The user likes coffee",
        confidence=0.85,
        context="conversation",
        timestamp=datetime.now(),
        approved=False,
    )


@pytest.fixture
def sample_transcript():
    """Create a sample ConversationTranscript for testing."""
    return ConversationTranscript(
        id=1,
        session_id="test-session-123",
        person_id=1,
        person_name="TestUser",
        transcript="Hello, how are you?\nI am fine, thank you!",
        audio_path="/tmp/test.wav",
        duration_seconds=15.5,
        started_at=datetime.now() - timedelta(minutes=5),
        ended_at=datetime.now(),
        metadata={"source": "test"},
        speaker_diarization=None,
        word_timestamps=None,
    )


@pytest.fixture
def sample_image_file(tmp_path: Path):
    """Create a sample image file for testing."""
    import cv2

    # Create a simple test image with a "face-like" region
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Draw a circle to simulate a face
    cv2.circle(img, (320, 240), 100, (200, 180, 160), -1)

    image_path = tmp_path / "test_photo.jpg"
    cv2.imwrite(str(image_path), img)
    return image_path


# ============================================================================
# Main CLI Tests
# ============================================================================


class TestMainCLI:
    """Tests for main CLI commands (status, version)."""

    def test_help_shows_commands(self) -> None:
        """Test that --help shows available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "user" in result.output
        assert "memory" in result.output
        assert "transcript" in result.output
        assert "status" in result.output
        assert "version" in result.output

    def test_status_command(self) -> None:
        """Test status command shows system stats."""
        mock_manager = Mock()
        mock_manager.get_stats.return_value = {
            "memories": 42,
            "persons": 3,
            "pending_facts": 5,
            "approved_facts": 10,
        }
        mock_manager.get_current_person.return_value = None

        with patch("reachy_mini_conversation_app.cli.common.get_memory_manager", return_value=mock_manager):
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "Database connected" in result.output
            assert "Memories:" in result.output
            assert "Known persons:" in result.output

    def test_version_command(self) -> None:
        """Test version command shows version info."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Lily CLI" in result.output


# ============================================================================
# User Command Tests
# ============================================================================


class TestUserCommands:
    """Tests for user management commands."""

    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    def test_user_list_empty(self, mock_get_manager: Mock, mock_memory_manager: Mock) -> None:
        """Test listing users when none exist."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_all_persons.return_value = []

        result = runner.invoke(app, ["user", "list"])
        assert result.exit_code == 0
        assert "No users registered" in result.output

    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    def test_user_list_with_users(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_person: KnownPerson
    ) -> None:
        """Test listing users with data."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_all_persons.return_value = [sample_person]
        mock_memory_manager.get_current_person.return_value = sample_person

        result = runner.invoke(app, ["user", "list"])
        assert result.exit_code == 0
        assert "TestUser" in result.output
        assert "Current" in result.output

    @patch("reachy_mini_conversation_app.cli.user.get_face_service")
    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    @patch("reachy_mini_conversation_app.cli.user.cv2")
    def test_user_add_from_photo(
        self,
        mock_cv2: Mock,
        mock_get_manager: Mock,
        mock_get_face: Mock,
        mock_memory_manager: Mock,
        mock_face_service: Mock,
        sample_image_file: Path,
    ) -> None:
        """Test adding a user from a photo file."""
        mock_get_manager.return_value = mock_memory_manager
        mock_get_face.return_value = mock_face_service
        mock_memory_manager.get_person_by_name.return_value = None
        mock_memory_manager.db.add_person.return_value = 1

        # Mock cv2 functions
        mock_cv2.imread.return_value = np.ones((480, 640, 3), dtype=np.uint8)
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        result = runner.invoke(app, ["user", "add", "NewUser", "--photo", str(sample_image_file)])
        assert result.exit_code == 0
        assert "Added user" in result.output

    def test_user_add_missing_photo_and_capture(self) -> None:
        """Test add command fails without --photo or --capture."""
        result = runner.invoke(app, ["user", "add", "TestUser"])
        assert result.exit_code == 1
        assert "Must provide" in result.output

    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    def test_user_delete_not_found(self, mock_get_manager: Mock, mock_memory_manager: Mock) -> None:
        """Test deleting a non-existent user."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_person_by_name.return_value = None

        result = runner.invoke(app, ["user", "delete", "NonExistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    def test_user_delete_with_force(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_person: KnownPerson
    ) -> None:
        """Test deleting a user with --force flag."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_person_by_name.return_value = sample_person

        result = runner.invoke(app, ["user", "delete", "TestUser", "--force"])
        assert result.exit_code == 0
        assert "Deleted user" in result.output

    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    def test_user_switch(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_person: KnownPerson
    ) -> None:
        """Test switching to a different user."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_person_by_name.return_value = sample_person

        result = runner.invoke(app, ["user", "switch", "TestUser"])
        assert result.exit_code == 0
        assert "Current user set to" in result.output

    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    def test_user_switch_not_found(self, mock_get_manager: Mock, mock_memory_manager: Mock) -> None:
        """Test switching to a non-existent user."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_person_by_name.return_value = None

        result = runner.invoke(app, ["user", "switch", "NonExistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    def test_user_clear(self, mock_get_manager: Mock, mock_memory_manager: Mock) -> None:
        """Test clearing current user."""
        mock_get_manager.return_value = mock_memory_manager

        result = runner.invoke(app, ["user", "clear"])
        assert result.exit_code == 0
        assert "Current user cleared" in result.output
        mock_memory_manager.set_current_person.assert_called_once_with(None)


# ============================================================================
# Memory Command Tests
# ============================================================================


class TestMemoryCommands:
    """Tests for memory management commands."""

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_list_empty(self, mock_get_manager: Mock, mock_memory_manager: Mock) -> None:
        """Test listing memories when none exist."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_all_memories.return_value = []

        result = runner.invoke(app, ["memory", "list"])
        assert result.exit_code == 0
        assert "No memories stored" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_list_with_data(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_memory: Memory
    ) -> None:
        """Test listing memories with data."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_all_memories.return_value = [sample_memory]

        result = runner.invoke(app, ["memory", "list"])
        assert result.exit_code == 0
        assert "robots" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_search(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_memory: Memory
    ) -> None:
        """Test searching memories."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.search_memories.return_value = [sample_memory]

        result = runner.invoke(app, ["memory", "search", "robots"])
        assert result.exit_code == 0
        assert "robots" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_search_no_results(
        self, mock_get_manager: Mock, mock_memory_manager: Mock
    ) -> None:
        """Test searching memories with no results."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.search_memories.return_value = []

        result = runner.invoke(app, ["memory", "search", "unicorns"])
        assert result.exit_code == 0
        assert "No memories found" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_add(self, mock_get_manager: Mock, mock_memory_manager: Mock) -> None:
        """Test adding a memory."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.add_memory.return_value = 1

        result = runner.invoke(app, ["memory", "add", "Test memory content"])
        assert result.exit_code == 0
        assert "Memory added" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_delete_with_force(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_memory: Memory
    ) -> None:
        """Test deleting a memory with --force."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_all_memories.return_value = [sample_memory]
        mock_memory_manager.db.delete_memory.return_value = True

        result = runner.invoke(app, ["memory", "delete", "1", "--force"])
        assert result.exit_code == 0
        assert "deleted" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_delete_not_found(
        self, mock_get_manager: Mock, mock_memory_manager: Mock
    ) -> None:
        """Test deleting a non-existent memory."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_all_memories.return_value = []

        result = runner.invoke(app, ["memory", "delete", "999", "--force"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_facts_pending(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_fact: LearnedFact
    ) -> None:
        """Test listing pending facts."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.get_pending_facts.return_value = [sample_fact]

        result = runner.invoke(app, ["memory", "facts", "--pending"])
        assert result.exit_code == 0
        assert "coffee" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_approve_fact(
        self, mock_get_manager: Mock, mock_memory_manager: Mock
    ) -> None:
        """Test approving a fact."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.approve_fact.return_value = True

        result = runner.invoke(app, ["memory", "approve", "1"])
        assert result.exit_code == 0
        assert "approved" in result.output

    @patch("reachy_mini_conversation_app.cli.memory.get_memory_manager")
    def test_memory_reject_fact(
        self, mock_get_manager: Mock, mock_memory_manager: Mock
    ) -> None:
        """Test rejecting a fact."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.delete_fact.return_value = True

        result = runner.invoke(app, ["memory", "reject", "1"])
        assert result.exit_code == 0
        assert "rejected" in result.output


# ============================================================================
# Transcript Command Tests
# ============================================================================


class TestTranscriptCommands:
    """Tests for transcript management commands."""

    @patch("reachy_mini_conversation_app.cli.transcript.get_memory_manager")
    def test_transcript_search_no_results(
        self, mock_get_manager: Mock, mock_memory_manager: Mock
    ) -> None:
        """Test searching transcripts with no results."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.search_transcripts.return_value = []

        result = runner.invoke(app, ["transcript", "search", "unicorns"])
        assert result.exit_code == 0
        assert "No transcripts found" in result.output

    @patch("reachy_mini_conversation_app.cli.transcript.get_memory_manager")
    def test_transcript_search_with_results(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_transcript: ConversationTranscript
    ) -> None:
        """Test searching transcripts with results."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.search_transcripts.return_value = [sample_transcript]

        result = runner.invoke(app, ["transcript", "search", "hello"])
        assert result.exit_code == 0
        assert "test-session" in result.output

    @patch("reachy_mini_conversation_app.cli.transcript.get_memory_manager")
    def test_transcript_show(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_transcript: ConversationTranscript
    ) -> None:
        """Test showing a specific transcript."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.get_transcript_by_session.return_value = sample_transcript

        result = runner.invoke(app, ["transcript", "show", "test-session-123"])
        assert result.exit_code == 0
        assert "Hello" in result.output
        assert "TestUser" in result.output

    @patch("reachy_mini_conversation_app.cli.transcript.get_memory_manager")
    def test_transcript_show_not_found(
        self, mock_get_manager: Mock, mock_memory_manager: Mock
    ) -> None:
        """Test showing a non-existent transcript."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.get_transcript_by_session.return_value = None

        result = runner.invoke(app, ["transcript", "show", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("reachy_mini_conversation_app.cli.transcript.get_memory_manager")
    def test_transcript_export_json(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_transcript: ConversationTranscript, tmp_path: Path
    ) -> None:
        """Test exporting a transcript to JSON."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.get_transcript_by_session.return_value = sample_transcript

        output_file = tmp_path / "export.json"
        result = runner.invoke(app, ["transcript", "export", "test-session-123", "-o", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert data["session_id"] == "test-session-123"
        assert "Hello" in data["transcript"]

    @patch("reachy_mini_conversation_app.cli.transcript.get_memory_manager")
    def test_transcript_export_txt(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_transcript: ConversationTranscript, tmp_path: Path
    ) -> None:
        """Test exporting a transcript to plain text."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.get_transcript_by_session.return_value = sample_transcript

        output_file = tmp_path / "export.txt"
        result = runner.invoke(app, ["transcript", "export", "test-session-123", "-f", "txt", "-o", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "Hello" in content
        assert "Session:" in content

    @patch("reachy_mini_conversation_app.cli.transcript.get_memory_manager")
    def test_transcript_export_srt(
        self, mock_get_manager: Mock, mock_memory_manager: Mock, sample_transcript: ConversationTranscript, tmp_path: Path
    ) -> None:
        """Test exporting a transcript to SRT format."""
        mock_get_manager.return_value = mock_memory_manager
        mock_memory_manager.db.get_transcript_by_session.return_value = sample_transcript

        output_file = tmp_path / "export.srt"
        result = runner.invoke(app, ["transcript", "export", "test-session-123", "-f", "srt", "-o", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "00:00:00" in content

    def test_transcript_export_invalid_format(self) -> None:
        """Test export with invalid format."""
        result = runner.invoke(app, ["transcript", "export", "test", "-f", "invalid"])
        assert result.exit_code == 1
        assert "Invalid format" in result.output

    def test_transcript_list_no_directory(self, tmp_path: Path) -> None:
        """Test listing transcripts when directory doesn't exist."""
        with patch("reachy_mini_conversation_app.cli.transcript.TRANSCRIPTS_PATH", tmp_path / "nonexistent"):
            result = runner.invoke(app, ["transcript", "list"])
            assert result.exit_code == 0
            assert "No recordings directory" in result.output


# ============================================================================
# Common Utilities Tests
# ============================================================================


class TestCommonUtilities:
    """Tests for common CLI utilities."""

    def test_validate_image_path_not_found(self, tmp_path: Path) -> None:
        """Test validation of non-existent image path."""
        from reachy_mini_conversation_app.cli.common import validate_image_path

        result = validate_image_path(str(tmp_path / "nonexistent.jpg"))
        assert result is None

    def test_validate_image_path_invalid_extension(self, tmp_path: Path) -> None:
        """Test validation of invalid image extension."""
        from reachy_mini_conversation_app.cli.common import validate_image_path

        text_file = tmp_path / "test.txt"
        text_file.write_text("not an image")

        result = validate_image_path(str(text_file))
        assert result is None

    def test_validate_image_path_valid(self, sample_image_file: Path) -> None:
        """Test validation of valid image path."""
        from reachy_mini_conversation_app.cli.common import validate_image_path

        result = validate_image_path(str(sample_image_file))
        assert result is not None
        assert result.exists()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in CLI commands."""

    def test_status_database_error(self) -> None:
        """Test status command handles database errors gracefully."""
        with patch("reachy_mini_conversation_app.cli.common.get_memory_manager") as mock_get_manager:
            mock_get_manager.side_effect = Exception("Database connection failed")

            result = runner.invoke(app, ["status"])
            # Should still exit gracefully
            assert "Error" in result.output

    def test_user_list_database_error(self) -> None:
        """Test user list handles database errors."""
        mock_manager = Mock()
        mock_manager.get_all_persons.side_effect = Exception("Database error")

        with patch("reachy_mini_conversation_app.cli.user.get_memory_manager", return_value=mock_manager):
            result = runner.invoke(app, ["user", "list"])
            assert result.exit_code == 1
            assert "Failed" in result.output


# ============================================================================
# Integration-like Tests (with mocked dependencies)
# ============================================================================


class TestUserWorkflow:
    """Test complete user management workflow."""

    @patch("reachy_mini_conversation_app.cli.user.get_face_service")
    @patch("reachy_mini_conversation_app.cli.user.get_memory_manager")
    @patch("reachy_mini_conversation_app.cli.user.cv2")
    def test_add_list_switch_delete_workflow(
        self,
        mock_cv2: Mock,
        mock_get_manager: Mock,
        mock_get_face: Mock,
        mock_memory_manager: Mock,
        mock_face_service: Mock,
        sample_person: KnownPerson,
        sample_image_file: Path,
    ) -> None:
        """Test complete workflow: add -> list -> switch -> delete."""
        mock_get_manager.return_value = mock_memory_manager
        mock_get_face.return_value = mock_face_service

        # Mock cv2 functions
        mock_cv2.imread.return_value = np.ones((480, 640, 3), dtype=np.uint8)
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        # 1. Add user
        mock_memory_manager.get_person_by_name.return_value = None
        mock_memory_manager.db.add_person.return_value = 1

        result = runner.invoke(app, ["user", "add", "WorkflowUser", "--photo", str(sample_image_file)])
        assert result.exit_code == 0, result.output
        assert "Added user" in result.output

        # 2. List users
        mock_memory_manager.get_all_persons.return_value = [sample_person]
        mock_memory_manager.get_current_person.return_value = None

        result = runner.invoke(app, ["user", "list"])
        assert result.exit_code == 0
        assert "TestUser" in result.output

        # 3. Switch to user
        mock_memory_manager.get_person_by_name.return_value = sample_person

        result = runner.invoke(app, ["user", "switch", "TestUser"])
        assert result.exit_code == 0
        assert "Current user set to" in result.output

        # 4. Delete user
        result = runner.invoke(app, ["user", "delete", "TestUser", "--force"])
        assert result.exit_code == 0
        assert "Deleted user" in result.output
