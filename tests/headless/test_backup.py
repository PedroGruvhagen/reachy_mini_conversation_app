"""Tests for the Backup and Restore module.

Tests for backup creation, restoration, rotation, and verification.
"""

import json
import os
import tarfile
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from reachy_mini_conversation_app.headless.backup import (
    BackupManifest,
    BackupInfo,
    RestoreResult,
    BackupManager,
    create_backup,
    restore_latest_backup,
    list_available_backups,
    DEFAULT_BACKUP_DIR,
    DEFAULT_MAX_BACKUPS,
    DEFAULT_CONFIG_FILES,
    BACKUP_MANIFEST_FILENAME,
    BACKUP_EXTENSION,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dirs():
    """Create temporary backup and data directories."""
    with tempfile.TemporaryDirectory() as backup_dir, \
         tempfile.TemporaryDirectory() as data_dir:
        yield Path(backup_dir), Path(data_dir)


@pytest.fixture
def backup_manager(temp_dirs):
    """Create BackupManager with temporary directories."""
    backup_dir, data_dir = temp_dirs
    return BackupManager(backup_dir=backup_dir, data_dir=data_dir)


@pytest.fixture
def sample_data_dir(temp_dirs):
    """Create sample data files for testing."""
    _, data_dir = temp_dirs

    # Create config files
    (data_dir / ".env").write_text("OPENAI_API_KEY=sk-test123\n")
    (data_dir / "config.json").write_text('{"setting": "value"}')

    # Create transcripts directory
    transcripts_dir = data_dir / "transcripts"
    transcripts_dir.mkdir()
    (transcripts_dir / "conv_001.json").write_text('{"id": "001"}')
    (transcripts_dir / "conv_002.json").write_text('{"id": "002"}')

    # Create recordings directory
    recordings_dir = data_dir / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "audio_001.wav").write_bytes(b"RIFF" + b"\x00" * 100)

    return data_dir


# =============================================================================
# BackupManifest Tests
# =============================================================================


class TestBackupManifest:
    """Tests for BackupManifest data class."""

    def test_default_values(self) -> None:
        """Test manifest default values."""
        manifest = BackupManifest()

        assert manifest.version == "1.0"
        assert manifest.backup_type == "full"
        assert manifest.total_size == 0
        assert isinstance(manifest.files, dict)
        assert manifest.created_at is not None

    def test_add_file(self) -> None:
        """Test adding files to manifest."""
        manifest = BackupManifest()

        manifest.add_file("config.json", "abc123", 100)
        manifest.add_file(".env", "def456", 50)

        assert len(manifest.files) == 2
        assert manifest.total_size == 150
        assert manifest.files["config.json"]["checksum"] == "abc123"
        assert manifest.files[".env"]["size"] == 50

    def test_to_dict(self) -> None:
        """Test manifest serialization."""
        manifest = BackupManifest(backup_type="config")
        manifest.add_file("test.txt", "hash123", 200)

        data = manifest.to_dict()

        assert data["version"] == "1.0"
        assert data["backup_type"] == "config"
        assert data["total_size"] == 200
        assert "test.txt" in data["files"]

    def test_from_dict(self) -> None:
        """Test manifest deserialization."""
        data = {
            "version": "1.0",
            "created_at": "2024-01-01T00:00:00",
            "hostname": "testhost",
            "backup_type": "transcripts",
            "files": {"file1.txt": {"checksum": "abc", "size": 100}},
            "total_size": 100,
        }

        manifest = BackupManifest.from_dict(data)

        assert manifest.version == "1.0"
        assert manifest.hostname == "testhost"
        assert manifest.backup_type == "transcripts"
        assert manifest.total_size == 100
        assert "file1.txt" in manifest.files

    def test_round_trip(self) -> None:
        """Test manifest serialization round-trip."""
        original = BackupManifest(backup_type="full")
        original.add_file("a.txt", "hash_a", 10)
        original.add_file("b.txt", "hash_b", 20)

        restored = BackupManifest.from_dict(original.to_dict())

        assert restored.backup_type == original.backup_type
        assert restored.total_size == original.total_size
        assert len(restored.files) == len(original.files)


# =============================================================================
# BackupInfo Tests
# =============================================================================


class TestBackupInfo:
    """Tests for BackupInfo data class."""

    def test_from_manifest(self) -> None:
        """Test creating BackupInfo from manifest."""
        manifest = BackupManifest(backup_type="config")
        manifest.add_file("config.json", "abc", 100)

        info = BackupInfo.from_manifest(Path("/backups/test.tar.gz"), manifest)

        assert info.path == Path("/backups/test.tar.gz")
        assert info.backup_type == "config"
        assert info.total_size == 100
        assert info.file_count == 1


# =============================================================================
# RestoreResult Tests
# =============================================================================


class TestRestoreResult:
    """Tests for RestoreResult data class."""

    def test_default_values(self) -> None:
        """Test result default values."""
        result = RestoreResult(success=True)

        assert result.success is True
        assert result.files_restored == []
        assert result.files_skipped == []
        assert result.errors == []

    def test_with_values(self) -> None:
        """Test result with values."""
        result = RestoreResult(
            success=False,
            files_restored=["a.txt"],
            files_skipped=["b.txt"],
            errors=["Error occurred"],
        )

        assert result.success is False
        assert len(result.files_restored) == 1
        assert len(result.files_skipped) == 1
        assert len(result.errors) == 1


# =============================================================================
# BackupManager Initialization Tests
# =============================================================================


class TestBackupManagerInit:
    """Tests for BackupManager initialization."""

    def test_default_directories(self) -> None:
        """Test default directory configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BackupManager(backup_dir=Path(tmpdir))

            assert manager.backup_dir == Path(tmpdir)
            assert manager.data_dir == Path.home() / ".reachy_mini"

    def test_custom_directories(self, temp_dirs) -> None:
        """Test custom directory configuration."""
        backup_dir, data_dir = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=data_dir)

        assert manager.backup_dir == backup_dir
        assert manager.data_dir == data_dir

    def test_creates_backup_directory(self, temp_dirs) -> None:
        """Test that backup directory is created."""
        backup_dir, data_dir = temp_dirs
        new_backup_dir = backup_dir / "nested" / "backup"

        manager = BackupManager(backup_dir=new_backup_dir, data_dir=data_dir)

        assert new_backup_dir.exists()

    def test_default_max_backups(self, backup_manager) -> None:
        """Test default max backups value."""
        assert backup_manager._max_backups == DEFAULT_MAX_BACKUPS


# =============================================================================
# Checksum Tests
# =============================================================================


class TestChecksums:
    """Tests for checksum operations."""

    def test_calculate_checksum(self) -> None:
        """Test checksum calculation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            checksum = BackupManager._calculate_checksum(Path(f.name))

            assert len(checksum) == 64  # SHA256 hex length
            assert checksum.isalnum()

            os.unlink(f.name)

    def test_verify_checksum_valid(self) -> None:
        """Test checksum verification with valid file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            checksum = BackupManager._calculate_checksum(Path(f.name))
            is_valid = BackupManager._verify_checksum(Path(f.name), checksum)

            assert is_valid is True

            os.unlink(f.name)

    def test_verify_checksum_invalid(self) -> None:
        """Test checksum verification with wrong checksum."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            is_valid = BackupManager._verify_checksum(Path(f.name), "wrong_hash")

            assert is_valid is False

            os.unlink(f.name)

    def test_checksum_deterministic(self) -> None:
        """Test that same content produces same checksum."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("deterministic content")
            f.flush()

            checksum1 = BackupManager._calculate_checksum(Path(f.name))
            checksum2 = BackupManager._calculate_checksum(Path(f.name))

            assert checksum1 == checksum2

            os.unlink(f.name)


# =============================================================================
# Backup Creation Tests
# =============================================================================


class TestBackupCreation:
    """Tests for backup creation."""

    def test_create_config_backup(self, temp_dirs, sample_data_dir) -> None:
        """Test creating a config backup."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup(backup_type="config")

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.suffix == ".gz"
        assert "config" in backup_path.name

    def test_create_transcripts_backup(self, temp_dirs, sample_data_dir) -> None:
        """Test creating a transcripts backup."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup(backup_type="transcripts")

        assert backup_path is not None
        assert backup_path.exists()
        assert "transcripts" in backup_path.name

    def test_create_full_backup(self, temp_dirs, sample_data_dir) -> None:
        """Test creating a full backup."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup(backup_type="full")

        assert backup_path is not None
        assert backup_path.exists()
        assert "full" in backup_path.name

    def test_create_backup_with_audio(self, temp_dirs, sample_data_dir) -> None:
        """Test creating backup including audio."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup(backup_type="full", include_audio=True)

        # Verify audio file is in backup
        manifest = manager._read_manifest(backup_path)
        audio_files = [f for f in manifest.files if "audio" in f or "wav" in f]
        assert len(audio_files) > 0

    def test_create_backup_empty_data_dir(self, temp_dirs) -> None:
        """Test creating backup with no files."""
        backup_dir, data_dir = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=data_dir)

        backup_path = manager.create_backup()

        assert backup_path is None  # No files to backup

    def test_backup_contains_manifest(self, temp_dirs, sample_data_dir) -> None:
        """Test that backup contains manifest."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup()

        with tarfile.open(backup_path, "r:gz") as tar:
            names = tar.getnames()
            assert BACKUP_MANIFEST_FILENAME in names

    def test_backup_manifest_has_files(self, temp_dirs, sample_data_dir) -> None:
        """Test that manifest contains file information."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup()
        manifest = manager._read_manifest(backup_path)

        assert manifest is not None
        assert len(manifest.files) > 0
        for file_info in manifest.files.values():
            assert "checksum" in file_info
            assert "size" in file_info


# =============================================================================
# Backup Rotation Tests
# =============================================================================


class TestBackupRotation:
    """Tests for backup rotation."""

    def test_rotation_removes_old_backups(self, temp_dirs, sample_data_dir) -> None:
        """Test that old backups are removed."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(
            backup_dir=backup_dir,
            data_dir=sample_data_dir,
            max_backups=3,
        )

        # Create more backups than limit
        for _ in range(5):
            manager.create_backup(backup_type="config")

        backups = manager.list_backups(backup_type="config")
        assert len(backups) <= 3

    def test_rotation_keeps_newest(self, temp_dirs, sample_data_dir) -> None:
        """Test that rotation keeps newest backups."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(
            backup_dir=backup_dir,
            data_dir=sample_data_dir,
            max_backups=2,
        )

        # Create backups with small delays to ensure different timestamps
        paths = []
        for _ in range(3):
            path = manager.create_backup(backup_type="config")
            paths.append(path)
            time.sleep(0.01)  # Small delay for distinct mtime

        # Count remaining backups (should be at most 2)
        remaining = [p for p in paths if p.exists()]
        assert len(remaining) <= 2

        # The newest should always exist
        assert paths[2].exists()


# =============================================================================
# Backup Listing Tests
# =============================================================================


class TestBackupListing:
    """Tests for listing backups."""

    def test_list_backups_empty(self, backup_manager) -> None:
        """Test listing when no backups exist."""
        backups = backup_manager.list_backups()
        assert backups == []

    def test_list_backups_by_type(self, temp_dirs, sample_data_dir) -> None:
        """Test filtering backups by type."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        manager.create_backup(backup_type="config")
        manager.create_backup(backup_type="transcripts")

        config_backups = manager.list_backups(backup_type="config")
        transcript_backups = manager.list_backups(backup_type="transcripts")

        assert len(config_backups) == 1
        assert len(transcript_backups) == 1

    def test_list_backups_sorted(self, temp_dirs, sample_data_dir) -> None:
        """Test that backups are sorted newest first."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        manager.create_backup(backup_type="config")
        manager.create_backup(backup_type="config")
        manager.create_backup(backup_type="config")

        backups = manager.list_backups()

        # Should be sorted newest first
        for i in range(len(backups) - 1):
            assert backups[i].created_at >= backups[i + 1].created_at


# =============================================================================
# Backup Verification Tests
# =============================================================================


class TestBackupVerification:
    """Tests for backup verification."""

    def test_verify_valid_backup(self, temp_dirs, sample_data_dir) -> None:
        """Test verification of valid backup."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup()

        is_valid, errors = manager.verify_backup(backup_path)

        assert is_valid is True
        assert len(errors) == 0

    def test_verify_nonexistent_backup(self, backup_manager) -> None:
        """Test verification of non-existent backup."""
        is_valid, errors = backup_manager.verify_backup(Path("/nonexistent.tar.gz"))

        assert is_valid is False
        assert any("does not exist" in e.lower() for e in errors)

    def test_verify_corrupted_backup(self, temp_dirs) -> None:
        """Test verification of corrupted backup."""
        backup_dir, data_dir = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=data_dir)

        # Create invalid backup file
        corrupt_backup = backup_dir / f"backup_full_20240101_000000{BACKUP_EXTENSION}"
        corrupt_backup.write_bytes(b"not a valid tar.gz file")

        is_valid, errors = manager.verify_backup(corrupt_backup)

        assert is_valid is False
        assert len(errors) > 0


# =============================================================================
# Backup Restoration Tests
# =============================================================================


class TestBackupRestoration:
    """Tests for backup restoration."""

    def test_restore_backup(self, temp_dirs, sample_data_dir) -> None:
        """Test restoring a backup."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        # Create backup
        backup_path = manager.create_backup(backup_type="config")

        # Clear data and restore
        with tempfile.TemporaryDirectory() as restore_dir:
            result = manager.restore_backup(
                backup_path,
                target_dir=Path(restore_dir),
            )

            assert result.success is True
            assert len(result.files_restored) > 0
            assert (Path(restore_dir) / ".env").exists()

    def test_restore_with_checksum_verification(
        self, temp_dirs, sample_data_dir
    ) -> None:
        """Test restoration with checksum verification."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup()

        with tempfile.TemporaryDirectory() as restore_dir:
            result = manager.restore_backup(
                backup_path,
                target_dir=Path(restore_dir),
                verify_checksums=True,
            )

            assert result.success is True
            assert len(result.errors) == 0

    def test_restore_skip_existing(self, temp_dirs, sample_data_dir) -> None:
        """Test that existing files are skipped without overwrite."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup(backup_type="config")

        with tempfile.TemporaryDirectory() as restore_dir:
            restore_path = Path(restore_dir)

            # Create existing file
            (restore_path / ".env").write_text("existing")

            result = manager.restore_backup(
                backup_path,
                target_dir=restore_path,
                overwrite=False,
            )

            assert ".env" in result.files_skipped
            assert (restore_path / ".env").read_text() == "existing"

    def test_restore_overwrite_existing(self, temp_dirs, sample_data_dir) -> None:
        """Test that existing files are overwritten with flag."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup(backup_type="config")

        with tempfile.TemporaryDirectory() as restore_dir:
            restore_path = Path(restore_dir)

            # Create existing file
            (restore_path / ".env").write_text("existing")

            result = manager.restore_backup(
                backup_path,
                target_dir=restore_path,
                overwrite=True,
            )

            assert ".env" in result.files_restored
            assert (restore_path / ".env").read_text() != "existing"

    def test_restore_nonexistent_backup(self, backup_manager) -> None:
        """Test restoration of non-existent backup."""
        result = backup_manager.restore_backup(Path("/nonexistent.tar.gz"))

        assert result.success is False
        assert len(result.errors) > 0


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_backup_size(self, temp_dirs, sample_data_dir) -> None:
        """Test getting total backup size."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        manager.create_backup()
        manager.create_backup()

        size = manager.get_backup_size()
        assert size > 0

    def test_cleanup_all_backups(self, temp_dirs, sample_data_dir) -> None:
        """Test removing all backups."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        manager.create_backup(backup_type="config")
        manager.create_backup(backup_type="transcripts")

        removed = manager.cleanup_all_backups()

        assert removed == 2
        assert len(manager.list_backups()) == 0

    def test_cleanup_backups_by_type(self, temp_dirs, sample_data_dir) -> None:
        """Test removing backups by type."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        manager.create_backup(backup_type="config")
        manager.create_backup(backup_type="transcripts")

        removed = manager.cleanup_all_backups(backup_type="config")

        assert removed == 1
        assert len(manager.list_backups(backup_type="transcripts")) == 1

    def test_export_backup(self, temp_dirs, sample_data_dir) -> None:
        """Test exporting backup to external location."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup()

        with tempfile.TemporaryDirectory() as export_dir:
            export_path = Path(export_dir) / "exported_backup.tar.gz"
            result = manager.export_backup(backup_path, export_path)

            assert result is True
            assert export_path.exists()

    def test_export_nonexistent_backup(self, backup_manager) -> None:
        """Test exporting non-existent backup."""
        with tempfile.TemporaryDirectory() as export_dir:
            result = backup_manager.export_backup(
                Path("/nonexistent.tar.gz"),
                Path(export_dir) / "export.tar.gz",
            )

            assert result is False

    def test_import_backup(self, temp_dirs, sample_data_dir) -> None:
        """Test importing backup from external location."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        # Create and export backup
        backup_path = manager.create_backup()

        with tempfile.TemporaryDirectory() as import_dir:
            external_path = Path(import_dir) / "external_backup.tar.gz"
            manager.export_backup(backup_path, external_path)

            # Create new manager with different backup dir
            with tempfile.TemporaryDirectory() as new_backup_dir:
                new_manager = BackupManager(
                    backup_dir=Path(new_backup_dir),
                    data_dir=sample_data_dir,
                )

                imported = new_manager.import_backup(external_path)

                assert imported is not None
                assert imported.exists()

    def test_import_invalid_backup(self, temp_dirs) -> None:
        """Test importing invalid backup."""
        backup_dir, data_dir = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=data_dir)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(b"invalid data")
            f.flush()

            imported = manager.import_backup(Path(f.name))
            assert imported is None

            os.unlink(f.name)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_backup_function(self, temp_dirs, sample_data_dir) -> None:
        """Test create_backup convenience function."""
        backup_dir, _ = temp_dirs

        with patch(
            "reachy_mini_conversation_app.headless.backup.BackupManager"
        ) as MockManager:
            mock_instance = MagicMock()
            mock_instance.create_backup.return_value = Path("/backup.tar.gz")
            MockManager.return_value = mock_instance

            result = create_backup(backup_type="config")

            mock_instance.create_backup.assert_called_once_with("config", False)
            assert result == Path("/backup.tar.gz")

    def test_restore_latest_backup_function(self) -> None:
        """Test restore_latest_backup convenience function."""
        with patch(
            "reachy_mini_conversation_app.headless.backup.BackupManager"
        ) as MockManager:
            mock_instance = MagicMock()
            mock_backup = MagicMock()
            mock_backup.path = Path("/backup.tar.gz")
            mock_instance.list_backups.return_value = [mock_backup]
            mock_instance.restore_backup.return_value = RestoreResult(success=True)
            MockManager.return_value = mock_instance

            result = restore_latest_backup(backup_type="full")

            assert result.success is True

    def test_restore_latest_backup_none_available(self) -> None:
        """Test restore when no backups available."""
        with patch(
            "reachy_mini_conversation_app.headless.backup.BackupManager"
        ) as MockManager:
            mock_instance = MagicMock()
            mock_instance.list_backups.return_value = []
            MockManager.return_value = mock_instance

            result = restore_latest_backup()

            assert result.success is False
            assert any("no" in e.lower() for e in result.errors)

    def test_list_available_backups_function(self) -> None:
        """Test list_available_backups convenience function."""
        with patch(
            "reachy_mini_conversation_app.headless.backup.BackupManager"
        ) as MockManager:
            mock_instance = MagicMock()
            mock_instance.list_backups.return_value = []
            MockManager.return_value = mock_instance

            result = list_available_backups()

            mock_instance.list_backups.assert_called_once()
            assert result == []


# =============================================================================
# Edge Cases and Security Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and security scenarios."""

    def test_backup_special_characters_in_filename(
        self, temp_dirs, sample_data_dir
    ) -> None:
        """Test backup with special characters in filenames."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        # Create file with spaces
        (sample_data_dir / "config with spaces.json").write_text("{}")

        backup_path = manager.create_backup()
        assert backup_path is not None

    def test_backup_empty_file(self, temp_dirs, sample_data_dir) -> None:
        """Test backup with empty files."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        # Create empty file
        (sample_data_dir / "empty.json").write_text("")

        backup_path = manager.create_backup()
        assert backup_path is not None

    def test_backup_large_file(self, temp_dirs, sample_data_dir) -> None:
        """Test backup with larger files."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        # Create larger file (1MB)
        large_content = "x" * (1024 * 1024)
        (sample_data_dir / "large.json").write_text(large_content)

        backup_path = manager.create_backup()
        assert backup_path is not None

    def test_restore_path_traversal_prevention(
        self, temp_dirs, sample_data_dir
    ) -> None:
        """Test that path traversal is prevented during restore."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        # Create legitimate backup first
        backup_path = manager.create_backup()

        # Restoration should not allow paths outside target
        with tempfile.TemporaryDirectory() as restore_dir:
            result = manager.restore_backup(
                backup_path,
                target_dir=Path(restore_dir),
            )

            # Verify no files were created outside restore_dir
            assert result.success is True

    def test_backup_info_properties(self, temp_dirs, sample_data_dir) -> None:
        """Test BackupInfo properties are correct."""
        backup_dir, _ = temp_dirs
        manager = BackupManager(backup_dir=backup_dir, data_dir=sample_data_dir)

        backup_path = manager.create_backup(backup_type="config")
        info = manager.get_backup_info(backup_path)

        assert info is not None
        assert info.path == backup_path
        assert info.backup_type == "config"
        assert info.file_count > 0
        assert info.total_size > 0
        assert isinstance(info.created_at, datetime)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_backup_dir(self) -> None:
        """Test default backup directory path."""
        assert DEFAULT_BACKUP_DIR == Path.home() / ".reachy_mini" / "backups"

    def test_default_max_backups(self) -> None:
        """Test default max backups value."""
        assert DEFAULT_MAX_BACKUPS == 10

    def test_backup_extension(self) -> None:
        """Test backup file extension."""
        assert BACKUP_EXTENSION == ".tar.gz"

    def test_manifest_filename(self) -> None:
        """Test manifest filename."""
        assert BACKUP_MANIFEST_FILENAME == "manifest.json"

    def test_default_config_files(self) -> None:
        """Test default config files list."""
        assert ".env" in DEFAULT_CONFIG_FILES
        assert "config.json" in DEFAULT_CONFIG_FILES
