"""Backup and restore utilities for headless operation.

Provides configuration backup, conversation transcript archival,
backup rotation, and restoration capabilities.
"""

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from reachy_mini_conversation_app.headless.security import (
    SecureFileHandler,
    InputSanitizer,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_BACKUP_DIR = Path.home() / ".reachy_mini" / "backups"
DEFAULT_MAX_BACKUPS = 10
DEFAULT_CONFIG_FILES = [".env", "config.json", "settings.yaml", "settings.json"]
BACKUP_MANIFEST_FILENAME = "manifest.json"
BACKUP_EXTENSION = ".tar.gz"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BackupManifest:
    """Manifest file for a backup archive.

    Contains metadata about the backup including checksums for verification.
    """

    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    hostname: str = field(default_factory=lambda: os.uname().nodename)
    files: dict[str, dict] = field(default_factory=dict)
    total_size: int = 0
    backup_type: str = "full"  # "full", "config", "transcripts"

    def add_file(self, relative_path: str, checksum: str, size: int) -> None:
        """Add file entry to manifest.

        Args:
            relative_path: Path relative to backup root.
            checksum: SHA256 checksum of file.
            size: File size in bytes.
        """
        self.files[relative_path] = {
            "checksum": checksum,
            "size": size,
        }
        self.total_size += size

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "hostname": self.hostname,
            "files": self.files,
            "total_size": self.total_size,
            "backup_type": self.backup_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BackupManifest":
        """Create manifest from dictionary."""
        manifest = cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            hostname=data.get("hostname", ""),
            backup_type=data.get("backup_type", "full"),
        )
        manifest.files = data.get("files", {})
        manifest.total_size = data.get("total_size", 0)
        return manifest


@dataclass
class BackupInfo:
    """Information about an existing backup."""

    path: Path
    created_at: datetime
    backup_type: str
    total_size: int
    file_count: int

    @classmethod
    def from_manifest(cls, path: Path, manifest: BackupManifest) -> "BackupInfo":
        """Create BackupInfo from manifest."""
        created_at = datetime.fromisoformat(manifest.created_at)
        return cls(
            path=path,
            created_at=created_at,
            backup_type=manifest.backup_type,
            total_size=manifest.total_size,
            file_count=len(manifest.files),
        )


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    success: bool
    files_restored: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Backup Manager
# =============================================================================


class BackupManager:
    """Manages backup and restore operations.

    Handles configuration backup, transcript archival, backup rotation,
    and restoration with verification.
    """

    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        max_backups: int = DEFAULT_MAX_BACKUPS,
    ) -> None:
        """Initialize backup manager.

        Args:
            backup_dir: Directory to store backups.
            data_dir: Application data directory.
            max_backups: Maximum number of backups to retain.
        """
        self._backup_dir = backup_dir or DEFAULT_BACKUP_DIR
        self._data_dir = data_dir or (Path.home() / ".reachy_mini")
        self._max_backups = max_backups

        # Ensure backup directory exists with secure permissions
        SecureFileHandler.secure_directory(self._backup_dir)

    @property
    def backup_dir(self) -> Path:
        """Get backup directory path."""
        return self._backup_dir

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self._data_dir

    # -------------------------------------------------------------------------
    # File Operations
    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.

        Args:
            file_path: Path to file.

        Returns:
            Hex string of SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _verify_checksum(file_path: Path, expected: str) -> bool:
        """Verify file checksum matches expected value.

        Args:
            file_path: Path to file.
            expected: Expected checksum.

        Returns:
            True if checksum matches.
        """
        actual = BackupManager._calculate_checksum(file_path)
        return actual == expected

    # -------------------------------------------------------------------------
    # Backup Operations
    # -------------------------------------------------------------------------

    def _generate_backup_name(self, backup_type: str) -> str:
        """Generate unique backup filename.

        Args:
            backup_type: Type of backup.

        Returns:
            Backup filename with timestamp.
        """
        # Include milliseconds for uniqueness when creating multiple backups quickly
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"backup_{backup_type}_{timestamp}{BACKUP_EXTENSION}"

    def _collect_config_files(self) -> list[Path]:
        """Collect configuration files to backup.

        Returns:
            List of configuration file paths.
        """
        files = []

        for filename in DEFAULT_CONFIG_FILES:
            file_path = self._data_dir / filename
            if file_path.exists():
                files.append(file_path)

        return files

    def _collect_transcript_files(self) -> list[Path]:
        """Collect transcript files to backup.

        Returns:
            List of transcript file paths.
        """
        files = []

        transcripts_dir = self._data_dir / "transcripts"
        if transcripts_dir.exists():
            for file_path in transcripts_dir.rglob("*"):
                if file_path.is_file():
                    files.append(file_path)

        return files

    def _collect_audio_files(self) -> list[Path]:
        """Collect audio recording files to backup.

        Returns:
            List of audio file paths.
        """
        files = []

        recordings_dir = self._data_dir / "recordings"
        if recordings_dir.exists():
            for file_path in recordings_dir.rglob("*"):
                if file_path.is_file():
                    files.append(file_path)

        return files

    def create_backup(
        self,
        backup_type: str = "full",
        include_audio: bool = False,
    ) -> Optional[Path]:
        """Create a backup archive.

        Args:
            backup_type: Type of backup ("full", "config", "transcripts").
            include_audio: Include audio recordings (can be large).

        Returns:
            Path to created backup, or None on failure.
        """
        try:
            # Collect files based on backup type
            files_to_backup: list[Path] = []

            if backup_type == "config":
                files_to_backup = self._collect_config_files()
            elif backup_type == "transcripts":
                files_to_backup = self._collect_transcript_files()
            else:  # full
                files_to_backup = (
                    self._collect_config_files()
                    + self._collect_transcript_files()
                )
                if include_audio:
                    files_to_backup += self._collect_audio_files()

            if not files_to_backup:
                logger.warning("No files to backup")
                return None

            # Create manifest
            manifest = BackupManifest(backup_type=backup_type)

            # Generate backup filename
            backup_name = self._generate_backup_name(backup_type)
            backup_path = self._backup_dir / backup_name

            # Create tar.gz archive
            with tarfile.open(backup_path, "w:gz") as tar:
                for file_path in files_to_backup:
                    # Calculate relative path
                    try:
                        rel_path = file_path.relative_to(self._data_dir)
                    except ValueError:
                        rel_path = file_path.name

                    # Add file to archive
                    tar.add(file_path, arcname=str(rel_path))

                    # Add to manifest
                    checksum = self._calculate_checksum(file_path)
                    manifest.add_file(
                        str(rel_path),
                        checksum,
                        file_path.stat().st_size,
                    )

                # Write manifest to archive
                manifest_json = json.dumps(manifest.to_dict(), indent=2)
                manifest_bytes = manifest_json.encode("utf-8")

                # Create tarinfo for manifest
                manifest_info = tarfile.TarInfo(name=BACKUP_MANIFEST_FILENAME)
                manifest_info.size = len(manifest_bytes)

                import io
                tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

            # Secure backup file permissions
            SecureFileHandler.secure_file(backup_path, "data")

            logger.info(
                f"Created backup: {backup_path} "
                f"({len(files_to_backup)} files, {manifest.total_size} bytes)"
            )

            # Rotate old backups
            self._rotate_backups(backup_type)

            return backup_path

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def _rotate_backups(self, backup_type: str) -> None:
        """Remove old backups exceeding max_backups limit.

        Args:
            backup_type: Type of backup to rotate.
        """
        pattern = f"backup_{backup_type}_*{BACKUP_EXTENSION}"
        backups = sorted(
            self._backup_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove oldest backups exceeding limit
        for old_backup in backups[self._max_backups:]:
            try:
                old_backup.unlink()
                logger.info(f"Rotated old backup: {old_backup.name}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")

    # -------------------------------------------------------------------------
    # List and Inspect Backups
    # -------------------------------------------------------------------------

    def list_backups(self, backup_type: Optional[str] = None) -> list[BackupInfo]:
        """List available backups.

        Args:
            backup_type: Filter by backup type, or None for all.

        Returns:
            List of BackupInfo objects sorted by date (newest first).
        """
        backups = []

        if backup_type:
            pattern = f"backup_{backup_type}_*{BACKUP_EXTENSION}"
        else:
            pattern = f"backup_*{BACKUP_EXTENSION}"

        for backup_path in self._backup_dir.glob(pattern):
            try:
                manifest = self._read_manifest(backup_path)
                if manifest:
                    info = BackupInfo.from_manifest(backup_path, manifest)
                    backups.append(info)
            except Exception as e:
                logger.warning(f"Could not read backup {backup_path}: {e}")

        # Sort by date, newest first
        backups.sort(key=lambda b: b.created_at, reverse=True)
        return backups

    def _read_manifest(self, backup_path: Path) -> Optional[BackupManifest]:
        """Read manifest from backup archive.

        Args:
            backup_path: Path to backup archive.

        Returns:
            BackupManifest or None if not found.
        """
        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                manifest_member = tar.getmember(BACKUP_MANIFEST_FILENAME)
                manifest_file = tar.extractfile(manifest_member)
                if manifest_file:
                    manifest_data = json.load(manifest_file)
                    return BackupManifest.from_dict(manifest_data)
        except (KeyError, tarfile.TarError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read manifest from {backup_path}: {e}")
        return None

    def get_backup_info(self, backup_path: Path) -> Optional[BackupInfo]:
        """Get detailed information about a backup.

        Args:
            backup_path: Path to backup archive.

        Returns:
            BackupInfo or None if invalid.
        """
        manifest = self._read_manifest(backup_path)
        if manifest:
            return BackupInfo.from_manifest(backup_path, manifest)
        return None

    # -------------------------------------------------------------------------
    # Restore Operations
    # -------------------------------------------------------------------------

    def verify_backup(self, backup_path: Path) -> tuple[bool, list[str]]:
        """Verify backup archive integrity.

        Args:
            backup_path: Path to backup archive.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        if not backup_path.exists():
            return False, ["Backup file does not exist"]

        manifest = self._read_manifest(backup_path)
        if not manifest:
            return False, ["Could not read backup manifest"]

        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                # Verify each file in manifest exists in archive
                for rel_path in manifest.files:
                    try:
                        tar.getmember(rel_path)
                    except KeyError:
                        errors.append(f"Missing file in archive: {rel_path}")

            if errors:
                return False, errors

            return True, []

        except tarfile.TarError as e:
            return False, [f"Archive error: {e}"]

    def restore_backup(
        self,
        backup_path: Path,
        target_dir: Optional[Path] = None,
        verify_checksums: bool = True,
        overwrite: bool = False,
    ) -> RestoreResult:
        """Restore files from a backup archive.

        Args:
            backup_path: Path to backup archive.
            target_dir: Directory to restore to (default: data_dir).
            verify_checksums: Verify file checksums after extraction.
            overwrite: Overwrite existing files.

        Returns:
            RestoreResult with details of the operation.
        """
        result = RestoreResult(success=True)
        target = target_dir or self._data_dir

        # Verify backup first
        is_valid, errors = self.verify_backup(backup_path)
        if not is_valid:
            result.success = False
            result.errors = errors
            return result

        manifest = self._read_manifest(backup_path)
        if not manifest:
            result.success = False
            result.errors.append("Could not read manifest")
            return result

        try:
            # Use a temporary directory for safe extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract to temporary directory
                with tarfile.open(backup_path, "r:gz") as tar:
                    # Security: filter for safe extraction
                    def safe_filter(tarinfo, path):
                        # Skip the manifest file during extraction
                        if tarinfo.name == BACKUP_MANIFEST_FILENAME:
                            return None

                        # Validate the archive member path is safe
                        # Prevent path traversal (../) and absolute paths
                        name = tarinfo.name
                        if name.startswith("/") or ".." in name:
                            logger.warning(
                                f"Skipping unsafe path: {tarinfo.name}"
                            )
                            return None

                        # Ensure the resolved path stays within temp_path
                        target_path = (temp_path / name).resolve()
                        if not str(target_path).startswith(str(temp_path.resolve())):
                            logger.warning(
                                f"Skipping path traversal attempt: {tarinfo.name}"
                            )
                            return None

                        return tarinfo

                    tar.extractall(temp_path, filter=safe_filter)

                # Verify checksums and move files
                for rel_path, file_info in manifest.files.items():
                    temp_file = temp_path / rel_path
                    target_file = target / rel_path

                    if not temp_file.exists():
                        result.errors.append(f"File not extracted: {rel_path}")
                        continue

                    # Verify checksum if requested
                    if verify_checksums:
                        if not self._verify_checksum(
                            temp_file, file_info["checksum"]
                        ):
                            result.errors.append(
                                f"Checksum mismatch: {rel_path}"
                            )
                            result.success = False
                            continue

                    # Check if target exists
                    if target_file.exists() and not overwrite:
                        result.files_skipped.append(rel_path)
                        continue

                    # Create parent directories
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    # Move file to target
                    shutil.copy2(temp_file, target_file)

                    # Secure permissions
                    SecureFileHandler.secure_file(target_file, "data")

                    result.files_restored.append(rel_path)

            logger.info(
                f"Restored {len(result.files_restored)} files from {backup_path}"
            )

            if result.errors:
                result.success = False

            return result

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            result.success = False
            result.errors.append(str(e))
            return result

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_backup_size(self) -> int:
        """Get total size of all backups in bytes.

        Returns:
            Total backup size in bytes.
        """
        total = 0
        for backup in self._backup_dir.glob(f"*{BACKUP_EXTENSION}"):
            total += backup.stat().st_size
        return total

    def cleanup_all_backups(self, backup_type: Optional[str] = None) -> int:
        """Remove all backups of specified type.

        Args:
            backup_type: Type to remove, or None for all.

        Returns:
            Number of backups removed.
        """
        count = 0

        if backup_type:
            pattern = f"backup_{backup_type}_*{BACKUP_EXTENSION}"
        else:
            pattern = f"backup_*{BACKUP_EXTENSION}"

        for backup in self._backup_dir.glob(pattern):
            try:
                backup.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {backup}: {e}")

        logger.info(f"Removed {count} backups")
        return count

    def export_backup(
        self,
        backup_path: Path,
        destination: Path,
    ) -> bool:
        """Export a backup to an external location.

        Args:
            backup_path: Path to backup archive.
            destination: Destination path.

        Returns:
            True if export succeeded.
        """
        if not backup_path.exists():
            logger.error(f"Backup does not exist: {backup_path}")
            return False

        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Copy backup
            shutil.copy2(backup_path, destination)
            logger.info(f"Exported backup to {destination}")
            return True

        except Exception as e:
            logger.error(f"Failed to export backup: {e}")
            return False

    def import_backup(
        self,
        source: Path,
    ) -> Optional[Path]:
        """Import a backup from an external location.

        Args:
            source: Source backup path.

        Returns:
            Path to imported backup, or None on failure.
        """
        if not source.exists():
            logger.error(f"Source backup does not exist: {source}")
            return None

        try:
            # Verify it's a valid backup
            is_valid, errors = self.verify_backup(source)
            if not is_valid:
                logger.error(f"Invalid backup: {errors}")
                return None

            # Copy to backup directory
            destination = self._backup_dir / source.name

            # Avoid overwriting existing backup
            if destination.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{source.stem}_{timestamp}{BACKUP_EXTENSION}"
                destination = self._backup_dir / new_name

            shutil.copy2(source, destination)
            SecureFileHandler.secure_file(destination, "data")

            logger.info(f"Imported backup: {destination}")
            return destination

        except Exception as e:
            logger.error(f"Failed to import backup: {e}")
            return None


# =============================================================================
# Convenience Functions
# =============================================================================


def create_backup(
    backup_type: str = "full",
    include_audio: bool = False,
) -> Optional[Path]:
    """Create a backup using default settings.

    Args:
        backup_type: Type of backup.
        include_audio: Include audio recordings.

    Returns:
        Path to backup or None on failure.
    """
    manager = BackupManager()
    return manager.create_backup(backup_type, include_audio)


def restore_latest_backup(
    backup_type: str = "full",
    overwrite: bool = False,
) -> RestoreResult:
    """Restore the most recent backup.

    Args:
        backup_type: Type of backup to restore.
        overwrite: Overwrite existing files.

    Returns:
        RestoreResult with operation details.
    """
    manager = BackupManager()
    backups = manager.list_backups(backup_type)

    if not backups:
        return RestoreResult(
            success=False,
            errors=[f"No {backup_type} backups found"],
        )

    latest = backups[0]
    return manager.restore_backup(latest.path, overwrite=overwrite)


def list_available_backups() -> list[BackupInfo]:
    """List all available backups.

    Returns:
        List of BackupInfo objects.
    """
    manager = BackupManager()
    return manager.list_backups()
