"""Security utilities for headless operation.

Provides API key validation, secure file handling, input sanitization,
and logging protection for safe operation without user intervention.
"""

import os
import re
import stat
import hashlib
import secrets
import logging
from pathlib import Path
from typing import Any, Optional
from functools import wraps
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


# =============================================================================
# API Key Validation
# =============================================================================


class APIKeyValidator:
    """Validates API keys for format and structure.

    Does NOT validate keys against the actual API - only checks format.
    Actual authentication is done by the API itself.
    """

    # Known API key patterns (prefix + length)
    PATTERNS = {
        "openai": {
            "prefix": "sk-",
            "min_length": 48,
            "max_length": 256,  # Allow for longer project keys
            "pattern": r"^sk-[a-zA-Z0-9_-]+$",
        },
        "anthropic": {
            "prefix": "sk-ant-",
            "min_length": 80,
            "max_length": 150,
            "pattern": r"^sk-ant-[a-zA-Z0-9_-]+$",
        },
    }

    @classmethod
    def validate_openai_key(cls, key: Optional[str]) -> tuple[bool, str]:
        """Validate OpenAI API key format.

        Args:
            key: API key to validate.

        Returns:
            Tuple of (is_valid, message).
        """
        if not key:
            return False, "API key is empty or None"

        key = key.strip()

        if not key.startswith("sk-"):
            return False, "OpenAI key must start with 'sk-'"

        pattern = cls.PATTERNS["openai"]

        if len(key) < pattern["min_length"]:
            return False, f"Key too short (min {pattern['min_length']} chars)"

        if len(key) > pattern["max_length"]:
            return False, f"Key too long (max {pattern['max_length']} chars)"

        if not re.match(pattern["pattern"], key):
            return False, "Key contains invalid characters"

        return True, "Valid format"

    @classmethod
    def mask_key(cls, key: Optional[str], visible_chars: int = 8) -> str:
        """Mask API key for safe logging.

        Args:
            key: API key to mask.
            visible_chars: Number of characters to show at end.

        Returns:
            Masked key string.
        """
        if not key:
            return "<empty>"

        key = key.strip()
        if len(key) <= visible_chars:
            return "*" * len(key)

        return "*" * (len(key) - visible_chars) + key[-visible_chars:]

    @classmethod
    def get_key_hash(cls, key: Optional[str]) -> str:
        """Get SHA256 hash of key for logging/comparison.

        Args:
            key: API key to hash.

        Returns:
            First 16 chars of SHA256 hash.
        """
        if not key:
            return "<empty>"

        key_hash = hashlib.sha256(key.strip().encode()).hexdigest()
        return key_hash[:16]


# =============================================================================
# Secure File Handling
# =============================================================================


class SecureFileHandler:
    """Handles files with security-focused permissions.

    Ensures sensitive files have appropriate permissions and ownership.
    """

    # Recommended permissions for different file types
    PERMISSIONS = {
        "env_file": 0o600,  # Owner read/write only
        "config": 0o644,  # Owner read/write, others read
        "log": 0o640,  # Owner read/write, group read
        "data": 0o600,  # Owner read/write only
        "executable": 0o700,  # Owner all, others none
        "directory": 0o700,  # Owner all, others none
    }

    @classmethod
    def secure_directory(cls, path: Path, create: bool = True) -> bool:
        """Ensure directory exists with secure permissions.

        Args:
            path: Directory path.
            create: Create if doesn't exist.

        Returns:
            True if directory is secure.
        """
        try:
            if not path.exists():
                if create:
                    path.mkdir(parents=True, mode=cls.PERMISSIONS["directory"])
                    logger.info(f"Created secure directory: {path}")
                else:
                    return False

            # Check and fix permissions
            current_mode = path.stat().st_mode & 0o777
            expected_mode = cls.PERMISSIONS["directory"]

            if current_mode != expected_mode:
                path.chmod(expected_mode)
                logger.info(
                    f"Fixed permissions on {path}: "
                    f"{oct(current_mode)} -> {oct(expected_mode)}"
                )

            return True

        except (OSError, PermissionError) as e:
            logger.error(f"Failed to secure directory {path}: {e}")
            return False

    @classmethod
    def secure_file(
        cls,
        path: Path,
        file_type: str = "data",
    ) -> bool:
        """Ensure file has secure permissions.

        Args:
            path: File path.
            file_type: Type of file for permission lookup.

        Returns:
            True if file is secure.
        """
        if not path.exists():
            return False

        try:
            expected_mode = cls.PERMISSIONS.get(file_type, 0o600)
            current_mode = path.stat().st_mode & 0o777

            if current_mode != expected_mode:
                path.chmod(expected_mode)
                logger.info(
                    f"Fixed permissions on {path}: "
                    f"{oct(current_mode)} -> {oct(expected_mode)}"
                )

            return True

        except (OSError, PermissionError) as e:
            logger.error(f"Failed to secure file {path}: {e}")
            return False

    @classmethod
    def write_secure(
        cls,
        path: Path,
        content: str,
        file_type: str = "data",
    ) -> bool:
        """Write content to file with secure permissions.

        Uses atomic write (temp file + rename) to prevent partial writes.

        Args:
            path: File path.
            content: Content to write.
            file_type: Type of file for permission lookup.

        Returns:
            True if write succeeded.
        """
        try:
            # Ensure parent directory is secure
            cls.secure_directory(path.parent)

            # Write to temp file first (atomic)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(content)

            # Set permissions before renaming
            temp_path.chmod(cls.PERMISSIONS.get(file_type, 0o600))

            # Atomic rename
            temp_path.rename(path)

            return True

        except Exception as e:
            logger.error(f"Failed to write secure file {path}: {e}")
            # Clean up temp file
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    @classmethod
    def is_world_readable(cls, path: Path) -> bool:
        """Check if file is world-readable (potential security issue).

        Args:
            path: File path.

        Returns:
            True if others have read permission.
        """
        if not path.exists():
            return False

        mode = path.stat().st_mode
        return bool(mode & stat.S_IROTH)

    @classmethod
    def check_env_file_security(cls, path: Path) -> list[str]:
        """Check .env file for security issues.

        Args:
            path: Path to .env file.

        Returns:
            List of security warnings.
        """
        warnings = []

        if not path.exists():
            return warnings

        # Check permissions
        if cls.is_world_readable(path):
            warnings.append(f"WARNING: {path} is world-readable!")

        # Check contents for common issues
        try:
            content = path.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Skip comments and empty lines
                if not line.strip() or line.strip().startswith("#"):
                    continue

                # Check for unquoted values with spaces
                if "=" in line:
                    key, _, value = line.partition("=")
                    value = value.strip()

                    # Check for accidentally committed real keys
                    if value.startswith("sk-") and len(value) > 20:
                        warnings.append(
                            f"Line {i}: Possible real API key detected. "
                            "Consider using environment variables."
                        )

        except Exception as e:
            warnings.append(f"Could not read file: {e}")

        return warnings


# =============================================================================
# Input Sanitization
# =============================================================================


class InputSanitizer:
    """Sanitizes user input to prevent injection attacks.

    Provides various sanitization methods for different input types.
    """

    # Maximum lengths for different input types
    MAX_LENGTHS = {
        "filename": 255,
        "path": 4096,
        "command": 1024,
        "text": 10000,
        "short_text": 500,
    }

    # Dangerous characters for different contexts
    DANGEROUS_CHARS = {
        "filename": r'[<>:"/\\|?*\x00-\x1f]',
        "path": r'[\x00-\x1f]',  # Only control chars for paths
        "shell": r'[;&|`$(){}[\]<>]',
        "sql": r"[';\"\\]",
    }

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent path traversal and special chars.

        Args:
            filename: Raw filename.

        Returns:
            Sanitized filename.
        """
        if not filename:
            return "unnamed"

        # Remove path separators (prevent traversal)
        filename = os.path.basename(filename)

        # Remove dangerous characters
        filename = re.sub(cls.DANGEROUS_CHARS["filename"], "_", filename)

        # Limit length
        max_len = cls.MAX_LENGTHS["filename"]
        if len(filename) > max_len:
            # Keep extension
            name, ext = os.path.splitext(filename)
            filename = name[: max_len - len(ext)] + ext

        # Ensure not empty after sanitization
        if not filename or filename in (".", ".."):
            filename = "unnamed"

        return filename

    @classmethod
    def sanitize_path(cls, path: str, base_dir: Optional[Path] = None) -> Optional[Path]:
        """Sanitize path and ensure it's within allowed directory.

        Args:
            path: Raw path string.
            base_dir: Base directory path must be within. None allows any path.

        Returns:
            Sanitized Path object, or None if invalid.
        """
        if not path:
            return None

        try:
            # Resolve to absolute path
            resolved = Path(path).resolve()

            # Check if within base directory
            if base_dir:
                base = base_dir.resolve()
                if not str(resolved).startswith(str(base)):
                    logger.warning(
                        f"Path traversal attempt blocked: {path} -> {resolved}"
                    )
                    return None

            return resolved

        except Exception as e:
            logger.warning(f"Invalid path: {path}: {e}")
            return None

    @classmethod
    def sanitize_text(cls, text: str, max_length: Optional[int] = None) -> str:
        """Sanitize text input (remove control characters).

        Args:
            text: Raw text.
            max_length: Maximum length (default from MAX_LENGTHS).

        Returns:
            Sanitized text.
        """
        if not text:
            return ""

        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Limit length
        limit = max_length or cls.MAX_LENGTHS["text"]
        if len(text) > limit:
            text = text[:limit]

        return text

    @classmethod
    def is_safe_command(cls, command: str) -> bool:
        """Check if command is safe to execute (no shell injection).

        Args:
            command: Command string to check.

        Returns:
            True if command appears safe.
        """
        if not command:
            return True

        # Check for dangerous shell characters
        if re.search(cls.DANGEROUS_CHARS["shell"], command):
            return False

        # Check for command substitution
        if "$(" in command or "`" in command:
            return False

        return True


# =============================================================================
# Secure Logging
# =============================================================================


class SecureLogger:
    """Logging utilities that prevent sensitive data leakage.

    Wraps logging to automatically redact sensitive information.
    """

    # Patterns to redact from logs
    SENSITIVE_PATTERNS = [
        # API keys
        (r"sk-[a-zA-Z0-9_-]{20,}", "<REDACTED_API_KEY>"),
        (r"sk-ant-[a-zA-Z0-9_-]{20,}", "<REDACTED_API_KEY>"),
        # Email addresses (optional)
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "<REDACTED_EMAIL>"),
        # JWT tokens
        (r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*", "<REDACTED_JWT>"),
        # Bearer tokens
        (r"Bearer\s+[a-zA-Z0-9_-]+", "Bearer <REDACTED>"),
        # Common password patterns in URLs
        (r"password=[^&\s]+", "password=<REDACTED>"),
        (r"passwd=[^&\s]+", "passwd=<REDACTED>"),
        (r"secret=[^&\s]+", "secret=<REDACTED>"),
        (r"token=[^&\s]+", "token=<REDACTED>"),
        (r"api_key=[^&\s]+", "api_key=<REDACTED>"),
    ]

    @classmethod
    def redact(cls, message: str) -> str:
        """Redact sensitive information from a message.

        Args:
            message: Message to redact.

        Returns:
            Redacted message.
        """
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            message = re.sub(pattern, replacement, message)
        return message

    @classmethod
    def log_safe(cls, logger: logging.Logger, level: int, message: str) -> None:
        """Log message with sensitive data redacted.

        Args:
            logger: Logger instance.
            level: Logging level.
            message: Message to log.
        """
        logger.log(level, cls.redact(message))


class SecureLoggingFilter(logging.Filter):
    """Logging filter that redacts sensitive information.

    Add to handlers to automatically redact all log messages.

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.addFilter(SecureLoggingFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact log record.

        Args:
            record: Log record to filter.

        Returns:
            True (always allow, but modify message).
        """
        if isinstance(record.msg, str):
            record.msg = SecureLogger.redact(record.msg)

        # Also redact args if present
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(SecureLogger.redact(arg))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)

        return True


# =============================================================================
# Rate Limiting
# =============================================================================


class RateLimiter:
    """Simple in-memory rate limiter.

    Limits operations per time window using token bucket algorithm.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
    ) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window.
            window_seconds: Time window in seconds.
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: list[datetime] = []

    def _cleanup(self) -> None:
        """Remove expired request timestamps."""
        cutoff = datetime.now() - timedelta(seconds=self._window_seconds)
        self._requests = [t for t in self._requests if t > cutoff]

    def allow(self) -> bool:
        """Check if request is allowed.

        Returns:
            True if request is allowed, False if rate limited.
        """
        self._cleanup()

        if len(self._requests) >= self._max_requests:
            return False

        self._requests.append(datetime.now())
        return True

    def remaining(self) -> int:
        """Get remaining requests in current window.

        Returns:
            Number of remaining allowed requests.
        """
        self._cleanup()
        return max(0, self._max_requests - len(self._requests))

    def reset_at(self) -> Optional[datetime]:
        """Get time when rate limit resets.

        Returns:
            Datetime when oldest request expires, or None if no active requests.
        """
        self._cleanup()
        if not self._requests:
            return None
        return self._requests[0] + timedelta(seconds=self._window_seconds)


def rate_limited(
    max_requests: int,
    window_seconds: float = 60.0,
    key_func: Optional[callable] = None,
) -> callable:
    """Decorator to rate limit a function.

    Args:
        max_requests: Maximum calls per window.
        window_seconds: Time window in seconds.
        key_func: Optional function to get rate limit key from args.

    Returns:
        Decorated function.

    Example:
        >>> @rate_limited(max_requests=10, window_seconds=60)
        ... def call_api():
        ...     return api.request()
    """
    limiters: dict[str, RateLimiter] = {}

    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"

            # Get or create limiter for this key
            if key not in limiters:
                limiters[key] = RateLimiter(max_requests, window_seconds)

            limiter = limiters[key]

            if not limiter.allow():
                reset_time = limiter.reset_at()
                wait_seconds = (
                    (reset_time - datetime.now()).total_seconds()
                    if reset_time
                    else window_seconds
                )
                raise RuntimeError(
                    f"Rate limit exceeded. Try again in {wait_seconds:.1f}s"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Security Audit
# =============================================================================


class SecurityAuditor:
    """Performs security audits on the system configuration.

    Checks for common security misconfigurations.
    """

    @classmethod
    def audit_environment(cls) -> list[tuple[str, str, str]]:
        """Audit environment for security issues.

        Returns:
            List of (severity, issue, recommendation) tuples.
        """
        issues = []

        # Check for API keys in environment
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            valid, msg = APIKeyValidator.validate_openai_key(api_key)
            if not valid:
                issues.append((
                    "ERROR",
                    f"Invalid OPENAI_API_KEY format: {msg}",
                    "Check your API key configuration",
                ))
        else:
            issues.append((
                "WARNING",
                "OPENAI_API_KEY not set",
                "Set OPENAI_API_KEY environment variable or in .env file",
            ))

        # Check common insecure settings
        if os.environ.get("DEBUG", "").lower() in ("true", "1", "yes"):
            issues.append((
                "WARNING",
                "DEBUG mode is enabled",
                "Disable DEBUG in production",
            ))

        # Check home directory permissions
        home = Path.home()
        reachy_dir = home / ".reachy_mini"
        if reachy_dir.exists():
            if SecureFileHandler.is_world_readable(reachy_dir):
                issues.append((
                    "ERROR",
                    f"{reachy_dir} is world-readable",
                    f"Run: chmod 700 {reachy_dir}",
                ))

            # Check .env file
            env_file = reachy_dir / ".env"
            if env_file.exists():
                warnings = SecureFileHandler.check_env_file_security(env_file)
                for warning in warnings:
                    issues.append(("WARNING", warning, "Review .env file security"))

        return issues

    @classmethod
    def audit_file_permissions(cls, directory: Path) -> list[tuple[str, str, str]]:
        """Audit file permissions in a directory.

        Args:
            directory: Directory to audit.

        Returns:
            List of (severity, issue, recommendation) tuples.
        """
        issues = []

        if not directory.exists():
            return issues

        # Sensitive file extensions and names
        sensitive_suffixes = {".env", ".key", ".pem", ".json"}
        sensitive_names = {".env", ".env.local", ".env.production", ".env.development"}

        for path in directory.rglob("*"):
            if path.is_file():
                # Check for world-readable sensitive files
                # Check both suffix and full name (for dotfiles like .env)
                is_sensitive = (
                    path.suffix in sensitive_suffixes
                    or path.name in sensitive_names
                )
                if is_sensitive and SecureFileHandler.is_world_readable(path):
                    issues.append((
                        "ERROR",
                        f"Sensitive file world-readable: {path}",
                        f"Run: chmod 600 {path}",
                    ))

        return issues

    @classmethod
    def generate_report(cls) -> str:
        """Generate a full security audit report.

        Returns:
            Formatted audit report.
        """
        lines = [
            "=" * 60,
            "SECURITY AUDIT REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
        ]

        # Environment audit
        lines.append("ENVIRONMENT AUDIT")
        lines.append("-" * 40)
        env_issues = cls.audit_environment()
        if env_issues:
            for severity, issue, recommendation in env_issues:
                lines.append(f"[{severity}] {issue}")
                lines.append(f"  -> {recommendation}")
        else:
            lines.append("No issues found.")
        lines.append("")

        # File permissions audit
        reachy_dir = Path.home() / ".reachy_mini"
        if reachy_dir.exists():
            lines.append("FILE PERMISSIONS AUDIT")
            lines.append("-" * 40)
            file_issues = cls.audit_file_permissions(reachy_dir)
            if file_issues:
                for severity, issue, recommendation in file_issues:
                    lines.append(f"[{severity}] {issue}")
                    lines.append(f"  -> {recommendation}")
            else:
                lines.append("No issues found.")
        lines.append("")

        # Summary
        all_issues = env_issues + (file_issues if reachy_dir.exists() else [])
        errors = sum(1 for s, _, _ in all_issues if s == "ERROR")
        warnings = sum(1 for s, _, _ in all_issues if s == "WARNING")

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Errors: {errors}")
        lines.append(f"Warnings: {warnings}")
        if errors > 0:
            lines.append("\nACTION REQUIRED: Fix errors before production deployment!")
        elif warnings > 0:
            lines.append("\nReview warnings to improve security posture.")
        else:
            lines.append("\nNo security issues detected.")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
