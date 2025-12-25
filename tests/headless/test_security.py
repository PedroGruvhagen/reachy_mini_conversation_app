"""Tests for the Security module.

Tests for API key validation, secure file handling, input sanitization,
secure logging, rate limiting, and security auditing.
"""

import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from reachy_mini_conversation_app.headless.security import (
    APIKeyValidator,
    SecureFileHandler,
    InputSanitizer,
    SecureLogger,
    SecureLoggingFilter,
    RateLimiter,
    rate_limited,
    SecurityAuditor,
)


# =============================================================================
# API Key Validator Tests
# =============================================================================


class TestAPIKeyValidator:
    """Tests for API key validation."""

    def test_validate_openai_key_valid(self) -> None:
        """Test validation of a valid OpenAI key format."""
        # Valid key with correct prefix and length
        valid_key = "sk-" + "a" * 48
        is_valid, message = APIKeyValidator.validate_openai_key(valid_key)
        assert is_valid is True
        assert message == "Valid format"

    def test_validate_openai_key_empty(self) -> None:
        """Test validation of empty key."""
        is_valid, message = APIKeyValidator.validate_openai_key(None)
        assert is_valid is False
        assert "empty" in message.lower()

        is_valid, message = APIKeyValidator.validate_openai_key("")
        assert is_valid is False
        assert "empty" in message.lower()

    def test_validate_openai_key_wrong_prefix(self) -> None:
        """Test validation of key with wrong prefix."""
        invalid_key = "pk-" + "a" * 48
        is_valid, message = APIKeyValidator.validate_openai_key(invalid_key)
        assert is_valid is False
        assert "sk-" in message

    def test_validate_openai_key_too_short(self) -> None:
        """Test validation of key that's too short."""
        short_key = "sk-abc123"
        is_valid, message = APIKeyValidator.validate_openai_key(short_key)
        assert is_valid is False
        assert "short" in message.lower()

    def test_validate_openai_key_too_long(self) -> None:
        """Test validation of key that's too long."""
        long_key = "sk-" + "a" * 300
        is_valid, message = APIKeyValidator.validate_openai_key(long_key)
        assert is_valid is False
        assert "long" in message.lower()

    def test_validate_openai_key_invalid_characters(self) -> None:
        """Test validation of key with invalid characters."""
        invalid_key = "sk-" + "a" * 40 + "!@#$%"
        is_valid, message = APIKeyValidator.validate_openai_key(invalid_key)
        assert is_valid is False
        assert "invalid" in message.lower()

    def test_validate_openai_key_with_whitespace(self) -> None:
        """Test validation strips whitespace."""
        valid_key = "  sk-" + "a" * 48 + "  "
        is_valid, message = APIKeyValidator.validate_openai_key(valid_key)
        assert is_valid is True

    def test_mask_key_valid(self) -> None:
        """Test masking a valid key."""
        key = "sk-1234567890abcdef"
        masked = APIKeyValidator.mask_key(key)
        assert masked.endswith("abcdef")
        assert masked.startswith("*")
        assert "1234567890" not in masked

    def test_mask_key_empty(self) -> None:
        """Test masking empty key."""
        assert APIKeyValidator.mask_key(None) == "<empty>"
        assert APIKeyValidator.mask_key("") == "<empty>"

    def test_mask_key_short(self) -> None:
        """Test masking key shorter than visible chars."""
        short_key = "abc"
        masked = APIKeyValidator.mask_key(short_key)
        assert masked == "***"

    def test_mask_key_custom_visible_chars(self) -> None:
        """Test masking with custom visible characters."""
        key = "sk-1234567890abcdef"
        masked = APIKeyValidator.mask_key(key, visible_chars=4)
        assert masked.endswith("cdef")

    def test_get_key_hash_valid(self) -> None:
        """Test getting hash of valid key."""
        key = "sk-test12345"
        hash1 = APIKeyValidator.get_key_hash(key)
        hash2 = APIKeyValidator.get_key_hash(key)

        # Same key should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA256

    def test_get_key_hash_empty(self) -> None:
        """Test getting hash of empty key."""
        assert APIKeyValidator.get_key_hash(None) == "<empty>"
        assert APIKeyValidator.get_key_hash("") == "<empty>"

    def test_get_key_hash_different_keys(self) -> None:
        """Test that different keys produce different hashes."""
        hash1 = APIKeyValidator.get_key_hash("sk-key1")
        hash2 = APIKeyValidator.get_key_hash("sk-key2")
        assert hash1 != hash2


# =============================================================================
# Secure File Handler Tests
# =============================================================================


class TestSecureFileHandler:
    """Tests for secure file handling."""

    def test_secure_directory_creates_new(self) -> None:
        """Test creating a new secure directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "secure_test"
            result = SecureFileHandler.secure_directory(new_dir, create=True)

            assert result is True
            assert new_dir.exists()
            assert (new_dir.stat().st_mode & 0o777) == 0o700

    def test_secure_directory_no_create(self) -> None:
        """Test that non-existent directory returns False when create=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent = Path(tmpdir) / "does_not_exist"
            result = SecureFileHandler.secure_directory(non_existent, create=False)
            assert result is False

    def test_secure_directory_fixes_permissions(self) -> None:
        """Test that directory permissions are fixed if wrong."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test"
            test_dir.mkdir(mode=0o755)

            result = SecureFileHandler.secure_directory(test_dir)
            assert result is True
            assert (test_dir.stat().st_mode & 0o777) == 0o700

    def test_secure_file_fixes_permissions(self) -> None:
        """Test that file permissions are fixed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")
            test_file.chmod(0o644)

            result = SecureFileHandler.secure_file(test_file, "data")
            assert result is True
            assert (test_file.stat().st_mode & 0o777) == 0o600

    def test_secure_file_nonexistent(self) -> None:
        """Test securing non-existent file returns False."""
        result = SecureFileHandler.secure_file(Path("/nonexistent/file.txt"))
        assert result is False

    def test_secure_file_different_types(self) -> None:
        """Test different file types get different permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for file_type, expected_mode in [
                ("env_file", 0o600),
                ("config", 0o644),
                ("log", 0o640),
                ("executable", 0o700),
            ]:
                test_file = Path(tmpdir) / f"test_{file_type}"
                test_file.write_text("test")

                SecureFileHandler.secure_file(test_file, file_type)
                actual_mode = test_file.stat().st_mode & 0o777
                assert actual_mode == expected_mode, f"Failed for {file_type}"

    def test_write_secure_creates_file(self) -> None:
        """Test secure write creates file with correct permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "secure_write.txt"
            result = SecureFileHandler.write_secure(test_file, "secret content")

            assert result is True
            assert test_file.exists()
            assert test_file.read_text() == "secret content"
            assert (test_file.stat().st_mode & 0o777) == 0o600

    def test_write_secure_atomic(self) -> None:
        """Test that write_secure uses atomic operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "atomic.txt"
            test_file.write_text("original")

            # Write new content
            SecureFileHandler.write_secure(test_file, "new content")

            # Temp file should not exist
            temp_file = test_file.with_suffix(".txt.tmp")
            assert not temp_file.exists()
            assert test_file.read_text() == "new content"

    def test_is_world_readable_true(self) -> None:
        """Test detection of world-readable files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "world_readable.txt"
            test_file.write_text("test")
            test_file.chmod(0o644)

            assert SecureFileHandler.is_world_readable(test_file) is True

    def test_is_world_readable_false(self) -> None:
        """Test detection of non-world-readable files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "private.txt"
            test_file.write_text("test")
            test_file.chmod(0o600)

            assert SecureFileHandler.is_world_readable(test_file) is False

    def test_is_world_readable_nonexistent(self) -> None:
        """Test is_world_readable for non-existent file."""
        assert SecureFileHandler.is_world_readable(Path("/nonexistent")) is False

    def test_check_env_file_security_world_readable(self) -> None:
        """Test env file security check detects world-readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("SOME_VAR=value")
            env_file.chmod(0o644)

            warnings = SecureFileHandler.check_env_file_security(env_file)
            assert any("world-readable" in w.lower() for w in warnings)

    def test_check_env_file_security_detects_api_key(self) -> None:
        """Test env file security check detects potential API keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("OPENAI_API_KEY=sk-1234567890abcdef1234567890")
            env_file.chmod(0o600)

            warnings = SecureFileHandler.check_env_file_security(env_file)
            assert any("api key" in w.lower() for w in warnings)

    def test_check_env_file_security_nonexistent(self) -> None:
        """Test env file security check for non-existent file."""
        warnings = SecureFileHandler.check_env_file_security(Path("/nonexistent"))
        assert len(warnings) == 0


# =============================================================================
# Input Sanitizer Tests
# =============================================================================


class TestInputSanitizer:
    """Tests for input sanitization."""

    def test_sanitize_filename_basic(self) -> None:
        """Test basic filename sanitization."""
        assert InputSanitizer.sanitize_filename("valid_file.txt") == "valid_file.txt"

    def test_sanitize_filename_empty(self) -> None:
        """Test sanitization of empty filename."""
        assert InputSanitizer.sanitize_filename("") == "unnamed"
        assert InputSanitizer.sanitize_filename(None) == "unnamed"

    def test_sanitize_filename_path_traversal(self) -> None:
        """Test that path traversal is prevented."""
        assert InputSanitizer.sanitize_filename("../../../etc/passwd") == "passwd"
        assert InputSanitizer.sanitize_filename("/etc/passwd") == "passwd"

    def test_sanitize_filename_dangerous_chars(self) -> None:
        """Test removal of dangerous characters."""
        result = InputSanitizer.sanitize_filename('file<>:"|?*.txt')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_sanitize_filename_length_limit(self) -> None:
        """Test filename length limiting."""
        long_name = "a" * 300 + ".txt"
        result = InputSanitizer.sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith(".txt")

    def test_sanitize_filename_dots(self) -> None:
        """Test that . and .. are replaced."""
        assert InputSanitizer.sanitize_filename(".") == "unnamed"
        assert InputSanitizer.sanitize_filename("..") == "unnamed"

    def test_sanitize_path_valid(self) -> None:
        """Test sanitization of valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = InputSanitizer.sanitize_path(tmpdir)
            assert result is not None
            assert result.exists()

    def test_sanitize_path_empty(self) -> None:
        """Test sanitization of empty path."""
        assert InputSanitizer.sanitize_path("") is None
        assert InputSanitizer.sanitize_path(None) is None

    def test_sanitize_path_traversal_blocked(self) -> None:
        """Test that path traversal is blocked when base_dir specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            traversal_path = str(base / ".." / ".." / "etc" / "passwd")

            result = InputSanitizer.sanitize_path(traversal_path, base_dir=base)
            assert result is None

    def test_sanitize_path_within_base(self) -> None:
        """Test that paths within base_dir are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir).resolve()
            subdir = base / "subdir"
            subdir.mkdir()

            result = InputSanitizer.sanitize_path(str(subdir), base_dir=base)
            assert result is not None
            # Both paths resolved, so compare resolved versions
            assert result.resolve() == subdir.resolve()

    def test_sanitize_text_basic(self) -> None:
        """Test basic text sanitization."""
        assert InputSanitizer.sanitize_text("Hello, World!") == "Hello, World!"

    def test_sanitize_text_empty(self) -> None:
        """Test sanitization of empty text."""
        assert InputSanitizer.sanitize_text("") == ""
        assert InputSanitizer.sanitize_text(None) == ""

    def test_sanitize_text_control_chars(self) -> None:
        """Test removal of control characters."""
        text_with_control = "Hello\x00World\x07!"
        result = InputSanitizer.sanitize_text(text_with_control)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "HelloWorld!" in result

    def test_sanitize_text_preserves_newlines(self) -> None:
        """Test that newlines and tabs are preserved."""
        text = "Line 1\nLine 2\tTabbed"
        result = InputSanitizer.sanitize_text(text)
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_text_length_limit(self) -> None:
        """Test text length limiting."""
        long_text = "a" * 20000
        result = InputSanitizer.sanitize_text(long_text)
        assert len(result) <= 10000

    def test_sanitize_text_custom_length(self) -> None:
        """Test text with custom max length."""
        text = "a" * 1000
        result = InputSanitizer.sanitize_text(text, max_length=100)
        assert len(result) == 100

    def test_is_safe_command_valid(self) -> None:
        """Test validation of safe commands."""
        assert InputSanitizer.is_safe_command("ls -la") is True
        assert InputSanitizer.is_safe_command("echo hello") is True
        assert InputSanitizer.is_safe_command("") is True

    def test_is_safe_command_injection(self) -> None:
        """Test detection of command injection."""
        assert InputSanitizer.is_safe_command("ls; rm -rf /") is False
        assert InputSanitizer.is_safe_command("echo $(whoami)") is False
        assert InputSanitizer.is_safe_command("cat `id`") is False
        assert InputSanitizer.is_safe_command("ls | grep test") is False
        assert InputSanitizer.is_safe_command("echo $HOME") is False


# =============================================================================
# Secure Logger Tests
# =============================================================================


class TestSecureLogger:
    """Tests for secure logging utilities."""

    def test_redact_api_key(self) -> None:
        """Test redaction of API keys."""
        message = "Using API key: sk-1234567890abcdef1234567890abcdef"
        result = SecureLogger.redact(message)
        assert "sk-1234567890" not in result
        assert "<REDACTED_API_KEY>" in result

    def test_redact_anthropic_key(self) -> None:
        """Test redaction of Anthropic API keys."""
        message = "Key: sk-ant-1234567890abcdef1234567890abcdef"
        result = SecureLogger.redact(message)
        assert "sk-ant-" not in result
        assert "<REDACTED_API_KEY>" in result

    def test_redact_email(self) -> None:
        """Test redaction of email addresses."""
        message = "Contact: user@example.com for support"
        result = SecureLogger.redact(message)
        assert "user@example.com" not in result
        assert "<REDACTED_EMAIL>" in result

    def test_redact_jwt(self) -> None:
        """Test redaction of JWT tokens."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        message = f"Token: {jwt}"
        result = SecureLogger.redact(message)
        assert jwt not in result
        assert "<REDACTED_JWT>" in result

    def test_redact_bearer_token(self) -> None:
        """Test redaction of Bearer tokens."""
        message = "Authorization: Bearer abc123xyz789"
        result = SecureLogger.redact(message)
        assert "abc123xyz789" not in result
        assert "Bearer <REDACTED>" in result

    def test_redact_password_in_url(self) -> None:
        """Test redaction of passwords in URLs."""
        message = "Connecting to https://api.example.com?password=secret123&user=test"
        result = SecureLogger.redact(message)
        assert "secret123" not in result
        assert "password=<REDACTED>" in result

    def test_redact_multiple_patterns(self) -> None:
        """Test redaction of multiple sensitive items."""
        message = "Key: sk-test123456789012345678901234 email: user@test.com"
        result = SecureLogger.redact(message)
        assert "sk-test" not in result
        assert "user@test.com" not in result

    def test_redact_preserves_normal_text(self) -> None:
        """Test that normal text is preserved."""
        message = "This is a normal log message without sensitive data."
        result = SecureLogger.redact(message)
        assert result == message

    def test_log_safe(self) -> None:
        """Test log_safe redacts before logging."""
        mock_logger = MagicMock()
        message = "API key: sk-test123456789012345678901234"

        SecureLogger.log_safe(mock_logger, logging.INFO, message)

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert "sk-test" not in call_args[0][1]
        assert "<REDACTED_API_KEY>" in call_args[0][1]


class TestSecureLoggingFilter:
    """Tests for secure logging filter."""

    def test_filter_redacts_message(self) -> None:
        """Test that filter redacts log messages."""
        filter_obj = SecureLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Key: sk-test123456789012345678901234",
            args=(),
            exc_info=None,
        )

        result = filter_obj.filter(record)

        assert result is True  # Always allows record
        assert "sk-test" not in record.msg
        assert "<REDACTED_API_KEY>" in record.msg

    def test_filter_redacts_args(self) -> None:
        """Test that filter redacts log arguments."""
        filter_obj = SecureLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Data: %s",
            args=("user@example.com",),
            exc_info=None,
        )

        filter_obj.filter(record)

        assert "<REDACTED_EMAIL>" in record.args[0]

    def test_filter_handles_non_string_args(self) -> None:
        """Test that filter handles non-string arguments."""
        filter_obj = SecureLoggingFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Count: %d, Value: %s",
            args=(42, "normal text"),
            exc_info=None,
        )

        # Should not raise
        result = filter_obj.filter(record)
        assert result is True
        assert record.args[0] == 42


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for rate limiting."""

    def test_allows_within_limit(self) -> None:
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            assert limiter.allow() is True

    def test_blocks_over_limit(self) -> None:
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        for _ in range(3):
            limiter.allow()

        assert limiter.allow() is False

    def test_remaining_count(self) -> None:
        """Test remaining request count."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        assert limiter.remaining() == 5
        limiter.allow()
        assert limiter.remaining() == 4
        limiter.allow()
        assert limiter.remaining() == 3

    def test_reset_at_returns_time(self) -> None:
        """Test reset_at returns correct time."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        assert limiter.reset_at() is None  # No requests yet

        limiter.allow()
        reset_time = limiter.reset_at()

        assert reset_time is not None
        assert reset_time > datetime.now()

    def test_window_expires(self) -> None:
        """Test that rate limit window expires."""
        limiter = RateLimiter(max_requests=2, window_seconds=0.1)

        # Use up the limit
        limiter.allow()
        limiter.allow()
        assert limiter.allow() is False

        # Wait for window to expire
        time.sleep(0.15)

        # Should be allowed again
        assert limiter.allow() is True


class TestRateLimitedDecorator:
    """Tests for rate_limited decorator."""

    def test_decorator_allows_calls(self) -> None:
        """Test decorator allows calls within limit."""
        call_count = 0

        @rate_limited(max_requests=5, window_seconds=60)
        def test_func() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        for _ in range(5):
            test_func()

        assert call_count == 5

    def test_decorator_raises_on_limit(self) -> None:
        """Test decorator raises when limit exceeded."""
        @rate_limited(max_requests=2, window_seconds=60)
        def limited_func() -> str:
            return "ok"

        limited_func()
        limited_func()

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            limited_func()

    def test_decorator_with_key_func(self) -> None:
        """Test decorator with key function for per-key limits."""
        @rate_limited(max_requests=2, window_seconds=60, key_func=lambda x: x)
        def keyed_func(key: str) -> str:
            return f"ok-{key}"

        # Each key gets its own limit
        keyed_func("a")
        keyed_func("a")
        keyed_func("b")
        keyed_func("b")

        # Key 'a' should be rate limited
        with pytest.raises(RuntimeError):
            keyed_func("a")

        # Key 'b' should also be rate limited
        with pytest.raises(RuntimeError):
            keyed_func("b")


# =============================================================================
# Security Auditor Tests
# =============================================================================


class TestSecurityAuditor:
    """Tests for security auditing."""

    def test_audit_environment_missing_api_key(self) -> None:
        """Test audit detects missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            issues = SecurityAuditor.audit_environment()
            assert any("OPENAI_API_KEY" in issue for _, issue, _ in issues)

    def test_audit_environment_invalid_api_key(self) -> None:
        """Test audit detects invalid API key format."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid-key"}, clear=True):
            issues = SecurityAuditor.audit_environment()
            assert any("format" in issue.lower() for _, issue, _ in issues)

    def test_audit_environment_debug_mode(self) -> None:
        """Test audit detects DEBUG mode enabled."""
        with patch.dict(os.environ, {"DEBUG": "true"}, clear=True):
            issues = SecurityAuditor.audit_environment()
            assert any("debug" in issue.lower() for _, issue, _ in issues)

    def test_audit_file_permissions(self) -> None:
        """Test file permissions audit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a world-readable sensitive file
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("SECRET=value")
            env_file.chmod(0o644)

            issues = SecurityAuditor.audit_file_permissions(Path(tmpdir))
            assert any("world-readable" in issue.lower() for _, issue, _ in issues)

    def test_audit_file_permissions_secure(self) -> None:
        """Test file permissions audit with secure files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a properly secured file
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("SECRET=value")
            env_file.chmod(0o600)

            issues = SecurityAuditor.audit_file_permissions(Path(tmpdir))
            assert len(issues) == 0

    def test_audit_file_permissions_nonexistent(self) -> None:
        """Test audit of non-existent directory."""
        issues = SecurityAuditor.audit_file_permissions(Path("/nonexistent"))
        assert len(issues) == 0

    def test_generate_report_structure(self) -> None:
        """Test that generate_report produces valid report."""
        with patch.dict(os.environ, {}, clear=True):
            report = SecurityAuditor.generate_report()

            assert "SECURITY AUDIT REPORT" in report
            assert "ENVIRONMENT AUDIT" in report
            assert "SUMMARY" in report
            assert "Errors:" in report
            assert "Warnings:" in report

    def test_generate_report_counts_issues(self) -> None:
        """Test that report correctly counts issues."""
        with patch.dict(os.environ, {"DEBUG": "true"}, clear=True):
            report = SecurityAuditor.generate_report()
            # Should have at least the missing API key warning and DEBUG warning
            assert "Warnings:" in report


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecurityIntegration:
    """Integration tests for security module."""

    def test_secure_file_workflow(self) -> None:
        """Test complete secure file workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create secure directory
            data_dir = base / "data"
            assert SecureFileHandler.secure_directory(data_dir) is True

            # Write secure file
            secret_file = data_dir / "secrets.json"
            assert SecureFileHandler.write_secure(
                secret_file, '{"key": "value"}'
            ) is True

            # Verify permissions
            assert not SecureFileHandler.is_world_readable(secret_file)
            assert (secret_file.stat().st_mode & 0o777) == 0o600

    def test_sanitize_and_write_workflow(self) -> None:
        """Test sanitizing input and writing securely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Sanitize filename
            unsafe_name = "../../../etc/passwd"
            safe_name = InputSanitizer.sanitize_filename(unsafe_name)
            assert ".." not in safe_name

            # Sanitize path
            safe_path = InputSanitizer.sanitize_path(str(base / safe_name), base_dir=base)
            assert safe_path is not None

            # Write securely
            SecureFileHandler.write_secure(safe_path, "content")
            assert safe_path.exists()

    def test_logging_with_filter(self) -> None:
        """Test logging with secure filter applied."""
        # Create logger with secure filter
        test_logger = logging.getLogger("test_secure")
        test_logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        handler.addFilter(SecureLoggingFilter())

        # Capture log output
        captured = []
        original_emit = handler.emit

        def capture_emit(record):
            captured.append(handler.format(record))
            return original_emit(record)

        handler.emit = capture_emit
        test_logger.addHandler(handler)

        # Log sensitive data
        test_logger.info("API key: sk-1234567890123456789012345678901234567890")

        # Verify redaction
        assert len(captured) > 0
        assert "sk-" not in captured[-1] or "<REDACTED" in captured[-1]

        # Cleanup
        test_logger.removeHandler(handler)
