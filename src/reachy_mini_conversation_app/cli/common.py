"""Common utilities for CLI commands.

Shared functions for database access, console output, and error handling.
"""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from reachy_mini_conversation_app.memory.manager import MemoryManager
from reachy_mini_conversation_app.vision.face_recognition import get_face_recognition_service


console = Console()


def get_memory_manager() -> MemoryManager:
    """Get a MemoryManager instance.

    Returns:
        Initialized MemoryManager.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    manager = MemoryManager(openai_api_key=openai_key)
    return manager


def get_face_service():
    """Get the face recognition service.

    Returns:
        Initialized FaceRecognitionService.
    """
    service = get_face_recognition_service()
    if not service.is_initialized:
        service.initialize()
    return service


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Message to print.
    """
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message.

    Args:
        message: Message to print.
    """
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Message to print.
    """
    console.print(f"[yellow]![/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Message to print.
    """
    console.print(f"[blue]ℹ[/blue] {message}")


def create_table(title: str, columns: list[str]) -> Table:
    """Create a formatted Rich table.

    Args:
        title: Table title.
        columns: List of column names.

    Returns:
        Rich Table object.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col in columns:
        table.add_column(col)
    return table


def validate_image_path(path: str) -> Optional[Path]:
    """Validate an image file path.

    Args:
        path: Path to image file.

    Returns:
        Path object if valid, None otherwise.
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        print_error(f"File not found: {path}")
        return None

    if not p.is_file():
        print_error(f"Not a file: {path}")
        return None

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if p.suffix.lower() not in valid_extensions:
        print_error(f"Invalid image format: {p.suffix}")
        print_info(f"Supported formats: {', '.join(valid_extensions)}")
        return None

    return p
