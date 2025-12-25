"""Main CLI application using Typer.

Provides the lily-cli command with subcommands for:
- User/face management
- Memory operations
- Transcript search
- System control
"""

import typer
from rich.console import Console

from reachy_mini_conversation_app.cli.user import app as user_app
from reachy_mini_conversation_app.cli.memory import app as memory_app
from reachy_mini_conversation_app.cli.transcript import app as transcript_app


# Create main app
app = typer.Typer(
    name="lily-cli",
    help="Reachy Mini Lily - Command Line Interface",
    no_args_is_help=True,
)

# Add subcommand apps
app.add_typer(user_app, name="user", help="Manage known users and face recognition")
app.add_typer(memory_app, name="memory", help="Manage memories")
app.add_typer(transcript_app, name="transcript", help="Search and manage conversation transcripts")

console = Console()


@app.command()
def status() -> None:
    """Show system status and statistics."""
    from reachy_mini_conversation_app.cli.common import get_memory_manager

    console.print("\n[bold blue]Lily System Status[/bold blue]\n")

    try:
        manager = get_memory_manager()
        stats = manager.get_stats()

        console.print(f"  [green]✓[/green] Database connected")
        console.print(f"  [cyan]Memories:[/cyan] {stats['memories']}")
        console.print(f"  [cyan]Known persons:[/cyan] {stats['persons']}")
        console.print(f"  [cyan]Pending facts:[/cyan] {stats['pending_facts']}")
        console.print(f"  [cyan]Approved facts:[/cyan] {stats['approved_facts']}")

        # Check current person
        current = manager.get_current_person()
        if current:
            console.print(f"  [cyan]Current user:[/cyan] {current.name}")
        else:
            console.print(f"  [cyan]Current user:[/cyan] None")

        manager.close()

    except Exception as e:
        console.print(f"  [red]✗[/red] Error: {e}")

    console.print()


@app.command()
def version() -> None:
    """Show version information."""
    from importlib.metadata import version as get_version

    try:
        ver = get_version("reachy_mini_conversation_app")
    except Exception:
        ver = "unknown"

    console.print(f"\n[bold]Lily CLI[/bold] version {ver}\n")


if __name__ == "__main__":
    app()
