"""Transcript management CLI commands.

Commands for searching and managing conversation transcripts.
"""

from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm

from reachy_mini_conversation_app.cli.common import (
    get_memory_manager,
    print_success,
    print_error,
    print_info,
    print_warning,
    create_table,
)


app = typer.Typer(help="Search and manage conversation transcripts")
console = Console()


# Transcripts directory
TRANSCRIPTS_PATH = Path.home() / ".reachy_mini" / "recordings"


@app.command("search")
def search_transcripts(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
) -> None:
    """Search conversation transcripts."""
    console.print(f"\n[bold]Searching transcripts: {query}[/bold]\n")

    manager = get_memory_manager()

    try:
        transcripts = manager.db.search_transcripts(query, limit=limit)

        if not transcripts:
            print_info(f"No transcripts found matching '{query}'")
            return

        table = create_table(f"Search Results for '{query}'", ["Session", "Person", "Preview", "Duration", "Date"])

        for t in transcripts:
            # Truncate transcript for preview
            preview = t.transcript[:60] + "..." if len(t.transcript) > 60 else t.transcript
            duration = f"{t.duration_seconds:.1f}s" if t.duration_seconds else "N/A"
            date_str = t.ended_at.strftime("%Y-%m-%d %H:%M") if t.ended_at else "Unknown"

            table.add_row(
                t.session_id,
                t.person_name or "Unknown",
                preview.replace("\n", " "),
                duration,
                date_str,
            )

        console.print(table)
        console.print(f"\nFound {len(transcripts)} matching transcripts\n")

    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("list")
def list_transcripts(
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum to show"),
) -> None:
    """List recent conversation transcripts."""
    console.print("\n[bold]Recent Transcripts[/bold]\n")

    if not TRANSCRIPTS_PATH.exists():
        print_info("No recordings directory found")
        print_info(f"Recordings will be stored in: {TRANSCRIPTS_PATH}")
        return

    # Find audio files
    audio_files = list(TRANSCRIPTS_PATH.glob("*.wav"))
    audio_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    audio_files = audio_files[:limit]

    if not audio_files:
        print_info("No recordings found")
        return

    table = create_table("Recordings", ["Filename", "Size", "Duration", "Date"])

    for audio_file in audio_files:
        stat = audio_file.stat()
        size_mb = stat.st_size / (1024 * 1024)

        # Estimate duration from file size (rough approximation)
        # 16-bit mono at 48kHz = ~96KB/s
        duration_sec = stat.st_size / 96000
        duration_min = int(duration_sec / 60)
        duration_sec = int(duration_sec % 60)

        from datetime import datetime
        date_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

        table.add_row(
            audio_file.name,
            f"{size_mb:.1f} MB",
            f"{duration_min}m {duration_sec}s",
            date_str,
        )

    console.print(table)
    console.print()


@app.command("show")
def show_transcript(
    session_id: str = typer.Argument(..., help="Session ID or filename"),
) -> None:
    """Show a specific transcript."""
    console.print(f"\n[bold]Transcript: {session_id}[/bold]\n")

    manager = get_memory_manager()

    try:
        transcript = manager.db.get_transcript_by_session(session_id)

        if not transcript:
            print_error(f"Transcript not found: {session_id}")
            raise typer.Exit(1)

        # Show metadata
        console.print(f"[cyan]Session ID:[/cyan] {transcript.session_id}")
        console.print(f"[cyan]Person:[/cyan] {transcript.person_name or 'Unknown'}")
        console.print(f"[cyan]Duration:[/cyan] {transcript.duration_seconds:.1f}s")
        if transcript.started_at:
            console.print(f"[cyan]Started:[/cyan] {transcript.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if transcript.ended_at:
            console.print(f"[cyan]Ended:[/cyan] {transcript.ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if transcript.audio_path:
            console.print(f"[cyan]Audio file:[/cyan] {transcript.audio_path}")

        console.print("\n[bold]Transcript:[/bold]")
        console.print("-" * 60)
        console.print(transcript.transcript)
        console.print("-" * 60)
        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to show transcript: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("export")
def export_transcript(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, txt, srt)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """Export a transcript to file."""
    console.print(f"\n[bold]Exporting transcript: {session_id}[/bold]\n")

    valid_formats = {"json", "txt", "srt"}
    if format not in valid_formats:
        print_error(f"Invalid format: {format}")
        print_info(f"Valid formats: {', '.join(valid_formats)}")
        raise typer.Exit(1)

    manager = get_memory_manager()

    try:
        transcript = manager.db.get_transcript_by_session(session_id)

        if not transcript:
            print_error(f"Transcript not found: {session_id}")
            raise typer.Exit(1)

        # Generate output filename if not provided
        if output is None:
            output = f"transcript-{session_id}.{format}"

        output_path = Path(output)

        # Export based on format
        if format == "json":
            import json
            export_data = {
                "session_id": transcript.session_id,
                "person_id": transcript.person_id,
                "person_name": transcript.person_name,
                "transcript": transcript.transcript,
                "audio_path": transcript.audio_path,
                "duration_seconds": transcript.duration_seconds,
                "started_at": transcript.started_at.isoformat() if transcript.started_at else None,
                "ended_at": transcript.ended_at.isoformat() if transcript.ended_at else None,
                "metadata": transcript.metadata,
            }
            output_path.write_text(json.dumps(export_data, indent=2))

        elif format == "txt":
            text_content = f"Session: {transcript.session_id}\n"
            text_content += f"Person: {transcript.person_name or 'Unknown'}\n"
            text_content += f"Date: {transcript.ended_at.strftime('%Y-%m-%d %H:%M:%S') if transcript.ended_at else 'Unknown'}\n"
            text_content += f"Duration: {transcript.duration_seconds:.1f}s\n"
            text_content += "\n" + "=" * 60 + "\n\n"
            text_content += transcript.transcript
            output_path.write_text(text_content)

        elif format == "srt":
            # Simple SRT format - treat entire transcript as one subtitle
            srt_content = "1\n"
            srt_content += "00:00:00,000 --> "
            duration_secs = int(transcript.duration_seconds)
            hours = duration_secs // 3600
            mins = (duration_secs % 3600) // 60
            secs = duration_secs % 60
            srt_content += f"{hours:02d}:{mins:02d}:{secs:02d},000\n"
            srt_content += transcript.transcript + "\n"
            output_path.write_text(srt_content)

        print_success(f"Exported to: {output_path}")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to export transcript: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("clean")
def clean_transcripts(
    days: int = typer.Option(30, "--days", "-d", help="Delete recordings older than N days"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Clean up old recordings."""
    console.print(f"\n[bold]Cleaning recordings older than {days} days[/bold]\n")

    if not TRANSCRIPTS_PATH.exists():
        print_info("No recordings directory found")
        return

    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    # Find old files
    old_files = []
    for audio_file in TRANSCRIPTS_PATH.glob("*.wav"):
        if audio_file.stat().st_mtime < cutoff_ts:
            old_files.append(audio_file)

    if not old_files:
        print_info("No old recordings to clean up")
        return

    total_size = sum(f.stat().st_size for f in old_files) / (1024 * 1024)

    print_info(f"Found {len(old_files)} old recordings ({total_size:.1f} MB)")

    if not force:
        if not Confirm.ask("Delete these recordings?"):
            print_info("Cancelled")
            raise typer.Exit(0)

    deleted = 0
    for audio_file in old_files:
        try:
            audio_file.unlink()
            deleted += 1
        except Exception as e:
            print_error(f"Failed to delete {audio_file.name}: {e}")

    print_success(f"Deleted {deleted} recordings")
