"""Memory management CLI commands.

Commands for viewing and managing stored memories.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm

from reachy_mini_conversation_app.cli.common import (
    get_memory_manager,
    print_success,
    print_error,
    print_info,
    create_table,
)


app = typer.Typer(help="Manage memories")
console = Console()


@app.command("list")
def list_memories(
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum memories to show"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source"),
) -> None:
    """List stored memories."""
    console.print("\n[bold]Stored Memories[/bold]\n")

    manager = get_memory_manager()

    try:
        if source:
            memories = manager.db.get_memories_by_source(source, limit=limit)
        else:
            memories = manager.get_all_memories(limit=limit)

        if not memories:
            print_info("No memories stored yet")
            return

        table = create_table("Memories", ["ID", "Content", "Source", "Created"])

        for memory in memories:
            # Truncate long content
            content = memory.content
            if len(content) > 60:
                content = content[:57] + "..."

            created = memory.timestamp.strftime("%Y-%m-%d %H:%M") if memory.timestamp else "Unknown"

            table.add_row(
                str(memory.id),
                content,
                memory.source,
                created,
            )

        console.print(table)
        console.print(f"\nShowing {len(memories)} memories (use --limit to see more)")
        console.print()

    except Exception as e:
        print_error(f"Failed to list memories: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("search")
def search_memories(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
) -> None:
    """Search memories by keyword."""
    console.print(f"\n[bold]Searching: {query}[/bold]\n")

    manager = get_memory_manager()

    try:
        memories = manager.db.search_memories(query, limit=limit)

        if not memories:
            print_info(f"No memories found matching '{query}'")
            return

        table = create_table(f"Search Results for '{query}'", ["ID", "Content", "Source"])

        for memory in memories:
            content = memory.content
            if len(content) > 80:
                content = content[:77] + "..."

            table.add_row(str(memory.id), content, memory.source)

        console.print(table)
        console.print()

    except Exception as e:
        print_error(f"Search failed: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("add")
def add_memory(
    content: str = typer.Argument(..., help="Memory content to add"),
) -> None:
    """Add a new memory."""
    console.print("\n[bold]Adding memory[/bold]\n")

    manager = get_memory_manager()

    try:
        memory_id = manager.db.add_memory(
            content=content,
            source="cli",
            metadata={"added_via": "lily-cli"},
        )
        print_success(f"Memory added (ID: {memory_id})")

    except Exception as e:
        print_error(f"Failed to add memory: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("delete")
def delete_memory(
    memory_id: int = typer.Argument(..., help="Memory ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a memory by ID."""
    console.print(f"\n[bold]Deleting memory ID: {memory_id}[/bold]\n")

    manager = get_memory_manager()

    try:
        # Get the memory first to show what will be deleted
        memories = manager.get_all_memories(limit=1000)
        memory = next((m for m in memories if m.id == memory_id), None)

        if not memory:
            print_error(f"Memory ID {memory_id} not found")
            raise typer.Exit(1)

        print_info(f"Content: {memory.content[:100]}...")

        if not force:
            if not Confirm.ask("Delete this memory?"):
                print_info("Cancelled")
                raise typer.Exit(0)

        if manager.db.delete_memory(memory_id):
            print_success(f"Memory {memory_id} deleted")
        else:
            print_error("Failed to delete memory")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to delete memory: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("facts")
def list_facts(
    pending: bool = typer.Option(False, "--pending", "-p", help="Show pending facts only"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum facts to show"),
) -> None:
    """List learned facts."""
    console.print("\n[bold]Learned Facts[/bold]\n")

    manager = get_memory_manager()

    try:
        if pending:
            facts = manager.get_pending_facts(limit=limit)
            title = "Pending Facts"
        else:
            facts = manager.get_approved_facts(limit=limit)
            title = "Approved Facts"

        if not facts:
            print_info(f"No {'pending' if pending else 'approved'} facts found")
            return

        table = create_table(title, ["ID", "Fact", "Confidence", "Status"])

        for fact in facts:
            content = fact.fact
            if len(content) > 60:
                content = content[:57] + "..."

            status = "Approved" if fact.approved else "Pending"
            confidence = f"{fact.confidence:.0%}"

            table.add_row(str(fact.id), content, confidence, status)

        console.print(table)
        console.print()

    except Exception as e:
        print_error(f"Failed to list facts: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("approve")
def approve_fact(
    fact_id: int = typer.Argument(..., help="Fact ID to approve"),
) -> None:
    """Approve a pending fact."""
    console.print(f"\n[bold]Approving fact ID: {fact_id}[/bold]\n")

    manager = get_memory_manager()

    try:
        if manager.db.approve_fact(fact_id):
            print_success(f"Fact {fact_id} approved")
        else:
            print_error("Failed to approve fact")

    except Exception as e:
        print_error(f"Failed to approve fact: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("reject")
def reject_fact(
    fact_id: int = typer.Argument(..., help="Fact ID to reject"),
) -> None:
    """Reject (delete) a pending fact."""
    console.print(f"\n[bold]Rejecting fact ID: {fact_id}[/bold]\n")

    manager = get_memory_manager()

    try:
        if manager.db.delete_fact(fact_id):
            print_success(f"Fact {fact_id} rejected")
        else:
            print_error("Failed to reject fact")

    except Exception as e:
        print_error(f"Failed to reject fact: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()
