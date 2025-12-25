"""User/face management CLI commands.

Commands for adding, listing, identifying, and managing known users.
"""

import base64
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm

import cv2
import numpy as np

from reachy_mini_conversation_app.cli.common import (
    get_memory_manager,
    get_face_service,
    print_success,
    print_error,
    print_warning,
    print_info,
    create_table,
    validate_image_path,
)


app = typer.Typer(help="Manage known users and face recognition")
console = Console()


@app.command("add")
def add_user(
    name: str = typer.Argument(..., help="Name of the person to add"),
    photo: Optional[str] = typer.Option(None, "--photo", "-p", help="Path to photo file"),
    capture: bool = typer.Option(False, "--capture", "-c", help="Capture from camera"),
) -> None:
    """Add a new known user with their photo."""
    console.print(f"\n[bold]Adding user: {name}[/bold]\n")

    if not photo and not capture:
        print_error("Must provide either --photo or --capture")
        raise typer.Exit(1)

    manager = get_memory_manager()

    try:
        # Check if user already exists
        existing = manager.get_person_by_name(name)
        if existing:
            print_warning(f"User '{name}' already exists (ID: {existing.id})")
            if not Confirm.ask("Update their photo?"):
                raise typer.Exit(0)

        # Get photo data
        if photo:
            # Load from file
            path = validate_image_path(photo)
            if not path:
                raise typer.Exit(1)

            frame = cv2.imread(str(path))
            if frame is None:
                print_error(f"Could not read image: {photo}")
                raise typer.Exit(1)

        elif capture:
            # Capture from camera
            print_info("Opening camera...")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print_error("Could not open camera")
                raise typer.Exit(1)

            print_info("Press ENTER to capture or 'q' to cancel...")

            frame = None
            while True:
                ret, current_frame = cap.read()
                if not ret:
                    print_error("Could not read from camera")
                    cap.release()
                    raise typer.Exit(1)

                # Show preview (headless mode - skip display)
                # In SSH we can't show preview, just capture immediately
                frame = current_frame
                break

            cap.release()

            if frame is None:
                print_error("No frame captured")
                raise typer.Exit(1)

        # Get face recognition service
        face_service = get_face_service()

        # Detect face
        faces = face_service.detect_faces(frame)
        if not faces:
            print_error("No face detected in image")
            raise typer.Exit(1)

        if len(faces) > 1:
            print_warning(f"Multiple faces detected ({len(faces)}), using the largest one")

        # Extract embedding
        embedding = face_service.extract_embedding(frame)
        if embedding is None:
            print_error("Could not extract face embedding")
            raise typer.Exit(1)

        # Encode photo as base64
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        photo_base64 = base64.b64encode(buffer).decode("utf-8")

        # Add or update user
        if existing:
            manager.db.update_person_photo(existing.id, photo_base64)
            manager.db.update_person_embedding(existing.id, embedding.tolist())
            print_success(f"Updated photo for '{name}' (ID: {existing.id})")
        else:
            person_id = manager.db.add_person(
                name=name,
                photo_base64=photo_base64,
                metadata={"source": "cli"},
                face_embedding=embedding.tolist(),
            )
            print_success(f"Added user '{name}' (ID: {person_id})")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to add user: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("delete")
def delete_user(
    name: str = typer.Argument(..., help="Name of the person to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a known user."""
    console.print(f"\n[bold]Deleting user: {name}[/bold]\n")

    manager = get_memory_manager()

    try:
        person = manager.get_person_by_name(name)
        if not person:
            print_error(f"User '{name}' not found")
            raise typer.Exit(1)

        if not force:
            if not Confirm.ask(f"Are you sure you want to delete '{name}'?"):
                print_info("Cancelled")
                raise typer.Exit(0)

        manager.db.delete_person(person.id)
        print_success(f"Deleted user '{name}'")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to delete user: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("list")
def list_users() -> None:
    """List all known users."""
    console.print("\n[bold]Known Users[/bold]\n")

    manager = get_memory_manager()

    try:
        persons = manager.get_all_persons()

        if not persons:
            print_info("No users registered yet")
            print_info("Use 'lily-cli user add <name> --photo <path>' to add a user")
            return

        current = manager.get_current_person()

        table = create_table("Known Users", ["ID", "Name", "Last Seen", "Recognition Count", "Status"])

        for person in persons:
            last_seen = person.last_seen.strftime("%Y-%m-%d %H:%M") if person.last_seen else "Never"
            is_current = "✓ Current" if current and current.id == person.id else ""

            table.add_row(
                str(person.id),
                person.name,
                last_seen,
                str(person.recognition_count),
                is_current,
            )

        console.print(table)
        console.print()

    except Exception as e:
        print_error(f"Failed to list users: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("identify")
def identify_user(
    save: bool = typer.Option(False, "--save", "-s", help="Set as current user if identified"),
) -> None:
    """Identify the person currently in front of the camera."""
    console.print("\n[bold]Identifying user...[/bold]\n")

    manager = get_memory_manager()

    try:
        # Get all known persons with embeddings
        persons = manager.get_all_persons()
        if not persons:
            print_warning("No users registered yet")
            print_info("Use 'lily-cli user add <name> --photo <path>' to add users first")
            raise typer.Exit(0)

        # Open camera
        print_info("Opening camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print_error("Could not open camera")
            raise typer.Exit(1)

        # Capture frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print_error("Could not capture frame")
            raise typer.Exit(1)

        # Get face service
        face_service = get_face_service()

        # Detect face
        faces = face_service.detect_faces(frame)
        if not faces:
            print_warning("No face detected in camera view")
            raise typer.Exit(0)

        # Extract embedding
        current_embedding = face_service.extract_embedding(frame)
        if current_embedding is None:
            print_error("Could not extract face embedding")
            raise typer.Exit(1)

        # Build list of persons with embeddings
        persons_with_embeddings = []
        for person in persons:
            if person.face_embedding:
                persons_with_embeddings.append({
                    "id": person.id,
                    "name": person.name,
                    "embedding": np.array(person.face_embedding, dtype=np.float32),
                })

        if not persons_with_embeddings:
            print_warning("No users have face embeddings stored")
            print_info("Re-add users with photos to generate embeddings")
            raise typer.Exit(0)

        # Find match
        match = face_service.find_matching_person(current_embedding, persons_with_embeddings)

        if match:
            person_id, name, confidence = match
            print_success(f"Recognized: {name} (confidence: {confidence:.2%})")

            if save:
                manager.set_current_person(person_id)
                print_success(f"Set '{name}' as current user")
        else:
            print_warning("No match found - unknown person")
            print_info("Use 'lily-cli user add <name> --capture' to add this person")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to identify user: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("switch")
def switch_user(
    name: str = typer.Argument(..., help="Name of the person to switch to"),
) -> None:
    """Manually set the current active user."""
    console.print(f"\n[bold]Switching to user: {name}[/bold]\n")

    manager = get_memory_manager()

    try:
        person = manager.get_person_by_name(name)
        if not person:
            print_error(f"User '{name}' not found")
            print_info("Use 'lily-cli user list' to see available users")
            raise typer.Exit(1)

        manager.set_current_person(person.id)
        print_success(f"Current user set to: {name}")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to switch user: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("update")
def update_user(
    name: str = typer.Argument(..., help="Name of the person to update"),
    photo: Optional[str] = typer.Option(None, "--photo", "-p", help="New photo file path"),
    capture: bool = typer.Option(False, "--capture", "-c", help="Capture new photo from camera"),
) -> None:
    """Update a user's photo."""
    console.print(f"\n[bold]Updating user: {name}[/bold]\n")

    if not photo and not capture:
        print_error("Must provide either --photo or --capture")
        raise typer.Exit(1)

    manager = get_memory_manager()

    try:
        person = manager.get_person_by_name(name)
        if not person:
            print_error(f"User '{name}' not found")
            raise typer.Exit(1)

        # Get photo data (same logic as add)
        if photo:
            path = validate_image_path(photo)
            if not path:
                raise typer.Exit(1)

            frame = cv2.imread(str(path))
            if frame is None:
                print_error(f"Could not read image: {photo}")
                raise typer.Exit(1)

        elif capture:
            print_info("Opening camera...")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print_error("Could not open camera")
                raise typer.Exit(1)

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                print_error("Could not capture frame")
                raise typer.Exit(1)

        # Get face service and extract embedding
        face_service = get_face_service()

        faces = face_service.detect_faces(frame)
        if not faces:
            print_error("No face detected in image")
            raise typer.Exit(1)

        embedding = face_service.extract_embedding(frame)
        if embedding is None:
            print_error("Could not extract face embedding")
            raise typer.Exit(1)

        # Encode photo as base64
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        photo_base64 = base64.b64encode(buffer).decode("utf-8")

        # Update
        manager.db.update_person_photo(person.id, photo_base64)
        manager.db.update_person_embedding(person.id, embedding.tolist())

        print_success(f"Updated photo for '{name}'")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to update user: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()


@app.command("clear")
def clear_current() -> None:
    """Clear the current active user."""
    console.print("\n[bold]Clearing current user[/bold]\n")

    manager = get_memory_manager()

    try:
        manager.set_current_person(None)
        print_success("Current user cleared")

    except Exception as e:
        print_error(f"Failed to clear current user: {e}")
        raise typer.Exit(1)
    finally:
        manager.close()
