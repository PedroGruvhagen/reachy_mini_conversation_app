"""Tool for learning new persons for face recognition."""

import base64
import logging
from typing import Any, Dict

import cv2

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class LearnPerson(Tool):
    """Learn a new person's face and name for future recognition."""

    name = "learn_person"
    description = (
        "Learn a new person's face and name so Lily can recognise them in the future. "
        "Use this when Pedro introduces you to someone new or asks you to remember "
        "who someone is."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's name to remember.",
            },
        },
        "required": ["name"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Capture a photo and store the person's face.

        Args:
            deps: Tool dependencies including camera_worker and memory_manager.
            **kwargs: Tool arguments including 'name'.

        Returns:
            Result dictionary with success status.
        """
        name = kwargs.get("name", "").strip()

        if not name:
            return {
                "success": False,
                "message": "I need to know the person's name, darling.",
            }

        # Get dependencies
        camera_worker = deps.camera_worker
        memory_manager = getattr(deps, "memory_manager", None)

        if camera_worker is None:
            return {
                "success": False,
                "message": "I can't see right now - the camera isn't available.",
            }

        if memory_manager is None:
            return {
                "success": False,
                "message": "My memory system isn't available right now.",
            }

        try:
            # Get current frame from camera
            frame = camera_worker.get_latest_frame()

            if frame is None:
                return {
                    "success": False,
                    "message": "I can't see anyone right now. Make sure they're in front of the camera.",
                }

            # Encode frame as JPEG then base64
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            photo_base64 = base64.b64encode(buffer).decode("utf-8")

            # Check if person already exists
            existing = memory_manager.get_person_by_name(name)
            if existing:
                # Update their photo
                await memory_manager.update_person_photo(existing.id, photo_base64)
                return {
                    "success": True,
                    "message": f"I've updated my memory of what {name} looks like.",
                    "person_id": existing.id,
                    "updated": True,
                }

            # Add new person
            person_id = await memory_manager.add_person(
                name=name,
                photo_base64=photo_base64,
                metadata={"learned_via": "voice_command"},
            )

            return {
                "success": True,
                "message": f"Lovely to meet you, {name}! I'll recognise you next time I see you.",
                "person_id": person_id,
                "updated": False,
            }

        except Exception as e:
            logger.exception(f"Failed to learn person: {e}")
            return {
                "success": False,
                "message": f"Sorry darling, I couldn't learn that face: {str(e)}",
            }
