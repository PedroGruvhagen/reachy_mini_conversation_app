"""Tool for identifying the current person in frame."""

import base64
import hashlib
import logging
from typing import Any, Dict, Optional

import cv2

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class WhoIsThis(Tool):
    """Identify the person currently in front of the camera."""

    name = "who_is_this"
    description = (
        "Identify who is currently in front of the camera. Use this when you want "
        "to know who you're talking to or to verify the person's identity."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Identify the person in the current camera frame.

        Uses OpenAI Vision API to compare with stored person photos.

        Args:
            deps: Tool dependencies.
            **kwargs: Tool arguments (none required).

        Returns:
            Result dictionary with identification result.
        """
        # Get dependencies
        camera_worker = deps.camera_worker
        memory_manager = getattr(deps, "memory_manager", None)

        if camera_worker is None:
            return {
                "success": False,
                "message": "I can't see right now - the camera isn't available.",
                "person": None,
            }

        if memory_manager is None:
            return {
                "success": False,
                "message": "My memory system isn't available right now.",
                "person": None,
            }

        try:
            # Get current frame
            frame = camera_worker.get_latest_frame()

            if frame is None:
                return {
                    "success": False,
                    "message": "I can't see anyone right now.",
                    "person": None,
                }

            # Get current person from session (if manually set)
            current_person = memory_manager.get_current_person()
            if current_person:
                return {
                    "success": True,
                    "message": f"I'm talking to {current_person.name}.",
                    "person": {
                        "id": current_person.id,
                        "name": current_person.name,
                        "confidence": 1.0,
                        "source": "manual_selection",
                    },
                }

            # Get all known persons
            known_persons = memory_manager.get_all_persons()

            if not known_persons:
                return {
                    "success": True,
                    "message": "I can see someone, but I haven't learned any faces yet. "
                               "Would you like to introduce yourself?",
                    "person": None,
                }

            # Encode current frame
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            current_photo_base64 = base64.b64encode(buffer).decode("utf-8")

            # Check cache first (Phase 3 preparation)
            # NOTE: Cache will be empty until Vision API is implemented,
            # but having this lookup ready reduces Phase 3 work.
            frame_hash = hashlib.md5(buffer).hexdigest()
            cached = memory_manager.db.get_cached_recognition(frame_hash)
            if cached:
                person_id, confidence = cached
                if person_id:
                    person = memory_manager.get_person_by_id(person_id)
                    if person:
                        return {
                            "success": True,
                            "message": f"I recognise {person.name}!",
                            "person": {
                                "id": person.id,
                                "name": person.name,
                                "confidence": confidence,
                                "source": "cached_recognition",
                            },
                        }

            # TODO(Phase 3): Implement OpenAI Vision API face comparison
            # Currently, automatic face recognition requires manual user selection
            # via the UI dropdown or the learn_person tool. The Vision API integration
            # will enable comparing the current frame against stored photos.
            #
            # For now, return that we see someone but need them to identify themselves
            # or be selected from the UI dropdown.
            return {
                "success": True,
                "message": "I can see someone, but I need to be introduced first. "
                           "Tell me your name, or select yourself from the Current User dropdown.",
                "person": None,
                "requires_manual_selection": True,
            }

        except Exception as e:
            logger.exception(f"Failed to identify person: {e}")
            return {
                "success": False,
                "message": f"Sorry darling, I had trouble with my vision: {str(e)}",
                "person": None,
            }
