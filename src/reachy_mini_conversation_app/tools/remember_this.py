"""Tool for storing user memories."""

from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class RememberThis(Tool):
    """Store something that Lily should remember about Pedro."""

    name = "remember_this"
    description = (
        "Store a piece of information that Lily should remember for future conversations. "
        "Use this when Pedro tells you something important about himself, his preferences, "
        "his family, or anything he wants you to remember."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember. Be specific and include context.",
            },
        },
        "required": ["content"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Store a memory in the database.

        Args:
            deps: Tool dependencies including memory_manager.
            **kwargs: Tool arguments including 'content'.

        Returns:
            Result dictionary with success status and message.
        """
        content = kwargs.get("content", "").strip()

        if not content:
            return {
                "success": False,
                "message": "No content provided to remember.",
            }

        # Get memory manager from deps
        memory_manager = getattr(deps, "memory_manager", None)

        if memory_manager is None:
            return {
                "success": False,
                "message": "Memory system is not available right now.",
            }

        try:
            memory_id = await memory_manager.add_memory(
                content=content,
                source="voice_command",
                metadata={"type": "user_request"},
            )

            return {
                "success": True,
                "message": f"Got it, darling. I'll remember that.",
                "memory_id": memory_id,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Sorry darling, I couldn't save that: {str(e)}",
            }
