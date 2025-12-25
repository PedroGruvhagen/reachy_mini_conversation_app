"""Tool for recalling stored memories."""

from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


class RecallMemory(Tool):
    """Search and recall stored memories."""

    name = "recall_memory"
    description = (
        "Search Lily's memories for specific information about Pedro, his preferences, "
        "family, or anything that was previously stored. Use this when you need to "
        "recall something specific."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in the memories.",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Search memories for relevant information.

        Args:
            deps: Tool dependencies including memory_manager.
            **kwargs: Tool arguments including 'query'.

        Returns:
            Result dictionary with matching memories.
        """
        query = kwargs.get("query", "").strip()

        if not query:
            return {
                "success": False,
                "message": "No search query provided.",
                "memories": [],
            }

        # Get memory manager from deps
        memory_manager = getattr(deps, "memory_manager", None)

        if memory_manager is None:
            return {
                "success": False,
                "message": "Memory system is not available right now.",
                "memories": [],
            }

        try:
            memories = await memory_manager.search_memories(query, limit=5)

            if not memories:
                return {
                    "success": True,
                    "message": "I don't have any memories matching that.",
                    "memories": [],
                }

            # Format memories for response
            memory_list = [
                {
                    "content": m.content,
                    "source": m.source,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                }
                for m in memories
            ]

            return {
                "success": True,
                "message": f"Found {len(memories)} relevant memories.",
                "memories": memory_list,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Sorry darling, I couldn't search my memories: {str(e)}",
                "memories": [],
            }
