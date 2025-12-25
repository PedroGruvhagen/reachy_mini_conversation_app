"""Gradio personality UI components and wiring.

This module encapsulates the UI elements and logic related to managing
conversation "personalities" (profiles), memory system, and person recognition.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, List, Optional
from pathlib import Path

import cv2
import gradio as gr

from .config import config
from .prompts import get_memory_manager


logger = logging.getLogger(__name__)


class PersonalityUI:
    """Container for personality-related Gradio components."""

    def __init__(self) -> None:
        """Initialize the PersonalityUI instance."""
        # Constants and paths
        self.DEFAULT_OPTION = "(built-in default)"
        self._profiles_root = Path(__file__).parent / "profiles"
        self._tools_dir = Path(__file__).parent / "tools"
        self._prompts_dir = Path(__file__).parent / "prompts"

        # Components (initialized in create_components)
        # Personality components
        self.personalities_dropdown: gr.Dropdown
        self.apply_btn: gr.Button
        self.status_md: gr.Markdown
        self.preview_md: gr.Markdown
        self.person_name_tb: gr.Textbox
        self.person_instr_ta: gr.TextArea
        self.tools_txt_ta: gr.TextArea
        self.voice_dropdown: gr.Dropdown
        self.new_personality_btn: gr.Button
        self.available_tools_cg: gr.CheckboxGroup
        self.save_btn: gr.Button

        # Person recognition components
        self.current_user_dropdown: gr.Dropdown
        self.learn_face_name_tb: gr.Textbox
        self.learn_face_btn: gr.Button
        self.person_status_md: gr.Markdown

        # Memory components
        self.memory_search_tb: gr.Textbox
        self.memory_list_df: gr.Dataframe
        self.refresh_memories_btn: gr.Button
        self.add_memory_tb: gr.Textbox
        self.add_memory_btn: gr.Button

    # ---------- Filesystem helpers ----------
    def _list_personalities(self) -> list[str]:
        names: list[str] = []
        try:
            if self._profiles_root.exists():
                for p in sorted(self._profiles_root.iterdir()):
                    if p.name == "user_personalities":
                        continue
                    if p.is_dir() and (p / "instructions.txt").exists():
                        names.append(p.name)
                user_dir = self._profiles_root / "user_personalities"
                if user_dir.exists():
                    for p in sorted(user_dir.iterdir()):
                        if p.is_dir() and (p / "instructions.txt").exists():
                            names.append(f"user_personalities/{p.name}")
        except Exception:
            pass
        return names

    def _resolve_profile_dir(self, selection: str) -> Path:
        return self._profiles_root / selection

    def _read_instructions_for(self, name: str) -> str:
        try:
            if name == self.DEFAULT_OPTION:
                default_file = self._prompts_dir / "default_prompt.txt"
                if default_file.exists():
                    return default_file.read_text(encoding="utf-8").strip()
                return ""
            target = self._resolve_profile_dir(name) / "instructions.txt"
            if target.exists():
                return target.read_text(encoding="utf-8").strip()
            return ""
        except Exception as e:
            return f"Could not load instructions: {e}"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        import re

        s = name.strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-zA-Z0-9_-]", "", s)
        return s

    def _get_known_persons(self) -> List[tuple]:
        """Get list of known persons for dropdown."""
        memory_manager = get_memory_manager()
        if memory_manager is None:
            return []
        try:
            persons = memory_manager.get_all_persons()
            return [(p.name, p.id) for p in persons]
        except Exception as e:
            logger.warning(f"Failed to get known persons: {e}")
            return []

    def _get_memories_for_display(self, search_query: str = "") -> List[List[str]]:
        """Get memories formatted for dataframe display."""
        memory_manager = get_memory_manager()
        if memory_manager is None:
            return []
        try:
            if search_query:
                # Use sync version since this is called from Gradio event
                memories = memory_manager.db.search_memories(search_query, limit=50)
            else:
                memories = memory_manager.db.get_all_memories(limit=50)

            return [
                [
                    str(m.id),
                    m.content[:100] + "..." if len(m.content) > 100 else m.content,
                    m.source,
                    m.timestamp.strftime("%Y-%m-%d %H:%M") if m.timestamp else "",
                ]
                for m in memories
            ]
        except Exception as e:
            logger.warning(f"Failed to get memories: {e}")
            return []

    # ---------- Public API ----------
    def create_components(self) -> None:
        """Instantiate Gradio components for the personality UI."""
        current_value = config.REACHY_MINI_CUSTOM_PROFILE or "lily"

        # Use lily as default if it exists
        available_profiles = self._list_personalities()
        if "lily" in available_profiles:
            current_value = "lily"

        # === Person Recognition Section ===
        with gr.Accordion("Person Recognition", open=True):
            with gr.Row():
                # Get known persons for dropdown
                known_persons = self._get_known_persons()
                person_choices = ["(Auto-detect)", "(No one)"] + [p[0] for p in known_persons]

                self.current_user_dropdown = gr.Dropdown(
                    label="Current User",
                    choices=person_choices,
                    value="(Auto-detect)",
                    info="Select who is talking to Lily, or let her auto-detect",
                )

            with gr.Row():
                self.learn_face_name_tb = gr.Textbox(
                    label="Person's Name",
                    placeholder="Enter name to learn...",
                    scale=3,
                )
                self.learn_face_btn = gr.Button(
                    "Learn Face",
                    variant="primary",
                    scale=1,
                )

            self.person_status_md = gr.Markdown(value="", visible=True)

        # === Memories Section ===
        with gr.Accordion("Memories", open=False):
            with gr.Row():
                self.memory_search_tb = gr.Textbox(
                    label="Search memories",
                    placeholder="Search...",
                    scale=3,
                )
                self.refresh_memories_btn = gr.Button("Refresh", scale=1)

            self.memory_list_df = gr.Dataframe(
                headers=["ID", "Content", "Source", "Date"],
                datatype=["str", "str", "str", "str"],
                value=self._get_memories_for_display(),
                label="Stored Memories",
                interactive=False,
                wrap=True,
            )

            with gr.Row():
                self.add_memory_tb = gr.Textbox(
                    label="Add new memory",
                    placeholder="Enter something for Lily to remember...",
                    scale=4,
                )
                self.add_memory_btn = gr.Button("Add", scale=1)

        # === Personality Section (collapsed by default) ===
        with gr.Accordion("Personality Settings", open=False):
            self.personalities_dropdown = gr.Dropdown(
                label="Select personality",
                choices=[self.DEFAULT_OPTION, *available_profiles],
                value=current_value,
            )
            self.apply_btn = gr.Button("Apply personality")
            self.status_md = gr.Markdown(visible=True)
            self.preview_md = gr.Markdown(value=self._read_instructions_for(current_value))
            self.person_name_tb = gr.Textbox(label="Personality name")
            self.person_instr_ta = gr.TextArea(label="Personality instructions", lines=10)
            self.tools_txt_ta = gr.TextArea(label="tools.txt", lines=10)
            self.voice_dropdown = gr.Dropdown(label="Voice", choices=["cedar", "nova"], value="nova")
            self.new_personality_btn = gr.Button("New personality")
            self.available_tools_cg = gr.CheckboxGroup(label="Available tools (helper)", choices=[], value=[])
            self.save_btn = gr.Button("Save personality (instructions + tools)")

    def additional_inputs_ordered(self) -> list[Any]:
        """Return the additional inputs in the expected order for Stream."""
        return [
            # Person recognition
            self.current_user_dropdown,
            self.learn_face_name_tb,
            self.learn_face_btn,
            self.person_status_md,
            # Memories
            self.memory_search_tb,
            self.memory_list_df,
            self.refresh_memories_btn,
            self.add_memory_tb,
            self.add_memory_btn,
            # Personality
            self.personalities_dropdown,
            self.apply_btn,
            self.new_personality_btn,
            self.status_md,
            self.preview_md,
            self.person_name_tb,
            self.person_instr_ta,
            self.tools_txt_ta,
            self.voice_dropdown,
            self.available_tools_cg,
            self.save_btn,
        ]

    # ---------- Event wiring ----------
    def wire_events(self, handler: Any, blocks: gr.Blocks) -> None:
        """Attach event handlers to components within a Blocks context."""

        # === Person Recognition Events ===
        def _update_current_user(selected: str) -> str:
            """Update the current user based on dropdown selection."""
            memory_manager = get_memory_manager()
            if memory_manager is None:
                return "Memory system not available"

            try:
                if selected == "(Auto-detect)":
                    memory_manager.set_current_person(None)
                    return "Auto-detection enabled"
                elif selected == "(No one)":
                    memory_manager.set_current_person(None)
                    return "No user selected"
                else:
                    # Find person by name
                    person = memory_manager.get_person_by_name(selected)
                    if person:
                        memory_manager.set_current_person(person.id)
                        return f"Now talking to {person.name}"
                    return f"Person '{selected}' not found"
            except Exception as e:
                return f"Error: {e}"

        def _learn_face(name: str) -> tuple:
            """Learn a new face from the current camera frame."""
            if not name or not name.strip():
                return "Please enter a name first", gr.update()

            name = name.strip()
            memory_manager = get_memory_manager()
            if memory_manager is None:
                return "Memory system not available", gr.update()

            # Get camera frame from handler's deps
            try:
                camera_worker = handler.deps.camera_worker if hasattr(handler, "deps") else None
                if camera_worker is None:
                    return "Camera not available", gr.update()

                frame = camera_worker.get_latest_frame()
                if frame is None:
                    return "No camera frame available. Make sure someone is in front of the camera.", gr.update()

                # Encode frame as JPEG then base64
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                photo_base64 = base64.b64encode(buffer).decode("utf-8")

                # Check if person already exists
                existing = memory_manager.get_person_by_name(name)
                if existing:
                    # Update their photo
                    memory_manager.db.update_person_photo(existing.id, photo_base64)
                    # Update dropdown
                    known_persons = self._get_known_persons()
                    person_choices = ["(Auto-detect)", "(No one)"] + [p[0] for p in known_persons]
                    return (
                        f"Updated photo for {name}",
                        gr.update(choices=person_choices, value=name),
                    )

                # Add new person
                person_id = memory_manager.db.add_person(
                    name=name,
                    photo_base64=photo_base64,
                    metadata={"learned_via": "ui_button"},
                )

                # Update dropdown
                known_persons = self._get_known_persons()
                person_choices = ["(Auto-detect)", "(No one)"] + [p[0] for p in known_persons]

                return (
                    f"Learned {name}! I'll recognise them next time.",
                    gr.update(choices=person_choices, value=name),
                )

            except Exception as e:
                logger.exception(f"Failed to learn face: {e}")
                return f"Error learning face: {e}", gr.update()

        # === Memory Events ===
        def _refresh_memories(search_query: str) -> List[List[str]]:
            """Refresh the memories list."""
            return self._get_memories_for_display(search_query)

        def _add_memory(content: str) -> tuple:
            """Add a new memory."""
            if not content or not content.strip():
                return self._get_memories_for_display(), "Please enter content to remember"

            memory_manager = get_memory_manager()
            if memory_manager is None:
                return self._get_memories_for_display(), "Memory system not available"

            try:
                memory_manager.db.add_memory(
                    content=content.strip(),
                    source="manual_ui",
                    metadata={"added_via": "ui"},
                )
                return self._get_memories_for_display(), f"Added: {content[:50]}..."
            except Exception as e:
                return self._get_memories_for_display(), f"Error: {e}"

        # === Personality Events ===
        async def _apply_personality(selected: str) -> tuple[str, str]:
            profile = None if selected == self.DEFAULT_OPTION else selected
            status = await handler.apply_personality(profile)
            preview = self._read_instructions_for(selected)
            return status, preview

        def _read_voice_for(name: str) -> str:
            try:
                if name == self.DEFAULT_OPTION:
                    return "cedar"
                vf = self._resolve_profile_dir(name) / "voice.txt"
                if vf.exists():
                    v = vf.read_text(encoding="utf-8").strip()
                    return v or "cedar"
            except Exception:
                pass
            return "cedar"

        async def _fetch_voices(selected: str) -> dict[str, Any]:
            try:
                voices = await handler.get_available_voices()
                current = _read_voice_for(selected)
                if current not in voices:
                    current = "nova"  # Default to nova for Lily
                return gr.update(choices=voices, value=current)
            except Exception:
                return gr.update(choices=["cedar", "nova"], value="nova")

        def _available_tools_for(selected: str) -> tuple[list[str], list[str]]:
            shared: list[str] = []
            try:
                for py in self._tools_dir.glob("*.py"):
                    if py.stem in {"__init__", "core_tools"}:
                        continue
                    shared.append(py.stem)
            except Exception:
                pass
            local: list[str] = []
            try:
                if selected != self.DEFAULT_OPTION:
                    for py in (self._profiles_root / selected).glob("*.py"):
                        local.append(py.stem)
            except Exception:
                pass
            return sorted(shared), sorted(local)

        def _parse_enabled_tools(text: str) -> list[str]:
            enabled: list[str] = []
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                enabled.append(s)
            return enabled

        def _load_profile_for_edit(selected: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
            instr = self._read_instructions_for(selected)
            tools_txt = ""
            if selected != self.DEFAULT_OPTION:
                tp = self._resolve_profile_dir(selected) / "tools.txt"
                if tp.exists():
                    tools_txt = tp.read_text(encoding="utf-8")
            shared, local = _available_tools_for(selected)
            all_tools = sorted(set(shared + local))
            enabled = _parse_enabled_tools(tools_txt)
            status_text = f"Loaded profile '{selected}'."
            return (
                gr.update(value=instr),
                gr.update(value=tools_txt),
                gr.update(choices=all_tools, value=enabled),
                status_text,
            )

        def _new_personality() -> tuple[
            dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any]
        ]:
            try:
                instr_val = """# Write your instructions here\n# e.g., Keep responses concise and friendly."""
                tools_txt_val = "# tools enabled for this profile\n"
                return (
                    gr.update(value=""),
                    gr.update(value=instr_val),
                    gr.update(value=tools_txt_val),
                    gr.update(choices=sorted(_available_tools_for(self.DEFAULT_OPTION)[0]), value=[]),
                    "Fill in a name, instructions and (optional) tools, then Save.",
                    gr.update(value="nova"),
                )
            except Exception:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "Failed to initialize new personality.",
                    gr.update(),
                )

        def _save_personality(
            name: str, instructions: str, tools_text: str, voice: str
        ) -> tuple[dict[str, Any], dict[str, Any], str]:
            name_s = self._sanitize_name(name)
            if not name_s:
                return gr.update(), gr.update(), "Please enter a valid name."
            try:
                target_dir = self._profiles_root / "user_personalities" / name_s
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "instructions.txt").write_text(instructions.strip() + "\n", encoding="utf-8")
                (target_dir / "tools.txt").write_text(tools_text.strip() + "\n", encoding="utf-8")
                (target_dir / "voice.txt").write_text((voice or "nova").strip() + "\n", encoding="utf-8")

                choices = self._list_personalities()
                value = f"user_personalities/{name_s}"
                if value not in choices:
                    choices.append(value)
                return (
                    gr.update(choices=[self.DEFAULT_OPTION, *sorted(choices)], value=value),
                    gr.update(value=instructions),
                    f"Saved personality '{name_s}'.",
                )
            except Exception as e:
                return gr.update(), gr.update(), f"Failed to save personality: {e}"

        def _sync_tools_from_checks(selected: list[str], current_text: str) -> dict[str, Any]:
            comments = [ln for ln in current_text.splitlines() if ln.strip().startswith("#")]
            body = "\n".join(selected)
            out = ("\n".join(comments) + ("\n" if comments else "") + body).strip() + "\n"
            return gr.update(value=out)

        # === Wire all events ===
        with blocks:
            # Person recognition events
            self.current_user_dropdown.change(
                fn=_update_current_user,
                inputs=[self.current_user_dropdown],
                outputs=[self.person_status_md],
            )

            self.learn_face_btn.click(
                fn=_learn_face,
                inputs=[self.learn_face_name_tb],
                outputs=[self.person_status_md, self.current_user_dropdown],
            )

            # Memory events
            self.refresh_memories_btn.click(
                fn=_refresh_memories,
                inputs=[self.memory_search_tb],
                outputs=[self.memory_list_df],
            )

            self.memory_search_tb.submit(
                fn=_refresh_memories,
                inputs=[self.memory_search_tb],
                outputs=[self.memory_list_df],
            )

            self.add_memory_btn.click(
                fn=_add_memory,
                inputs=[self.add_memory_tb],
                outputs=[self.memory_list_df, self.person_status_md],
            )

            # Personality events
            self.apply_btn.click(
                fn=_apply_personality,
                inputs=[self.personalities_dropdown],
                outputs=[self.status_md, self.preview_md],
            )

            self.personalities_dropdown.change(
                fn=_load_profile_for_edit,
                inputs=[self.personalities_dropdown],
                outputs=[self.person_instr_ta, self.tools_txt_ta, self.available_tools_cg, self.status_md],
            )

            blocks.load(
                fn=_fetch_voices,
                inputs=[self.personalities_dropdown],
                outputs=[self.voice_dropdown],
            )

            self.available_tools_cg.change(
                fn=_sync_tools_from_checks,
                inputs=[self.available_tools_cg, self.tools_txt_ta],
                outputs=[self.tools_txt_ta],
            )

            self.new_personality_btn.click(
                fn=_new_personality,
                inputs=[],
                outputs=[
                    self.person_name_tb,
                    self.person_instr_ta,
                    self.tools_txt_ta,
                    self.available_tools_cg,
                    self.status_md,
                    self.voice_dropdown,
                ],
            )

            self.save_btn.click(
                fn=_save_personality,
                inputs=[self.person_name_tb, self.person_instr_ta, self.tools_txt_ta, self.voice_dropdown],
                outputs=[self.personalities_dropdown, self.person_instr_ta, self.status_md],
            ).then(
                fn=_apply_personality,
                inputs=[self.personalities_dropdown],
                outputs=[self.status_md, self.preview_md],
            )
