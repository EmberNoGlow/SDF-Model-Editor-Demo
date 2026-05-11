"""Shader compilation and management."""
from OpenGL.GL import glDeleteProgram
from typing import Tuple
from src.app.data.states import st


def recompile_shader(scene_builder) -> Tuple[bool, dict]:
    """Recompile the active shader program and return new uniforms."""
    new_shader, new_uniforms = st.shader_manager.get_or_compile(scene_builder)

    if new_shader is None:
        return False, None

    # Clean up old shader if it changed
    if st.shader is not None and st.shader != new_shader:
        old_hash = _find_shader_hash_in_cache(st.shader)
        if old_hash is None:
            glDeleteProgram(st.shader)

    st.shader = new_shader
    st.uniform_locs = new_uniforms
    return True, new_uniforms


def _find_shader_hash_in_cache(shader) -> str:
    """Find a shader's hash in the cache."""
    for cached_hash, (cached_shader, _) in st.shader_cache.items():
        if cached_shader == shader:
            return cached_hash
    return None


def handle_shader_monitor(monitor_changes_callback):
    """Monitor shader changes and reset accumulation if needed."""
    if st.monitor_shader_changes and st.shader_choice == 1:
        st.monitor_shader_changes = False
        st.frame_count = 0
        monitor_changes_callback()


def handle_frame_accumulation():
    """Increment frame counter for accumulation-based rendering."""
    if st.shader_choice == 1:  # Cycles shader
        st.frame_count = min(st.frame_count + 1, st.max_frames)
    else:
        st.frame_count = 0