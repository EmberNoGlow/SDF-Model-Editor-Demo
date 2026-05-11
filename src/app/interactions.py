"""Compatibility wrappers for primitive interaction handlers."""

from src.core.helpers.dragging_helper import dragging_primitive
from src.core.helpers.rotation_helper import rotate_privitive


def handle_primitive_dragging(window, scene_builder, camera):
    """Handle primitive drag interactions."""
    return dragging_primitive(window, scene_builder, camera)


def handle_primitive_rotation(window, scene_builder):
    """Handle primitive rotation interactions."""
    return rotate_privitive(window, scene_builder)
