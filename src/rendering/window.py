import glfw
import imgui
from ..app.data.consts import cn
from ..app.data.states import scaled_rendering_width, scaled_rendering_height, resolution_scale

def handle_window_resize(window):
    """Get current window dimensions and calculate derived sizes."""
    width, height = glfw.get_framebuffer_size(window)
    menu_bar_height = int(imgui.get_frame_height())
    panel_width = int(width * cn["PANEL_WIDTH_RATIO"])
    rendering_width = width - 2 * panel_width
    rendering_height = height - menu_bar_height

    return (
        width,
        height,
        menu_bar_height,
        panel_width,
        rendering_width,
        rendering_height,
    )


def apply_resolution_scale(rendering_width, rendering_height):
    """Apply resolution scaling to render dimensions."""
    scaled_rendering_width = int(rendering_width * resolution_scale)
    scaled_rendering_height = int(rendering_height * resolution_scale)
