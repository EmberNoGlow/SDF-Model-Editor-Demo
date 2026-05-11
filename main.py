"""
SDF Model Editor - Main Entry Point

Refactored application with modular architecture:
- Rendering logic separated into rendering/
- UI components split into ui/
- Core event handling in app/core/
- Clean, extensible structure
"""

import glfw
import imgui
import time

from src.app.data.states import st
from src.app.data.consts import cn
from src.app.init import (
    init_application,
    init_scene,
    init_shader,
    init_opengl_resources,
    setup_glfw_callbacks,
    setup_time_tracking,
    load_user_configuration,
)

from src.ui.input_controls import rebuild_imgui_fonts

from src.app.core import (
    handle_delta_time,
    handle_fps_calculation,
    handle_keyboard_and_scene_input,
    handle_mouse_input,
    handle_camera_input,
    handle_home_key,
    detect_camera_changes,
)

from src.rendering import (
    recompile_shader,
    handle_shader_monitor,
    handle_frame_accumulation,
    render_scene_main,
    render_framebuffer_scaled,
    render_direct_screen,
)

from src.ui import (
    render_menu_bar,
    render_settings_window,
    render_editor_settings_window,
    render_export_windows,
    render_status_message_window,
    render_error_window,
    render_scene_tree_panel,
    render_inspector_panel,
    render_add_change_window,
    render_property_change_window,
    render_reparent_window,
    render_fps_overlay,
    render_orientation_overlay,
)

from src.ui.windows import (
    render_exit_confirmation_window,
    render_restart_confirmation_window,
    render_about_window,
)

from src.rendering.cleanup import cleanup_gl_resources
import src.ui.themes as ui_themes

from src.core.HistoryManager import History

def _setup_initial_state(scene_builder):
    """Initialize camera tracking variables and framebuffer state."""
    st.prev_cam_yaw = st.cam_yaw
    st.prev_cam_pitch = st.cam_pitch
    st.prev_cam_radius = st.cam_radius
    st.prev_cam_orbit = st.cam_orbit

    st.fbo = None
    st.render_texture = None
    st.fbo_width = 0
    st.fbo_height = 0


def _get_io():
    """Get ImGui I/O object."""
    return imgui.get_io()


def _recompile_with_error_handling():
    """Recompile shader with error handling."""
    success, new_uniforms = recompile_shader(st.scene_builder)
    if success:
        st.uniform_locs = new_uniforms
    return success


def _handle_dragging_interactions(window, scene_builder, camera):
    """Handle dragging and rotation interactions."""
    from src.app.interactions import (
        handle_primitive_dragging,
        handle_primitive_rotation,
    )

    if handle_primitive_dragging(window, scene_builder, camera):
        _recompile_with_error_handling()

    if handle_primitive_rotation(window, scene_builder):
        _recompile_with_error_handling()


def _handle_code_editor():
    """Check code editor for updates."""
    if st.CE_app is not None and st.CE_app.rec:
        st.additional_scene_code = st.CE_app.get_plain_text()
        st.CE_app.rec = False
        return True
    return False


def _render_all_ui(width, height, menu_bar_height, panel_width, rendering_width, rendering_height, scene_builder):
    """Render all UI elements."""
    render_menu_bar(st.window, scene_builder)
    render_settings_window(width, height, scene_builder, rendering_width, rendering_height)
    render_editor_settings_window(width, height)
    render_export_windows(st.window, scene_builder)
    render_status_message_window(width)
    render_error_window()
    render_scene_tree_panel(width, height, menu_bar_height, panel_width, scene_builder)
    render_inspector_panel(width, height, menu_bar_height, panel_width, scene_builder)
    render_add_change_window(width, height, scene_builder)
    render_property_change_window(width, height, scene_builder)
    render_reparent_window(width, height, scene_builder)
    render_fps_overlay(width, panel_width)
    render_orientation_overlay(width, panel_width)
    render_exit_confirmation_window(width, height)
    render_restart_confirmation_window(width, height)
    render_about_window(width, height)


def main():
    """Main application entry point and event loop."""
    # Initialization
    window, impl, ICONS, camera = init_application()
    scene_builder = init_scene()

    st.shader, st.uniform_locs = init_shader(scene_builder, st.shader_manager)
    if st.shader is None:
        print("Failed to compile initial shader. Exiting.")
        impl.shutdown()
        glfw.terminate()
        return

    vao, vbo, st.display_vao, st.display_vbo, st.display_shader = init_opengl_resources()

    st.scene_builder = scene_builder
    st.window = window
    st.vao = vao

    st.default_uconfig, _ = load_user_configuration()
    rebuild_imgui_fonts(impl, "assets/fonts/Roboto-Medium.ttf", 16.0)

    setup_glfw_callbacks(window)
    setup_time_tracking()
    _setup_initial_state(scene_builder)

    # Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        ui_themes.setup_theme()

        # Update timing
        handle_delta_time()
        handle_fps_calculation()

        # Handle input
        io = _get_io()
        handle_keyboard_and_scene_input(window, io, scene_builder, _recompile_with_error_handling)
        handle_mouse_input(window)
        handle_camera_input(window, io, camera)
        handle_home_key(io)

        # Update camera state
        detect_camera_changes(lambda: None)

        # Handle shader changes
        handle_shader_monitor(lambda: None)
        handle_frame_accumulation()

        # Handle interactions
        _handle_dragging_interactions(window, scene_builder, camera)

        if _handle_code_editor():
            _recompile_with_error_handling()

        # Render scene
        use_accumulation, width, height, menu_bar_height, panel_width, rendering_width, rendering_height = (
            render_scene_main(window, scene_builder, camera, vao, vbo, st.display_vao, st.display_vbo, st.display_shader)
        )

        # Framebuffer rendering
        if not (st.shader is not None and st.shader_choice == 1 and use_accumulation == 1):
            render_framebuffer_scaled(
                width, height, menu_bar_height, panel_width, rendering_width, rendering_height, st.display_shader
            )
            render_direct_screen(width, height, menu_bar_height, panel_width, rendering_width, rendering_height)

        # UI Rendering
        _render_all_ui(width, height, menu_bar_height, panel_width, rendering_width, rendering_height, scene_builder)

        # Render ImGui
        imgui.render()
        impl.render(imgui.get_draw_data())

        # Swap buffers
        glfw.swap_buffers(window)

    # Cleanup
    cleanup_gl_resources(
        st.fbo,
        st.render_texture,
        st.display_shader,
        st.display_vao,
        st.display_vbo,
        vao,
        vbo,
    )
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()