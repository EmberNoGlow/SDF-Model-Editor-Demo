import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import math
import hashlib
import ctypes
import imgui
import imgui.core
import src.app.Exporter as sdfexp
import src.app.CodeEditor as CodeEdit

from PIL import Image
from typing import Dict, List, Any

import os
import json
import numpy as np
import math
import copy

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import webbrowser
import pyperclip

from src import *
from src.ui import *
from src.utils import *
from src.rendering import *

from src.app.data.consts import cn
from src.app.data.states import st

import src.ui.themes as ui_themes

# --- GLOBAL STATE & INITIALIZATION ---

vertex_shader, fragment_shader_template, sdf_library = load_shaders()
shader_manager = ShaderManager(
    vertex_shader_src=vertex_shader, sdf_library_src=sdf_library, state=st
)
glob_history = History()


# --- SHADER RECOMPILATION DECORATOR ---


def MonitorChanges(func):
    """Decorator to track when shader recompilation is needed."""

    def wrapper(*args, **kwargs):
        st.monitor_shader_changes = True
        result = func(*args, **kwargs)
        return result

    return wrapper


# --- INITIALIZATION FUNCTIONS ---


def init_application():
    """Initialize GLFW, ImGui, and core application state."""
    window, impl = init_glfw_impl(cn["SCREEN_SIZE"])
    ICONS = load_all_textures()

    camera = Camera()
    st.theme = ui_themes.default_theme

    return window, impl, ICONS, camera


def init_scene():
    """Initialize the SDF scene builder with default primitives."""
    scene_builder = SDFSceneBuilder(glob_history, st.selected_item_id)

    scene_builder.add_standalone_primitive(
        "box", position=[0, 0, 0], size_or_radius=[0.5, 0.2, 0.8], ui_name="Cube"
    )

    return scene_builder


def init_shader(scene_builder):
    """Initialize the shader program and uniform locations."""
    shader, uniform_locs = shader_manager.get_or_compile(scene_builder)

    if shader is None:
        return None, None

    return shader, uniform_locs


def init_opengl_resources():
    """Initialize VAO, VBO, and display shader resources."""
    vao, vbo, display_vao, display_vbo, display_shader = init_vao_vbo()
    return vao, vbo, display_vao, display_vbo, display_shader


def load_user_configuration():
    """Load user configuration from disk, with fallback to defaults."""
    default_uconfig = {"Theme": st.theme, "UIScale": 1.0}

    try:
        UConfig = load_user_config("UserData/User.data")
    except:
        UConfig = default_uconfig

    if not UConfig or not isinstance(UConfig, dict):
        save_user_config("UserData/User.data", default_uconfig)
        UConfig = default_uconfig
    else:
        st.theme = UConfig["Theme"]
        for label, color in list(st.theme.items()):
            setattr(ui_themes, label, st.theme[label])
            ui_themes.setup_theme()

    return UConfig, default_uconfig


def setup_glfw_callbacks(window):
    """Set up GLFW window callbacks."""

    def on_window_close(wnd):
        glfw.set_window_should_close(wnd, False)
        st.show_exit_window = True

    glfw.set_window_close_callback(window, on_window_close)


def setup_time_tracking():
    """Initialize timing for delta time and FPS calculations."""
    st.start_time = time.time()
    st.prev_time = time.time()


# --- SHADER RECOMPILATION ---


def recompile_shader(scene_builder):
    """Recompile the active shader program and return new uniforms."""
    new_shader, new_uniforms = shader_manager.get_or_compile(scene_builder)

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


def _find_shader_hash_in_cache(shader):
    """Find a shader's hash in the cache."""
    for cached_hash, (cached_shader, _) in st.shader_cache.items():
        if cached_shader == shader:
            return cached_hash
    return None


# --- INPUT & INTERACTION HANDLING ---


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
    st.scaled_rendering_width = int(rendering_width * st.resolution_scale)
    st.scaled_rendering_height = int(rendering_height * st.resolution_scale)


def handle_delta_time():
    """Calculate delta time for frame-independent updates."""
    current_time = time.time()
    st.delta_time = current_time - st.prev_time
    st.prev_time = current_time


def handle_fps_calculation():
    """Update FPS counter every second."""
    st.fps_frames += 1
    current_time = time.time()

    if current_time - st.fps_clock >= 1.0:
        st.fps_value = st.fps_frames
        st.fps_frames = 0
        st.fps_clock = current_time


def handle_keyboard_and_scene_input(window, io, scene_builder):
    """Process keyboard input and update scene based on user actions."""
    handle = handler(
        window, io, scene_builder, glob_history, st.selected_item_id, st.selected_items
    )

    if handle[0]:  # Shader recompile needed
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    st.selected_item_id = handle[1]
    st.selected_items = handle[2]


def handle_mouse_input(window):
    """Handle middle mouse button and pan/zoom interactions."""
    # Middle mouse button press detection
    if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
        shift_pressed = _is_shift_pressed(window)

        if not st.is_mmb_pressed:
            st.is_mmb_pressed = True
            st.is_shift_mmb_pressed = shift_pressed
            st.last_x, st.last_y = glfw.get_cursor_pos(window)
            if shift_pressed:
                st.last_pan_x, st.last_pan_y = st.last_x, st.last_y
    elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.RELEASE:
        if st.is_mmb_pressed:
            st.is_mmb_pressed = False
            st.is_shift_mmb_pressed = False

    # Update cursor mode
    if st.is_mmb_pressed or st.dragging or st.R_dragging:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    else:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)


def _is_shift_pressed(window):
    """Check if either shift key is pressed."""
    return (
        glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )


def handle_camera_input(window, io, camera):
    """Handle camera zoom, rotation, and panning."""
    # Mouse wheel zoom
    if io.mouse_wheel != 0:
        st.target_radius -= io.mouse_wheel * cn["ZOOM_SENSITIVITY"]
        st.target_radius = max(
            cn["MIN_RADIUS"], min(cn["MAX_RADIUS"], st.target_radius)
        )

    # Smooth camera radius interpolation
    st.cam_radius += (st.target_radius - st.cam_radius) * (
        cn["CAMERA_LERP_FACTOR"] * st.delta_time
    )

    # Update camera angles only when MMB is pressed
    if st.is_mmb_pressed:
        current_x, current_y = glfw.get_cursor_pos(window)

        if st.is_shift_mmb_pressed:
            # Panning mode
            _handle_camera_pan(current_x, current_y)
        else:
            # Rotation mode
            _handle_camera_rotation(current_x, current_y)

    # Update camera vectors
    st.cam_yaw, st.cam_pitch = camera.update(
        st.target_yaw,
        st.target_pitch,
        st.target_pan_y,
        st.target_pan_x,
        cn["CAMERA_LERP_FACTOR"] * st.delta_time,
    )
    st.cam_orbit = camera.get_orbit()


def _handle_camera_pan(current_x, current_y):
    """Handle camera panning with Shift+MMB."""
    dx = current_x - st.last_pan_x
    dy = current_y - st.last_pan_y
    st.last_pan_x, st.last_pan_y = current_x, current_y
    st.target_pan_x += dx * cn["PAN_SENSITIVITY"]
    st.target_pan_y += dy * cn["PAN_SENSITIVITY"]


def _handle_camera_rotation(current_x, current_y):
    """Handle camera rotation with MMB."""
    dx = current_x - st.last_x
    dy = current_y - st.last_y
    st.last_x, st.last_y = current_x, current_y
    st.target_yaw -= dx * cn["MOUSE_SENSITIVITY"]
    st.target_pitch += dy * cn["MOUSE_SENSITIVITY"]
    st.target_pitch = max(cn["MIN_PITCH"], min(cn["MAX_PITCH"], st.target_pitch))


def handle_home_key(io):
    """Reset camera to home position when Home key is pressed."""
    if io.keys_down[glfw.KEY_HOME]:
        st.target_pan_x = st.target_pan_y = 0.0
        st.cam_orbit = [0.0, 0.0, 0.0]


def detect_camera_changes():
    """Detect if camera position/rotation changed and reset accumulation."""
    epsilon = 0.0001

    camera_changed = (
        abs(st.cam_yaw - st.prev_cam_yaw) > epsilon
        or abs(st.cam_pitch - st.prev_cam_pitch) > epsilon
        or abs(st.cam_radius - st.prev_cam_radius) > epsilon
        or any(abs(st.cam_orbit[i] - st.prev_cam_orbit[i]) > epsilon for i in range(3))
    )

    if camera_changed:
        st.frame_count = 0
        clear_accumulation_fbos()
        st.current_accum_index = 0

    # Update previous values
    st.prev_cam_yaw = st.cam_yaw
    st.prev_cam_pitch = st.cam_pitch
    st.prev_cam_radius = st.cam_radius
    st.prev_cam_orbit = st.cam_orbit


def handle_shader_monitor(scene_builder):
    """Monitor shader changes and reset accumulation if needed."""
    if st.monitor_shader_changes and st.shader_choice == 1:
        st.monitor_shader_changes = False
        st.frame_count = 0
        clear_accumulation_fbos()
        st.current_accum_index = 0


def handle_frame_accumulation():
    """Increment frame counter for accumulation-based rendering."""
    if st.shader_choice == 1:  # Cycles shader
        st.frame_count = min(st.frame_count + 1, st.max_frames)
    else:
        st.frame_count = 0


def handle_primitive_dragging(window, scene_builder, camera):
    """Handle primitive dragging with G key."""
    drag_result = dragging_primitive(window, scene_builder, camera)
    return drag_result


def handle_primitive_rotation(window, scene_builder):
    """Handle primitive rotation with R key."""
    rotate_result = rotate_privitive(window, scene_builder)
    return rotate_result


def handle_code_editor_updates():
    """Check code editor for updates and recompile if necessary."""
    if st.CE_app is not None and st.CE_app.rec:
        st.additional_scene_code = st.CE_app.get_plain_text()
        st.CE_app.rec = False
        return True
    return False


# RENDERING


def render_scene(
    window, scene_builder, camera, vao, vbo, display_vao, display_vbo, display_shader
):
    """Main rendering pipeline."""
    width, height, menu_bar_height, panel_width, rendering_width, rendering_height = (
        handle_window_resize(window)
    )

    apply_resolution_scale(rendering_width, rendering_height)

    use_accumulation = rendering_pass(
        st,
        st.shader,
        display_shader,
        vao,
        display_vao,
        st.uniform_locs,
        rendering_width,
        rendering_height,
        width,
        height,
        panel_width,
        menu_bar_height,
        setup_accumulation_buffer,
        bind_sprite_textures,
        set_move_pos_uniform,
        set_move_rot_uniform,
    )

    return (
        use_accumulation,
        width,
        height,
        menu_bar_height,
        panel_width,
        rendering_width,
        rendering_height,
    )


def render_framebuffer_scaling(
    width,
    height,
    menu_bar_height,
    panel_width,
    rendering_width,
    rendering_height,
    display_shader,
):
    """Render to framebuffer at scaled resolution."""
    # Skip if accumulation rendering already handled
    if st.shader is None or st.shader_choice == 1:
        return None, None, None, None, None

    if display_shader is None or st.resolution_scale == 1.0:
        return None, None, None, None, None

    (
        framebuffer_output,
        st.scaled_rendering_width,
        st.scaled_rendering_height,
        fbo,
        render_texture,
        fbo_width,
        fbo_height,
    ) = setup_framebuffer(
        st.scaled_rendering_width,
        st.scaled_rendering_height,
        st.fbo,
        st.render_texture,
        st.fbo_width,
        st.fbo_height,
    )

    if not framebuffer_output:
        return None, None, None, None, None

    # Render to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, st.scaled_rendering_width, st.scaled_rendering_height)
    glClear(GL_COLOR_BUFFER_BIT)

    _render_scene_to_framebuffer(
        display_shader,
        render_texture,
        width,
        height,
        rendering_width,
        rendering_height,
        panel_width,
        menu_bar_height,
    )

    return fbo, render_texture, st.fbo_width, st.fbo_height, framebuffer_output


def _render_scene_to_framebuffer(
    display_shader,
    render_texture,
    width,
    height,
    rendering_width,
    rendering_height,
    panel_width,
    menu_bar_height,
):
    """Internal: render scaled scene to framebuffer texture."""
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, width, height)

    glUseProgram(display_shader)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, render_texture)
    glUniform1i(glGetUniformLocation(display_shader, "renderTexture"), 0)

    glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
    glBindVertexArray(st.display_vao)
    glDrawArrays(GL_QUADS, 0, 4)
    glBindVertexArray(0)

    glViewport(0, 0, width, height)


def render_direct_to_screen(
    width, height, menu_bar_height, panel_width, rendering_width, rendering_height
):
    """Direct rendering to screen when scale is 1.0."""
    if st.shader is None:
        return

    glUseProgram(st.shader)

    if st.uniform_locs is not None:
        current_time_uniform = time.time() - st.start_time
        glUniform1f(st.uniform_locs["time"], current_time_uniform)
        glUniform2f(st.uniform_locs["resolution"], rendering_width, rendering_height)
        glUniform2f(
            st.uniform_locs["viewportOffset"],
            float(panel_width),
            float(menu_bar_height),
        )
        glUniform1f(st.uniform_locs["camYaw"], st.cam_yaw)
        glUniform1f(st.uniform_locs["camPitch"], st.cam_pitch)
        glUniform1f(st.uniform_locs["radius"], st.cam_radius)
        glUniform3f(
            st.uniform_locs["CamOrbit"],
            st.cam_orbit[0],
            st.cam_orbit[1],
            st.cam_orbit[2],
        )
        set_move_pos_uniform(st.shader, st.uniform_locs, st.drag_position)
        set_move_rot_uniform(st.shader, st.uniform_locs, st.drag_rot_position)

    if rendering_width > 0 and rendering_height > 0:
        glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
        glBindVertexArray(st.vao)
        bind_sprite_textures(st.uniform_locs, st.sprites_array)
        glDrawArrays(GL_QUADS, 0, 4)

    glViewport(0, 0, width, height)


# --- UI RENDERING - MENU BAR ---


def render_menu_bar(window, scene_builder):
    """Render the main menu bar and handle menu actions."""
    if not imgui.begin_main_menu_bar():
        return

    _render_file_menu(window, scene_builder)
    _render_edit_menu(scene_builder)
    _render_view_menu()
    _render_editor_menu()
    _render_about_menu()
    _render_shader_mode_buttons(window)

    imgui.end_main_menu_bar()


def _render_file_menu(window, scene_builder):
    """Render File menu with save/load/export options."""
    if not imgui.begin_menu("File", True):
        return

    if imgui.menu_item("Save Scene", "Ctrl+S")[0]:
        success, message = save_scene_dialog(scene_builder, window)
        st.save_load_message = message
        st.save_load_message_time = time.time()

    if imgui.menu_item("Load Scene", "Ctrl+O")[0]:
        success, message = load_scene_dialog(scene_builder)
        st.save_load_message = message
        st.save_load_message_time = time.time()
        if success:
            glob_history.undo_stack.clear()
            glob_history.redo_stack.clear()
            scene_builder.update_glob_history(glob_history)
            st.selected_item_id = None
            scene_builder.update_selected_item_id(st.selected_item_id)
            st.selection_mode = None
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

    imgui.separator()
    imgui.spacing()

    if imgui.begin_menu("Export..."):
        if imgui.menu_item("As Volume")[0]:
            st.show_export_vol_window = True
        if imgui.menu_item("To OBJ")[0]:
            st.show_export_obj_window = True
        imgui.end_menu()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    if imgui.menu_item("Exit", "Alt+F4")[0]:
        glfw.set_window_should_close(window, False)
        st.show_exit_window = True

    imgui.end_menu()


def _render_edit_menu(scene_builder):
    """Render Edit menu with primitive/operation addition and compilation."""
    if not imgui.begin_menu("Edit", True):
        return

    if imgui.menu_item("Add Primitive/Operation", "Ctrl+A")[0]:
        st.show_add_change_window = True
        st.pending_change_node_id = None

    if imgui.menu_item("Compile Shader", "Ctrl+B")[0]:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.end_menu()


def _render_view_menu():
    """Render View menu with settings."""
    if not imgui.begin_menu("View", True):
        return

    if imgui.menu_item("Settings", "F10")[0]:
        st.show_settings_window = True

    imgui.end_menu()


def _render_editor_menu():
    """Render Editor menu with editor settings."""
    if not imgui.begin_menu("Editor", True):
        return

    if imgui.menu_item("Settings")[0]:
        st.show_editor_settings_window = True

    imgui.end_menu()


def _render_about_menu():
    """Render About menu with information."""
    if not imgui.begin_menu("About", True):
        return

    if imgui.menu_item("Information")[0]:
        st.show_about_window = True

    imgui.end_menu()


def _render_shader_mode_buttons(window):
    """Render shader mode selection buttons."""
    cursor_pos = imgui.get_cursor_pos()
    window_width = imgui.get_window_width()
    remaining_width = window_width - cursor_pos.x

    button_width = 100
    spacing = 20
    total_buttons_width = 3 * button_width + 2 * spacing
    start_x = (cursor_pos.x + (remaining_width - total_buttons_width)) / 2

    # Template button
    imgui.set_cursor_pos_x(start_x)
    if imgui.button("Template", button_width):
        st.shader_choice = 0
        success, new_uniforms = recompile_shader(st.scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    # Cycles button
    imgui.set_cursor_pos_x(start_x + button_width + spacing)
    if imgui.button("Cycles", button_width):
        st.shader_choice = 1
        success, new_uniforms = recompile_shader(st.scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    # Script button
    imgui.set_cursor_pos_x(start_x + 2 * (button_width + spacing))
    if imgui.button("Script", button_width):
        _launch_code_editor()


def _launch_code_editor():
    """Launch the GLSL code editor in a separate thread."""

    def run_tkinter_app():
        if st.CE_app is None:
            st.CE_app = CodeEdit.GLSLEditor()

            def on_close():
                st.CE_app.destroy()
                st.CE_app = None

            st.CE_app.protocol("WM_DELETE_WINDOW", on_close)
            st.CE_app.mainloop()

    st.tkinter_thread = threading.Thread(target=run_tkinter_app, daemon=True)
    st.tkinter_thread.start()


# --- UI RENDERING - WINDOWS & DIALOGS ---


def render_settings_window(
    width, height, scene_builder, rendering_width, rendering_height
):
    """Render the main settings window."""
    if not st.show_settings_window:
        return

    imgui.set_next_window_position(width // 2 - 200, height // 2 - 150)
    imgui.set_next_window_size(400, 300)
    is_open, st.show_settings_window = imgui.begin(
        "Settings", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_settings_window = False
        imgui.end()
        return

    _render_shader_selection_ui(scene_builder)
    _render_resolution_scale_ui()
    _render_sky_color_ui(scene_builder)
    _render_grid_or_samples_ui(scene_builder)
    _render_sun_direction_ui(scene_builder)
    _render_render_size_info(rendering_width, rendering_height)

    imgui.spacing()
    if imgui.button("Close", -1):
        st.show_settings_window = False

    imgui.end()


def _render_shader_selection_ui(scene_builder):
    """Render shader selection combo."""
    imgui.text("Rendering Settings")
    imgui.separator()
    imgui.text("Fragment Shader:")

    clicked, st.shader_choice = imgui.combo(
        "##shader_select",
        st.shader_choice,
        [name.replace("shaders/fragment/", "") for name in st.shader_names],
    )

    if clicked:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.spacing()
    imgui.separator()
    imgui.spacing()


def _render_resolution_scale_ui():
    """Render resolution scale slider."""
    imgui.text("Resolution Scale:")
    imgui.same_line()
    imgui.text(f"{st.resolution_scale:.2f}x")

    changed, st.resolution_scale = imgui.slider_float(
        "##st.resolution_scale", st.resolution_scale, 0.25, 2.0, "%.2f"
    )

    if changed:
        st.frame_count = 0

    imgui.spacing()
    imgui.text_colored("1.0 = Normal resolution", 0.7, 0.7, 0.7, 1.0)
    imgui.text_colored("2.0 = Oversampling (better quality)", 0.7, 0.7, 0.7, 1.0)
    imgui.text_colored("<1.0 = Low resolution (better performance)", 0.7, 0.7, 0.7, 1.0)
    imgui.spacing()
    imgui.separator()


def _render_sky_color_ui(scene_builder):
    """Render sky color pickers."""
    imgui.text("Sky Top Color:")
    top_color_changed, top_color_rgba = imgui.color_edit3(
        "SkyTopColor##color",
        st.sky_top_color[0],
        st.sky_top_color[1],
        st.sky_top_color[2],
    )

    if top_color_changed:
        st.sky_top_color = list(top_color_rgba[:3])
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.text("Sky Bottom Color:")
    bottom_color_changed, bottom_color_rgba = imgui.color_edit3(
        "SkyBottomColor##color",
        st.sky_bottom_color[0],
        st.sky_bottom_color[1],
        st.sky_bottom_color[2],
    )

    if bottom_color_changed:
        st.sky_bottom_color = list(bottom_color_rgba[:3])
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_grid_or_samples_ui(scene_builder):
    """Render grid or samples UI based on shader choice."""
    if st.shader_choice == 0:
        imgui.text("Grid Enabled:")
        changed, st.GridEnabled = imgui.checkbox("", st.GridEnabled)
        if changed:
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

        imgui.spacing()
        imgui.separator()

    elif st.shader_choice == 1:
        imgui.text("Max Samples count:")
        changed, st.max_frames = imgui.input_int("", st.max_frames)
        st.max_frames = max(st.max_frames, 8)
        if changed:
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

        imgui.spacing()
        imgui.separator()


def _render_sun_direction_ui(scene_builder):
    """Render sun direction input."""
    imgui.text("Sun:")
    changed, st.LightDir = input_vec3("Sun Direction", st.LightDir)
    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.spacing()
    imgui.separator()


def _render_render_size_info(rendering_width, rendering_height):
    """Display current render sizes."""
    scaled_w = int(rendering_width * st.resolution_scale)
    scaled_h = int(rendering_height * st.resolution_scale)
    imgui.text(f"Current render size: {scaled_w}x{scaled_h}")
    imgui.text(f"Base size: {rendering_width}x{rendering_height}")


def render_editor_settings_window(width, height):
    """Render the editor settings window with tabbed interface."""
    if not st.show_editor_settings_window:
        return

    imgui.set_next_window_position(width // 2 - 400, height // 2 - 300)
    imgui.set_next_window_size(800, 600)

    is_open, st.show_editor_settings_window = imgui.begin(
        "Editor Settings", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_editor_settings_window = False
        imgui.end()
        return

    if imgui.begin_child("SettingsTabs", 0, 0, border=False):
        sidebar_width = 150

        # Left sidebar navigation
        imgui.begin_group()
        if imgui.button("Themes", width=sidebar_width):
            st.current_settings_tab = "Themes"
        imgui.separator()
        if imgui.button("User", width=sidebar_width):
            st.current_settings_tab = "User"
        imgui.separator()
        if imgui.button("Shortcuts", width=sidebar_width):
            st.current_settings_tab = "Shortcuts"
        imgui.end_group()

        imgui.set_cursor_pos_x(sidebar_width + 10)
        imgui.same_line()

        # Right content panel
        window_width = imgui.get_window_width()
        content_width = window_width - sidebar_width - 30

        if imgui.begin_child("SettingsContent", content_width, 400, border=False):
            _render_settings_tab_content()
            imgui.end_child()

        imgui.end_child()

    imgui.end()


def _render_settings_tab_content():
    """Render the active settings tab content."""
    if st.current_settings_tab == "Themes":
        _render_themes_tab()
    elif st.current_settings_tab == "User":
        _render_user_tab()
    elif st.current_settings_tab == "Shortcuts":
        _render_shortcuts_tab()


def _render_themes_tab():
    """Render themes customization tab."""
    changes = []
    for label in st.theme:
        item = st.theme[label]
        if isinstance(item, list) and len(item) == 4:
            changed, color_rgba = imgui.color_edit4(label, *item)
            if changed:
                changes.append((label, list(color_rgba)))
        elif isinstance(item, list) and len(item) == 2:
            changed, size = input_vec2(label, item)
            if changed:
                changes.append((label, list(size)))

    for label, new_value in changes:
        st.theme[label] = new_value
        setattr(ui_themes, label, new_value)

    if changes:
        ui_themes.setup_theme()

    imgui.spacing()
    if imgui.button("Reset Theme", -1):
        st.theme = copy.deepcopy(st.default_uconfig["Theme"])
        for label, item in st.theme.items():
            setattr(ui_themes, label, item)
        ui_themes.setup_theme()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()


def _render_user_tab():
    """Render user profile settings tab."""
    imgui.text("User Profile Settings Content Here... WIP")


def _render_shortcuts_tab():
    """Render keyboard shortcuts tab."""
    for name, keys in ShortCuts.items():
        imgui.text(name)
        imgui.same_line()
        for key in (keys,):
            imgui.text(str(key))
            imgui.same_line()
        imgui.spacing()


def render_export_windows(window, scene_builder):
    """Render volume and OBJ export windows."""
    _render_export_volume_window(window, scene_builder)
    _render_export_obj_window(window, scene_builder)


def _render_export_volume_window(window, scene_builder):
    """Render the volume export dialog."""
    if not st.show_export_vol_window:
        return

    width = glfw.get_window_size(window)[0]
    height = glfw.get_window_size(window)[1]

    imgui.set_next_window_position(width // 2 - 150, height // 2 - 125)
    imgui.set_next_window_size(300, 250)
    is_open, st.show_export_vol_window = imgui.begin(
        "Export as Volume", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_export_vol_window = False
        imgui.end()
        return

    imgui.text("Grid Size:")
    changed, st.grid_size = imgui.input_int("##GridSize", st.grid_size, 8)
    imgui.text_colored(
        "Note that its dimensions range \nfrom -GridSize/2 to +GridSize/2.",
        0.56,
        0.93,
        0.56,
    )

    imgui.spacing()
    changed, st.vox_quality = input_float("Vox. Quality", st.vox_quality, 0.25, 100)
    changed, st.exp_use_color = imgui.checkbox("Use Color", st.exp_use_color)

    imgui.separator()
    imgui.spacing()

    file_preview_size = sdfexp.calculate_sdf_file_size(
        st.grid_size, st.vox_quality, st.exp_use_color
    )
    if file_preview_size[1] > 1:
        imgui.text(f"File size = {file_preview_size[1]:.2f} mb")
    else:
        imgui.text(f"File size = {file_preview_size[0]:.2f} kb")

    imgui.spacing()
    imgui.spacing()

    if imgui.button("Cancel", 135, 30):
        st.show_export_vol_window = False

    imgui.same_line(150)

    if imgui.button("Export", 135, 30):
        code = scene_builder.generate_raymarch_code()
        comp_bin = sdfexp.compute_sdf_3d(
            st.grid_size,
            st.vox_quality,
            code,
            st.additional_scene_code,
            st.exp_use_color,
            window,
        )
        save_sdfvol_dialog(sdfexp, comp_bin)
        st.show_export_vol_window = False

    imgui.end()


def _render_export_obj_window(window, scene_builder):
    """Render the OBJ export dialog."""
    if not st.show_export_obj_window:
        return

    width = glfw.get_window_size(window)[0]
    height = glfw.get_window_size(window)[1]

    imgui.set_next_window_position(width // 2 - 150, height // 2 - 130)
    imgui.set_next_window_size(300, 260)
    is_open, st.show_export_obj_window = imgui.begin(
        "Export to OBJ", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_export_obj_window = False
        imgui.end()
        return

    imgui.text("Grid Size:")
    changed, st.grid_size = imgui.input_int("##GridSize", st.grid_size, 8)
    imgui.text_colored(
        "Note that its dimensions range \nfrom -GridSize/2 to +GridSize/2.",
        0.56,
        0.93,
        0.56,
    )

    imgui.spacing()
    changed, st.vox_quality = input_float(
        "Voxelization Quality", st.vox_quality, 0.25, 100
    )
    imgui.separator()
    imgui.spacing()

    changed, st.export_level = input_float("Level", st.export_level, 0.05, 100)
    st.export_level = np.clip(st.export_level, 0.0, 1.0)

    imgui.spacing()
    changed, st.export_z_up = imgui.checkbox("Z up", st.export_z_up)
    imgui.same_line()
    changed, st.exp_use_color = imgui.checkbox("Use Color", st.exp_use_color)

    imgui.separator()
    imgui.spacing()

    if imgui.button("Cancel", 135, 30):
        st.show_export_obj_window = False

    imgui.same_line(150)

    if imgui.button("Export", 135, 30):
        code = scene_builder.generate_raymarch_code()
        comp_bin = sdfexp.compute_sdf_3d(
            st.grid_size,
            st.vox_quality,
            code,
            st.additional_scene_code,
            st.exp_use_color,
            window,
        )

        dist_sdf = None
        color_sdf = None

        if isinstance(comp_bin, tuple):
            elvl = np.interp(
                st.export_level, [0, 1], [comp_bin[0].min(), comp_bin[0].max()]
            )
            dist_sdf, color_sdf = comp_bin
        else:
            elvl = np.interp(st.export_level, [0, 1], [comp_bin.min(), comp_bin.max()])
            dist_sdf = comp_bin

        success, message = save_sdfobj_dialog(
            sdfexp, dist_sdf, color_sdf, st.export_z_up, elvl, st.exp_use_color
        )
        st.export_obj_message = [success, message]
        st.export_obj_message_time = time.time()
        st.show_export_obj_window = False

    imgui.end()


def render_status_message_window(width):
    """Render transient status messages."""
    if st.save_load_message is not None:
        if time.time() - st.save_load_message_time < 3.0:
            imgui.set_next_window_position(width // 2 - 150, 100)
            imgui.begin(
                "Status", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE
            )

            is_success = (
                "saved" in st.save_load_message.lower()
                or "loaded" in st.save_load_message.lower()
            )
            color = (0.0, 1.0, 0.0, 1.0) if is_success else (1.0, 0.0, 0.0, 1.0)
            imgui.text_colored(st.save_load_message, *color)

            imgui.same_line(350, 0)
            if imgui.button("copy"):
                pyperclip.copy(st.save_load_message)

            imgui.end()
        else:
            st.save_load_message = None

    if st.export_obj_message is not None:
        if time.time() - st.export_obj_message_time < 3.0:
            imgui.set_next_window_position(width // 2 - 150, 100)
            imgui.begin(
                "Status", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE
            )

            is_success = st.export_obj_message[0]
            color = (0.0, 1.0, 0.0, 1.0) if is_success else (1.0, 0.0, 0.0, 1.0)
            imgui.text_colored(st.export_obj_message[1], *color)

            imgui.same_line(350, 0)
            if imgui.button("copy"):
                pyperclip.copy(st.export_obj_message[1])

            imgui.end()
        else:
            st.export_obj_message = None


def render_error_window():
    """Render shader compilation error display."""
    if not st.shader_compile_error:
        return

    width = (
        glfw.get_window_size(glfw.get_current_context())[0]
        if glfw.get_current_context()
        else 800
    )
    height = (
        glfw.get_window_size(glfw.get_current_context())[1]
        if glfw.get_current_context()
        else 600
    )

    imgui.set_next_window_position(width // 2 - 200, height // 2 - 50)
    imgui.set_next_window_size(400, 100)
    imgui.begin("Shader Compilation Error", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
    imgui.text_colored("Error:", 1.0, 0.0, 0.0, 1.0)
    imgui.same_line()
    imgui.text_wrapped(st.shader_compile_error)
    if imgui.button("Dismiss"):
        st.shader_compile_error = None
    imgui.end()


def render_scene_tree_panel(width, height, menu_bar_height, panel_width, scene_builder):
    """Render the left panel with scene hierarchy."""
    imgui.set_next_window_position(0, menu_bar_height)
    imgui.set_next_window_size(panel_width, height - menu_bar_height)
    imgui.begin(
        "Scene Tree",
        False,
        imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
    )

    imgui.text("Scene Hierarchy:")
    imgui.separator()

    for root_id in scene_builder.root_children:
        _render_scene_node_recursive(root_id, scene_builder, panel_width, 0)

    imgui.spacing()
    imgui.separator()

    if imgui.button(f"Add (Ctrl+A)", -1):
        st.show_add_change_window = True
        st.pending_change_node_id = None

    imgui.end()


def _render_scene_node_recursive(node_id, scene_builder, panel_width, depth):
    """Recursively render scene hierarchy nodes."""
    node = scene_builder.get_node(node_id)
    if not node:
        return

    item_data = node.item_data
    children = node.children
    is_leaf = len(children) == 0

    label = _format_node_label(item_data.ui_name, node_id)

    # Determine tree node flags
    flags = 0
    if not is_leaf:
        flags |= imgui.TREE_NODE_DEFAULT_OPEN
    else:
        flags |= imgui.TREE_NODE_LEAF

    if st.selected_item_id == node_id:
        flags |= imgui.TREE_NODE_SELECTED

    # Root level movement controls
    if node.parent_id is None and node_id in scene_builder.root_children:
        _render_root_node_movement_buttons(node_id, scene_builder)

    # Delete button
    imgui.push_id(f"delete_{node_id}")
    clicked_delete = imgui.button("X", 20, 20)
    imgui.pop_id()

    if clicked_delete:
        scene_builder.delete_node(node_id)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms
        return

    imgui.same_line()

    # Tree node
    node_open = imgui.tree_node(label, flags)

    # Handle selection
    if imgui.is_item_clicked():
        _handle_scene_node_selection(node_id, scene_builder)

    # Right-click context menu
    _render_node_context_menu(node_id, node, scene_builder)

    # Render children
    if children:
        for child_id in list(children):
            _render_scene_node_recursive(
                child_id, scene_builder, panel_width, depth + 1
            )

    if node_open:
        imgui.tree_pop()


def _format_node_label(name, op_id, max_chars=16):
    """Format node label with truncation."""
    if len(name) > max_chars:
        truncated_name = name[: max_chars - 3] + "..."
    else:
        truncated_name = name
    return f"{truncated_name} ({op_id})"


def _render_root_node_movement_buttons(node_id, scene_builder):
    """Render up/down buttons for root-level nodes."""
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (1, 1))
    root_idx = scene_builder.root_children.index(node_id)

    if imgui.arrow_button(f"##up_{node_id}", 2):
        if root_idx > 0:
            scene_builder.move_root_node(node_id, root_idx - 1)
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

    imgui.same_line()

    if imgui.arrow_button(f"##down_{node_id}", 3):
        if root_idx < len(scene_builder.root_children) - 1:
            scene_builder.move_root_node(node_id, root_idx + 1)
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

    imgui.pop_style_var(1)
    imgui.same_line()


def _handle_scene_node_selection(node_id, scene_builder):
    """Handle left-click node selection with Ctrl for multi-select."""
    io_local = imgui.get_io()
    if io_local.key_ctrl:
        if node_id in st.selected_items:
            st.selected_items.remove(node_id)
        else:
            st.selected_items.add(node_id)
        if len(st.selected_items) > 0:
            st.selected_item_id = None
    else:
        st.selected_items.clear()
        st.selected_item_id = node_id
        scene_builder.update_selected_item_id(st.selected_item_id)
        st.selection_mode = "node"
        st.renaming_item_id = None
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_node_context_menu(node_id, node, scene_builder):
    """Render right-click context menu for nodes."""
    popup_id = f"node_ctx_{node_id}"
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
        imgui.open_popup(popup_id)

    if imgui.begin_popup(popup_id):
        if node.node_type == "operation":
            if imgui.menu_item("Change Operation Type")[0]:
                st.pending_change_node_id = node_id
                st.show_add_change_window = True
                imgui.close_current_popup()
        else:
            if imgui.menu_item("Change Type")[0]:
                st.pending_change_node_id = node_id
                st.show_add_change_window = True
                imgui.close_current_popup()

        imgui.separator()

        if imgui.menu_item("Change Properties")[0]:
            st.property_change_node_id = node_id
            st.show_property_change_window = True
            imgui.close_current_popup()

        if imgui.menu_item("Reparent")[0]:
            st.reparent_node_id = node_id
            st.show_reparent_window = True
            imgui.close_current_popup()

        imgui.end_popup()


def render_inspector_panel(width, height, menu_bar_height, panel_width, scene_builder):
    """Render the right panel with property inspector."""
    imgui.set_next_window_position(width - panel_width, menu_bar_height)
    imgui.set_next_window_size(panel_width, height - menu_bar_height)
    imgui.begin(
        "Inspector",
        False,
        imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
    )

    if (
        st.selected_item_id is not None
        and st.selected_item_id in scene_builder.id_to_node
    ):
        node = scene_builder.get_node(st.selected_item_id)
        if node:
            _render_node_inspector(node, scene_builder, panel_width)
    else:
        imgui.text("No node selected")
        imgui.text("Click on a node in the Scene Tree")

    imgui.end()


def _render_node_inspector(node, scene_builder, panel_width):
    """Render inspector content for a selected node."""
    item_data = node.item_data

    imgui.text(f"Type: {node.node_type}")
    if node.node_type == "operation":
        imgui.text(f"Operation: {item_data.operation_type}")
    else:
        imgui.text(f"Primitive: {item_data.primitive_type}")

    imgui.separator()
    imgui.text(f"Selected: {item_data.ui_name}")

    _render_node_rename_ui(node, scene_builder)
    imgui.separator()

    if node.node_type == "primitive":
        _render_primitive_inspector(node, scene_builder, item_data, panel_width)
    elif node.node_type == "operation":
        _render_operation_inspector(node, scene_builder, item_data)


def _render_node_rename_ui(node, scene_builder):
    """Render rename input field."""
    if imgui.button("Rename"):
        st.renaming_item_id = st.selected_item_id
        st.rename_text = node.item_data.ui_name

    if st.renaming_item_id == st.selected_item_id:
        changed, st.rename_text = imgui.input_text("##rename", st.rename_text, 256)

        if imgui.button("OK", 60):
            scene_builder.rename_node(st.selected_item_id, st.rename_text)
            st.renaming_item_id = None
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

        imgui.same_line()
        if imgui.button("Cancel", 60):
            st.renaming_item_id = None


def _render_primitive_inspector(node, scene_builder, item_data, panel_width):
    """Render inspector for primitive nodes."""
    panel_elem_width_vec3 = (panel_width / 4) - 14
    panel_elem_width_float = (panel_width / 2) - 14

    primitive = node.item_data

    # Type-specific properties
    _render_primitive_type_properties(
        primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float
    )

    # Common transform properties
    imgui.begin_group()
    imgui.spacing()
    imgui.separator()
    imgui.dummy((panel_width / 4) - 8, 0)
    imgui.same_line()
    imgui.text_colored("Transform", 1.0, 0.7, 0.5, 1.0)
    imgui.spacing()
    imgui.end_group()

    changed, item_data.position = input_vec3(
        "Position", item_data.position, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    changed, item_data.rotation = input_vec3(
        "Rotation", item_data.rotation, cn["STEP_VARIABLE_ANGLE"], panel_elem_width_vec3
    )
    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    changed, item_data.scale = input_vec3(
        "Scale", item_data.scale, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    # Color properties
    imgui.begin_group()
    imgui.spacing()
    imgui.separator()
    imgui.dummy((panel_width / 3) - 12, 0)
    imgui.same_line()
    imgui.text_colored("Color", 1.0, 0.7, 0.5, 1.0)
    imgui.spacing()
    imgui.end_group()

    color_changed, color_rgba = imgui.color_edit3("Color##color", *primitive.color)
    if color_changed:
        primitive.color = list(color_rgba[:3])
        scene_builder.modify_primitive_property(node.item_id, "color", primitive.color)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.spacing()
    imgui.text("RGB Sliders:")
    r_changed, primitive.color[0] = imgui.slider_float(
        "R##color_r", primitive.color[0], 0.0, 1.0
    )
    g_changed, primitive.color[1] = imgui.slider_float(
        "G##color_g", primitive.color[1], 0.0, 1.0
    )
    b_changed, primitive.color[2] = imgui.slider_float(
        "B##color_b", primitive.color[2], 0.0, 1.0
    )

    if r_changed or g_changed or b_changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_primitive_type_properties(
    primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float
):
    """Render type-specific primitive properties."""
    prim_type = primitive.primitive_type

    if prim_type == "sprite":
        _render_sprite_properties(
            primitive,
            node,
            scene_builder,
            panel_elem_width_vec3,
            panel_elem_width_float,
        )
    elif prim_type == "cone":
        _render_cone_properties(primitive, scene_builder, panel_elem_width_float)
    elif prim_type == "plane":
        _render_plane_properties(
            primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float
        )
    elif prim_type == "pointer":
        _render_pointer_properties(
            primitive,
            node,
            scene_builder,
            panel_elem_width_vec3,
            panel_elem_width_float,
        )
    elif prim_type == "curve":
        _render_curve_properties(
            primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float
        )
    else:
        _render_standard_primitive_properties(
            primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float
        )

    # Round box special parameter
    if prim_type == "round_box":
        imgui.spacing()
        changed, primitive.kwargs["radius"] = input_float(
            "Radius",
            primitive.kwargs.get("radius", 0.1),
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        if changed:
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms


def _render_sprite_properties(
    primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float
):
    """Render sprite-specific properties."""
    sprite_idx = primitive.kwargs.get("sprite_index", None)
    if sprite_idx is None or sprite_idx >= len(st.sprites_array):
        imgui.text_colored("Sprite data missing or corrupted", 1.0, 0.0, 0.0, 1.0)
        return

    spr = st.sprites_array[sprite_idx]
    imgui.text("Plane parameters:")

    changed, primitive.position = input_vec3(
        "Point", primitive.position, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    changed2, spr.planeNormal = input_vec3(
        "Normal", spr.planeNormal, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    changed3, spr.planeWidth = input_float(
        "Width", spr.planeWidth, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    changed4, spr.planeHeight = input_float(
        "Height", spr.planeHeight, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    spr.planePoint = primitive.position

    if changed or changed2 or changed3 or changed4:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.separator()
    imgui.text("Mapping:")

    uv2 = spr.uvSize
    changed_uv, uv2 = input_vec2("UV Size", uv2, 0.1, panel_elem_width_vec3)
    spr.uvSize[0], spr.uvSize[1] = uv2[0], uv2[1]

    changed_alpha, spr.Alpha = input_float(
        "Alpha", spr.Alpha, 0.01, panel_elem_width_float
    )
    changed_lod, spr.LOD = input_float("LOD", spr.LOD, 0.1, panel_elem_width_float)

    if changed_uv or changed_alpha or changed_lod:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    # Texture status and loading
    if spr.texture_id:
        imgui.text(f"Texture loaded: {spr.tex_size[0]}x{spr.tex_size[1]}")
    else:
        imgui.text_colored("No texture loaded", 0.9, 0.3, 0.3, 1.0)

    imgui.spacing()
    if imgui.button("Load Texture", -1):
        _load_sprite_texture(spr, sprite_idx, scene_builder)


def _load_sprite_texture(sprite, sprite_idx, scene_builder):
    """Load texture file for sprite."""
    root = tk.Tk()
    root.withdraw()
    filetypes = [
        ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tga")),
        ("All files", "*.*"),
    ]
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    root.destroy()

    if filepath:
        ok = sprite.load_texture_from_file(filepath)
        if ok:
            sprite.SprTexture = f"sprTex{sprite_idx}"
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms


def _render_cone_properties(primitive, scene_builder, panel_elem_width_float):
    """Render cone-specific properties."""
    c_sin = primitive.kwargs.get("c_sin", 0.5)
    c_cos = primitive.kwargs.get("c_cos", 0.866)
    height = primitive.kwargs.get("height", 1.0)

    changed1, c_sin = input_float(
        "Sin(Angle)", c_sin, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    changed2, c_cos = input_float(
        "Cos(Angle)", c_cos, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    changed3, height = input_float(
        "Height", height, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )

    if changed1 or changed2 or changed3:
        primitive.kwargs["c_sin"] = c_sin
        primitive.kwargs["c_cos"] = c_cos
        primitive.kwargs["height"] = height
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_plane_properties(
    primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float
):
    """Render plane-specific properties."""
    normal = primitive.kwargs.get("normal", [0.0, 1.0, 0.0])
    h = primitive.kwargs.get("h", 0.0)

    changed1, normal = input_vec3(
        "Normal", normal, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    changed2, h = input_float(
        "Offset (h)", h, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )

    if changed1 or changed2:
        # Normalize the normal vector
        norm_len = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        if norm_len > 0.001:
            normal = [normal[0] / norm_len, normal[1] / norm_len, normal[2] / norm_len]

        primitive.kwargs["normal"] = normal
        primitive.kwargs["h"] = h
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_pointer_properties(
    primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float
):
    """Render pointer function properties."""
    changed_pos, primitive.position = input_vec3(
        "Position", primitive.position, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )

    if changed_pos:
        scene_builder.modify_primitive_property(
            node.item_id, "position", primitive.position
        )
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    # Pointer function selection
    pointer_funcs = [
        "pointer_identity",
        "pointer_symmetry_x",
        "pointer_symmetry_y",
        "pointer_symmetry_z",
    ]

    current_func = primitive.kwargs.get("func", "pointer_identity")
    try:
        current_index = pointer_funcs.index(current_func)
    except ValueError:
        pointer_funcs.append(current_func)
        current_index = len(pointer_funcs) - 1

    clicked, new_index = imgui.combo("Function", current_index, pointer_funcs)
    if clicked:
        new_func = pointer_funcs[new_index]
        primitive.kwargs["func"] = new_func
        scene_builder.modify_primitive_property(node.item_id, "kwargs.func", new_func)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.separator()
    imgui.text(
        "Pointer functions mutate \nthe raymarch point `p` \nfor subsequent primitives."
    )
    imgui.text_colored(
        "Place a pointer earlier in \nthe tree to affect later objects.",
        0.9,
        0.8,
        0.2,
        1.0,
    )


def _render_curve_properties(
    primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float
):
    """Render curve-specific properties."""
    imgui.spacing()

    # Points array editor
    points = primitive.kwargs.get("points", [[0, 0, 0], [1, 1, 1]])
    imgui.text("Curve Points:")

    points_to_remove = None
    for i, pt in enumerate(points):
        changed, new_pt = input_vec3(
            f"Point {i}", list(pt), cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
        )
        if changed:
            points[i] = new_pt
            primitive.kwargs["points"] = points
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

        imgui.same_line()
        if imgui.button(f"Remove##pt{i}", width=60):
            points_to_remove = i

    if points_to_remove is not None and len(points) > 2:
        points.pop(points_to_remove)
        primitive.kwargs["points"] = points
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    if imgui.button("Add Point", width=-1):
        points.append([0.0, 0.0, 0.0])
        primitive.kwargs["points"] = points
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.spacing()

    # Thickness parameter
    thickness = primitive.kwargs.get("thickness", 0.1)
    changed, thickness = input_float(
        "Thickness", thickness, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    if changed:
        primitive.kwargs["thickness"] = thickness
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_standard_primitive_properties(
    primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float
):
    """Render standard size/radius properties for most primitives."""
    primitive.size_or_radius = (
        list(primitive.size_or_radius)
        if isinstance(primitive.size_or_radius, tuple)
        else primitive.size_or_radius
    )
    primitive.size_or_radius = (
        [primitive.size_or_radius]
        if isinstance(primitive.size_or_radius, float)
        else primitive.size_or_radius
    )

    changed = False
    prim_type = primitive.primitive_type

    if prim_type == "sphere":
        changed, primitive.size_or_radius[0] = input_float(
            "Radius",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )

    elif prim_type == "torus":
        changed1, primitive.size_or_radius[0] = input_float(
            "Major Radius",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Minor Radius",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2

    elif prim_type == "hex_prism":
        changed1, primitive.size_or_radius[0] = input_float(
            "Hex Radius",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Height",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2

    elif prim_type == "vertical_capsule":
        changed1, primitive.size_or_radius[0] = input_float(
            "Height",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Radius",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2

    elif prim_type == "capped_cylinder":
        changed1, primitive.size_or_radius[0] = input_float(
            "Radius",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Height",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2

    elif prim_type == "rounded_cylinder":
        changed1, primitive.size_or_radius[0] = input_float(
            "Radius A",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Radius B",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed3, primitive.kwargs["height"] = input_float(
            "Height",
            primitive.kwargs.get("height", 1.0),
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2 or changed3

    elif prim_type not in ["cone", "plane", "pointer", "sprite", "curve"]:
        changed, primitive.size_or_radius = input_vec3(
            "Size",
            primitive.size_or_radius,
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_vec3,
        )

    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_operation_inspector(node, scene_builder, item_data):
    """Render inspector for operation nodes."""
    imgui.text(f"Operation Type: {item_data.operation_type}")

    # Show operands
    imgui.text("Operands:")
    for i, operand_id in enumerate(node.children):
        operand_node = scene_builder.get_node(operand_id)
        if operand_node:
            imgui.text(f"  {i+1}. {operand_node.item_data.ui_name} ({operand_id})")

    # Show smooth_k if applicable
    if hasattr(item_data, "smooth_k") and item_data.smooth_k is not None:
        changed, new_k = imgui.slider_float("Smooth K", item_data.smooth_k, 0.0, 1.0)
        if changed:
            item_data.smooth_k = new_k
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms


def render_add_change_window(width, height, scene_builder):
    """Render the combined add/change type dialog."""
    if not st.show_add_change_window:
        return

    imgui.set_next_window_position(width // 2 - 300, height // 2 - 235)
    imgui.set_next_window_size(600, 470)
    is_open, st.show_add_change_window = imgui.begin(
        "Add / Change Type", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_add_change_window = False
        st.pending_change_node_id = None
        imgui.end()
        return

    primitives_list = _get_primitives_list()
    operations_list = _get_operations_list()

    # Two-column layout
    imgui.columns(2, "add_change_cols", border=True)
    imgui.set_column_width(0, 290)

    # Left column: Primitives
    imgui.text("Primitives")
    imgui.separator()

    for label, prim_type, size in primitives_list:
        if imgui.button(label, -1, 24):
            _handle_primitive_selection(prim_type, label, size, scene_builder)

    # Right column: Operations
    imgui.next_column()
    imgui.text("Operations")
    imgui.separator()

    for label, op_type, operand_count, description in operations_list:
        if imgui.button(label, -1, 24):
            _handle_operation_selection(op_type, label, scene_builder)

        if imgui.is_item_hovered():
            imgui.set_tooltip(description)

    # Bottom buttons
    imgui.columns(1)
    imgui.separator()
    imgui.spacing()
    imgui.same_line(20, 0)

    if imgui.button("Cancel", 265, 28):
        st.show_add_change_window = False
        st.pending_change_node_id = None

    imgui.same_line(305, 0)

    if imgui.button("Close", 265, 28):
        st.show_add_change_window = False
        st.pending_change_node_id = None

    imgui.end()


def _get_primitives_list():
    """Return list of available primitives."""
    return [
        ("Box", "box", (0.5, 0.5, 0.5)),
        ("Sphere", "sphere", 0.5),
        ("Round Box", "round_box", (0.5, 0.5, 0.5)),
        ("Torus", "torus", (0.5, 0.25)),
        ("Cone", "cone", None),
        ("Plane", "plane", None),
        ("Hex Prism", "hex_prism", (0.5, 0.5)),
        ("Vertical Capsule", "vertical_capsule", (1.0, 0.3)),
        ("Capped Cylinder", "capped_cylinder", (0.3, 1.0)),
        ("Rounded Cylinder", "rounded_cylinder", (0.3, 0.3)),
    ]


def _get_operations_list():
    """Return list of available operations."""
    return [
        ("Union", "union", 2, "Combines two shapes (minimum distance)"),
        ("Subtraction", "sub", 2, "Subtracts second from first"),
        ("Intersection", "inter", 2, "Keeps only overlapping parts"),
        ("Smooth Union", "sunion", 2, "Union with smooth blending"),
        ("Smooth Subtraction", "ssub", 2, "Subtraction with smooth blending"),
        ("Smooth Intersection", "sinter", 2, "Intersection with smooth blending"),
        ("Mix", "mix", 2, "Blends between two distances"),
        ("Invert", "invert", 1, "Inverts the shape"),
        ("Round", "round", 1, "Rounds the shape"),
        ("Onion", "onion", 1, "Creates a shell effect"),
        ("XOR", "xor", 2, "Exclusive OR operation"),
        ("snoiseDisp", "snoiseDisp", 1, "Noise displacement"),
    ]


def _handle_primitive_selection(prim_type, label, size, scene_builder):
    """Handle primitive type selection in add/change dialog."""
    if st.pending_change_node_id is None:
        # Add new primitive
        new_id = scene_builder.add_standalone_primitive(
            prim_type,
            position=[0.0, 0.0, 0.0],
            size_or_radius=size if size is not None else 0.5,
            ui_name=label,
        )
        if new_id:
            st.selected_items.clear()
            st.selected_item_id = new_id
            scene_builder.update_selected_item_id(st.selected_item_id)
            st.selection_mode = "node"
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms
    else:
        # Change existing node to this primitive
        scene_builder.change_node_to_primitive(
            st.pending_change_node_id,
            prim_type,
            position=None,
            size_or_radius=(size if size is not None else 0.5),
        )
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

        st.pending_change_node_id = None

    st.show_add_change_window = False


def _handle_operation_selection(op_type, label, scene_builder):
    """Handle operation type selection in add/change dialog."""
    if st.pending_change_node_id is None:
        # Add new operation
        new_op_id = scene_builder.add_operation_with_auto_primitives(
            op_type, auto_primitive_type="box", ui_name=label
        )
        if new_op_id:
            st.selected_items.clear()
            st.selected_item_id = new_op_id
            scene_builder.update_selected_item_id(st.selected_item_id)
            st.selection_mode = "node"
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms
    else:
        # Convert existing node to this operation type
        scene_builder.change_node_to_operation(
            st.pending_change_node_id, op_type, auto_primitive_type="box"
        )
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

        st.pending_change_node_id = None

    st.show_add_change_window = False


def render_property_change_window(width, height, scene_builder):
    """Render the property change window for symmetry and other properties."""
    if not st.show_property_change_window:
        return

    imgui.set_next_window_position(width // 2 - 150, height // 2 - 125)
    imgui.set_next_window_size(300, 250)
    is_open, st.show_property_change_window = imgui.begin(
        "Change Properties", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_property_change_window = False
        st.property_change_node_id = None
        imgui.end()
        return

    node = scene_builder.get_node(st.property_change_node_id)
    if not node:
        imgui.end()
        return

    prim = node.item_data
    sym = prim.properties.get("symmetry")
    if sym is None:
        sym = [False, False, False]
        prim.update_property("symmetry", sym)

    imgui.spacing()
    imgui.text("Symmetry:")
    imgui.same_line()
    changed_x, sym[0] = imgui.checkbox("X", sym[0])
    imgui.same_line()
    changed_y, sym[1] = imgui.checkbox("Y", sym[1])
    imgui.same_line()
    changed_z, sym[2] = imgui.checkbox("Z", sym[2])
    imgui.spacing()

    if changed_x or changed_y or changed_z:
        prim.update_property("symmetry", sym)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    if imgui.button("Close", -1):
        st.show_property_change_window = False
        st.property_change_node_id = None

    imgui.end()


def render_reparent_window(width, height, scene_builder):
    """Render the reparent node window."""
    if not st.show_reparent_window or not st.reparent_node_id:
        return

    imgui.set_next_window_size(400, 500)
    st.show_reparent_window, _ = imgui.begin("Reparent Node", True)

    if not st.show_reparent_window:
        imgui.end()
        return

    reparent_node = scene_builder.get_node(st.reparent_node_id)
    if not reparent_node:
        imgui.end()
        return

    imgui.text(f"Reparenting: {reparent_node.item_data.ui_name}")
    imgui.separator()
    imgui.text("Select new parent operation:")

    all_descendants = scene_builder.get_all_children_recursive(st.reparent_node_id)
    all_descendants.append(st.reparent_node_id)

    parent_selected = False
    for root_id in scene_builder.root_children:
        parent_selected |= _render_reparent_node_list(
            root_id, st.reparent_node_id, all_descendants, scene_builder, "  "
        )

    if parent_selected and st.reparent_target_parent:
        _render_reparent_operand_selection(st.reparent_target_parent, scene_builder)

    imgui.spacing()
    imgui.separator()

    if imgui.button("Cancel", 100, 30):
        st.show_reparent_window = False
        st.reparent_node_id = None
        st.reparent_target_parent = None
        st.reparent_child_to_replace = None

    imgui.same_line(150)

    can_reparent = _can_reparent_node(st.reparent_target_parent, scene_builder)
    if imgui.button("Reparent", 100, 30):
        if can_reparent:
            if scene_builder.reparent_node(
                st.reparent_node_id,
                st.reparent_target_parent,
                st.reparent_child_to_replace,
            ):
                success, new_uniforms = recompile_shader(scene_builder)
                if success:
                    st.uniform_locs = new_uniforms

            st.show_reparent_window = False
            st.reparent_node_id = None
            st.reparent_target_parent = None
            st.reparent_child_to_replace = None

    imgui.end()


def _render_reparent_node_list(
    node_id, exclude_node_id, exclude_descendants, scene_builder, indent=""
):
    """Recursively render selectable nodes for reparent window."""
    if node_id in exclude_descendants or node_id == exclude_node_id:
        return False

    node = scene_builder.get_node(node_id)
    if not node or node.node_type != "operation":
        return False

    label = f"{indent}{node.item_data.ui_name} ({node_id})"
    clicked, _ = imgui.selectable(label, False)

    if clicked:
        st.reparent_target_parent = node_id
        return True

    result = imgui.is_item_clicked()

    # Recursively render children
    for child_id in node.children:
        child_node = scene_builder.get_node(child_id)
        if child_node and child_node.node_type == "operation":
            result |= _render_reparent_node_list(
                child_id,
                exclude_node_id,
                exclude_descendants,
                scene_builder,
                indent + "  ",
            )

    return result


def _render_reparent_operand_selection(parent_id, scene_builder):
    """Render operand selection if parent is at capacity."""
    parent_node = scene_builder.get_node(parent_id)
    if not parent_node or parent_node.node_type != "operation":
        return

    required_operands = scene_builder._get_operand_count(
        parent_node.item_data.operation_type
    )
    current_operands = len(parent_node.children)

    if current_operands >= required_operands:
        imgui.separator()
        imgui.text(f"Parent is full ({current_operands}/{required_operands} operands)")
        imgui.text("Select child to replace:")

        for i, child_id in enumerate(parent_node.children):
            child_node = scene_builder.get_node(child_id)
            if child_node:
                if imgui.selectable(
                    f"{child_node.item_data.ui_name} ({child_id})",
                    st.reparent_child_to_replace == child_id,
                )[0]:
                    st.reparent_child_to_replace = child_id


def _can_reparent_node(target_parent_id, scene_builder):
    """Check if reparenting to target parent is valid."""
    if target_parent_id is None:
        return False

    parent_node = scene_builder.get_node(target_parent_id)
    if not parent_node or parent_node.node_type != "operation":
        return False

    required_operands = scene_builder._get_operand_count(
        parent_node.item_data.operation_type
    )
    current_operands = len(parent_node.children)

    if current_operands >= required_operands and st.reparent_child_to_replace is None:
        return False

    return True


def render_fps_overlay(width, panel_width):
    """Render FPS/sample counter overlay."""
    fps_x = width - panel_width - cn["FPS_WINDOW_WIDTH"] - cn["FPS_WINDOW_OFFSET"]
    imgui.set_next_window_position(fps_x, cn["FPS_WINDOW_OFFSET"])
    imgui.set_next_window_size(cn["FPS_WINDOW_WIDTH"], cn["FPS_WINDOW_HEIGHT"])

    imgui.begin(
        "FPS",
        False,
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE
        | imgui.WINDOW_NO_SCROLLBAR,
    )

    if st.shader_choice == 0:
        imgui.text_colored("FPS: " + str(st.fps_value), 0.0, 1.0, 0.0, 1.0)
    elif st.shader_choice == 1:
        imgui.text_colored("Sample: " + str(st.frame_count), 1.0, 1.0, 0.0, 1.0)

    imgui.end()


def render_orientation_overlay(width, panel_width):
    """Render camera orientation guide overlay."""
    fps_x = width - panel_width - cn["FPS_WINDOW_WIDTH"] - cn["FPS_WINDOW_OFFSET"]
    ori_x = fps_x + 70

    imgui.set_next_window_position(ori_x, cn["ORI_WINDOW_OFFSET"])
    imgui.set_next_window_size(cn["ORI_WINDOW_WIDTH"], cn["ORI_WINDOW_HEIGHT"])

    imgui.begin(
        "ORI",
        False,
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_ALWAYS_AUTO_RESIZE
        | imgui.WINDOW_NO_SCROLLBAR,
    )

    imgui.same_line(17, 0)
    imgui.text_colored("VIEW", 0.8, 0.8, 1.0)
    imgui.spacing()

    # X axis
    if imgui.small_button("X##Ori"):
        st.target_yaw = 0.0
        st.target_pitch = 0.0
    imgui.same_line()
    if imgui.small_button("-X##Ori"):
        st.target_yaw = 3.14
        st.target_pitch = 0.0

    imgui.spacing()

    # Y axis
    if imgui.small_button("Y##Ori"):
        st.target_pitch = 1.57
    imgui.same_line()
    if imgui.small_button("-Y##Ori"):
        st.target_pitch = -1.57

    imgui.spacing()

    # Z axis
    if imgui.small_button("Z##Ori"):
        st.target_yaw = 1.57
        st.target_pitch = 0.0
    imgui.same_line()
    if imgui.small_button("-Z##Ori"):
        st.target_yaw = -1.57
        st.target_pitch = 0.0

    imgui.end()


def render_exit_confirmation_window(width, height):
    """Render the exit confirmation dialog."""
    if not st.show_exit_window:
        return

    imgui.set_next_window_position(width // 2 - 150, height // 2 - 65)
    imgui.set_next_window_size(300, 130)
    is_open, st.show_exit_window = imgui.begin(
        "Confirm Exit", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_exit_window = False
        imgui.end()
        return

    imgui.spacing()
    imgui.text(f"Are you sure you want to exit?\nUnsaved data may be lost.")
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    if imgui.button("Cancel", 130, 30):
        st.show_exit_window = False

    imgui.same_line(0, 15)

    if imgui.button("YES", 130, 30):
        config = {"Theme": st.theme}
        save_user_config("UserData/User.data", config)
        glfw.set_window_should_close(st.window, True)

    imgui.end()


def render_restart_confirmation_window(width, height):
    """Render the restart confirmation dialog."""
    if not st.show_restart_window:
        return

    imgui.set_next_window_position(width // 2 - 150, height // 2 - 65)
    imgui.set_next_window_size(300, 130)
    is_open, st.show_restart_window = imgui.begin(
        "Confirm Restart", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_restart_window = False
        imgui.end()
        return

    imgui.spacing()
    imgui.text(
        f"Are you sure you want to restart the app?\nThis may result in loss of unsaved data."
    )
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    if imgui.button("Cancel", 130, 30):
        st.show_restart_window = False

    imgui.same_line(0, 15)

    if imgui.button("YES", 130, 30):
        config = {"Theme": st.theme}
        save_user_config("UserData/User.data", config)

        import sys
        import subprocess

        if getattr(sys, "frozen", False):
            subprocess.Popen([sys.executable])
        else:
            subprocess.Popen([sys.executable] + sys.argv)
        exit()

    imgui.end()


def render_about_window(width, height):
    """Render the about/information window."""
    if not st.show_about_window:
        return

    imgui.set_next_window_position(width // 2 - 250, height // 2 - 200)
    imgui.set_next_window_size(500, 400)
    is_open, st.show_about_window = imgui.begin("About", True, imgui.WINDOW_NO_COLLAPSE)

    if not is_open:
        st.show_about_window = False
        imgui.end()
        return

    about_text = """
MIT License

Copyright (c) 2025-present EmberNoGlow

------------------

This is a project in which I created rendering and full interaction with sdf primitives. Using Python, GLSL, Imgui, glfw, pyopengl.

------------------

Thank you for using this project! If you liked the project, give it a star on github.

You can also support the project by reporting an error, or by suggesting an improvement by opening a Pull Request (PR).
    """

    imgui.begin_child("LicenseText", width=490, height=300, border=True)
    imgui.text_wrapped(about_text)
    imgui.end_child()

    imgui.spacing()

    # GitHub link
    imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
    imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.6, 0.4, 0.1, 0.5)
    imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0, 0, 0, 0)
    imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 0.5, 1.0)

    if imgui.selectable("Visit project page in Github (Double click)", False):
        if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
            webbrowser.open("https://github.com/EmberNoGlow/SDF-Model-Editor-Demo")

    imgui.pop_style_color()
    imgui.pop_style_color(3)

    imgui.spacing()
    if imgui.button("Close", -1):
        st.show_about_window = False

    imgui.end()


def cleanup_resources(
    fbo, render_texture, display_shader, display_vao, display_vbo, vao, vbo
):
    """Clean up OpenGL resources."""
    # Clean up accumulation buffers
    for i in range(2):
        if st.accumulation_fbos[i] is not None:
            try:
                glDeleteFramebuffers(1, [st.accumulation_fbos[i]])
            except Exception:
                pass
            st.accumulation_fbos[i] = None

        if st.accumulation_textures[i] is not None:
            try:
                glDeleteTextures(1, [st.accumulation_textures[i]])
            except Exception:
                pass
            st.accumulation_textures[i] = None

    # Delete cached shaders
    for cached_shader, _ in st.shader_cache.values():
        if cached_shader is not None:
            glDeleteProgram(cached_shader)
    st.shader_cache.clear()

    # Clean up framebuffer
    if fbo is not None:
        glDeleteFramebuffers(1, [fbo])
    if render_texture is not None:
        glDeleteTextures(1, [render_texture])

    # Clean up display shader
    if display_shader is not None:
        glDeleteProgram(display_shader)
    if display_vao is not None:
        glDeleteVertexArrays(1, [display_vao])
    if display_vbo is not None:
        glDeleteBuffers(1, [display_vbo])

    # Clean up main VAO/VBO
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])


# MAIN LOOP


def main():
    """Main application entry point and event loop."""
    # Initialization phase
    window, impl, ICONS, camera = init_application()
    scene_builder = init_scene()

    st.shader, st.uniform_locs = init_shader(scene_builder)
    if st.shader is None:
        print("Failed to compile initial shader. Exiting.")
        impl.shutdown()
        glfw.terminate()
        return

    vao, vbo, st.display_vao, st.display_vbo, st.display_shader = (
        init_opengl_resources()
    )

    # Store scene builder in state for access throughout
    st.scene_builder = scene_builder
    st.window = window
    st.vao = vao

    # User configuration
    st.default_uconfig, default_uconfig = load_user_configuration()
    rebuild_imgui_fonts(impl, "assets/fonts/Roboto-Medium.ttf", 16.0)

    # Setup callbacks and timing
    setup_glfw_callbacks(window)
    setup_time_tracking()

    # Framebuffer state
    st.fbo = None
    st.render_texture = None
    st.fbo_width = 0
    st.fbo_height = 0

    # Initialize camera tracking variables
    st.prev_cam_yaw = st.cam_yaw
    st.prev_cam_pitch = st.cam_pitch
    st.prev_cam_radius = st.cam_radius
    st.prev_cam_orbit = st.cam_orbit

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
        io = get_io()
        handle_keyboard_and_scene_input(window, io, scene_builder)
        handle_mouse_input(window)
        handle_camera_input(window, io, camera)
        handle_home_key(io)

        # Update camera state
        detect_camera_changes()

        # Handle shader changes
        handle_shader_monitor(scene_builder)
        handle_frame_accumulation()

        # Handle primitive interactions
        if handle_primitive_dragging(window, scene_builder, camera):
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

        if handle_primitive_rotation(window, scene_builder):
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

        if handle_code_editor_updates():
            recompile_shader(scene_builder)

        # Render scene
        (
            use_accumulation,
            width,
            height,
            menu_bar_height,
            panel_width,
            rendering_width,
            rendering_height,
        ) = render_scene(
            window,
            scene_builder,
            camera,
            vao,
            vbo,
            st.display_vao,
            st.display_vbo,
            st.display_shader,
        )

        # Framebuffer rendering
        if not (
            st.shader is not None and st.shader_choice == 1 and use_accumulation == 1
        ):
            render_framebuffer_scaling(
                width,
                height,
                menu_bar_height,
                panel_width,
                rendering_width,
                rendering_height,
                st.display_shader,
            )
            render_direct_to_screen(
                width,
                height,
                menu_bar_height,
                panel_width,
                rendering_width,
                rendering_height,
            )

        # UI Rendering
        render_menu_bar(window, scene_builder)
        render_settings_window(
            width, height, scene_builder, rendering_width, rendering_height
        )
        render_editor_settings_window(width, height)
        render_export_windows(window, scene_builder)
        render_status_message_window(width)
        render_error_window()
        render_scene_tree_panel(
            width, height, menu_bar_height, panel_width, scene_builder
        )
        render_inspector_panel(
            width, height, menu_bar_height, panel_width, scene_builder
        )
        render_add_change_window(width, height, scene_builder)
        render_property_change_window(width, height, scene_builder)
        render_reparent_window(width, height, scene_builder)
        render_fps_overlay(width, panel_width)
        render_orientation_overlay(width, panel_width)
        render_exit_confirmation_window(width, height)
        render_restart_confirmation_window(width, height)
        render_about_window(width, height)

        # Render ImGui
        imgui.render()
        impl.render(imgui.get_draw_data())

        # Swap buffers
        glfw.swap_buffers(window)

    # Cleanup
    cleanup_resources(
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
