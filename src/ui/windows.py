"""All UI windows and dialogs."""
import imgui
import glfw
import time
import webbrowser
import pyperclip
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

from src.app.data.states import st
from src.app.data.consts import cn
from src.ui.helpers import input_vec3, input_vec2, input_float
from src.rendering.shader_compiler import recompile_shader
import src.app.Exporter as sdfexp


def render_settings_window(width, height, scene_builder, rendering_width, rendering_height):
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
        from src.ui.dialogs import save_sdfvol_dialog
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

        from src.ui.dialogs import save_sdfobj_dialog
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
        from src.utils import save_user_config
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
        from src.utils import save_user_config
        import sys
        import subprocess

        config = {"Theme": st.theme}
        save_user_config("UserData/User.data", config)

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