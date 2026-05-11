"""Menu bar rendering and actions."""
import imgui
import glfw
import threading
from src.app.data.states import st
from src.rendering.shader_compiler import recompile_shader
import src.app.CodeEditor as CodeEdit
from src.ui.dialogs import save_scene_dialog, load_scene_dialog
import time


def render_menu_bar(window, scene_builder):
    """Render the main menu bar and handle menu actions."""
    if not imgui.begin_main_menu_bar():
        return

    _render_file_menu(window, scene_builder)
    _render_edit_menu(scene_builder)
    _render_view_menu()
    _render_editor_menu()
    _render_about_menu()
    _render_shader_mode_buttons(window, scene_builder)

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
            st.glob_history.undo_stack.clear()
            st.glob_history.redo_stack.clear()
            scene_builder.update_glob_history(st.glob_history)
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


def _render_shader_mode_buttons(window, scene_builder):
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
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    # Cycles button
    imgui.set_cursor_pos_x(start_x + button_width + spacing)
    if imgui.button("Cycles", button_width):
        st.shader_choice = 1
        success, new_uniforms = recompile_shader(scene_builder)
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