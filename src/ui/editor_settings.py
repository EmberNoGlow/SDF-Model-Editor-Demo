"""Editor settings window."""
import imgui
import copy
import src.ui.themes as ui_themes
from src.app.data.states import st
from src.ui.helpers import input_vec2


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
    from src.ui.helpers import ShortCuts
    
    for name, keys in ShortCuts.items():
        imgui.text(name)
        imgui.same_line()
        for key in (keys,):
            imgui.text(str(key))
            imgui.same_line()
        imgui.spacing()