from ..__init__ import *


def init_application():
    """Initialize GLFW, ImGui, and core application state."""
    window, impl = init_glfw_impl(cn["SCREEN_SIZE"])
    ICONS = load_all_textures()

    camera = Camera()
    st.theme = ui_themes.default_theme

    return window, impl, ICONS, camera


def init_scene():
    """Initialize the SDF scene builder with default primitives."""
    scene_builder = SDFSceneBuilder(st.glob_history, st.selected_item_id)

    scene_builder.add_standalone_primitive(
        "box", position=[0, 0, 0], size_or_radius=[0.5, 0.2, 0.8], ui_name="Cube"
    )

    return scene_builder


def init_shader(scene_builder, shader_manager):
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

