import imgui
import glfw

from imgui.integrations.glfw import GlfwRenderer


def init_glfw_impl(SCREEN_SIZE):
    # Initialize GLFW
    if not glfw.init():
        print("--- GLFW IS NOT INIT ---")
        return

    # Create a windowed mode window and its OpenGL context
    try:
        window = glfw.create_window(
            SCREEN_SIZE[0], SCREEN_SIZE[1], "Viewport", None, None
        )
        if not window:
            glfw.terminate()
            return

    except Exception as e:
        print(f"{e}")
        glfw.terminate()
        return

    # Make the window's context current
    try:
        glfw.make_context_current(window)
    except Exception as e:
        print(f"{e}")

    # Initialize ImGui
    imgui.create_context()
    impl = GlfwRenderer(window)

    return window, impl
