"""FPS and orientation overlays."""
import imgui
from src.app.data.states import st
from src.app.data.consts import cn


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