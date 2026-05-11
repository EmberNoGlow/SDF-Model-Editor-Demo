"""Handle keyboard, mouse, and camera input events."""
import glfw
import imgui
import time
from typing import Tuple
from src.app.data.states import st
from src.app.data.consts import cn


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


def handle_keyboard_and_scene_input(window, io, scene_builder, recompile_callback):
    """Process keyboard input and update scene based on user actions."""
    from ...io.handler import handler
    
    handle = handler(
        window, io, scene_builder, st.glob_history, st.selected_item_id, st.selected_items
    )

    if handle[0]:  # Shader recompile needed
        recompile_callback()

    st.selected_item_id = handle[1]
    st.selected_items = handle[2]


def handle_mouse_input(window):
    """Handle middle mouse button and pan/zoom interactions."""
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


def _is_shift_pressed(window) -> bool:
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
            _handle_camera_pan(current_x, current_y)
        else:
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


def _handle_camera_pan(current_x: float, current_y: float):
    """Handle camera panning with Shift+MMB."""
    dx = current_x - st.last_pan_x
    dy = current_y - st.last_pan_y
    st.last_pan_x, st.last_pan_y = current_x, current_y
    st.target_pan_x += dx * cn["PAN_SENSITIVITY"]
    st.target_pan_y += dy * cn["PAN_SENSITIVITY"]


def _handle_camera_rotation(current_x: float, current_y: float):
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


def detect_camera_changes(clear_accumulation_callback):
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
        clear_accumulation_callback()
        st.current_accum_index = 0

    # Update previous values
    st.prev_cam_yaw = st.cam_yaw
    st.prev_cam_pitch = st.cam_pitch
    st.prev_cam_radius = st.cam_radius
    st.prev_cam_orbit = st.cam_orbit