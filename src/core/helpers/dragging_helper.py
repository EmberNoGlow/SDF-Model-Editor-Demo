from ...io.input import input_handle
from ...app.data.states import st
from ...app.data.consts import cn
from ...rendering.fbo import clear_accumulation_fbos

import glfw
import numpy as np

def dragging_primitive(window, scene_builder, camera, selected_item_id, accumulation_fbos, scaled_rendering_width, scaled_rendering_height):
    ret_recompile_shader: bool = False

    # Drag on G
    key_g_is_down = input_handle("Move")
    key_x_is_down = input_handle("X")
    key_y_is_down = input_handle("Y")
    key_z_is_down = input_handle("Z")

    # Toggle st.dragging on G press (edge detect)
    if key_g_is_down and not st.last_key_g_pressed:
        # Toggle st.dragging state
        st.dragging = not st.dragging

        if st.dragging:
            # Start st.dragging: capture which item and initialize drag state
            st.dragging_op_id = selected_item_id

            if st.dragging_op_id:
                node = scene_builder.get_node(st.dragging_op_id)
                if node and node.node_type == 'primitive':
                    prim = node.item_data
                    # Copy the primitive start position
                    st.drag_start_pos = prim.position[:]
                    # Reset accumulated movement
                    st.drag_accum = [0.0, 0.0, 0.0]
                    # Record starting mouse cursor
                    st.drag_last_x, st.drag_last_y = glfw.get_cursor_pos(window)
                else:
                    # Not a primitive, can't drag
                    st.dragging_op_id = None
                    st.drag_start_pos = None
                    st.drag_accum = [0.0, 0.0, 0.0]
            else:
                st.dragging_op_id = None
                st.drag_start_pos = None
                st.drag_accum = [0.0, 0.0, 0.0]

            # Reset axis toggles when starting a new drag
            st.axis_toggled_gx = st.axis_toggled_gy = st.axis_toggled_gz = False

        else:
            # Stop st.dragging: commit final position
            if st.dragging_op_id:
                node = scene_builder.get_node(st.dragging_op_id)
                if node and node.node_type == 'primitive':
                    prim = node.item_data
                    final_pos = prim.position
                    # Register only if changed
                    if st.drag_start_pos is not None and final_pos != st.drag_start_pos:
                        # Directly update (no undo needed for now)
                        ret_recompile_shader |= True

            # Clear drag state
            st.dragging_op_id = None
            st.drag_start_pos = None
            st.drag_accum = [0.0, 0.0, 0.0]
            st.axis_toggled_gx = st.axis_toggled_gy = st.axis_toggled_gz = False

    # Always update st.last_key_g_pressed for proper edge detection
    st.last_key_g_pressed = key_g_is_down

    # Handle axis toggles (Blender-style)
    if st.dragging:
        if key_x_is_down and not st.last_key_gx_pressed:
            state = not st.axis_toggled_gx
            st.axis_toggled_gx, st.axis_toggled_gy, st.axis_toggled_gz = state, False, False

        if key_y_is_down and not st.last_key_gy_pressed:
            state = not st.axis_toggled_gy
            st.axis_toggled_gx, st.axis_toggled_gy, st.axis_toggled_gz = False, state, False

        if key_z_is_down and not st.last_key_gz_pressed:
            state = not st.axis_toggled_gz
            st.axis_toggled_gx, st.axis_toggled_gy, st.axis_toggled_gz = False, False, state

    # Update the "last key" flags for X/Y/Z
    st.last_key_gx_pressed = key_x_is_down
    st.last_key_gy_pressed = key_y_is_down
    st.last_key_gz_pressed = key_z_is_down

    # Determine active axis
    active_axis = None
    if st.axis_toggled_gx:
        active_axis = 0
    elif st.axis_toggled_gy:
        active_axis = 1
    elif st.axis_toggled_gz:
        active_axis = 2

    # Per-frame drag movement
    if st.dragging and st.dragging_op_id:
        node = scene_builder.get_node(st.dragging_op_id)
        if node and node.node_type == 'primitive':
            # Read current mouse and compute delta
            current_x, current_y = glfw.get_cursor_pos(window)
            dx = current_x - st.drag_last_x
            dy = current_y - st.drag_last_y
            # Store for next frame
            st.drag_last_x, st.drag_last_y = current_x, current_y

            # Convert to mouse-space movement
            mouse_delta_x = dx * cn['DRAG_SENSITIVITY']
            mouse_delta_y = -dy * cn['DRAG_SENSITIVITY']

            if np.linalg.norm(np.array([mouse_delta_x, mouse_delta_y])) > 0.01:
                st.frame_count = 0
                clear_accumulation_fbos(accumulation_fbos, scaled_rendering_width, scaled_rendering_height)

            # Transform mouse deltas into world-space
            move_delta_x, move_delta_y, move_delta_z = camera.get_move_delta(mouse_delta_x, mouse_delta_y)

            # Axis constraints
            if active_axis is not None:
                if active_axis == 0:
                    move_delta_y = 0.0
                    move_delta_x = 0.0
                elif active_axis == 1:
                    move_delta_x = 0.0
                    move_delta_z = 0.0
                elif active_axis == 2:
                    move_delta_z = 0.0
                    move_delta_y = 0.0

            # Accumulate world movement
            st.drag_accum[0] += move_delta_z
            st.drag_accum[1] += move_delta_y
            st.drag_accum[2] += move_delta_x

            # Compute new position
            prim = node.item_data
            if st.drag_start_pos is None:
                st.drag_start_pos = prim.position.copy()

            new_pos = [
                st.drag_start_pos[0] + st.drag_accum[0],
                st.drag_start_pos[1] + st.drag_accum[1],
                st.drag_start_pos[2] + st.drag_accum[2],
            ]

            # Apply live position
            prim.position = new_pos
            drag_position = new_pos.copy()

    else:
        # When not st.dragging
        if selected_item_id:
            node = scene_builder.get_node(selected_item_id)
            if node and node.node_type == 'primitive':
                prim = node.item_data
                drag_position = prim.position
        else:
            drag_position = [0.0, 0.0, 0.0]
    
    return ret_recompile_shader, drag_position