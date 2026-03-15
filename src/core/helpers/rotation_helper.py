from ...io.input import input_handle
from ...app.data.states import st
from ...app.data.consts import cn
from ...rendering.fbo import clear_accumulation_fbos

import glfw
import numpy as np

def rotate_privitive(window, scene_builder):
    ret_recompile_shader = False

    key_r_is_down = input_handle("Rotate")
    key_x_is_down = input_handle("X")
    key_y_is_down = input_handle("Y")
    key_z_is_down = input_handle("Z")

    # Edge-detect R press to toggle rotation mode
    if key_r_is_down and not st.last_key_r_pressed:
        st.R_dragging = not st.R_dragging

        if st.R_dragging:
            # Start rotation: capture selected item and initialize rotation state
            st.R_dragging_op_id = st.selected_item_id

            if st.R_dragging_op_id and st.R_dragging_op_id in scene_builder.id_to_node:
                node = scene_builder.get_node(st.R_dragging_op_id)
                if node and node.node_type == 'primitive':
                    prim = node.item_data
                    st.R_drag_start_pos = prim.rotation.copy()
                    st.R_drag_accum = [0.0, 0.0, 0.0]
                    st.R_drag_last_x, st.R_drag_last_y = glfw.get_cursor_pos(window)
                else:
                    st.R_dragging_op_id = None
                    st.R_drag_start_pos = None
                    st.R_drag_accum = [0.0, 0.0, 0.0]
            else:
                st.R_dragging_op_id = None
                st.R_drag_start_pos = None
                st.R_drag_accum = [0.0, 0.0, 0.0]

            st.axis_toggled_rx = st.axis_toggled_ry = st.axis_toggled_rz = False

        else:
            # Stop rotation: commit final rotation (register undo/redo)
            if st.R_dragging_op_id and st.R_dragging_op_id in scene_builder.id_to_node:
                node = scene_builder.get_node(st.R_dragging_op_id)
                if node and node.node_type == 'primitive':
                    prim = node.item_data
                    final_rot = prim.rotation
                    if st.R_drag_start_pos is not None and final_rot != st.R_drag_start_pos:
                        # Use scene_builder to register the change (compatibility method)
                        scene_builder.modify_primitive_property(st.R_dragging_op_id, 'rotation', final_rot)
                        ret_recompile_shader |= True

            st.R_dragging_op_id = None
            st.R_drag_start_pos = None
            st.R_drag_accum = [0.0, 0.0, 0.0]
            st.axis_toggled_rx = st.axis_toggled_ry = st.axis_toggled_rz = False

    # Update last R state
    st.last_key_r_pressed = key_r_is_down

    # Rotation axis toggles (Blender-style)
    if st.R_dragging:
        if key_x_is_down and not st.last_key_rx_pressed:
            state = not st.axis_toggled_rx
            st.axis_toggled_rx, st.axis_toggled_ry, st.axis_toggled_rz = state, False, False
        if key_y_is_down and not st.last_key_ry_pressed:
            state = not st.axis_toggled_ry
            st.axis_toggled_rx, st.axis_toggled_ry, st.axis_toggled_rz = False, state, False
        if key_z_is_down and not st.last_key_rz_pressed:
            state = not st.axis_toggled_rz
            st.axis_toggled_rx, st.axis_toggled_ry, st.axis_toggled_rz = False, False, state

    st.last_key_rx_pressed = key_x_is_down
    st.last_key_ry_pressed = key_y_is_down
    st.last_key_rz_pressed = key_z_is_down

    # Per-frame rotation update while st.R_dragging is active
    if st.R_dragging and st.R_dragging_op_id and st.R_dragging_op_id in scene_builder.id_to_node:
        current_x, current_y = glfw.get_cursor_pos(window)
        dx = current_x - st.R_drag_last_x
        dy = current_y - st.R_drag_last_y
        st.R_drag_last_x, st.R_drag_last_y = current_x, current_y

        rot_delta_x = -dy * cn['R_ROT_SENSITIVITY']
        rot_delta_y = -dx * cn['R_ROT_SENSITIVITY']
        rot_delta_z = 0.0

        if st.axis_toggled_rx:
            rot_delta_y = 0.0
            rot_delta_z = 0.0
        elif st.axis_toggled_ry:
            rot_delta_x = 0.0
            rot_delta_z = 0.0
        elif st.axis_toggled_rz:
            rot_delta_x = 0.0
            rot_delta_y = 0.0
            rot_delta_z = -dx * cn['R_ROT_SENSITIVITY']

        if abs(rot_delta_x) + abs(rot_delta_y) + abs(rot_delta_z) > 1e-5:
            st.frame_count = 0
            clear_accumulation_fbos()

        st.R_drag_accum[0] += rot_delta_x
        st.R_drag_accum[1] += rot_delta_y
        st.R_drag_accum[2] += rot_delta_z

        node = scene_builder.get_node(st.R_dragging_op_id)
        if node and node.node_type == 'primitive':
            prim = node.item_data
            if st.R_drag_start_pos is None:
                st.R_drag_start_pos = prim.rotation.copy()
            new_rot = [
                st.R_drag_start_pos[0] + st.R_drag_accum[0],
                st.R_drag_start_pos[1] + st.R_drag_accum[1],
                st.R_drag_start_pos[2] + st.R_drag_accum[2],
            ]
            prim.rotation = new_rot
            st.drag_rot_position = new_rot.copy()

    else:
        # keep shader MoveRot aligned with selection (or zero)
        if st.selected_item_id and st.selected_item_id in scene_builder.id_to_node:
            node = scene_builder.get_node(st.selected_item_id)
            if node and node.node_type == 'primitive':
                prim = node.item_data
                st.drag_rot_position = prim.rotation
        else:
            st.drag_rot_position = [0.0, 0.0, 0.0]
    
    return ret_recompile_shader