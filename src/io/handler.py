from .input import get_io, input_handle
from src.app.data.states import st
from src.core.classes.save_load_helpers.SaveLoadUtils import load_scene_dialog, save_scene_dialog, take_screenshot

import time
import copy
import glfw

def handler(window, io, scene_builder, glob_history, selected_item_id, selected_items):
    # A flag that determines whether we need to recompile the shader
    ret_recompile_shader: bool = False
            
    # Check Ctrl+A for add window 
    if input_handle("Add"):
        if not st.last_key_a_pressed:
            # Open Add Operation dialog (keeps same code path as the menu)
            st.show_add_change_window = True
            st.pending_change_node_id = None
            st.last_key_a_pressed = True
    else:
        st.last_key_a_pressed = False
    
    # Check F2 for rename (with debouncing)
    if input_handle("Rename") and selected_item_id is not None and st.renaming_item_id is None:
        if not st.last_key_f2_pressed:
            st.renaming_item_id = selected_item_id
            st.rename_text = scene_builder.get_item_name(selected_item_id)
            st.last_key_f2_pressed = True
    else:
        st.last_key_f2_pressed = False
    
    # Check Delete key for deletion (with debouncing)
    # Only allow deletion if node is direct child of root (depth = 1)
    if input_handle("Delete") and selected_item_id is not None:
        if not st.last_key_delete_pressed:
            node_to_delete = scene_builder.get_node(selected_item_id)
            if node_to_delete:
                # Check depth: only delete if parent is None (direct root child)
                depth = scene_builder.get_node_depth(selected_item_id)
                if depth == 1:  # Direct child of root
                    if scene_builder.delete_node(selected_item_id):
                        ret_recompile_shader |= True
                        selected_item_id = None
                        scene_builder.update_selected_item_id(selected_item_id)
                        st.selection_mode = None
                else:
                    # Cannot delete
                    pass
            st.last_key_delete_pressed = True
    else:
        st.last_key_delete_pressed = False
    
    # Check Ctrl+B for compile (with debouncing)
    if input_handle("Compile"):
        if not st.last_key_compile_pressed:
            ret_recompile_shader |= True
            st.last_key_compile_pressed = True
    else:
        st.last_key_compile_pressed = False


    if glfw.get_key(window, glfw.KEY_F12) == glfw.PRESS:
        take_screenshot(window)
    

    # Check Ctrl + S/O
    if input_handle("Open"):
        if not st.last_key_o_pressed: 
            success, message = load_scene_dialog(scene_builder)
            st.save_load_message = message
            st.save_load_message_time = time.time()
            if success:
                ret_recompile_shader |= True
                selected_item_id = None
                scene_builder.update_selected_item_id(selected_item_id)
                st.selection_mode = None
            st.last_key_o_pressed = True
    else:
        st.last_key_o_pressed = False


    # --- Duplicate (Ctrl+D) ---
    if input_handle("Duplicate"):
        if not st.last_key_d_pressed:
            # Duplicate selected items (multi-select supported)
            duplicated_ids = []
            if len(selected_items) > 0:
                # Duplicate all selected items
                for sid in list(selected_items):
                    node = scene_builder.get_node(sid)
                    if node and node.node_type == 'primitive':
                        prim = node.item_data
                        # Recreate with same properties
                        new_id = scene_builder.add_standalone_primitive(
                            prim.primitive_type,
                            copy.deepcopy(prim.position),
                            copy.deepcopy(prim.size_or_radius),
                            copy.deepcopy(prim.rotation),
                            copy.deepcopy(prim.scale),
                            prim.ui_name + " (copy)",
                            copy.deepcopy(prim.color),
                            **copy.deepcopy(prim.kwargs)
                        )
                        duplicated_ids.append(new_id)
            elif selected_item_id:
                node = scene_builder.get_node(selected_item_id)
                if node and node.node_type == 'primitive':
                    prim = node.item_data
                    new_id = scene_builder.add_standalone_primitive(
                        prim.primitive_type,
                        copy.deepcopy(prim.position),
                        copy.deepcopy(prim.size_or_radius),
                        copy.deepcopy(prim.rotation),
                        copy.deepcopy(prim.scale),
                        prim.ui_name + " (copy)",
                        copy.deepcopy(prim.color),
                        **copy.deepcopy(prim.kwargs)
                    )
                    duplicated_ids.append(new_id)

            # Select the most recent duplicated id if any
            if duplicated_ids:
                selected_items.clear()
                selected_item_id = duplicated_ids[-1]
                scene_builder.update_selected_item_id(selected_item_id)
                st.selection_mode = 'node'
                # Recompile shader to pick up new primitives
                ret_recompile_shader |= True

            st.last_key_d_pressed = True
    else:
        st.last_key_d_pressed = False
    


    if input_handle("Save"):
        if not st.last_key_s_pressed: 
            success, message = save_scene_dialog(scene_builder, window)
            st.save_load_message = message
            st.save_load_message_time = time.time()
            if success:
                ret_recompile_shader |= True
                selected_item_id = None
                scene_builder.update_selected_item_id(selected_item_id)
                st.selection_mode = None
            st.last_key_s_pressed = True
    else:
        st.last_key_s_pressed = False


    # Check Undo/Redo keys Ctrl+Z/Y
    if input_handle("Undo") and io.key_ctrl and not io.key_shift:
        if not st.last_key_z_pressed: 
            undo_success = glob_history.undo()
            scene_builder.update_glob_history(glob_history)
            if undo_success:
                ret_recompile_shader |= True
            st.last_key_z_pressed = True
    else:
        st.last_key_z_pressed = False


    if input_handle("Redo") or input_handle("Redo2"):
        if not st.last_key_y_pressed: 
            undo_success = glob_history.redo()
            scene_builder.update_glob_history(glob_history)
            if undo_success:
                ret_recompile_shader |= True
            st.last_key_y_pressed = True
    else:
        st.last_key_y_pressed = False


    # Check F10 for settings
    if io.keys_down[glfw.KEY_F10]:
        if not st.last_key_f10_pressed:
            st.show_settings_window = True
            st.last_key_f10_pressed = True
    else:
        st.last_key_f10_pressed = False


    # In this frame we must recompile the shader
    return ret_recompile_shader, selected_item_id, selected_items