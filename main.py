import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import math
import hashlib
import ctypes
import imgui
import imgui.core
import src.app.Exporter as sdfexp
import src.app.CodeEditor as CodeEdit

from PIL import Image
from typing import Dict, List, Any

import os
import json
import numpy as np
import math
import copy

import tkinter as tk
from tkinter import filedialog, messagebox

from src import *
from src.ui import *
from src.utils import *
from src.rendering import *

import src.ui.themes as ui_themes


# Load shaders
vertex_shader, fragment_shader_template, sdf_library = load_shaders()


# --- Configuration ---
SCREEN_SIZE = (1280, 720)
FOV_ANGLE = math.radians(75)  # Field of View - Used for ray direction calculation
STEP_VARIABLE_FLOAT = 0.1
STEP_VARIABLE_ROTATION = 5.0


# UI Constants
PANEL_WIDTH_RATIO = 0.2  # Left and right panel width as ratio of window width
FPS_WINDOW_OFFSET = 25  # Offset from top for FPS window
FPS_WINDOW_WIDTH = 140
FPS_WINDOW_HEIGHT = 30

ORI_WINDOW_OFFSET = 60  # Offset from top for Orientation window
ORI_WINDOW_WIDTH = 70
ORI_WINDOW_HEIGHT = 110

# Camera Constants
MOUSE_SENSITIVITY = 0.005
PAN_SENSITIVITY = 0.1
CAMERA_LERP_FACTOR = 7.5
ZOOM_SENSITIVITY = 0.5
MIN_RADIUS = 1.0
MAX_RADIUS = 100.0
MIN_PITCH = -math.radians(90)
MAX_PITCH = math.radians(90)

# Moved variables
drag_position = [0,0,0] # Track calculation result
drag_rot_position = [0,0,0]
selected_item_id = None  # Track which item is selected in the tree

# Multi-selection support (CTRL+click to toggle)
selected_items = set()

glob_history = History()

start_drag = False
end_drag = False

R_start_drag = False
R_end_drag = False

# A variable to track what we recompiled the shader
# in cycles mode for later updating the fbo
monitor = False


def MonitorChanges(func):
    def wrapper(*args, **kwargs):
        global monitor; monitor = True
        result = func(*args, **kwargs)
        return result
    return wrapper

CE_app = None
tkinter_thread = None


def main():
    # Globals
    global start_drag, end_drag, dragging, R_dragging, selected_item_id, drag_position, drag_rot_position, selection_mode


    # Initialize GLFW & Imgui
    window, impl = init_glfw_impl(SCREEN_SIZE)
    ICONS = load_all_textures()

    # --- Camera State ---
    target_yaw = 0.0
    target_pitch = 0.0
    target_pan_y = 0.0
    target_pan_x = 0.0
    target_radius = 5.0
    cam_yaw = 0.0
    cam_pitch = 0.0
    last_x, last_y = 0.0, 0.0
    last_pan_x, last_pan_y = 0.0, 0.0  # Separate tracking for panning
    cam_radius = 5.0
    cam_orbit = [0.0, 0.0, 0.0]
    last_x, last_y = 0.0, 0.0

    PAN_SENSITIVITY = 0.01  # Adjust this to control pan speed
    DRAG_SENSITIVITY = 0.01 # Adjust this to control drag (primitive) speed

    is_mmb_pressed = False
    is_shift_mmb_pressed = False

    # --- SaveLoad ---
    save_load_message = None
    save_load_message_time = None
    export_obj_message = None
    export_obj_message_time = None

    # --- Keys ---
    last_key_s_pressed = False
    last_key_o_pressed = False
    last_key_z_pressed = False
    last_key_y_pressed = False
    last_key_g_pressed = False
    axis_toggled_gx = False
    axis_toggled_gy = False
    axis_toggled_gz = False
    last_key_gx_pressed = False
    last_key_gy_pressed = False
    last_key_gz_pressed = False
    last_key_r_pressed = False
    # Rotation-specific axis toggles and key debounces (separate from move G-toggles)
    axis_toggled_rx = False
    axis_toggled_ry = False
    axis_toggled_rz = False
    last_key_rx_pressed = False
    last_key_ry_pressed = False
    last_key_rz_pressed = False

    last_key_f10_pressed = False  # Add this if not present
    
    last_key_d_pressed = False # Duplicate key debounce

    # --- Draging ---
    dragging = False
    dragging_op_id = None           # op_id of the item currently being dragged
    drag_last_x = 0.0               # last mouse x while dragging (separate from camera last_x/last_y)
    drag_last_y = 0.0
    drag_start_pos = None           # original primitive position at drag start (copied list)
    drag_accum = [0.0, 0.0, 0.0]    # accumulated world-space movement since drag start
    DRAG_SENSITIVITY = 0.01         # adjust for speed; consider scaling with cam_radius for consistent feel
    
    # --- Rotate ---
    R_dragging = False
    R_dragging_op_id = None           # op_id of the item currently being dragged
    R_drag_last_x = 0.0               # last mouse x while dragging (separate from camera last_x/last_y)
    R_drag_last_y = 0.0
    R_drag_start_pos = None           # original primitive position at drag start (copied list)
    R_drag_accum = [0.0, 0.0, 0.0]    # accumulated world-space movement since drag start


    # --- Delta time --- 
    delta_time = 0.0 


    # --- Pipeline ---
    camera = Camera()


    # --- Scene Definition ---
    scene_builder = SDFSceneBuilder(glob_history, selected_item_id)

    # --- Default Scene ---
    scene_builder.add_standalone_primitive(
        'box',
        position=[0, 0, 0],
        size_or_radius=[0.5,0.2,0.8],
        ui_name='Cube'
    )

    # --- UI State ---
    show_operation_selection_window = False
    show_primitive_selection_window = False
    show_settings_window = False
    show_add_change_window = False
    pending_change_node_id = None
    property_change_node_id = None
    show_property_change_window = False
    show_reparent_window = False
    reparent_node_id = None
    reparent_target_parent = None  # Selected new parent
    reparent_child_to_replace = None  # Child to delete if parent is full
    show_editor_settings_window = False
    current_settings_tab = "Themes"  # State to track which tab is active
    show_export_vol_window = False
    show_export_obj_window = False
    show_about_window = False
    show_exit_window = False
    show_restart_window = False
    selection_mode = None  # 'primitive' or 'operation'
    renaming_item_id = None  # Item being renamed
    rename_text = ""
    last_key_a_pressed = False  # Track if Ctrl+A was pressed
    last_key_f2_pressed = False  # Track if F2 was pressed
    last_key_delete_pressed = False  # Track if Delete was pressed
    last_key_compile_pressed = False  # Track if Ctrl+B was pressed


    # --- Defined Palette ---
    theme = ui_themes.default_theme



    # Shader selection
    shader_choice = 0  # 0 = template, 1 = cycles
    shader_names = ["shaders/fragment/template.glsl", "shaders/fragment/cycles.glsl"]

    # Sky shaders uniforms (cycles)
    sky_top_color = [0.7, 0.8, 1.0]
    sky_bottom_color = [0.1, 0.15, 0.25]

    # Grid (template)
    GridEnabled = True

    # Light
    LightDir = [0.5, 1.0, 0.7]

    # --- Settings ---
    resolution_scale = 1.0  # 1.0 = normal, 2.0 = oversampling, <1.0 = low res for performance

    # Export Config
    grid_size = 16
    vox_quality = 1.0
    export_z_up = True
    export_level = 0.0
    exp_use_color = True

    # Sprites
    sprites_array = []


    # --- FPS tracking ---
    fps_clock = time.time()
    fps_frames = 0
    fps_value = 0


    # --- Shader compilation and error tracking ---
    shader_compile_error = None
    shader_cache = {}  # Cache for compiled shaders: {hash: (shader_program, uniforms)}

    additional_scene_code = ""
    
    def get_shader_hash():
        """Generate a hash of the current shader code for caching."""
        scene_code = scene_builder.generate_raymarch_code()
        postproc_code = generate_postproc_code(sprites_array)
        selected_fragment_shader = load_shader_code(shader_names[shader_choice])
        fragment_shader = selected_fragment_shader.replace("{SDF_LIBRARY}", sdf_library)
        fragment_shader = fragment_shader.replace("{SCENE_CODE}", scene_code)
        fragment_shader = fragment_shader.replace("{FOV_ANGLE_VAL}", str(FOV_ANGLE))
        fragment_shader = fragment_shader.replace("{POSTPROC}", postproc_code[0])
        fragment_shader = fragment_shader.replace("{ADDITIONAL_UNIFORMS}", postproc_code[1])
        fragment_shader = fragment_shader.replace("{ADDITIONAL_SCENE_CODE}", additional_scene_code)
        
        # Create hash of the complete shader code (including shader choice)
        shader_code = f"{vertex_shader}\n{fragment_shader}\n{shader_names[shader_choice]}"
        return hashlib.md5(shader_code.encode('utf-8')).hexdigest()
    


    def compile_shader():
        """Compile the shader program from the current scene.  Uses caching."""
        nonlocal shader_compile_error
        
        # Check cache first
        shader_hash = get_shader_hash()
        if shader_hash in shader_cache:
            cached_shader, cached_uniforms = shader_cache[shader_hash]
            shader_compile_error = None
            return cached_shader, cached_uniforms
        
        # Not in cache, compile new shader
        try:
            scene_code = scene_builder.generate_raymarch_code()
            # Use selected shader
            postproc_code = generate_postproc_code(sprites_array)
            selected_fragment_shader = load_shader_code(shader_names[shader_choice])
            fragment_shader = selected_fragment_shader.replace("{SDF_LIBRARY}", sdf_library)
            fragment_shader = fragment_shader.replace("{SCENE_CODE}", scene_code)
            fragment_shader = fragment_shader.replace("{FOV_ANGLE_VAL}", str(FOV_ANGLE))
            fragment_shader = fragment_shader.replace("{POSTPROC}", postproc_code[0])
            fragment_shader = fragment_shader.replace("{ADDITIONAL_UNIFORMS}", postproc_code[1])
            fragment_shader = fragment_shader.replace("{ADDITIONAL_SCENE_CODE}", additional_scene_code)
            
            shader_program = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
            
            # Get uniform locations
            uniforms = get_uniform_locations(shader_program)
            
            # Cache the compiled shader
            shader_cache[shader_hash] = (shader_program, uniforms)
            
            shader_compile_error = None
            return shader_program, uniforms
        except Exception as e:
            shader_compile_error = str(e)
            print(f"Shader compilation error:  {e}")
            return None, None


    @MonitorChanges
    def recompile_shader():
        """Recompile shader and update uniform locations.  Returns (success, uniforms_dict). Uses caching."""
        nonlocal shader, uniform_locs
        
        new_shader, new_uniforms = compile_shader()
        if new_shader is None:
            return False, None
        
        if shader is not None and shader != new_shader:
            old_hash = None
            for cached_hash, (cached_shader, _) in shader_cache.items():
                if cached_shader == shader:
                    old_hash = cached_hash
                    break
            
            if old_hash is None:
                glDeleteProgram(shader)
        
        shader = new_shader
        uniform_locs = new_uniforms
        return True, new_uniforms

    shader, uniform_locs = compile_shader()
    if shader is None:
        print("Failed to compile initial shader. Exiting.")
        impl.shutdown()
        glfw.terminate()
        return

    # --- OpenGL Setup ---
    vao, vbo, display_vao, display_vbo, display_shader = init_vao_vbo()

    # --- Framebuffer Setup for Resolution Scaling ---
    fbo = None
    render_texture = None
    fbo_width = 0
    fbo_height = 0

    # --- Accumulation Buffer Setup ---
    accumulation_width = 0
    accumulation_height = 0
    frame_count = 0
    max_frames = 128
    accumulation_textures = [None, None]  # Double buffer
    accumulation_fbos = [None, None]
    current_accum_index = 0  # Which one to write to


    def on_window_close(window):
        nonlocal show_exit_window
        glfw.set_window_should_close(window, False)
        show_exit_window = True

    def restart():
        nonlocal show_restart_window
        show_restart_window = True


    # --- Main Loop ---
    start_time = time.time()
    prev_time = time.time() 

    glfw.set_window_close_callback(window, on_window_close)



    # Load User Config
    # I use JSON format with data extension to avoid confusion with one extension
    default_uconfig = {"Theme": theme, "UIScale" : 1.0}
    try:
        UConfig = load_user_config("UserData/User.data")
    except:
        UConfig = default_uconfig

    if not UConfig or not isinstance(UConfig, dict):
        save_user_config("UserData/User.data", default_uconfig)
        UConfig = default_uconfig
    else:
        theme = UConfig["Theme"]
        for label, color in list(theme.items()):
                # Update the dictionary key with the new list/tuple value
                setattr(ui_themes, label, theme[label])
                ui_themes.setup_theme()

    rebuild_imgui_fonts(impl, "assets/fonts/Roboto-Medium.ttf", 16.0)

    while not glfw.window_should_close(window):
        # calc Delta time 
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        ui_themes.setup_theme()



        # --- FPS calculation ---
        fps_frames += 1
        current_time = time.time()
        if current_time - fps_clock >= 1.0:
            fps_value = fps_frames
            fps_frames = 0
            fps_clock = current_time

        # --- Handle keyboard input ---
        io = get_io()
                
        # Check Ctrl+A for add window 
        if input_handle("Add"):
            if not last_key_a_pressed:
                # Open Add Operation dialog (keeps same code path as the menu)
                show_add_change_window = True
                pending_change_node_id = None
                last_key_a_pressed = True
        else:
            last_key_a_pressed = False
        
        # Check F2 for rename (with debouncing)
        if input_handle("Rename") and selected_item_id is not None and renaming_item_id is None:
            if not last_key_f2_pressed:
                renaming_item_id = selected_item_id
                rename_text = scene_builder.get_item_name(selected_item_id)
                last_key_f2_pressed = True
        else:
            last_key_f2_pressed = False
        
        # Check Delete key for deletion (with debouncing)
        # Only allow deletion if node is direct child of root (depth = 1)
        if input_handle("Delete") and selected_item_id is not None:
            if not last_key_delete_pressed:
                node_to_delete = scene_builder.get_node(selected_item_id)
                if node_to_delete:
                    # Check depth: only delete if parent is None (direct root child)
                    depth = scene_builder.get_node_depth(selected_item_id)
                    if depth == 1:  # Direct child of root
                        if scene_builder.delete_node(selected_item_id):
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                            selected_item_id = None
                            scene_builder.update_selected_item_id(selected_item_id)
                            selection_mode = None
                    else:
                        # Cannot delete - show message (optional)
                        pass
                last_key_delete_pressed = True
        else:
            last_key_delete_pressed = False
        
        # Check Ctrl+B for compile (with debouncing)
        if input_handle("Compile"):
            if not last_key_compile_pressed:
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms
                last_key_compile_pressed = True
        else:
            last_key_compile_pressed = False


        if glfw.get_key(window, glfw.KEY_F12) == glfw.PRESS:
            take_screenshot(window)



        # Increment frame counter only when using cycles shader
        if shader_choice == 1:   # cycles_fragment_shader.glsl
            frame_count = min(frame_count + 1, max_frames)
        else: 
            frame_count = 0  # Reset accumulation when switching shaders
        
        # Get window and rendering dimensions
        width, height = glfw.get_framebuffer_size(window)
        menu_bar_height = int(imgui.get_frame_height())
        panel_width = int(width * PANEL_WIDTH_RATIO)
        rendering_width = width - 2 * panel_width
        rendering_height = height - menu_bar_height
        panel_elem_width_vec3 = (panel_width/4)-14
        panel_elem_width_float = (panel_width/2)-14

        
        scaled_rendering_width = int(rendering_width * resolution_scale)
        scaled_rendering_height = int(rendering_height * resolution_scale)



        # Get the current window size
        width, height = glfw.get_framebuffer_size(window)
        # Get menu bar height (needed for calculations) - convert to int for glViewport
        menu_bar_height = int(imgui.get_frame_height())
        panel_width = int(width * PANEL_WIDTH_RATIO)
        rendering_width = width - 2 * panel_width
        rendering_height = height - menu_bar_height
        
        # Apply resolution scale
        scaled_rendering_width = int(rendering_width * resolution_scale)
        scaled_rendering_height = int(rendering_height * resolution_scale)


        # If we recompiled the shader, we will update the fbo
        global monitor
        if monitor == True and shader_choice == 1:
            monitor = False
            frame_count = 0
            clear_accumulation_fbos(accumulation_fbos,scaled_rendering_width, scaled_rendering_height)
            current_accum_index = 0



        # Handle MMB press and release for camera control
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            shift_pressed = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
            
            if not is_mmb_pressed:
                is_mmb_pressed = True
                is_shift_mmb_pressed = shift_pressed
                last_x, last_y = glfw.get_cursor_pos(window)
                if shift_pressed:
                    last_pan_x, last_pan_y = last_x, last_y
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.RELEASE:
            if is_mmb_pressed:
                is_mmb_pressed = False
                is_shift_mmb_pressed = False

        if is_mmb_pressed or dragging or R_dragging:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        else:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)


        prev_cam_yaw = cam_yaw
        prev_cam_pitch = cam_pitch
        prev_cam_radius = cam_radius
        prev_cam_orbit = cam_orbit


        # Handle mouse wheel input for camera zoom
        if io.mouse_wheel != 0:
            target_radius -= io.mouse_wheel * ZOOM_SENSITIVITY
            target_radius = max(MIN_RADIUS, min(MAX_RADIUS, target_radius))

        cam_radius += (target_radius - cam_radius) * (CAMERA_LERP_FACTOR * delta_time)

        # Only update target camera angles if MMB is pressed
        if is_mmb_pressed:
            current_x, current_y = glfw.get_cursor_pos(window)
            if is_shift_mmb_pressed:
                # Panning mode: Shift + MMB
                dx = current_x - last_pan_x
                dy = current_y - last_pan_y
                last_pan_x, last_pan_y = current_x, current_y
                target_pan_x += dx * PAN_SENSITIVITY
                target_pan_y += dy * PAN_SENSITIVITY
            else:
                # Rotation mode: MMB only
                dx = current_x - last_x
                dy = current_y - last_y
                last_x, last_y = current_x, current_y
                target_yaw -= dx * MOUSE_SENSITIVITY
                target_pitch += dy * MOUSE_SENSITIVITY
                target_pitch = max(MIN_PITCH, min(MAX_PITCH, target_pitch))


        # --- Camera vectors ---
        cam_yaw, cam_pitch = camera.update(target_yaw, target_pitch, target_pan_y, target_pan_x, CAMERA_LERP_FACTOR*delta_time)
        cam_orbit = camera.get_orbit()

        # -----

        if io.keys_down[glfw.KEY_HOME]:
            target_pan_x = target_pan_y = 0.0
            cam_orbit = [0.0,0.0,0.0]



        elip = 0.0001
        if (abs(cam_yaw - prev_cam_yaw) > elip or 
            abs(cam_pitch - prev_cam_pitch) > elip or
            abs(cam_radius - prev_cam_radius) > elip or
            any(abs(cam_orbit[i] - prev_cam_orbit[i]) > elip for i in range(3))):

            # Reset accumulation buffers so no stale data is read later
            frame_count = 0
            clear_accumulation_fbos(accumulation_fbos,scaled_rendering_width, scaled_rendering_height)
            current_accum_index = 0


        prev_cam_yaw = cam_yaw
        prev_cam_pitch = cam_pitch
        prev_cam_radius = cam_radius
        prev_cam_orbit = cam_orbit


        # TODO: Unsuccessful attempt
        #circle_points = proj_3d22d(np.array([[0.0, 0.0, 100.0]]), cam_yaw, cam_pitch)
        #print(circle_points)
        
        #bg_draw_list = imgui.get_background_draw_list()
        
        #bg_draw_list.add_circle_filled(
        #    circle_points[0][0]+(width//2),
        #    circle_points[0][1]+(height//2),
        #    25, 
        #    imgui.get_color_u32_rgba(1, 0, 0, 1)
        #)


        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)
        
        
        # --- Setup accumulation buffer if using cycles shader ---
        use_accumulation = 0
        accbuffer_output = False
        accbuffer_output, \
        scaled_rendering_width, scaled_rendering_height, \
        accumulation_fbos, accumulation_textures, \
        accumulation_width, accumulation_height = setup_accumulation_buffer(
                scaled_rendering_width, scaled_rendering_height,
                accumulation_fbos, accumulation_textures,
                accumulation_width, accumulation_height
            )

        if shader_choice == 1:  # cycles.glsl
            if accbuffer_output:
                use_accumulation = 1

        # --- RENDER TO ACCUMULATION BUFFER ---
        if shader is not None and shader_choice == 1 and use_accumulation == 1:
            write_buffer = current_accum_index
            read_buffer = 1 - current_accum_index
            glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[write_buffer])
            glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)

            if frame_count == 0:
                glClear(GL_COLOR_BUFFER_BIT)

            if frame_count < max_frames:
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], scaled_rendering_width, scaled_rendering_height)
                    glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                    glUniform1f(uniform_locs['camYaw'], cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], cam_pitch)
                    glUniform1f(uniform_locs['radius'], cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                    glUniform1i(uniform_locs['frameIndex'], frame_count)
                    glUniform1i(uniform_locs['maxFrames'], max_frames)
                    set_move_pos_uniform(shader, uniform_locs, drag_position)
                    set_move_rot_uniform(shader, uniform_locs, drag_rot_position)

                    # Bind accumulation texture for reading
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, accumulation_textures[read_buffer])
                    glUniform1i(uniform_locs['accumulationTexture'], 0)
                    glUniform1i(uniform_locs['useAccumulation'], 1)

                    glUniform3f(uniform_locs['col_sky_top'], sky_top_color[0], sky_top_color[1], sky_top_color[2])
                    glUniform3f(uniform_locs['col_sky_bottom'], sky_bottom_color[0], sky_bottom_color[1], sky_bottom_color[2])
                    
                    glUniform3f(uniform_locs['LightDir'], LightDir[0], LightDir[1], LightDir[2])

                bind_sprite_textures(uniform_locs, sprites_array)
                glBindVertexArray(vao)
                glDrawArrays(GL_QUADS, 0, 4)

            # Switch back to default framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, width, height)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, accumulation_textures[write_buffer])

            # Display accumulated result
            glUseProgram(display_shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, accumulation_textures[write_buffer])
            glUniform1i(glGetUniformLocation(display_shader, "renderTexture"), 0)
            

            # Set isAccumulation to 1 if rendering is complete
            if frame_count >= max_frames:
                glUniform1i(glGetUniformLocation(display_shader, "isAccumulation"), 1)
            else:
                glUniform1i(glGetUniformLocation(display_shader, "isAccumulation"), 0)
            #print(glGetUniformLocation(display_shader,"isAccumulation"))

            glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
            bind_sprite_textures(uniform_locs, sprites_array)
            glBindVertexArray(display_vao)
            glDrawArrays(GL_QUADS, 0, 4)
            glBindVertexArray(0)

            glViewport(0, 0, width, height)
            current_accum_index = 1 - current_accum_index
        
        # --- RENDER DIRECTLY (if NOT using cycles or accumulation disabled) ---
        elif shader is not None: 
            glUseProgram(shader)
            if uniform_locs is not None:
                current_time_uniform = time.time() - start_time
                glUniform1f(uniform_locs['time'], current_time_uniform)
                glUniform2f(uniform_locs['resolution'], rendering_width, rendering_height)
                # When rendering directly into the screen viewport we must subtract the panel/menu offset
                glUniform2f(uniform_locs['viewportOffset'], float(panel_width), float(menu_bar_height))
                glUniform1f(uniform_locs['camYaw'], cam_yaw)
                glUniform1f(uniform_locs['camPitch'], cam_pitch)
                glUniform1f(uniform_locs['radius'], cam_radius)
                glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                glUniform1i(uniform_locs['frameIndex'], 0)
                glUniform1i(uniform_locs['useAccumulation'], 0)
                set_move_pos_uniform(shader, uniform_locs, drag_position)
                set_move_rot_uniform(shader, uniform_locs, drag_rot_position)

                glUniform3f(uniform_locs['col_sky_top'], sky_top_color[0], sky_top_color[1], sky_top_color[2])
                glUniform3f(uniform_locs['col_sky_bottom'], sky_bottom_color[0], sky_bottom_color[1], sky_bottom_color[2])

                glUniform1i(uniform_locs['grid_enabled'], GridEnabled)
                glUniform3f(uniform_locs['LightDir'], LightDir[0], LightDir[1], LightDir[2])


            # Check if viewport is minimized
            if rendering_width > 0 and rendering_height > 0:
                glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
                glBindVertexArray(vao)
                bind_sprite_textures(uniform_locs, sprites_array)
                glDrawArrays(GL_QUADS, 0, 4)

            glViewport(0, 0, width, height)



        # --- TOP MENU BAR ---
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                if imgui.menu_item("Save Scene", "Ctrl+S")[0]:
                    # Trigger save dialog
                    success, message = save_scene_dialog(scene_builder, window)
                    save_load_message = message
                    save_load_message_time = time.time()
        
                if imgui.menu_item("Load Scene", "Ctrl+O")[0]:
                    # Trigger load dialog
                    success, message = load_scene_dialog(scene_builder)
                    save_load_message = message
                    save_load_message_time = time.time()
                    if success:
                        glob_history.undo_stack.clear()
                        glob_history.redo_stack.clear() 
                        scene_builder.update_glob_history(glob_history)
                        selected_item_id = None
                        scene_builder.update_selected_item_id(selected_item_id)
                        selection_mode = None
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms

                imgui.separator()
                imgui.spacing()
                if imgui.begin_menu("Export..."):
                    if imgui.menu_item("As Volume")[0]:
                        show_export_vol_window = True
                    if imgui.menu_item("To OBJ")[0]:
                        show_export_obj_window = True
                    imgui.end_menu()

                imgui.spacing()

                imgui.separator()
                imgui.spacing()
                if imgui.menu_item("Exit", "Alt+F4")[0]:
                    on_window_close(window)

                imgui.end_menu()

            if imgui.begin_menu("Edit", True):
                if imgui.menu_item("Add Primitive/Operation", "Ctrl+A")[0]:
                    show_add_change_window = True
                    pending_change_node_id = None
                if imgui.menu_item("Compile Shader", "Ctrl+B")[0]:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                imgui.end_menu()
    
            if imgui.begin_menu("View", True):
                if imgui.menu_item("Settings", "F10")[0]:
                    show_settings_window = True
                imgui.end_menu()

            if imgui.begin_menu("Editor", True):
                if imgui.menu_item("Settings")[0]:
                    show_editor_settings_window = True
                imgui.end_menu()
    
            if imgui.begin_menu("About", True):
                if imgui.menu_item("Information")[0]:
                    show_about_window = True
                imgui.end_menu()


            # --- Fast Change Rendering mode ---
            cursor_pos = imgui.get_cursor_pos()
            window_width = imgui.get_window_width()
            remaining_width = window_width - cursor_pos.x

            # Calculate positions for centered buttons
            button_width = 100
            spacing = 20
            total_buttons_width = 3 * button_width + 2 * spacing 
            start_x = (cursor_pos.x + (remaining_width - total_buttons_width)) / 2

            imgui.set_cursor_pos_x(start_x)
            if imgui.button("Template", button_width):
                shader_choice = 0
                # Recompile with new shader
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.set_cursor_pos_x(start_x + button_width + spacing)
            if imgui.button("Cycles", button_width):
                shader_choice = 1
                # Recompile with new shader
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.set_cursor_pos_x(start_x + 2 * (button_width + spacing))

            import threading

            if imgui.button("Script", button_width):
                def run_tkinter_app():
                    global CE_app, tkinter_thread
                    if CE_app is None:
                        CE_app = CodeEdit.GLSLEditor()

                        def on_close():
                            global CE_app
                            CE_app.destroy()
                            CE_app = None

                        CE_app.protocol("WM_DELETE_WINDOW", on_close)
                        CE_app.mainloop()

                # Start Tkinter in a new thread (non-blocking)
                tkinter_thread = threading.Thread(target=run_tkinter_app, daemon=True)
                tkinter_thread.start()
                
            if CE_app != None:
                if CE_app.rec == True: # I don't think it's reliable, but oh well!
                    additional_scene_code = CE_app.get_plain_text()
                    recompile_shader()
                    CE_app.rec = False


            imgui.end_main_menu_bar()
        

        # Check Ctrl + S/O
        if input_handle("Open"):
            if not last_key_o_pressed: 
                success, message = load_scene_dialog(scene_builder)
                save_load_message = message
                save_load_message_time = time.time()
                if success:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    selected_item_id = None
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = None
                last_key_o_pressed = True
        else:
            last_key_o_pressed = False


        # --- Duplicate (Ctrl+D) ---
        if input_handle("Duplicate"):
            if not last_key_d_pressed:
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
                    selection_mode = 'node'
                    # Recompile shader to pick up new primitives
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

                last_key_d_pressed = True
        else:
            last_key_d_pressed = False
        


        if input_handle("Save"):
            if not last_key_s_pressed: 
                success, message = save_scene_dialog(scene_builder, window)
                save_load_message = message
                save_load_message_time = time.time()
                if success:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    selected_item_id = None
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = None
                last_key_s_pressed = True
        else:
            last_key_s_pressed = False


        # Check Undo/Redo keys Ctrl+Z/Y
        if input_handle("Undo") and io.key_ctrl and not io.key_shift:
            if not last_key_z_pressed: 
                undo_success = glob_history.undo()
                scene_builder.update_glob_history(glob_history)
                if undo_success:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                last_key_z_pressed = True
        else:
            last_key_z_pressed = False


        if input_handle("Redo") or input_handle("Redo2"):
            if not last_key_y_pressed: 
                undo_success = glob_history.redo()
                scene_builder.update_glob_history(glob_history)
                if undo_success:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                last_key_y_pressed = True
        else:
            last_key_y_pressed = False



        # Drag on G
        key_g_is_down = input_handle("Move")
        key_x_is_down = input_handle("X")
        key_y_is_down = input_handle("Y")
        key_z_is_down = input_handle("Z")

        # Toggle dragging on G press (edge detect)
        if key_g_is_down and not last_key_g_pressed:
            # Toggle dragging state
            dragging = not dragging

            if dragging:
                # Start dragging: capture which item and initialize drag state
                dragging_op_id = selected_item_id

                if dragging_op_id:
                    node = scene_builder.get_node(dragging_op_id)
                    if node and node.node_type == 'primitive':
                        prim = node.item_data
                        # Copy the primitive start position
                        drag_start_pos = prim.position[:]
                        # Reset accumulated movement
                        drag_accum = [0.0, 0.0, 0.0]
                        # Record starting mouse cursor
                        drag_last_x, drag_last_y = glfw.get_cursor_pos(window)
                    else:
                        # Not a primitive, can't drag
                        dragging_op_id = None
                        drag_start_pos = None
                        drag_accum = [0.0, 0.0, 0.0]
                else:
                    dragging_op_id = None
                    drag_start_pos = None
                    drag_accum = [0.0, 0.0, 0.0]

                # Reset axis toggles when starting a new drag
                axis_toggled_gx = axis_toggled_gy = axis_toggled_gz = False

            else:
                # Stop dragging: commit final position
                if dragging_op_id:
                    node = scene_builder.get_node(dragging_op_id)
                    if node and node.node_type == 'primitive':
                        prim = node.item_data
                        final_pos = prim.position
                        # Register only if changed
                        if drag_start_pos is not None and final_pos != drag_start_pos:
                            # Directly update (no undo needed for now)
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                # Clear drag state
                dragging_op_id = None
                drag_start_pos = None
                drag_accum = [0.0, 0.0, 0.0]
                axis_toggled_gx = axis_toggled_gy = axis_toggled_gz = False

        # Always update last_key_g_pressed for proper edge detection
        last_key_g_pressed = key_g_is_down

        # Handle axis toggles (Blender-style)
        if dragging:
            if key_x_is_down and not last_key_gx_pressed:
                state = not axis_toggled_gx
                axis_toggled_gx, axis_toggled_gy, axis_toggled_gz = state, False, False

            if key_y_is_down and not last_key_gy_pressed:
                state = not axis_toggled_gy
                axis_toggled_gx, axis_toggled_gy, axis_toggled_gz = False, state, False

            if key_z_is_down and not last_key_gz_pressed:
                state = not axis_toggled_gz
                axis_toggled_gx, axis_toggled_gy, axis_toggled_gz = False, False, state

        # Update the "last key" flags for X/Y/Z
        last_key_gx_pressed = key_x_is_down
        last_key_gy_pressed = key_y_is_down
        last_key_gz_pressed = key_z_is_down

        # Determine active axis
        active_axis = None
        if axis_toggled_gx:
            active_axis = 0
        elif axis_toggled_gy:
            active_axis = 1
        elif axis_toggled_gz:
            active_axis = 2

        # Per-frame drag movement
        if dragging and dragging_op_id:
            node = scene_builder.get_node(dragging_op_id)
            if node and node.node_type == 'primitive':
                # Read current mouse and compute delta
                current_x, current_y = glfw.get_cursor_pos(window)
                dx = current_x - drag_last_x
                dy = current_y - drag_last_y
                # Store for next frame
                drag_last_x, drag_last_y = current_x, current_y

                # Convert to mouse-space movement
                mouse_delta_x = dx * DRAG_SENSITIVITY
                mouse_delta_y = -dy * DRAG_SENSITIVITY

                if np.linalg.norm(np.array([mouse_delta_x, mouse_delta_y])) > 0.01:
                    frame_count = 0
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
                drag_accum[0] += move_delta_z
                drag_accum[1] += move_delta_y
                drag_accum[2] += move_delta_x

                # Compute new position
                prim = node.item_data
                if drag_start_pos is None:
                    drag_start_pos = prim.position.copy()

                new_pos = [
                    drag_start_pos[0] + drag_accum[0],
                    drag_start_pos[1] + drag_accum[1],
                    drag_start_pos[2] + drag_accum[2],
                ]

                # Apply live position
                prim.position = new_pos
                drag_position = new_pos.copy()

        else:
            # When not dragging
            if selected_item_id:
                node = scene_builder.get_node(selected_item_id)
                if node and node.node_type == 'primitive':
                    prim = node.item_data
                    drag_position = prim.position
            else:
                drag_position = [0.0, 0.0, 0.0]




        # ---- Rotate (MoveRot) using R key ----
        key_r_is_down = input_handle("Rotate")
        key_x_is_down = input_handle("X")
        key_y_is_down = input_handle("Y")
        key_z_is_down = input_handle("Z")

        # Edge-detect R press to toggle rotation mode
        if key_r_is_down and not last_key_r_pressed:
            R_dragging = not R_dragging

            if R_dragging:
                # Start rotation: capture selected item and initialize rotation state
                R_dragging_op_id = selected_item_id

                if R_dragging_op_id and R_dragging_op_id in scene_builder.id_to_node:
                    node = scene_builder.get_node(R_dragging_op_id)
                    if node and node.node_type == 'primitive':
                        prim = node.item_data
                        R_drag_start_pos = prim.rotation.copy()
                        R_drag_accum = [0.0, 0.0, 0.0]
                        R_drag_last_x, R_drag_last_y = glfw.get_cursor_pos(window)
                    else:
                        R_dragging_op_id = None
                        R_drag_start_pos = None
                        R_drag_accum = [0.0, 0.0, 0.0]
                else:
                    R_dragging_op_id = None
                    R_drag_start_pos = None
                    R_drag_accum = [0.0, 0.0, 0.0]

                axis_toggled_rx = axis_toggled_ry = axis_toggled_rz = False

            else:
                # Stop rotation: commit final rotation (register undo/redo)
                if R_dragging_op_id and R_dragging_op_id in scene_builder.id_to_node:
                    node = scene_builder.get_node(R_dragging_op_id)
                    if node and node.node_type == 'primitive':
                        prim = node.item_data
                        final_rot = prim.rotation
                        if R_drag_start_pos is not None and final_rot != R_drag_start_pos:
                            # Use scene_builder to register the change (compatibility method)
                            scene_builder.modify_primitive_property(R_dragging_op_id, 'rotation', final_rot)
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                R_dragging_op_id = None
                R_drag_start_pos = None
                R_drag_accum = [0.0, 0.0, 0.0]
                axis_toggled_rx = axis_toggled_ry = axis_toggled_rz = False

        # Update last R state
        last_key_r_pressed = key_r_is_down

        # Rotation axis toggles (Blender-style)
        if R_dragging:
            if key_x_is_down and not last_key_rx_pressed:
                state = not axis_toggled_rx
                axis_toggled_rx, axis_toggled_ry, axis_toggled_rz = state, False, False
            if key_y_is_down and not last_key_ry_pressed:
                state = not axis_toggled_ry
                axis_toggled_rx, axis_toggled_ry, axis_toggled_rz = False, state, False
            if key_z_is_down and not last_key_rz_pressed:
                state = not axis_toggled_rz
                axis_toggled_rx, axis_toggled_ry, axis_toggled_rz = False, False, state

        last_key_rx_pressed = key_x_is_down
        last_key_ry_pressed = key_y_is_down
        last_key_rz_pressed = key_z_is_down

        # Per-frame rotation update while R_dragging is active
        if R_dragging and R_dragging_op_id and R_dragging_op_id in scene_builder.id_to_node:
            current_x, current_y = glfw.get_cursor_pos(window)
            dx = current_x - R_drag_last_x
            dy = current_y - R_drag_last_y
            R_drag_last_x, R_drag_last_y = current_x, current_y

            R_ROT_SENSITIVITY = 0.005

            rot_delta_x = -dy * R_ROT_SENSITIVITY
            rot_delta_y = -dx * R_ROT_SENSITIVITY
            rot_delta_z = 0.0

            if axis_toggled_rx:
                rot_delta_y = 0.0
                rot_delta_z = 0.0
            elif axis_toggled_ry:
                rot_delta_x = 0.0
                rot_delta_z = 0.0
            elif axis_toggled_rz:
                rot_delta_x = 0.0
                rot_delta_y = 0.0
                rot_delta_z = -dx * R_ROT_SENSITIVITY

            if abs(rot_delta_x) + abs(rot_delta_y) + abs(rot_delta_z) > 1e-5:
                frame_count = 0
                clear_accumulation_fbos(accumulation_fbos, scaled_rendering_width, scaled_rendering_height)

            R_drag_accum[0] += rot_delta_x
            R_drag_accum[1] += rot_delta_y
            R_drag_accum[2] += rot_delta_z

            node = scene_builder.get_node(R_dragging_op_id)
            if node and node.node_type == 'primitive':
                prim = node.item_data
                if R_drag_start_pos is None:
                    R_drag_start_pos = prim.rotation.copy()
                new_rot = [
                    R_drag_start_pos[0] + R_drag_accum[0],
                    R_drag_start_pos[1] + R_drag_accum[1],
                    R_drag_start_pos[2] + R_drag_accum[2],
                ]
                prim.rotation = new_rot
                drag_rot_position = new_rot.copy()

        else:
            # keep shader MoveRot aligned with selection (or zero)
            if selected_item_id and selected_item_id in scene_builder.id_to_node:
                node = scene_builder.get_node(selected_item_id)
                if node and node.node_type == 'primitive':
                    prim = node.item_data
                    drag_rot_position = prim.rotation
            else:
                drag_rot_position = [0.0, 0.0, 0.0]





        # Check F10 for settings
        if io.keys_down[glfw.KEY_F10]:
            if not last_key_f10_pressed:
                show_settings_window = True
                last_key_f10_pressed = True
        else:
            last_key_f10_pressed = False
        
        # --- RENDER TO FRAMEBUFFER AT SCALED RESOLUTION ---
        # If we've already rendered & displayed the accumulation buffer above (cycles shader),
        # skip the further framebuffer / direct rendering to avoid double-draw and viewport offset.
        if shader is not None and shader_choice == 1 and use_accumulation == 1:
            # accumulation rendering & display already handled above
            pass

        elif shader is not None and display_shader is not None and resolution_scale != 1.0:
            # Setup framebuffer
            framebuffer_output = False # ouu!
            framebuffer_output, \
            scaled_rendering_width, scaled_rendering_height, \
            fbo, render_texture, \
            fbo_width, fbo_height = setup_framebuffer(
                                    scaled_rendering_width, scaled_rendering_height,
                                    fbo, render_texture, fbo_width, fbo_height
                                    )

            if framebuffer_output:
                # Render to framebuffer
                glBindFramebuffer(GL_FRAMEBUFFER, fbo)
                glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)
                glClear(GL_COLOR_BUFFER_BIT)
                
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], scaled_rendering_width, scaled_rendering_height)
                    glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                    glUniform1f(uniform_locs['camYaw'], cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], cam_pitch)
                    glUniform1f(uniform_locs['radius'], cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                    set_move_pos_uniform(shader, uniform_locs, drag_position)
                    set_move_rot_uniform(shader, uniform_locs, drag_rot_position)

                bind_sprite_textures(uniform_locs, sprites_array)


                glBindVertexArray(vao)
                glDrawArrays(GL_QUADS, 0, 4)
                
                # Switch back to default framebuffer
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glViewport(0, 0, width, height)
                
                # Display the texture stretched to the viewport
                glUseProgram(display_shader)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, render_texture)
                glUniform1i(glGetUniformLocation(display_shader, "renderTexture"), 0)
                
                # Set viewport to the rendering area (accounting for menu bar)
                glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
                glBindVertexArray(display_vao)
                glDrawArrays(GL_QUADS, 0, 4)
                glBindVertexArray(0)
                
                # Reset viewport
                glViewport(0, 0, width, height)
            else:
                # Fallback to direct rendering if framebuffer fails
                if shader is not None:
                    glUseProgram(shader)
                    if uniform_locs is not None:
                        current_time_uniform = time.time() - start_time
                        glUniform1f(uniform_locs['time'], current_time_uniform)
                        glUniform2f(uniform_locs['resolution'], scaled_rendering_width, scaled_rendering_height)
                        glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                        glUniform1f(uniform_locs['camYaw'], cam_yaw)
                        glUniform1f(uniform_locs['camPitch'], cam_pitch)
                        glUniform1f(uniform_locs['radius'], cam_radius)
                        glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                        set_move_pos_uniform(shader, uniform_locs, drag_position)
                        set_move_rot_uniform(shader, uniform_locs, drag_rot_position)

                    glViewport(panel_width, menu_bar_height, scaled_rendering_width, scaled_rendering_height)
                    glBindVertexArray(vao)
                    bind_sprite_textures(uniform_locs, sprites_array)
                    glDrawArrays(GL_QUADS, 0, 4)
                    glViewport(0, 0, width, height)
        else:
            # Direct rendering when scale is 1.0 or display shader not available
            # Skip if accumulation handled above (see guard at top)
            if shader is not None:
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], rendering_width, rendering_height)
                    # Default framebuffer viewport is offset by the left panel and menu bar
                    glUniform2f(uniform_locs['viewportOffset'], float(panel_width), float(menu_bar_height))
                    glUniform1f(uniform_locs['camYaw'], cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], cam_pitch)
                    glUniform1f(uniform_locs['radius'], cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                    set_move_pos_uniform(shader, uniform_locs, drag_position)
                    set_move_rot_uniform(shader, uniform_locs, drag_rot_position)

                # Check if viewport is minimized
                if rendering_width > 0 and rendering_height > 0:
                    glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
                    glBindVertexArray(vao)
                    bind_sprite_textures(uniform_locs, sprites_array)
                    glDrawArrays(GL_QUADS, 0, 4)

                glViewport(0, 0, width, height)
        

        # --- SETTINGS WINDOW ---
        if show_settings_window:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 150)
            imgui.set_next_window_size(400, 300)  # Increased height
            is_open, show_settings_window = imgui.begin("Settings", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                show_settings_window = False
            
            imgui.text("Rendering Settings")
            imgui.separator()
            
            # Shader Selection
            imgui.text("Fragment Shader:")
            clicked, shader_choice = imgui.combo(
                "##shader_select",
                shader_choice,
                [name.replace("shaders/fragment/", "") for name in shader_names]
            )
            if clicked:
                # Recompile with new shader
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms
            
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            # Resolution Scale
            imgui.text("Resolution Scale:")
            imgui.same_line()
            imgui.text(f"{resolution_scale:.2f}x")
            
            changed, resolution_scale = imgui.slider_float("##resolution_scale", resolution_scale, 0.25, 2.0, "%.2f")
            if changed:
                frame_count = 0


            imgui.spacing()
            imgui.text_colored("1.0 = Normal resolution", 0.7, 0.7, 0.7, 1.0)
            imgui.text_colored("2.0 = Oversampling (better quality)", 0.7, 0.7, 0.7, 1.0)
            imgui.text_colored("<1.0 = Low resolution (better performance)", 0.7, 0.7, 0.7, 1.0)
            
            imgui.spacing()
            imgui.separator()


            # Show Sky colors
            imgui.text("Sky Top Color:")
            top_color_changed, top_color_rgba = imgui.color_edit3("SkyTopColor##color", sky_top_color[0], sky_top_color[1], sky_top_color[2])
            if top_color_changed:
                sky_top_color = list(top_color_rgba[:3])  # Only use RGB, ignore alpha
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.text("Sky Bottom Color:")
            bottom_color_changed, bottom_color_rgba = imgui.color_edit3("SkyBottomColor##color", sky_bottom_color[0], sky_bottom_color[1], sky_bottom_color[2])
            if bottom_color_changed:
                sky_bottom_color = list(bottom_color_rgba[:3])  # Only use RGB, ignore alpha
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            if shader_choice == 0:
                imgui.text("Grid Enabled:")
                changed, GridEnabled = imgui.checkbox("", GridEnabled)
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

                imgui.spacing()
                imgui.separator()

            elif shader_choice == 1:
                imgui.text("Max Samples count:")
                changed, max_frames = imgui.input_int("", max_frames)
                max_frames = max(max_frames, 8)
                if changed:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

                imgui.spacing()
                imgui.separator()
            

            imgui.text("Sun:")
            changed, LightDir = input_vec3("Sun Direction", LightDir)
            if changed:
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.spacing()
            imgui.separator()


            # Calculate scaled size for display
            scaled_w = int(rendering_width * resolution_scale)
            scaled_h = int(rendering_height * resolution_scale)
            imgui.text(f"Current render size: {scaled_w}x{scaled_h}")
            imgui.text(f"Base size: {rendering_width}x{rendering_height}")
            
            imgui.spacing()
            if imgui.button("Close", -1):
                show_settings_window = False
            
            
            imgui.end()




        # --- Editor Settings Window ---
        # --- Content Functions (Placeholders) ---
        def render_themes_tab():
            nonlocal theme
            changes = []
            for label in theme:
                item = theme[label]
                if isinstance(item, list) and len(item) == 4:
                    changed, color_rgba = imgui.color_edit4(label, *item)
                    if changed:
                        changes.append((label, list(color_rgba)))
                elif isinstance(item, list) and len(item) == 2:
                    changed, size = input_vec2(label, item)
                    if changed:
                        changes.append((label, list(size)))

            for label, new_value in changes:
                theme[label] = new_value
                setattr(ui_themes, label, new_value)
            if changes:
                ui_themes.setup_theme()
            
            imgui.spacing()
            if imgui.button("Reset Theme", -1):
                theme = copy.deepcopy(default_uconfig["Theme"])
                for label, item in theme.items():
                    setattr(ui_themes, label, item)
                ui_themes.setup_theme()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        def render_user_tab():
            imgui.text("User Profile Settings Content Here... WIP")

        def render_shortcuts_tab():
            for name, keys in ShortCuts.items():
                imgui.text(name)
                imgui.same_line()

                for key in (keys,):
                    imgui.text(str(key))
                    imgui.same_line()

                imgui.spacing()


        if show_editor_settings_window:
            # Set initial positioning and size for the main window container
            imgui.set_next_window_position(width // 2 - 400, height // 2 - 300)
            imgui.set_next_window_size(800, 600)
            
            is_open, show_editor_settings_window = imgui.begin("Editor Settings", True, imgui.WINDOW_NO_COLLAPSE)
            
            if is_open:
                # 1. Setup two columns: one narrow for navigation, one wide for content
                # We use a fixed width ratio or absolute width if preferred.
                # Here, we use the available space divided by 5 (1/5 for sidebar, 4/5 for content)
                if imgui.begin_child("SettingsTabs", 0, 0, border=False):
                    
                    # Sidebar Width (e.g., 150 pixels wide)
                    sidebar_width = 150
                    
                    # --- Left Panel: Navigation Buttons ---
                    imgui.begin_group()
                    
                    # Button 1: Themes
                    if imgui.button("Themes", width=sidebar_width):
                        current_settings_tab = "Themes"
                    
                    imgui.separator()

                    # Button 2: User
                    if imgui.button("User", width=sidebar_width):
                        current_settings_tab = "User"
                        
                    imgui.separator()

                    # Button 3: Shortcuts
                    if imgui.button("Shortcuts", width=sidebar_width):
                        current_settings_tab = "Shortcuts"
                        
                    imgui.end_group()
                    
                    # --- Content Separator (Visual separation if columns aren't perfect) ---
                    # Move cursor over to where the content panel should start
                    imgui.set_cursor_pos_x(sidebar_width + 10)
                    imgui.same_line()

                    # --- Right Panel: Content Area ---
                    # Calculate remaining width for content area
                    window_width = imgui.get_window_width()
                    content_width = window_width - sidebar_width - 30 # Subtract sidebar + padding/separator

                    # Start the content rendering block
                    if imgui.begin_child("SettingsContent", content_width, 400, border=False):
                        
                        if current_settings_tab == "Themes":
                            render_themes_tab()
                        elif current_settings_tab == "User":
                            render_user_tab()
                        elif current_settings_tab == "Shortcuts":
                            render_shortcuts_tab()

                        imgui.end_child() # End SettingsContent

                    imgui.end_child()
                    
            if not is_open:
                show_editor_settings_window = False

            imgui.end()



        if show_export_vol_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 125)
            imgui.set_next_window_size(300, 250)
            is_open, show_export_vol_window = imgui.begin("Export as Volume", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                show_export_vol_window = False

            imgui.text("Grid Size:")
            changed, grid_size = imgui.input_int("##GridSize", grid_size, 8)
            imgui.text_colored(
                "Note that its dimensions range \nfrom -GridSize/2 to +GridSize/2.",
                0.56, 0.93, 0.56
            )

            imgui.spacing()

            changed, vox_quality = input_float("Vox. Quality", vox_quality, 0.25, 100)

            changed, exp_use_color = imgui.checkbox("Use Color", exp_use_color)

            imgui.separator()
            imgui.spacing()

            file_preview_size = sdfexp.calculate_sdf_file_size(grid_size, vox_quality, exp_use_color)
            if file_preview_size[1]>1:
                imgui.text(f"File size = {file_preview_size[1]:.2f} mb")
            else:
                imgui.text(f"File size = {file_preview_size[0]:.2f} kb")

            imgui.spacing()
            imgui.spacing()

            if imgui.button("Cancel", 135,30):
                show_export_vol_window = False

            imgui.same_line(150)

            if imgui.button("Export", 135,30):
                code = scene_builder.generate_raymarch_code()
                comp_bin = sdfexp.compute_sdf_3d(grid_size, vox_quality, code, additional_scene_code, exp_use_color, window)
                save_sdfvol_dialog(sdfexp, comp_bin)

                show_export_vol_window = False

            imgui.end()

        if show_export_obj_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 130)
            imgui.set_next_window_size(300, 260)
            is_open, show_export_obj_window = imgui.begin("Export to OBJ", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                show_export_obj_window = False

            imgui.text("Grid Size:")
            changed, grid_size = imgui.input_int("##GridSize", grid_size, 8)
            imgui.text_colored(
                "Note that its dimensions range \nfrom -GridSize/2 to +GridSize/2.",
                0.56, 0.93, 0.56
            )

            imgui.spacing()

            changed, vox_quality = input_float("Voxelization Quality", vox_quality, 0.25, 100)

            imgui.separator()
            imgui.spacing() 

            changed, export_level = input_float("Level", export_level, 0.05, 100)
            export_level = np.clip(export_level, 0.0, 1.0)

            imgui.spacing()

            changed, export_z_up = imgui.checkbox("Z up", export_z_up)

            imgui.same_line()

            changed, exp_use_color = imgui.checkbox("Use Color", exp_use_color)


            imgui.separator()
            imgui.spacing()

            if imgui.button("Cancel", 135,30):
                show_export_obj_window = False

            imgui.same_line(150)

            if imgui.button("Export", 135,30):
                code = scene_builder.generate_raymarch_code()
                comp_bin = sdfexp.compute_sdf_3d(grid_size, vox_quality, code, additional_scene_code, exp_use_color, window)
                dist_sdf = None
                color_sdf = None

                if isinstance(comp_bin, tuple):
                    elvl = np.interp(export_level, [0,1], [comp_bin[0].min(), comp_bin[0].max()])
                    dist_sdf, color_sdf = comp_bin
                else:
                    elvl = np.interp(export_level, [0,1], [comp_bin.min(), comp_bin.max()])
                    dist_sdf = comp_bin

                success, message = save_sdfobj_dialog(sdfexp, dist_sdf, color_sdf, export_z_up, elvl, exp_use_color)
                export_obj_message = [success, message]
                export_obj_message_time = time.time()

                show_export_obj_window = False

            imgui.end()


        if show_about_window:
            imgui.set_next_window_position(width // 2 - 250, height // 2 - 200)
            imgui.set_next_window_size(500, 400)  # Increased height
            is_open, show_about_window = imgui.begin("About", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                show_about_window = False
            
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

            # --- Github project page URL ---
            import webbrowser
            
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
                show_about_window = False

            imgui.end()

        # --- FPS OVERLAY (Top Right, above right panel) ---
        fps_x = width - panel_width - FPS_WINDOW_WIDTH - FPS_WINDOW_OFFSET
        imgui.set_next_window_position(fps_x, FPS_WINDOW_OFFSET)
        imgui.set_next_window_size(FPS_WINDOW_WIDTH, FPS_WINDOW_HEIGHT)
        imgui.begin("FPS", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        if shader_choice == 0:
            imgui.text_colored("FPS: " + str(fps_value), 0.0, 1.0, 0.0, 1.0)
        elif shader_choice == 1:
            imgui.text_colored("Sample: " + str(frame_count), 1.0, 1.0, 0.0, 1.0)

        imgui.end()

        # Orientation Overlay
        ori_x = width - panel_width - ORI_WINDOW_WIDTH - ORI_WINDOW_OFFSET
        imgui.set_next_window_position(fps_x+70, ORI_WINDOW_OFFSET)
        imgui.set_next_window_size(ORI_WINDOW_WIDTH, ORI_WINDOW_HEIGHT)
        imgui.begin("ORI", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        
        imgui.same_line(17,0) # At Center
        imgui.text_colored("VIEW", 0.8,0.8,1.0)
        imgui.spacing()
        if imgui.small_button("X##Ori"):
            target_yaw = 0.0
            target_pitch = 0.0
        imgui.same_line()
        if imgui.small_button("-X##Ori"):
            target_yaw = 3.14
            target_pitch = 0.0
        imgui.spacing()
        if imgui.small_button("Y##Ori"):
            target_pitch = 1.57
        imgui.same_line()
        if imgui.small_button("-Y##Ori"):
            target_pitch = -1.57     
        imgui.spacing()
        if imgui.small_button("Z##Ori"):
            target_yaw = 1.57
            target_pitch = 0.0
        imgui.same_line()
        if imgui.small_button("-Z##Ori"):
            target_yaw = -1.57 
            target_pitch = 0.0    

        imgui.end()
        

        if show_exit_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 65)
            imgui.set_next_window_size(300, 130)  # Increased height
            is_open, show_exit_window = imgui.begin("Confirm Exit", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                show_exit_window = False
            
            imgui.spacing()
            imgui.text(f"Are you sure you want to exit?\nUnsaved data may be lost.")
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui.button("Cancel", 130,30):
                show_exit_window = False
            imgui.same_line(0,15)
            if imgui.button("YES", 130,30):
                # Save Data
                config = {"Theme": theme}
                save_user_config("UserData/User.data", config)


                glfw.set_window_should_close(window, True)

            imgui.end()

        if show_restart_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 65)
            imgui.set_next_window_size(300, 130)  # Increased height
            is_open, show_restart_window = imgui.begin("Confirm Restart", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                show_restart_window = False
            
            imgui.spacing()
            imgui.text(f"Are you sure you want to restart the app?\nThis may result in loss of unsaved data.")
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui.button("Cancel", 130,30):
                show_restart_window = False
            imgui.same_line(0,15)
            if imgui.button("YES", 130,30):
                # Save Data
                config = {"Theme": theme}
                save_user_config("UserData/User.data", config)

                import sys
                import subprocess
                if getattr(sys, 'frozen', False):
                    subprocess.Popen([sys.executable])
                else:
                    subprocess.Popen([sys.executable] + sys.argv)
                exit()

            imgui.end()       



        # Display save/load status message
        import pyperclip

        if save_load_message is not None:
            # Show message for 3 seconds
            if time.time() - save_load_message_time < 3.0:
                imgui.set_next_window_position(width // 2 - 150, 100)
                imgui.begin("Status", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)

                # Color based on success
                is_success = "saved" in save_load_message.lower() or "loaded" in save_load_message.lower()
                color = (0.0, 1.0, 0.0, 1.0) if is_success else (1.0, 0.0, 0.0, 1.0)
                imgui.text_colored(save_load_message, *color)

                imgui.same_line(350, 0)

                if imgui.button("copy"):
                    pyperclip.copy(save_load_message)

                imgui.end()
            else:
                save_load_message = None


        if export_obj_message is not None:
            # Show message for 3 seconds
            if time.time() - export_obj_message_time < 3.0:
                imgui.set_next_window_position(width // 2 - 150, 100)
                imgui.begin("Status", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)

                # Color based on success
                is_success = export_obj_message[0]
                color = (0.0, 1.0, 0.0, 1.0) if is_success else (1.0, 0.0, 0.0, 1.0)
                imgui.text_colored(export_obj_message[1], *color)
                
                imgui.same_line(350, 0)

                if imgui.button("copy"):
                    pyperclip.copy(save_load_message)
                
                imgui.end()
            else:
                export_obj_message[1] = None



        # --- Error Display (if shader compilation failed) ---
        if shader_compile_error:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 50)
            imgui.set_next_window_size(400, 100)
            imgui.begin("Shader Compilation Error", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            imgui.text_colored("Error:", 1.0, 0.0, 0.0, 1.0)
            imgui.same_line()
            imgui.text_wrapped(shader_compile_error)
            if imgui.button("Dismiss"):
                shader_compile_error = None
            imgui.end()
        

        # --- LEFT PANEL: Scene Tree (HIERARCHICAL) ---
        # Offset panels below menu bar
        imgui.set_next_window_position(0, menu_bar_height)
        imgui.set_next_window_size(panel_width, height - menu_bar_height)
        imgui.begin("Scene Tree", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        
        def format_label(name, op_id, max_chars=16):
            """Format label text with truncation."""
            if len(name) > max_chars:
                truncated_name = name[:max_chars - 3] + "..."
            else:
                truncated_name = name
            return f"{truncated_name} ({op_id})"
    

        def render_node_recursive(node_id, depth=0):
            """
            Recursively render a scene node and its children.
            
            Adds:
            - right-click context popup per-node to Add a child primitive or child operation
                (only for operation nodes that still accept operands).
            """
            nonlocal pending_change_node_id, show_add_change_window, property_change_node_id, show_property_change_window, show_reparent_window, reparent_node_id

            node = scene_builder.get_node(node_id)
            if not node:
                return

            item_data = node.item_data
            children = node.children
            is_leaf = len(children) == 0

            # Format the display label
            label = format_label(item_data.ui_name, node_id)

            # Determine tree node flags
            flags = 0
            if not is_leaf:
                flags |= imgui.TREE_NODE_DEFAULT_OPEN  # Open branches by default
            else:
                flags |= imgui.TREE_NODE_LEAF  # Leaves have no expand arrow

            global selected_item_id
            if selected_item_id == node_id:
                flags |= imgui.TREE_NODE_SELECTED

            # Movement controls for root-level nodes
            if node.parent_id is None and node_id in scene_builder.root_children:
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (1, 1))
                root_idx = scene_builder.root_children.index(node_id)

                # Up arrow button
                if imgui.arrow_button(f"##up_{node_id}", 2):  # 2 = up arrow
                    if root_idx > 0:
                        scene_builder.move_root_node(node_id, root_idx - 1)
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms

                imgui.same_line()

                # Down arrow button
                if imgui.arrow_button(f"##down_{node_id}", 3):  # 3 = down arrow
                    if root_idx < len(scene_builder.root_children) - 1:
                        scene_builder.move_root_node(node_id, root_idx + 1)
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms

                imgui.pop_style_var(1)
                imgui.same_line()

            # Delete button (balanced push/pop)
            imgui.push_id(f"delete_{node_id}")
            clicked_delete = imgui.button("X", 20, 20)
            imgui.pop_id()

            if clicked_delete:
                scene_builder.delete_node(node_id)
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms
                return

            imgui.same_line()

            # Render tree node
            node_open = imgui.tree_node(label, flags)

            # Handle left-click selection
            if imgui.is_item_clicked():
                io_local = imgui.get_io()
                if io_local.key_ctrl:
                    if node_id in selected_items:
                        selected_items.remove(node_id)
                    else:
                        selected_items.add(node_id)
                    if len(selected_items) > 0:
                        selected_item_id = None
                else:
                    selected_items.clear()
                    selected_item_id = node_id
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = 'node'
                    renaming_item_id = None
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

            # Right-click context menu: open popup when right-clicking the tree item
            popup_id = f"node_ctx_{node_id}"
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
                imgui.open_popup(popup_id)

            # Begin popup (if opened)
            if imgui.begin_popup(popup_id):
                # Only offer add options for operation nodes
                if node.node_type == 'operation':
                    if imgui.menu_item("Change Operation Type")[0]:
                        pending_change_node_id = node_id
                        show_add_change_window = True
                        imgui.close_current_popup()
                else:
                    # For primitives, offer Change Type (in-place) rather than forcing delete+create
                    if imgui.menu_item("Change Type")[0]:
                        pending_change_node_id = node_id
                        show_add_change_window = True
                        imgui.close_current_popup()
                
                imgui.separator()
                
                if imgui.menu_item("Change Properties")[0]:
                    property_change_node_id = node_id
                    show_property_change_window = True
                    imgui.close_current_popup()
                
                # NEW: Reparent option
                if imgui.menu_item("Reparent")[0]:
                    reparent_node_id = node_id
                    show_reparent_window = True
                    imgui.close_current_popup()
                
                imgui.end_popup()

            # Render children recursively
            if children:
                for child_id in list(children):  # list() to be safe if children mutate
                    render_node_recursive(child_id, depth + 1)

            if node_open:
                imgui.tree_pop()

            
        # ====== RENDER ALL ROOT NODES ======
        imgui.text("Scene Hierarchy:")
        imgui.separator()
        
        for root_id in scene_builder.root_children:
            render_node_recursive(root_id, depth=0)
        
        # ====== ADD BUTTONS ======
        imgui.spacing()
        imgui.separator()
        
        if imgui.button(f"Add (Ctrl+A)", -1):
            show_add_change_window = True
            pending_change_node_id = None
        
        imgui.end()  # End Scene Tree window
        


        # --- RIGHT PANEL: Properties/Inspector ---
        imgui.set_next_window_position(width - panel_width, menu_bar_height)
        imgui.set_next_window_size(panel_width, height - menu_bar_height)
        imgui.begin("Inspector", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        if selected_item_id is not None and selected_item_id in scene_builder.id_to_node:
            node = scene_builder.get_node(selected_item_id)
            if node:
                item_data = node.item_data
                
                # Display node type
                imgui.text(f"Type: {node.node_type}")
                if node.node_type == 'operation':
                    imgui.text(f"Operation: {item_data.operation_type}")
                else:
                    imgui.text(f"Primitive: {item_data.primitive_type}")
                
                imgui.separator()
                imgui.text(f"Selected: {item_data.ui_name}")
                
                # Rename functionality
                if imgui.button("Rename"):
                    renaming_item_id = selected_item_id
                    rename_text = item_data.ui_name
                
                if renaming_item_id == selected_item_id:
                    changed, rename_text = imgui.input_text("##rename", rename_text, 256)
                    
                    if imgui.button("OK", width / 5):
                        scene_builder.rename_node(selected_item_id, rename_text)
                        renaming_item_id = None
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
                    
                    imgui.same_line()
                    if imgui.button("Cancel", width / 5):
                        renaming_item_id = None
                
                imgui.separator()
                
                # Show node-specific properties
                if node.node_type == 'primitive':
                    # Primitive properties




                    # =======================
                    primitive_type = node.item_data.primitive_type
                    primitive = node.item_data

 
                    if primitive_type == "sprite":
                        # sprite_index is stored in primitive.kwargs at creation time
                        sprite_idx = primitive.kwargs.get('sprite_index', None)
                        if sprite_idx is None or sprite_idx >= len(sprites_array):
                            imgui.text_colored("Sprite data missing or corrupted", 1.0, 0.0, 0.0, 1.0)
                        else:
                            spr = sprites_array[sprite_idx]
                            imgui.text("Plane parameters:")
                            changed, primitive.position = input_vec3("Point", primitive.position, STEP_VARIABLE_FLOAT, panel_elem_width_vec3)
                            changed2, spr.planeNormal = input_vec3("Normal", spr.planeNormal, STEP_VARIABLE_FLOAT, panel_elem_width_vec3)
                            changed3, spr.planeWidth = input_float("Width", spr.planeWidth, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                            changed4, spr.planeHeight = input_float("Height", spr.planeHeight, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                            spr.planePoint = primitive.position
                            if changed or changed2 or changed3 or changed4:
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            imgui.separator()
                            imgui.text("Mapping:")
                            uv2 = spr.uvSize
                            changed_uv, uv2 = input_vec2("UV Size", uv2, 0.1, panel_elem_width_vec3)
                            spr.uvSize[0], spr.uvSize[1] = uv2[0], uv2[1]
                            changed_alpha, spr.Alpha = input_float("Alpha", spr.Alpha, 0.01, panel_elem_width_float)
                            changed_lod, spr.LOD = input_float("LOD", spr.LOD, 0.1, panel_elem_width_float)

                            if changed_uv or changed_alpha or changed_lod:
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            # Show texture status and "Load Texture" button
                            if spr.texture_id:
                                imgui.text(f"Texture loaded: {spr.tex_size[0]}x{spr.tex_size[1]}")
                            else:
                                imgui.text_colored("No texture loaded", 0.9, 0.3, 0.3, 1.0)

                            imgui.spacing()
                            if imgui.button("Load Texture", -1):
                                # Use tkinter filedialog (as in other parts of the code)
                                root = tk.Tk()
                                root.withdraw()
                                filetypes = [("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tga")), ("All files", "*.*")]
                                filepath = filedialog.askopenfilename(filetypes=filetypes)
                                root.destroy()
                                if filepath:
                                    ok = spr.load_texture_from_file(filepath)
                                    if ok:
                                        # Ensure sampler name is unique and recompile so the sampler uniform is declared/located
                                        spr.SprTexture = f"sprTex{sprite_idx}"
                                        success, new_uniforms = recompile_shader()
                                        if success:
                                            uniform_locs = new_uniforms


                    primitive.size_or_radius = list(primitive.size_or_radius) if isinstance(primitive.size_or_radius, tuple) else primitive.size_or_radius 
                    primitive.size_or_radius = [primitive.size_or_radius] if isinstance(primitive.size_or_radius, float) else primitive.size_or_radius 

                    # Size/Radius - varies by primitive type
                    match primitive.primitive_type:    
                        case "sphere":
                            changed, primitive.size_or_radius[0] = input_float(
                                "Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                        case "torus":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Major Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Minor Radius", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        case "hex_prism":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Hex Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Height", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        case "vertical_capsule":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Height", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Radius", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        case "capped_cylinder":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Height", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        case "rounded_cylinder":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Radius A", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Radius B", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        # Special parameters for specific primitives
                        case "cone":
                            c_sin = primitive.kwargs.get('c_sin', 0.5)
                            c_cos = primitive.kwargs.get('c_cos', 0.866)
                            height = primitive.kwargs.get('height', 1.0)
                            changed1, c_sin = input_float(
                                "Sin(Angle)", c_sin, STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, c_cos = input_float(
                                "Cos(Angle)", c_cos, STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed3, height = input_float(
                                "Height", height, STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            if changed1 or changed2 or changed3:
                                primitive.kwargs['c_sin'] = c_sin
                                primitive.kwargs['c_cos'] = c_cos
                                primitive.kwargs['height'] = height
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        case "plane":
                            normal = primitive.kwargs.get('normal', [0.0, 1.0, 0.0])
                            h = primitive.kwargs.get('h', 0.0)
                            changed1, normal = input_vec3("Normal", normal, STEP_VARIABLE_FLOAT, panel_elem_width_vec3)
                            changed2, h = input_float("Offset (h)", h, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                            if changed1 or changed2:
                                # Normalize the normal vector
                                norm_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
                                if norm_len > 0.001:
                                    normal = [normal[0]/norm_len, normal[1]/norm_len, normal[2]/norm_len]
                                primitive.kwargs['normal'] = normal
                                primitive.kwargs['h'] = h
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        case "rounded_cylinder":
                            height = primitive.kwargs.get('height', 1.0)
                            changed, height = input_float("Height", height, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                            if changed:
                                primitive.kwargs['height'] = height
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # --- Inspector: add UI to edit pointer function selection (inside the primitive inspector branch) ---
                        case "pointer":
                            changed_pos, primitive.position = input_vec3(
                                "Position", primitive.position, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                            )
                            if changed_pos:
                                scene_builder.modify_primitive_property(node.item_id, 'position', primitive.position)
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            # List of available pointer functions (must exist in sdf_library.glsl)
                            pointer_funcs = [
                                "pointer_identity",
                                "pointer_symmetry_x",
                                "pointer_symmetry_y",
                                "pointer_symmetry_z",
                                # add your custom pointer function names here...
                            ]
                            current_func = primitive.kwargs.get('func', 'pointer_identity')
                            try:
                                current_index = pointer_funcs.index(current_func)
                            except ValueError:
                                pointer_funcs.append(current_func)
                                current_index = len(pointer_funcs)-1

                            clicked, new_index = imgui.combo("Function", current_index, pointer_funcs)
                            if clicked:
                                new_func = pointer_funcs[new_index]
                                primitive.kwargs['func'] = new_func
                                # Record change in history for undo/redo
                                scene_builder.modify_primitive_property(node.item_id, "kwargs.func", new_func)
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            imgui.separator()
                            imgui.text("Pointer functions mutate \nthe raymarch point `p` \nfor subsequent primitives.")
                            imgui.text_colored("Place a pointer earlier in \nthe tree to affect later objects.", 0.9, 0.8, 0.2, 1.0)

                        case "sprite":
                            pass # Skip Transforms and Color

                        case "curve":
                            imgui.spacing()
                            
                            # Points array editor
                            points = primitive.kwargs.get('points', [[0, 0, 0], [1, 1, 1]])
                            imgui.text("Curve Points:")
                            
                            points_to_remove = None
                            for i, pt in enumerate(points):
                                changed, new_pt = input_vec3(
                                    f"Point {i}", list(pt), STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                                )
                                if changed:
                                    points[i] = new_pt
                                    primitive.kwargs['points'] = points
                                    success, new_uniforms = recompile_shader()
                                    if success:
                                        uniform_locs = new_uniforms
                                
                                imgui.same_line()
                                if imgui.button(f"Remove##pt{i}", width=60):
                                    points_to_remove = i
                            
                            if points_to_remove is not None and len(points) > 2:
                                points.pop(points_to_remove)
                                primitive.kwargs['points'] = points
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                            
                            if imgui.button("Add Point", width=panel_elem_width_float):
                                points.append([0.0, 0.0, 0.0])
                                primitive.kwargs['points'] = points
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                            
                            imgui.spacing()
                            
                            # Thickness parameter
                            thickness = primitive.kwargs.get('thickness', 0.1)
                            changed, thickness = input_float(
                                "Thickness", thickness, STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            if changed:
                                primitive.kwargs['thickness'] = thickness
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        case _:
                            if primitive.primitive_type not in ["cone", "plane", "rounded_cylinder", "pointer", "sprite", "curve"]:
                                changed, primitive.size_or_radius = input_vec3(
                                    "Size", primitive.size_or_radius, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                                )
                        
                    if primitive.primitive_type not in ["pointer", "sprite", "curve"]:
                        if changed:
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                        


                        else:
                            # Special parameters for specific primitives
                            if primitive.primitive_type == "round_box":
                                imgui.spacing()
                                changed, primitive.kwargs['radius'] = input_float(
                                    "Radius", primitive.kwargs.get('radius', 0.1),STEP_VARIABLE_FLOAT, panel_elem_width_float
                                    )
                                if changed:
                                    success, new_uniforms = recompile_shader()
                                    if success:
                                        uniform_locs = new_uniforms
                            

                        imgui.begin_group()

                        imgui.spacing()
                        imgui.separator()
                        imgui.dummy((panel_width/4)-8, 0)
                        imgui.same_line()
                        imgui.text_colored("Transform", 1.0,0.7,0.5,1.0)
                        imgui.spacing()
            
                        imgui.end_group()


                    # ==============================================================


                    changed, item_data.position = input_vec3(
                        "Position",
                        item_data.position,
                        STEP_VARIABLE_FLOAT,
                        panel_elem_width_vec3
                    )
                    if changed:
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
                    
                    changed, item_data.rotation = input_vec3(
                        "Rotation",
                        item_data.rotation,
                        STEP_VARIABLE_FLOAT,
                        panel_elem_width_vec3
                    )
                    if changed:
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
                    
                    changed, item_data.scale = input_vec3(
                        "Scale",
                        item_data.scale,
                        STEP_VARIABLE_FLOAT,
                        panel_elem_width_vec3
                    )
                    if changed:
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
                    
                    changed, item_data.color = input_vec3(
                        "Color",
                        item_data.color,
                        STEP_VARIABLE_FLOAT,
                        panel_elem_width_vec3
                    )
                    if changed:
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
    


                    # Color picker
                    imgui.begin_group()

                    imgui.spacing()
                    imgui.separator()
                    imgui.dummy((panel_width/3)-12, 0)
                    imgui.same_line()
                    imgui.text_colored("Color", 1.0,0.7,0.5,1.0)
                    imgui.spacing()
        
                    imgui.end_group()

                    # Color edit - imgui automatically shows a picker button
                    color_changed, color_rgba = imgui.color_edit3("Color##color", *primitive.color)
                    if color_changed:
                        primitive.color = list(color_rgba[: 3])
                        scene_builder.modify_primitive_property(node.item_id, 'color', primitive.color)
                        success, new_uniforms = recompile_shader()
                        if success: 
                            uniform_locs = new_uniforms
                    
                    # Alternative: RGB sliders for fine control
                    imgui.spacing()
                    imgui.text("RGB Sliders:")
                    r_changed, primitive.color[0] = imgui.slider_float("R##color_r", primitive.color[0], 0.0, 1.0)
                    g_changed, primitive.color[1] = imgui.slider_float("G##color_g", primitive.color[1], 0.0, 1.0)
                    b_changed, primitive.color[2] = imgui.slider_float("B##color_b", primitive.color[2], 0.0, 1.0)
                    if r_changed or g_changed or b_changed:
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms




                elif node.node_type == 'operation':
                    # Operation properties
                    imgui.text(f"Operation Type: {item_data.operation_type}")
                    
                    # Show operands
                    imgui.text("Operands:")
                    for i, operand_id in enumerate(node.children):
                        operand_node = scene_builder.get_node(operand_id)
                        if operand_node:
                            imgui.text(f"  {i+1}. {operand_node.item_data.ui_name} ({operand_id})")
                    
                    # Show smooth_k if applicable
                    if hasattr(item_data, 'smooth_k') and item_data.smooth_k is not None:
                        changed, new_k = imgui.slider_float(
                            "Smooth K",
                            item_data.smooth_k,
                            0.0,
                            1.0
                        )
                        if changed:
                            item_data.smooth_k = new_k
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
        
        else:
            imgui.text("No node selected")
            imgui.text("Click on a node in the Scene Tree")
        
        imgui.end()
        
        # --- OPERATION/PRIMITIVE SELECTION DIALOG (HIERARCHICAL) ---
        if show_operation_selection_window:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 200)
            imgui.set_next_window_size(400, 400)
            
            is_open, show_operation_selection_window = imgui.begin(
                "Add Operation",
                True,
                imgui.WINDOW_NO_COLLAPSE
            )
            
            if not is_open:
                show_operation_selection_window = False
            
            imgui.text("Select Operation Type:")
            imgui.separator()
            
            # Define available operations with their properties
            operations_list = [
                ("Union", "union", 2, "Combines two shapes (minimum distance)"),
                ("Subtraction", "sub", 2, "Subtracts second from first"),
                ("Intersection", "inter", 2, "Keeps only overlapping parts"),
                ("Smooth Union", "sunion", 2, "Union with smooth blending"),
                ("Smooth Subtraction", "ssub", 2, "Subtraction with smooth blending"),
                ("Smooth Intersection", "sinter", 2, "Intersection with smooth blending"),
                ("Mix", "mix", 2, "Blends between two distances"),
                ("Invert", "invert", 1, "Inverts the shape"),
                ("Round", "round", 1, "Rounds the shape"),
                ("Onion", "onion", 1, "Creates a shell effect"),
                ("XOR", "xor", 2, "Exclusive OR operation"),
                ("snoiseDisp", "snoiseDisp", 1, "Noise displacement"),
            ]
            
            # Let user choose what primitive to auto-create
            imgui.text("Auto-create Primitives:")
            auto_prim_options = [
                "Box", "Sphere", "Torus", "Cone", "Hex Prism",
                "Vertical Capsule", "Capped Cylinder", "Rounded Cylinder"
            ]
            auto_prim_type_index = 1  # Default to Sphere
            clicked, auto_prim_type_index = imgui.combo(
                "##auto_prim_type",
                auto_prim_type_index,
                auto_prim_options
            )
            
            auto_prim_map = {
                0: 'box', 1: 'sphere', 2: 'torus', 3: 'cone',
                4: 'hex_prism', 5: 'vertical_capsule',
                6: 'capped_cylinder', 7: 'rounded_cylinder'
            }
            selected_auto_prim = auto_prim_map[auto_prim_type_index]
            
            imgui.separator()
            
            # Display operations
            for label, op_type, operand_count, description in operations_list:
                if imgui.button(f"{label} ({operand_count} operands)", -1):
                    # Create operation with auto-generated primitives
                    new_op_id = scene_builder.add_operation_with_auto_primitives(
                        op_type,
                        auto_primitive_type=selected_auto_prim,
                        ui_name=label
                    )
                    
                    # Recompile shader
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    
                    # Select the new operation
                    selected_item_id = new_op_id
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = 'node'
                    show_operation_selection_window = False
                
                if imgui.is_item_hovered():
                    imgui.set_tooltip(description)
            
            imgui.end()
        
        # --- ADD PRIMITIVE DIALOG ---
        if show_primitive_selection_window:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 150)
            imgui.set_next_window_size(400, 400)
            
            is_open, show_primitive_selection_window = imgui.begin(
                "Add Standalone Primitive",
                True,
                imgui.WINDOW_NO_COLLAPSE
            )
            
            if not is_open:
                show_primitive_selection_window = False
            
            imgui.text("Select Primitive Type:")
            imgui.text("(These are root-level, not part of an operation)")
            imgui.separator()
            
            primitives_list = [
                ("Box", "box", (0.5, 0.5, 0.5)),
                ("Sphere", "sphere", 0.5),
                ("Round Box", "round_box", (0.5, 0.5, 0.5)),
                ("Torus", "torus", (0.5, 0.25)),
                ("Cone", "cone", None),
                ("Plane", "plane", None),
                ("Hex Prism", "hex_prism", (0.5, 0.5)),
                ("Vertical Capsule", "vertical_capsule", (1.0, 0.3)),
                ("Capped Cylinder", "capped_cylinder", (0.3, 1.0)),
                ("Rounded Cylinder", "rounded_cylinder", (0.3, 0.3)),
            ]
            
            for label, prim_type, size_or_radius in primitives_list:
                if imgui.button(f"{label}", -1):
                    # Create standalone primitive at origin
                    new_prim_id = scene_builder.add_standalone_primitive(
                        prim_type,
                        position=[0.0, 0.0, 0.0],
                        size_or_radius=size_or_radius,
                        ui_name=label
                    )
                    
                    # Recompile shader
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    
                    # Select new primitive
                    selected_item_id = new_prim_id
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = 'node'
                    show_primitive_selection_window = False
            

            imgui.end()


        # Combined Add/Change Window - Two columns (left = primitives, right = operations)
        if show_add_change_window:
            imgui.set_next_window_position(width // 2 - 300, height // 2 - 235)
            imgui.set_next_window_size(600, 470)
            is_open, show_add_change_window = imgui.begin("Add / Change Type", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                show_add_change_window = False
                pending_change_node_id = None

            # Define lists (same primitives_list and operations_list)
            primitives_list = [
                ("Box", "box", (0.5, 0.5, 0.5)),
                ("Sphere", "sphere", 0.5),
                ("Round Box", "round_box", (0.5, 0.5, 0.5)),
                ("Torus", "torus", (0.5, 0.25)),
                ("Cone", "cone", None),
                ("Plane", "plane", None),
                ("Hex Prism", "hex_prism", (0.5, 0.5)),
                ("Vertical Capsule", "vertical_capsule", (1.0, 0.3)),
                ("Capped Cylinder", "capped_cylinder", (0.3, 1.0)),
                ("Rounded Cylinder", "rounded_cylinder", (0.3, 0.3)),
            ]

            operations_list = [
                ("Union", "union", 2, "Combines two shapes (minimum distance)"),
                ("Subtraction", "sub", 2, "Subtracts second from first"),
                ("Intersection", "inter", 2, "Keeps only overlapping parts"),
                ("Smooth Union", "sunion", 2, "Union with smooth blending"),
                ("Smooth Subtraction", "ssub", 2, "Subtraction with smooth blending"),
                ("Smooth Intersection", "sinter", 2, "Intersection with smooth blending"),
                ("Mix", "mix", 2, "Blends between two distances"),
                ("Invert", "invert", 1, "Inverts the shape"),
                ("Round", "round", 1, "Rounds the shape"),
                ("Onion", "onion", 1, "Creates a shell effect"),
                ("XOR", "xor", 2, "Exclusive OR operation"),
                ("snoiseDisp", "snoiseDisp", 1, "Noise displacement"),
            ]

            # Layout: two columns
            imgui.columns(2, "add_change_cols", border=True)
            imgui.set_column_width(0, 290)  # primitives column
            imgui.text("Primitives")
            imgui.separator()

            for label, prim_type, size in primitives_list:
                if imgui.button(label, -1, 24):
                    # If pending_change_node_id is None -> ADD, else -> CHANGE TYPE
                    if pending_change_node_id is None:
                        # Add new primitive at origin with defaults
                        new_id = scene_builder.add_standalone_primitive(
                            prim_type,
                            position=[0.0, 0.0, 0.0],
                            size_or_radius=size if size is not None else 0.5,
                            ui_name=label
                        )
                        if new_id:
                            selected_items.clear()
                            selected_item_id = new_id
                            scene_builder.update_selected_item_id(selected_item_id)
                            selection_mode = 'node'
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                    else:
                        # Change the pending node to this primitive (in-place)
                        node = scene_builder.get_node(pending_change_node_id)
                        if node:
                            # If it was operation -> convert to primitive
                            scene_builder.change_node_to_primitive(pending_change_node_id, prim_type, position=None, size_or_radius=(size if size is not None else 0.5))
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                        # Clear pending state
                        pending_change_node_id = None
                        show_add_change_window = False
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Add / Change to {label}")
            

            imgui.next_column()
            imgui.text("Operations")
            imgui.separator()

            for label, op_type, operand_count, description in operations_list:
                if imgui.button(label, -1, 24):
                    if pending_change_node_id is None:
                        # Add new operation (auto-create primitives)
                        new_op_id = scene_builder.add_operation_with_auto_primitives(
                            op_type,
                            auto_primitive_type='box',
                            ui_name=label
                        )
                        if new_op_id:
                            selected_items.clear()
                            selected_item_id = new_op_id
                            scene_builder.update_selected_item_id(selected_item_id)
                            selection_mode = 'node'
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                    else:
                        # Convert pending node to this operation type (in-place)
                        scene_builder.change_node_to_operation(pending_change_node_id, op_type, auto_primitive_type='box')
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms

                        pending_change_node_id = None
                        show_add_change_window = False

                if imgui.is_item_hovered():
                    imgui.set_tooltip(description)

            imgui.columns(1)
            imgui.separator()
            imgui.spacing()
            imgui.same_line(20,0)
            if imgui.button("Cancel", 265, 28):
                show_add_change_window = False
                pending_change_node_id = None

            imgui.same_line(305,0)
            if imgui.button("Close", 265, 28):
                show_add_change_window = False
                pending_change_node_id = None

            imgui.end()


        if show_property_change_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 125)
            imgui.set_next_window_size(300, 250)
            is_open, show_property_change_window = imgui.begin("Change Properties", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                show_property_change_window = False
                property_change_node_id = None


            node = scene_builder.get_node(property_change_node_id)
            prim = node.item_data
            sym = prim.properties.get("symmetry")
            if sym is None:
                sym = [False, False, False]
                prim.update_property("symmetry", sym)

            imgui.spacing()
            imgui.text("Symmetry:")
            imgui.same_line()
            changed_x, sym[0] = imgui.checkbox("X", sym[0])
            imgui.same_line()
            changed_y, sym[1] = imgui.checkbox("Y", sym[1])
            imgui.same_line()
            changed_z, sym[2] = imgui.checkbox("Z", sym[2])
            imgui.spacing()

            if changed_x or changed_y or changed_z:
                prim.update_property("symmetry", sym)
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            if imgui.button("Close", -1):
                show_property_change_window = False
                property_change_node_id = None

            imgui.end()


        def _render_reparent_node_list(node_id, exclude_node_id, exclude_descendants, indent=""):
            """
            Recursively render selectable nodes for reparent window.
            Only show operation nodes (valid parents).
            Returns True if a node was selected.
            """
            global reparent_target_parent, reparent_child_to_replace

            if node_id in exclude_descendants or node_id == exclude_node_id:
                return False

            node = scene_builder.get_node(node_id)
            if not node or node.node_type != 'operation':
                return False

            label = f"{indent}{node.item_data.ui_name} ({node_id})"
            # NOTE: the second arg is the 'selected' boolean. We pass False for predictable behaviour.
            clicked, _ = imgui.selectable(label, False)
            if clicked:
                reparent_target_parent = node_id
                return True

            result = imgui.is_item_clicked()

            # Recursively render children (only operations)
            for child_id in node.children:
                child_node = scene_builder.get_node(child_id)
                if child_node and child_node.node_type == 'operation':
                    result |= _render_reparent_node_list(
                        child_id,
                        exclude_node_id,
                        exclude_descendants,
                        indent + "  "
                    )

            return result


        # === REPARENT WINDOW ===
        if show_reparent_window and reparent_node_id:
            imgui.set_next_window_size(400, 500)
            show_reparent_window, _ = imgui.begin("Reparent Node", True)
            
            if show_reparent_window:
                reparent_node = scene_builder.get_node(reparent_node_id)
                if reparent_node:
                    imgui.text(f"Reparenting: {reparent_node.item_data.ui_name}")
                    imgui.separator()
                    imgui.text("Select new parent operation:")
                    
                    # List all operation nodes (excluding the node being reparented and its descendants)
                    all_descendants = scene_builder.get_all_children_recursive(reparent_node_id)
                    all_descendants.append(reparent_node_id)
                    
                    parent_selected = False
                    for root_id in scene_builder.root_children:
                        parent_selected |= _render_reparent_node_list(
                            root_id, 
                            reparent_node_id, 
                            all_descendants,
                            "  "
                        )
                    
                    if parent_selected and reparent_target_parent:
                        new_parent_node = scene_builder.get_node(reparent_target_parent)
                        if new_parent_node and new_parent_node.node_type == 'operation':
                            required_operands = scene_builder._get_operand_count(new_parent_node.item_data.operation_type)
                            current_operands = len(new_parent_node.children)
                            
                            if current_operands >= required_operands:
                                imgui.separator()
                                imgui.text(f"Parent is full ({current_operands}/{required_operands} operands)")
                                imgui.text("Select child to replace:")
                                
                                for i, child_id in enumerate(new_parent_node.children):
                                    child_node = scene_builder.get_node(child_id)
                                    if child_node:
                                        if imgui.selectable(
                                            f"{child_node.item_data.ui_name} ({child_id})",
                                            reparent_child_to_replace == child_id
                                        )[0]:
                                            reparent_child_to_replace = child_id
                    
                    imgui.spacing()
                    imgui.separator()
                    
                    if imgui.button("Cancel", 100, 30):
                        show_reparent_window = False
                        reparent_node_id = None
                        reparent_target_parent = None
                        reparent_child_to_replace = None
                    
                    imgui.same_line(150)
                    
                    can_reparent = reparent_target_parent is not None
                    if reparent_target_parent:
                        new_parent_node = scene_builder.get_node(reparent_target_parent)
                        required_operands = scene_builder._get_operand_count(new_parent_node.item_data.operation_type)
                        current_operands = len(new_parent_node.children)
                        if current_operands >= required_operands and reparent_child_to_replace is None:
                            can_reparent = False
                    
                    if not can_reparent:
                        pass
                    
                    if imgui.button("Reparent", 100, 30):
                        if scene_builder.reparent_node(reparent_node_id, reparent_target_parent, reparent_child_to_replace):
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                        show_reparent_window = False
                        reparent_node_id = None
                        reparent_target_parent = None
                        reparent_child_to_replace = None
                    
                    if not can_reparent:
                        pass
            
            imgui.end()


        # Render ImGui
        imgui.render()
        impl.render(imgui.get_draw_data())

        # Swap front and back buffers
        glfw.swap_buffers(window)

    for i in range(2):
        if accumulation_fbos[i] is not None:
            try:
                glDeleteFramebuffers(1, [accumulation_fbos[i]])
            except Exception:
                pass
            accumulation_fbos[i] = None
        if accumulation_textures[i] is not None:
            try:
                glDeleteTextures(1, [accumulation_textures[i]])
            except Exception:
                pass
            accumulation_textures[i] = None

    # Clean up
    # Delete all cached shaders
    for cached_shader, _ in shader_cache.values():
        if cached_shader is not None:
            glDeleteProgram(cached_shader)
    shader_cache.clear()
    
    # Clean up framebuffer
    if fbo is not None:
        glDeleteFramebuffers(1, [fbo])
    if render_texture is not None:
        glDeleteTextures(1, [render_texture])
    
    # Clean up display shader
    if display_shader is not None:
        glDeleteProgram(display_shader)
    if display_vao is not None:
        glDeleteVertexArrays(1, [display_vao])
    if display_vbo is not None:
        glDeleteBuffers(1, [display_vbo])
    
    impl.shutdown()
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glfw.terminate()




if __name__ == "__main__":
    main()