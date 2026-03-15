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

from src.app.data.consts import cn
from src.app.data.states import st

import src.ui.themes as ui_themes


# Load shaders
vertex_shader, fragment_shader_template, sdf_library = load_shaders()

# Create shader manager
shader_manager = ShaderManager(
    vertex_shader_src=vertex_shader,
    sdf_library_src=sdf_library,
    state=st
)

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
    # Initialize GLFW & Imgui
    window, impl = init_glfw_impl(cn['SCREEN_SIZE'])
    ICONS = load_all_textures()

    camera = Camera()

    # --- Defined Palette ---
    st.theme = ui_themes.default_theme


    # --- Scene Definition ---
    scene_builder = SDFSceneBuilder(glob_history, st.selected_item_id)

    # --- Default Scene ---
    scene_builder.add_standalone_primitive(
        'box',
        position=[0, 0, 0],
        size_or_radius=[0.5,0.2,0.8],
        ui_name='Cube'
    )


    @MonitorChanges
    def recompile_shader():
        nonlocal shader, uniform_locs

        new_shader, new_uniforms = shader_manager.get_or_compile(scene_builder)

        if new_shader is None:
            return False, None

        # If shader changed, delete old one (unless it was cached)
        if shader is not None and shader != new_shader:
            old_hash = None
            for cached_hash, (cached_shader, _) in st.shader_cache.items():
                if cached_shader == shader:
                    old_hash = cached_hash
                    break

            if old_hash is None:
                glDeleteProgram(shader)

        shader = new_shader
        uniform_locs = new_uniforms
        return True, new_uniforms


    shader, uniform_locs = shader_manager.get_or_compile(scene_builder)
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

    def on_window_close(window):
        glfw.set_window_should_close(window, False)
        st.show_exit_window = True

    def restart():
        st.show_restart_window = True


    # --- Main Loop ---
    st.start_time = time.time()
    st.prev_time = time.time() 

    glfw.set_window_close_callback(window, on_window_close)



    # Load User Config
    # I use JSON format with data extension to avoid confusion with one extension
    default_uconfig = {"Theme": st.theme, "UIScale" : 1.0}
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
                # Update the dictionary key with the new list/tuple value
                setattr(ui_themes, label, st.theme[label])
                ui_themes.setup_theme()

    rebuild_imgui_fonts(impl, "assets/fonts/Roboto-Medium.ttf", 16.0)

    while not glfw.window_should_close(window):
        # calc Delta time 
        current_time = time.time()
        st.delta_time = current_time - st.prev_time
        st.prev_time = current_time

        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        ui_themes.setup_theme()



        # --- FPS calculation ---
        st.fps_frames += 1
        current_time = time.time()
        if current_time - st.fps_clock >= 1.0:
            st.fps_value = st.fps_frames
            st.fps_frames = 0
            st.fps_clock = current_time

        # --- Handle keyboard input ---
        io = get_io()

        # Return: Flag (Recompile shader), st.selected_item_id, st.selected_items
        handle = handler(window, io, scene_builder, glob_history, st.selected_item_id, st.selected_items)
        if handle[0]:
            success, new_uniforms = recompile_shader()
            if success:
                uniform_locs = new_uniforms
        st.selected_item_id = handle[1]
        st.selected_items = handle[2]


        # Increment frame counter only when using cycles shader
        if st.shader_choice == 1:   # cycles_fragment_shader.glsl
            st.frame_count = min(st.frame_count + 1, st.max_frames)
        else: 
            st.frame_count = 0  # Reset accumulation when switching shaders
        
        # Get window and rendering dimensions
        width, height = glfw.get_framebuffer_size(window)
        menu_bar_height = int(imgui.get_frame_height())
        panel_width = int(width * cn['PANEL_WIDTH_RATIO'])
        rendering_width = width - 2 * panel_width
        rendering_height = height - menu_bar_height
        panel_elem_width_vec3 = (panel_width/4)-14
        panel_elem_width_float = (panel_width/2)-14

        
        st.scaled_rendering_width = int(rendering_width * st.resolution_scale)
        st.scaled_rendering_height = int(rendering_height * st.resolution_scale)



        # Get the current window size
        width, height = glfw.get_framebuffer_size(window)
        # Get menu bar height (needed for calculations) - convert to int for glViewport
        menu_bar_height = int(imgui.get_frame_height())
        panel_width = int(width * cn['PANEL_WIDTH_RATIO'])
        rendering_width = width - 2 * panel_width
        rendering_height = height - menu_bar_height
        
        # Apply resolution scale
        st.scaled_rendering_width = int(rendering_width * st.resolution_scale)
        st.scaled_rendering_height = int(rendering_height * st.resolution_scale)


        # If we recompiled the shader, we will update the fbo
        global monitor
        if monitor == True and st.shader_choice == 1:
            monitor = False
            st.frame_count = 0
            clear_accumulation_fbos()
            st.current_accum_index = 0



        # Handle MMB press and release for st.camera control
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            shift_pressed = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
            
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

        if st.is_mmb_pressed or st.dragging or st.R_dragging:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        else:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)


        prev_cam_yaw = st.cam_yaw
        prev_cam_pitch = st.cam_pitch
        prev_cam_radius = st.cam_radius
        prev_cam_orbit = st.cam_orbit


        # Handle mouse wheel input for st.camera zoom
        if io.mouse_wheel != 0:
            st.target_radius -= io.mouse_wheel * cn['ZOOM_SENSITIVITY']
            st.target_radius = max(cn['MIN_RADIUS'], min(cn['MAX_RADIUS'], st.target_radius))

        st.cam_radius += (st.target_radius - st.cam_radius) * (cn['CAMERA_LERP_FACTOR'] * st.delta_time)

        # Only update target st.camera angles if MMB is pressed
        if st.is_mmb_pressed:
            current_x, current_y = glfw.get_cursor_pos(window)
            if st.is_shift_mmb_pressed:
                # Panning mode: Shift + MMB
                dx = current_x - st.last_pan_x
                dy = current_y - st.last_pan_y
                st.last_pan_x, st.last_pan_y = current_x, current_y
                st.target_pan_x += dx * cn['PAN_SENSITIVITY']
                st.target_pan_y += dy * cn['PAN_SENSITIVITY']
            else:
                # Rotation mode: MMB only
                dx = current_x - st.last_x
                dy = current_y - st.last_y
                st.last_x, st.last_y = current_x, current_y
                st.target_yaw -= dx * cn['MOUSE_SENSITIVITY']
                st.target_pitch += dy * cn['MOUSE_SENSITIVITY']
                st.target_pitch = max(cn['MIN_PITCH'], min(cn['MAX_PITCH'], st.target_pitch))


        # --- st.Camera vectors ---
        st.cam_yaw, st.cam_pitch = camera.update(st.target_yaw, st.target_pitch, st.target_pan_y, st.target_pan_x, cn['CAMERA_LERP_FACTOR']*st.delta_time)
        st.cam_orbit = camera.get_orbit()

        # -----

        if io.keys_down[glfw.KEY_HOME]:
            st.target_pan_x = st.target_pan_y = 0.0
            st.cam_orbit = [0.0,0.0,0.0]



        elip = 0.0001
        if (abs(st.cam_yaw - prev_cam_yaw) > elip or 
            abs(st.cam_pitch - prev_cam_pitch) > elip or
            abs(st.cam_radius - prev_cam_radius) > elip or
            any(abs(st.cam_orbit[i] - prev_cam_orbit[i]) > elip for i in range(3))):

            # Reset accumulation buffers so no stale data is read later
            st.frame_count = 0
            clear_accumulation_fbos()
            st.current_accum_index = 0


        prev_cam_yaw = st.cam_yaw
        prev_cam_pitch = st.cam_pitch
        prev_cam_radius = st.cam_radius
        prev_cam_orbit = st.cam_orbit

        use_accumulation =  rendering_pass(
            st, shader, display_shader, vao, display_vao, uniform_locs,
            rendering_width, rendering_height,
            width, height,
            panel_width, menu_bar_height,
            setup_accumulation_buffer,
            bind_sprite_textures,
            set_move_pos_uniform,
            set_move_rot_uniform
        )




        # --- TOP MENU BAR ---
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                if imgui.menu_item("Save Scene", "Ctrl+S")[0]:
                    # Trigger save dialog
                    success, message = save_scene_dialog(scene_builder, window)
                    st.save_load_message = message
                    st.save_load_message_time = time.time()
        
                if imgui.menu_item("Load Scene", "Ctrl+O")[0]:
                    # Trigger load dialog
                    success, message = load_scene_dialog(scene_builder)
                    st.save_load_message = message
                    st.save_load_message_time = time.time()
                    if success:
                        glob_history.undo_stack.clear()
                        glob_history.redo_stack.clear() 
                        scene_builder.update_glob_history(glob_history)
                        st.selected_item_id = None
                        scene_builder.update_selected_item_id(st.selected_item_id)
                        st.selection_mode = None
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms

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
                    on_window_close(window)

                imgui.end_menu()

            if imgui.begin_menu("Edit", True):
                if imgui.menu_item("Add Primitive/Operation", "Ctrl+A")[0]:
                    st.show_add_change_window = True
                    st.pending_change_node_id = None
                if imgui.menu_item("Compile Shader", "Ctrl+B")[0]:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                imgui.end_menu()
    
            if imgui.begin_menu("View", True):
                if imgui.menu_item("Settings", "F10")[0]:
                    st.show_settings_window = True
                imgui.end_menu()

            if imgui.begin_menu("Editor", True):
                if imgui.menu_item("Settings")[0]:
                    st.show_editor_settings_window = True
                imgui.end_menu()
    
            if imgui.begin_menu("About", True):
                if imgui.menu_item("Information")[0]:
                    st.show_about_window = True
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
                st.shader_choice = 0
                # Recompile with new shader
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.set_cursor_pos_x(start_x + button_width + spacing)
            if imgui.button("Cycles", button_width):
                st.shader_choice = 1
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
                    st.additional_scene_code = CE_app.get_plain_text()
                    recompile_shader()
                    CE_app.rec = False


            imgui.end_main_menu_bar()


        # Drag on G
        drag_result = dragging_primitive(window,
            scene_builder,
            camera
        )

        if drag_result:
            success, new_uniforms = recompile_shader()
            if success:
                uniform_locs = new_uniforms


        # ---- Rotate (MoveRot) using R key ----
        rotate_result = rotate_privitive(window,
            scene_builder
        )

        if rotate_result:
            success, new_uniforms = recompile_shader()
            if success:
                uniform_locs = new_uniforms

        
        # --- RENDER TO FRAMEBUFFER AT SCALED RESOLUTION ---
        # If we've already rendered & displayed the accumulation buffer above (cycles shader),
        # skip the further framebuffer / direct rendering to avoid double-draw and viewport offset.
        if shader is not None and st.shader_choice == 1 and use_accumulation == 1:
            # accumulation rendering & display already handled above
            pass

        elif shader is not None and display_shader is not None and st.resolution_scale != 1.0:
            # Setup framebuffer
            framebuffer_output = False # ouu!
            framebuffer_output, \
            st.scaled_rendering_width, st.scaled_rendering_height, \
            fbo, render_texture, \
            fbo_width, fbo_height = setup_framebuffer(
                                    st.scaled_rendering_width, st.scaled_rendering_height,
                                    fbo, render_texture, fbo_width, fbo_height
                                    )

            if framebuffer_output:
                # Render to framebuffer
                glBindFramebuffer(GL_FRAMEBUFFER, fbo)
                glViewport(0, 0, st.scaled_rendering_width, st.scaled_rendering_height)
                glClear(GL_COLOR_BUFFER_BIT)
                
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - st.start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], st.scaled_rendering_width, st.scaled_rendering_height)
                    glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                    glUniform1f(uniform_locs['camYaw'], st.cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], st.cam_pitch)
                    glUniform1f(uniform_locs['radius'], st.cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], st.cam_orbit[0], st.cam_orbit[1], st.cam_orbit[2])
                    set_move_pos_uniform(shader, uniform_locs, st.drag_position)
                    set_move_rot_uniform(shader, uniform_locs, st.drag_rot_position)

                bind_sprite_textures(uniform_locs, st.sprites_array)


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
                        current_time_uniform = time.time() - st.start_time
                        glUniform1f(uniform_locs['time'], current_time_uniform)
                        glUniform2f(uniform_locs['resolution'], st.scaled_rendering_width, st.scaled_rendering_height)
                        glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                        glUniform1f(uniform_locs['camYaw'], st.cam_yaw)
                        glUniform1f(uniform_locs['camPitch'], st.cam_pitch)
                        glUniform1f(uniform_locs['radius'], st.cam_radius)
                        glUniform3f(uniform_locs['CamOrbit'], st.cam_orbit[0], st.cam_orbit[1], st.cam_orbit[2])
                        set_move_pos_uniform(shader, uniform_locs, st.drag_position)
                        set_move_rot_uniform(shader, uniform_locs, st.drag_rot_position)

                    glViewport(panel_width, menu_bar_height, st.scaled_rendering_width, st.scaled_rendering_height)
                    glBindVertexArray(vao)
                    bind_sprite_textures(uniform_locs, st.sprites_array)
                    glDrawArrays(GL_QUADS, 0, 4)
                    glViewport(0, 0, width, height)
        else:
            # Direct rendering when scale is 1.0 or display shader not available
            # Skip if accumulation handled above (see guard at top)
            if shader is not None:
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - st.start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], rendering_width, rendering_height)
                    # Default framebuffer viewport is offset by the left panel and menu bar
                    glUniform2f(uniform_locs['viewportOffset'], float(panel_width), float(menu_bar_height))
                    glUniform1f(uniform_locs['camYaw'], st.cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], st.cam_pitch)
                    glUniform1f(uniform_locs['radius'], st.cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], st.cam_orbit[0], st.cam_orbit[1], st.cam_orbit[2])
                    set_move_pos_uniform(shader, uniform_locs, st.drag_position)
                    set_move_rot_uniform(shader, uniform_locs, st.drag_rot_position)

                # Check if viewport is minimized
                if rendering_width > 0 and rendering_height > 0:
                    glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
                    glBindVertexArray(vao)
                    bind_sprite_textures(uniform_locs, st.sprites_array)
                    glDrawArrays(GL_QUADS, 0, 4)

                glViewport(0, 0, width, height)
        

        # --- SETTINGS WINDOW ---
        if st.show_settings_window:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 150)
            imgui.set_next_window_size(400, 300)  # Increased height
            is_open, st.show_settings_window = imgui.begin("Settings", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                st.show_settings_window = False
            
            imgui.text("Rendering Settings")
            imgui.separator()
            
            # Shader Selection
            imgui.text("Fragment Shader:")
            clicked, st.shader_choice = imgui.combo(
                "##shader_select",
                st.shader_choice,
                [name.replace("shaders/fragment/", "") for name in st.shader_names]
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
            imgui.text(f"{st.resolution_scale:.2f}x")
            
            changed, st.resolution_scale = imgui.slider_float("##st.resolution_scale", st.resolution_scale, 0.25, 2.0, "%.2f")
            if changed:
                st.frame_count = 0


            imgui.spacing()
            imgui.text_colored("1.0 = Normal resolution", 0.7, 0.7, 0.7, 1.0)
            imgui.text_colored("2.0 = Oversampling (better quality)", 0.7, 0.7, 0.7, 1.0)
            imgui.text_colored("<1.0 = Low resolution (better performance)", 0.7, 0.7, 0.7, 1.0)
            
            imgui.spacing()
            imgui.separator()


            # Show Sky colors
            imgui.text("Sky Top Color:")
            top_color_changed, top_color_rgba = imgui.color_edit3("SkyTopColor##color", st.sky_top_color[0], st.sky_top_color[1], st.sky_top_color[2])
            if top_color_changed:
                st.sky_top_color = list(top_color_rgba[:3])  # Only use RGB, ignore alpha
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.text("Sky Bottom Color:")
            bottom_color_changed, bottom_color_rgba = imgui.color_edit3("SkyBottomColor##color", st.sky_bottom_color[0], st.sky_bottom_color[1], st.sky_bottom_color[2])
            if bottom_color_changed:
                st.sky_bottom_color = list(bottom_color_rgba[:3])  # Only use RGB, ignore alpha
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            if st.shader_choice == 0:
                imgui.text("Grid Enabled:")
                changed, st.GridEnabled = imgui.checkbox("", st.GridEnabled)
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

                imgui.spacing()
                imgui.separator()

            elif st.shader_choice == 1:
                imgui.text("Max Samples count:")
                changed, st.max_frames = imgui.input_int("", st.max_frames)
                st.max_frames = max(st.max_frames, 8)
                if changed:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

                imgui.spacing()
                imgui.separator()
            

            imgui.text("Sun:")
            changed, st.LightDir = input_vec3("Sun Direction", st.LightDir)
            if changed:
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.spacing()
            imgui.separator()


            # Calculate scaled size for display
            scaled_w = int(rendering_width * st.resolution_scale)
            scaled_h = int(rendering_height * st.resolution_scale)
            imgui.text(f"Current render size: {scaled_w}x{scaled_h}")
            imgui.text(f"Base size: {rendering_width}x{rendering_height}")
            
            imgui.spacing()
            if imgui.button("Close", -1):
                st.show_settings_window = False
            
            
            imgui.end()




        # --- Editor Settings Window ---
        # --- Content Functions (Placeholders) ---
        def render_themes_tab():
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
                st.theme = copy.deepcopy(default_uconfig["Theme"])
                for label, item in st.theme.items():
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


        if st.show_editor_settings_window:
            # Set initial positioning and size for the main window container
            imgui.set_next_window_position(width // 2 - 400, height // 2 - 300)
            imgui.set_next_window_size(800, 600)
            
            is_open, st.show_editor_settings_window = imgui.begin("Editor Settings", True, imgui.WINDOW_NO_COLLAPSE)
            
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
                        st.current_settings_tab = "Themes"
                    
                    imgui.separator()

                    # Button 2: User
                    if imgui.button("User", width=sidebar_width):
                        st.current_settings_tab = "User"
                        
                    imgui.separator()

                    # Button 3: Shortcuts
                    if imgui.button("Shortcuts", width=sidebar_width):
                        st.current_settings_tab = "Shortcuts"
                        
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
                        
                        if st.current_settings_tab == "Themes":
                            render_themes_tab()
                        elif st.current_settings_tab == "User":
                            render_user_tab()
                        elif st.current_settings_tab == "Shortcuts":
                            render_shortcuts_tab()

                        imgui.end_child() # End SettingsContent

                    imgui.end_child()
                    
            if not is_open:
                st.show_editor_settings_window = False

            imgui.end()



        if st.show_export_vol_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 125)
            imgui.set_next_window_size(300, 250)
            is_open, st.show_export_vol_window = imgui.begin("Export as Volume", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                st.show_export_vol_window = False

            imgui.text("Grid Size:")
            changed, st.grid_size = imgui.input_int("##GridSize", st.grid_size, 8)
            imgui.text_colored(
                "Note that its dimensions range \nfrom -GridSize/2 to +GridSize/2.",
                0.56, 0.93, 0.56
            )

            imgui.spacing()

            changed, st.vox_quality = input_float("Vox. Quality", st.vox_quality, 0.25, 100)

            changed, st.exp_use_color = imgui.checkbox("Use Color", st.exp_use_color)

            imgui.separator()
            imgui.spacing()

            file_preview_size = sdfexp.calculate_sdf_file_size(st.grid_size, st.vox_quality, st.exp_use_color)
            if file_preview_size[1]>1:
                imgui.text(f"File size = {file_preview_size[1]:.2f} mb")
            else:
                imgui.text(f"File size = {file_preview_size[0]:.2f} kb")

            imgui.spacing()
            imgui.spacing()

            if imgui.button("Cancel", 135,30):
                st.show_export_vol_window = False

            imgui.same_line(150)

            if imgui.button("Export", 135,30):
                code = scene_builder.generate_raymarch_code()
                comp_bin = sdfexp.compute_sdf_3d(st.grid_size, st.vox_quality, code, st.additional_scene_code, st.exp_use_color, window)
                save_sdfvol_dialog(sdfexp, comp_bin)

                st.show_export_vol_window = False

            imgui.end()

        if st.show_export_obj_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 130)
            imgui.set_next_window_size(300, 260)
            is_open, st.show_export_obj_window = imgui.begin("Export to OBJ", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                st.show_export_obj_window = False

            imgui.text("Grid Size:")
            changed, st.grid_size = imgui.input_int("##GridSize", st.grid_size, 8)
            imgui.text_colored(
                "Note that its dimensions range \nfrom -GridSize/2 to +GridSize/2.",
                0.56, 0.93, 0.56
            )

            imgui.spacing()

            changed, st.vox_quality = input_float("Voxelization Quality", st.vox_quality, 0.25, 100)

            imgui.separator()
            imgui.spacing() 

            changed, st.export_level = input_float("Level", st.export_level, 0.05, 100)
            st.export_level = np.clip(st.export_level, 0.0, 1.0)

            imgui.spacing()

            changed, st.export_z_up = imgui.checkbox("Z up", st.export_z_up)

            imgui.same_line()

            changed, st.exp_use_color = imgui.checkbox("Use Color", st.exp_use_color)


            imgui.separator()
            imgui.spacing()

            if imgui.button("Cancel", 135,30):
                st.show_export_obj_window = False

            imgui.same_line(150)

            if imgui.button("Export", 135,30):
                code = scene_builder.generate_raymarch_code()
                comp_bin = sdfexp.compute_sdf_3d(st.grid_size, st.vox_quality, code, st.additional_scene_code, st.exp_use_color, window)
                dist_sdf = None
                color_sdf = None

                if isinstance(comp_bin, tuple):
                    elvl = np.interp(st.export_level, [0,1], [comp_bin[0].min(), comp_bin[0].max()])
                    dist_sdf, color_sdf = comp_bin
                else:
                    elvl = np.interp(st.export_level, [0,1], [comp_bin.min(), comp_bin.max()])
                    dist_sdf = comp_bin

                success, message = save_sdfobj_dialog(sdfexp, dist_sdf, color_sdf, st.export_z_up, elvl, st.exp_use_color)
                st.export_obj_message = [success, message]
                st.export_obj_message_time = time.time()

                st.show_export_obj_window = False

            imgui.end()


        if st.show_about_window:
            imgui.set_next_window_position(width // 2 - 250, height // 2 - 200)
            imgui.set_next_window_size(500, 400)  # Increased height
            is_open, st.show_about_window = imgui.begin("About", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                st.show_about_window = False
            
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
                st.show_about_window = False

            imgui.end()

        # --- FPS OVERLAY (Top Right, above right panel) ---
        fps_x = width - panel_width - cn['FPS_WINDOW_WIDTH'] - cn['FPS_WINDOW_OFFSET']
        imgui.set_next_window_position(fps_x, cn['FPS_WINDOW_OFFSET'])
        imgui.set_next_window_size(cn['FPS_WINDOW_WIDTH'], cn['FPS_WINDOW_HEIGHT'])
        imgui.begin("FPS", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        if st.shader_choice == 0:
            imgui.text_colored("FPS: " + str(st.fps_value), 0.0, 1.0, 0.0, 1.0)
        elif st.shader_choice == 1:
            imgui.text_colored("Sample: " + str(st.frame_count), 1.0, 1.0, 0.0, 1.0)

        imgui.end()

        # Orientation Overlay
        ori_x = width - panel_width - cn['ORI_WINDOW_WIDTH'] - cn['ORI_WINDOW_OFFSET']
        imgui.set_next_window_position(fps_x+70, cn['ORI_WINDOW_OFFSET'])
        imgui.set_next_window_size(cn['ORI_WINDOW_WIDTH'], cn['ORI_WINDOW_HEIGHT'])
        imgui.begin("ORI", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        
        imgui.same_line(17,0) # At Center
        imgui.text_colored("VIEW", 0.8,0.8,1.0)
        imgui.spacing()
        if imgui.small_button("X##Ori"):
            st.target_yaw = 0.0
            st.target_pitch = 0.0
        imgui.same_line()
        if imgui.small_button("-X##Ori"):
            st.target_yaw = 3.14
            st.target_pitch = 0.0
        imgui.spacing()
        if imgui.small_button("Y##Ori"):
            st.target_pitch = 1.57
        imgui.same_line()
        if imgui.small_button("-Y##Ori"):
            st.target_pitch = -1.57     
        imgui.spacing()
        if imgui.small_button("Z##Ori"):
            st.target_yaw = 1.57
            st.target_pitch = 0.0
        imgui.same_line()
        if imgui.small_button("-Z##Ori"):
            st.target_yaw = -1.57 
            st.target_pitch = 0.0    

        imgui.end()
        

        if st.show_exit_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 65)
            imgui.set_next_window_size(300, 130)  # Increased height
            is_open, st.show_exit_window = imgui.begin("Confirm Exit", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                st.show_exit_window = False
            
            imgui.spacing()
            imgui.text(f"Are you sure you want to exit?\nUnsaved data may be lost.")
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui.button("Cancel", 130,30):
                st.show_exit_window = False
            imgui.same_line(0,15)
            if imgui.button("YES", 130,30):
                # Save Data
                config = {"Theme": st.theme}
                save_user_config("UserData/User.data", config)


                glfw.set_window_should_close(window, True)

            imgui.end()

        if st.show_restart_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 65)
            imgui.set_next_window_size(300, 130)  # Increased height
            is_open, st.show_restart_window = imgui.begin("Confirm Restart", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                st.show_restart_window = False
            
            imgui.spacing()
            imgui.text(f"Are you sure you want to restart the app?\nThis may result in loss of unsaved data.")
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui.button("Cancel", 130,30):
                st.show_restart_window = False
            imgui.same_line(0,15)
            if imgui.button("YES", 130,30):
                # Save Data
                config = {"Theme": st.theme}
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

        if st.save_load_message is not None:
            # Show message for 3 seconds
            if time.time() - st.save_load_message_time < 3.0:
                imgui.set_next_window_position(width // 2 - 150, 100)
                imgui.begin("Status", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)

                # Color based on success
                is_success = "saved" in st.save_load_message.lower() or "loaded" in st.save_load_message.lower()
                color = (0.0, 1.0, 0.0, 1.0) if is_success else (1.0, 0.0, 0.0, 1.0)
                imgui.text_colored(st.save_load_message, *color)

                imgui.same_line(350, 0)

                if imgui.button("copy"):
                    pyperclip.copy(st.save_load_message)

                imgui.end()
            else:
                st.save_load_message = None


        if st.export_obj_message is not None:
            # Show message for 3 seconds
            if time.time() - st.export_obj_message_time < 3.0:
                imgui.set_next_window_position(width // 2 - 150, 100)
                imgui.begin("Status", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)

                # Color based on success
                is_success = st.export_obj_message[0]
                color = (0.0, 1.0, 0.0, 1.0) if is_success else (1.0, 0.0, 0.0, 1.0)
                imgui.text_colored(st.export_obj_message[1], *color)
                
                imgui.same_line(350, 0)

                if imgui.button("copy"):
                    pyperclip.copy(st.save_load_message)
                
                imgui.end()
            else:
                st.export_obj_message[1] = None



        # --- Error Display (if shader compilation failed) ---
        if st.shader_compile_error:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 50)
            imgui.set_next_window_size(400, 100)
            imgui.begin("Shader Compilation Error", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            imgui.text_colored("Error:", 1.0, 0.0, 0.0, 1.0)
            imgui.same_line()
            imgui.text_wrapped(st.shader_compile_error)
            if imgui.button("Dismiss"):
                st.shader_compile_error = None
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

            if st.selected_item_id == node_id:
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
                    if node_id in st.selected_items:
                        st.selected_items.remove(node_id)
                    else:
                        st.selected_items.add(node_id)
                    if len(st.selected_items) > 0:
                        st.selected_item_id = None
                else:
                    st.selected_items.clear()
                    st.selected_item_id = node_id
                    scene_builder.update_selected_item_id(st.selected_item_id)
                    st.selection_mode = 'node'
                    st.renaming_item_id = None
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
                        st.pending_change_node_id = node_id
                        st.show_add_change_window = True
                        imgui.close_current_popup()
                else:
                    # For primitives, offer Change Type (in-place) rather than forcing delete+create
                    if imgui.menu_item("Change Type")[0]:
                        st.pending_change_node_id = node_id
                        st.show_add_change_window = True
                        imgui.close_current_popup()
                
                imgui.separator()
                
                if imgui.menu_item("Change Properties")[0]:
                    st.property_change_node_id = node_id
                    st.show_property_change_window = True
                    imgui.close_current_popup()
                
                # NEW: Reparent option
                if imgui.menu_item("Reparent")[0]:
                    st.reparent_node_id = node_id
                    st.show_reparent_window = True
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
            st.show_add_change_window = True
            st.pending_change_node_id = None
        
        imgui.end()  # End Scene Tree window
        


        # --- RIGHT PANEL: Properties/Inspector ---
        imgui.set_next_window_position(width - panel_width, menu_bar_height)
        imgui.set_next_window_size(panel_width, height - menu_bar_height)
        imgui.begin("Inspector", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        if st.selected_item_id is not None and st.selected_item_id in scene_builder.id_to_node:
            node = scene_builder.get_node(st.selected_item_id)
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
                    st.renaming_item_id = st.selected_item_id
                    st.rename_text = item_data.ui_name
                
                if st.renaming_item_id == st.selected_item_id:
                    changed, st.rename_text = imgui.input_text("##rename", st.rename_text, 256)
                    
                    if imgui.button("OK", width / 5):
                        scene_builder.rename_node(st.selected_item_id, st.rename_text)
                        st.renaming_item_id = None
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
                    
                    imgui.same_line()
                    if imgui.button("Cancel", width / 5):
                        st.renaming_item_id = None
                
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
                        if sprite_idx is None or sprite_idx >= len(st.sprites_array):
                            imgui.text_colored("Sprite data missing or corrupted", 1.0, 0.0, 0.0, 1.0)
                        else:
                            spr = st.sprites_array[sprite_idx]
                            imgui.text("Plane parameters:")
                            changed, primitive.position = input_vec3("Point", primitive.position, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_vec3)
                            changed2, spr.planeNormal = input_vec3("Normal", spr.planeNormal, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_vec3)
                            changed3, spr.planeWidth = input_float("Width", spr.planeWidth, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float)
                            changed4, spr.planeHeight = input_float("Height", spr.planeHeight, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float)
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
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                        case "torus":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Major Radius", primitive.size_or_radius[0], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Minor Radius", primitive.size_or_radius[1], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        case "hex_prism":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Hex Radius", primitive.size_or_radius[0], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Height", primitive.size_or_radius[1], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        case "vertical_capsule":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Height", primitive.size_or_radius[0], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Radius", primitive.size_or_radius[1], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        case "capped_cylinder":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Radius", primitive.size_or_radius[0], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Height", primitive.size_or_radius[1], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        case "rounded_cylinder":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Radius A", primitive.size_or_radius[0], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Radius B", primitive.size_or_radius[1], 
                                cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        # Special parameters for specific primitives
                        case "cone":
                            c_sin = primitive.kwargs.get('c_sin', 0.5)
                            c_cos = primitive.kwargs.get('c_cos', 0.866)
                            height = primitive.kwargs.get('height', 1.0)
                            changed1, c_sin = input_float(
                                "Sin(Angle)", c_sin, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed2, c_cos = input_float(
                                "Cos(Angle)", c_cos, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            changed3, height = input_float(
                                "Height", height, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
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
                            changed1, normal = input_vec3("Normal", normal, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_vec3)
                            changed2, h = input_float("Offset (h)", h, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float)
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
                            changed, height = input_float("Height", height, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float)
                            if changed:
                                primitive.kwargs['height'] = height
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # --- Inspector: add UI to edit pointer function selection (inside the primitive inspector branch) ---
                        case "pointer":
                            changed_pos, primitive.position = input_vec3(
                                "Position", primitive.position, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_vec3
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
                                    f"Point {i}", list(pt), cn['STEP_VARIABLE_FLOAT'], panel_elem_width_vec3
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
                                "Thickness", thickness, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
                            )
                            if changed:
                                primitive.kwargs['thickness'] = thickness
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        case _:
                            if primitive.primitive_type not in ["cone", "plane", "rounded_cylinder", "pointer", "sprite", "curve"]:
                                changed, primitive.size_or_radius = input_vec3(
                                    "Size", primitive.size_or_radius, cn['STEP_VARIABLE_FLOAT'], panel_elem_width_vec3
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
                                    "Radius", primitive.kwargs.get('radius', 0.1),cn['STEP_VARIABLE_FLOAT'], panel_elem_width_float
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
                        cn['STEP_VARIABLE_FLOAT'],
                        panel_elem_width_vec3
                    )
                    if changed:
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
                    
                    changed, item_data.rotation = input_vec3(
                        "Rotation",
                        item_data.rotation,
                        cn['STEP_VARIABLE_ANGLE'],
                        panel_elem_width_vec3
                    )
                    if changed:
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
                    
                    changed, item_data.scale = input_vec3(
                        "Scale",
                        item_data.scale,
                        cn['STEP_VARIABLE_FLOAT'],
                        panel_elem_width_vec3
                    )
                    if changed:
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms
                    
                    changed, item_data.color = input_vec3(
                        "Color",
                        item_data.color,
                        cn['STEP_VARIABLE_FLOAT'],
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
        if st.show_operation_selection_window:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 200)
            imgui.set_next_window_size(400, 400)
            
            is_open, st.show_operation_selection_window = imgui.begin(
                "Add Operation",
                True,
                imgui.WINDOW_NO_COLLAPSE
            )
            
            if not is_open:
                st.show_operation_selection_window = False
            
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
                    st.selected_item_id = new_op_id
                    scene_builder.update_selected_item_id(st.selected_item_id)
                    st.selection_mode = 'node'
                    st.show_operation_selection_window = False
                
                if imgui.is_item_hovered():
                    imgui.set_tooltip(description)
            
            imgui.end()
        
        # --- ADD PRIMITIVE DIALOG ---
        if st.show_primitive_selection_window:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 150)
            imgui.set_next_window_size(400, 400)
            
            is_open, st.show_primitive_selection_window = imgui.begin(
                "Add Standalone Primitive",
                True,
                imgui.WINDOW_NO_COLLAPSE
            )
            
            if not is_open:
                st.show_primitive_selection_window = False
            
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
                    st.selected_item_id = new_prim_id
                    scene_builder.update_selected_item_id(st.selected_item_id)
                    st.selection_mode = 'node'
                    st.show_primitive_selection_window = False
            

            imgui.end()


        # Combined Add/Change Window - Two columns (left = primitives, right = operations)
        if st.show_add_change_window:
            imgui.set_next_window_position(width // 2 - 300, height // 2 - 235)
            imgui.set_next_window_size(600, 470)
            is_open, st.show_add_change_window = imgui.begin("Add / Change Type", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                st.show_add_change_window = False
                st.pending_change_node_id = None

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
                    # If st.pending_change_node_id is None -> ADD, else -> CHANGE TYPE
                    if st.pending_change_node_id is None:
                        # Add new primitive at origin with defaults
                        new_id = scene_builder.add_standalone_primitive(
                            prim_type,
                            position=[0.0, 0.0, 0.0],
                            size_or_radius=size if size is not None else 0.5,
                            ui_name=label
                        )
                        if new_id:
                            st.selected_items.clear()
                            st.selected_item_id = new_id
                            scene_builder.update_selected_item_id(st.selected_item_id)
                            st.selection_mode = 'node'
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                    else:
                        # Change the pending node to this primitive (in-place)
                        node = scene_builder.get_node(st.pending_change_node_id)
                        if node:
                            # If it was operation -> convert to primitive
                            scene_builder.change_node_to_primitive(st.pending_change_node_id, prim_type, position=None, size_or_radius=(size if size is not None else 0.5))
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                        # Clear pending state
                        st.pending_change_node_id = None
                        st.show_add_change_window = False
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"Add / Change to {label}")
            

            imgui.next_column()
            imgui.text("Operations")
            imgui.separator()

            for label, op_type, operand_count, description in operations_list:
                if imgui.button(label, -1, 24):
                    if st.pending_change_node_id is None:
                        # Add new operation (auto-create primitives)
                        new_op_id = scene_builder.add_operation_with_auto_primitives(
                            op_type,
                            auto_primitive_type='box',
                            ui_name=label
                        )
                        if new_op_id:
                            st.selected_items.clear()
                            st.selected_item_id = new_op_id
                            scene_builder.update_selected_item_id(st.selected_item_id)
                            st.selection_mode = 'node'
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                    else:
                        # Convert pending node to this operation type (in-place)
                        scene_builder.change_node_to_operation(st.pending_change_node_id, op_type, auto_primitive_type='box')
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms

                        st.pending_change_node_id = None
                        st.show_add_change_window = False

                if imgui.is_item_hovered():
                    imgui.set_tooltip(description)

            imgui.columns(1)
            imgui.separator()
            imgui.spacing()
            imgui.same_line(20,0)
            if imgui.button("Cancel", 265, 28):
                st.show_add_change_window = False
                st.pending_change_node_id = None

            imgui.same_line(305,0)
            if imgui.button("Close", 265, 28):
                st.show_add_change_window = False
                st.pending_change_node_id = None

            imgui.end()


        if st.show_property_change_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 125)
            imgui.set_next_window_size(300, 250)
            is_open, st.show_property_change_window = imgui.begin("Change Properties", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                st.show_property_change_window = False
                st.property_change_node_id = None


            node = scene_builder.get_node(st.property_change_node_id)
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
                st.show_property_change_window = False
                st.property_change_node_id = None

            imgui.end()


        def _render_reparent_node_list(node_id, exclude_node_id, exclude_descendants, indent=""):
            """
            Recursively render selectable nodes for reparent window.
            Only show operation nodes (valid parents).
            Returns True if a node was selected.
            """

            if node_id in exclude_descendants or node_id == exclude_node_id:
                return False

            node = scene_builder.get_node(node_id)
            if not node or node.node_type != 'operation':
                return False

            label = f"{indent}{node.item_data.ui_name} ({node_id})"
            # NOTE: the second arg is the 'selected' boolean. We pass False for predictable behaviour.
            clicked, _ = imgui.selectable(label, False)
            if clicked:
                st.reparent_target_parent = node_id
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
        if st.show_reparent_window and st.reparent_node_id:
            imgui.set_next_window_size(400, 500)
            st.show_reparent_window, _ = imgui.begin("Reparent Node", True)
            
            if st.show_reparent_window:
                reparent_node = scene_builder.get_node(st.reparent_node_id)
                if reparent_node:
                    imgui.text(f"Reparenting: {reparent_node.item_data.ui_name}")
                    imgui.separator()
                    imgui.text("Select new parent operation:")
                    
                    # List all operation nodes (excluding the node being reparented and its descendants)
                    all_descendants = scene_builder.get_all_children_recursive(st.reparent_node_id)
                    all_descendants.append(st.reparent_node_id)
                    
                    parent_selected = False
                    for root_id in scene_builder.root_children:
                        parent_selected |= _render_reparent_node_list(
                            root_id, 
                            st.reparent_node_id, 
                            all_descendants,
                            "  "
                        )
                    
                    if parent_selected and st.reparent_target_parent:
                        new_parent_node = scene_builder.get_node(st.reparent_target_parent)
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
                                            st.reparent_child_to_replace == child_id
                                        )[0]:
                                            st.reparent_child_to_replace = child_id
                    
                    imgui.spacing()
                    imgui.separator()
                    
                    if imgui.button("Cancel", 100, 30):
                        st.show_reparent_window = False
                        st.reparent_node_id = None
                        st.reparent_target_parent = None
                        st.reparent_child_to_replace = None
                    
                    imgui.same_line(150)
                    
                    can_reparent = st.reparent_target_parent is not None
                    if st.reparent_target_parent:
                        new_parent_node = scene_builder.get_node(st.reparent_target_parent)
                        required_operands = scene_builder._get_operand_count(new_parent_node.item_data.operation_type)
                        current_operands = len(new_parent_node.children)
                        if current_operands >= required_operands and st.reparent_child_to_replace is None:
                            can_reparent = False
                    
                    if not can_reparent:
                        pass
                    
                    if imgui.button("Reparent", 100, 30):
                        if scene_builder.reparent_node(st.reparent_node_id, st.reparent_target_parent, st.reparent_child_to_replace):
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                        st.show_reparent_window = False
                        st.reparent_node_id = None
                        st.reparent_target_parent = None
                        st.reparent_child_to_replace = None
                    
                    if not can_reparent:
                        pass
            
            imgui.end()


        # Render ImGui
        imgui.render()
        impl.render(imgui.get_draw_data())

        # Swap front and back buffers
        glfw.swap_buffers(window)

    for i in range(2):
        if st.accumulation_fbos[i] is not None:
            try:
                glDeleteFramebuffers(1, [st.accumulation_fbos[i]])
            except Exception:
                pass
            st.accumulation_fbos[i] = None
        if st.accumulation_textures[i] is not None:
            try:
                glDeleteTextures(1, [st.accumulation_textures[i]])
            except Exception:
                pass
            st.accumulation_textures[i] = None

    # Clean up
    # Delete all cached shaders
    for cached_shader, _ in st.shader_cache.values():
        if cached_shader is not None:
            glDeleteProgram(cached_shader)
    st.shader_cache.clear()
    
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