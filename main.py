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

from imgui.integrations.glfw import GlfwRenderer
from PIL import Image
from typing import Dict, List, Any

import os
import json
import numpy as np
import math
import copy

import tkinter as tk
from tkinter import filedialog, messagebox

from src.classes import *
from src.app import *
from src.ui import *
from src.utils import *
from src.render import *

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
UI_SCALE = 1.0

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
    global start_drag, end_drag, dragging, R_dragging, selected_item_id, drag_position, drag_rot_position


    # Initialize GLFW
    if not glfw.init():
        print("---GLFW IS NOT INIT---")
        return

    # Create a windowed mode window and its OpenGL context
    try:
        window = glfw.create_window(SCREEN_SIZE[0], SCREEN_SIZE[1], "Viewport", None, None)
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

    # --- Camera State ---
    target_yaw = 0.0
    target_pitch = 0.0
    target_pan_y = 0.0
    target_pan_x = 0.0
    target_radius = 5.0
    cam_yaw = 0.0
    cam_pitch = 0.0
    cam_pan_y = 0.0
    cam_pan_x = 0.0
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


    # Helper: safely set MovePos uniform (call this wherever you were directly doing glUniform3f for MovePos)
    def set_move_pos_uniform(shader_program, uniform_locs, pos):
        """
        Safely set the MovePos uniform. If the cached uniform location is missing (-1 or None),
        query it dynamically and cache it. Only call glUniform if the location exists.
        """
        if uniform_locs is None or shader_program is None:
            return
        move_key = 'move_pos'
        loc = uniform_locs.get(move_key, None)
        if loc is None or loc == -1:
            # Query the active program for the location (this is safe and will return -1 if not declared)
            loc = glGetUniformLocation(shader_program, "MovePos")
            uniform_locs[move_key] = loc
        if loc != -1:
            glUniform3f(loc, float(pos[0]), float(pos[1]), float(pos[2]))


    def set_move_rot_uniform(shader_program, uniform_locs, rot):
        """
        Safely set the MoveRot uniform. If the cached uniform location is missing (-1 or None),
        query it dynamically and cache it. Only call glUniform if the location exists.
        """
        if uniform_locs is None or shader_program is None:
            return
        move_key = 'move_rot'
        loc = uniform_locs.get(move_key, None)
        if loc is None or loc == -1:
            # Query the active program for the location (this is safe and will return -1 if not declared)
            loc = glGetUniformLocation(shader_program, "MoveRot")
            uniform_locs[move_key] = loc
        if loc != -1:
            glUniform3f(loc, float(rot[0]), float(rot[1]), float(rot[2]))



    def bind_sprite_textures(uniforms):
        """
        Bind loaded sprite textures to texture units and upload the sampler uniform indices.
        Assumes texture unit 0 may be used for accumulation/render targets, so start at unit 1.
        """
        base_unit = 1
        for i, spr in enumerate(sprites_array):
            loc = uniforms.get(spr.SprTexture, -1) if uniforms else -1
            unit = base_unit + i
            if spr.texture_id is not None and loc is not None and loc != -1:
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, spr.texture_id)
                # Tell shader which texture unit to sample from
                glUniform1i(loc, unit)
            else:
                # If texture not loaded, bind 0 to keep behavior stable
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, 0)
                if loc != -1:
                    glUniform1i(loc, unit)
        # restore active texture to 0
        glActiveTexture(GL_TEXTURE0)


    # --- Delta time --- 
    delta_time = 0.0 


    # --- Scene Definition ---
    scene_builder = SDFSceneBuilder(glob_history, selected_item_id)

    box = scene_builder.add_box((0.0, -0.5+2.0, 0.0), (0.5, 0.5, 0.5), ui_name="Box 1", color=[0.8, 0.2, 0.2])
    sphere_id = scene_builder.add_sphere((0.0, -0.75+2.0, 0.0), 0.5, ui_name="Sphere 1", color=[0.2, 0.8, 0.2])
    sphere_id = scene_builder.ssub(sphere_id, box, 0.05, ui_name="Subtract 1")
    box_id = scene_builder.add_roundbox((0.0, -2.0+2.0, 0.0), (3.0, 1.0, 3.0), 0.1, ui_name="Round Box 1", color=[0.4, 0.4, 0.8])
    box2_id = scene_builder.add_box((0.0, -1.5+2.0, 0.0), (2.0, 1.0, 2.0), ui_name="Box 2", color=[0.8, 0.8, 0.4])
    final_id = scene_builder.ssub(box2_id, box_id, 0.05, ui_name="Subtract 2")
    final_id = scene_builder.sunion(final_id, sphere_id, 0.05, ui_name="Union 1")
    

    # --- UI State ---
    show_selection_window = False
    show_settings_window = False
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
    theme = {
        "bg_dark" : [0.12, 0.11, 0.09, 1.0],        # Very dark background
        "panel_dark" : [0.18, 0.16, 0.13, 1.0],     # Slightly lighter panel/frame
        "accent" : [0.608, 0.067, 0.118, 1.0],      # Primary Dark Red/Crimson (Buttons, Active Header)
        "hover" : [0.902, 0.125, 0.125, 1.0],       # Bright Red/Scarlet (Hover/Active State)
        "text_light" : [0.92, 0.90, 0.80, 1.0],     # Off-white text

        "muted_accent" : [0.4, 0.04, 0.08, 1.0], 
        "child_bg" : [0.20, 0.18, 0.15, 1.0],
        "dim_background" : [0.0, 0.0, 0.0, 0.7], 
        "border_color" : [0.25, 0.23, 0.20, 1.0]
    }



    # Shader selection
    shader_choice = 0  # 0 = template, 1 = cycles
    shader_names = ["shaders/fragment/template.glsl", "shaders/fragment/cycles.glsl"]

    # Sky shaders uniforms (cycles)
    sky_top_color = [0.7, 0.8, 1.0]
    sky_bottom_color = [0.1, 0.15, 0.25]

    # Grid (template)
    GridEnabled = True

    # Light
    LightDir = [0.5, 1.0, -0.7]

    # --- Settings ---
    resolution_scale = 1.0  # 1.0 = normal, 2.0 = oversampling, <1.0 = low res for performance

    # Export Config
    grid_size = 16
    vox_quality = 1.0
    export_z_up = True
    export_level = 0.0

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
    
    def get_uniform_locations(shader_program):
        # Get all uniform locations for the shader program.
        uniforms = {
            'time'                 :       glGetUniformLocation(shader_program, "time"),
            'resolution'           :       glGetUniformLocation(shader_program, "resolution"),
            'viewportOffset'       :       glGetUniformLocation(shader_program, "viewportOffset"),
            'camYaw'               :       glGetUniformLocation(shader_program, "camYaw"),
            'camPitch'             :       glGetUniformLocation(shader_program, "camPitch"),
            'radius'               :       glGetUniformLocation(shader_program, "radius"),
            'CamOrbit'             :       glGetUniformLocation(shader_program, "CamOrbit"),
            'frameIndex'           :       glGetUniformLocation(shader_program, "frameIndex"),
            'accumulationTexture'  :       glGetUniformLocation(shader_program, "accumulationTexture"),
            'useAccumulation'      :       glGetUniformLocation(shader_program, "useAccumulation"),
            'col_sky_top'          :       glGetUniformLocation(shader_program, "SkyColorTop"),
            'col_sky_bottom'       :       glGetUniformLocation(shader_program, "SkyColorBottom"),
            'grid_enabled'         :       glGetUniformLocation(shader_program, "GridEnabled"),
            'move_pos'             :       glGetUniformLocation(shader_program, "MovePos"),
            'move_rot'             :       glGetUniformLocation(shader_program, "MoveRot"),
            'maxFrames'            :       glGetUniformLocation(shader_program, "MaxFrames"),
            'LightDir'             :       glGetUniformLocation(shader_program, "LightDir")
        }

        # Register sprite sampler uniforms (dynamic)
        # sprites_array is in outer scope; it's the list of Sprite objects used for postprocessing
        try:
            for spr in sprites_array:
                # Use sampler name string as key, store location (may be -1 if unused)
                uniforms[spr.SprTexture] = glGetUniformLocation(shader_program, spr.SprTexture)
        except Exception:
            # If sprites_array is not defined yet, skip (defensive)
            pass

        return uniforms


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

    # --- OpenGL Setup (Quad VAO/VBO) ---
    vertices = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    vertices = (GLfloat * len(vertices))(*vertices)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    
    # --- Framebuffer Setup for Resolution Scaling ---
    fbo = None
    render_texture = None
    fbo_width = 0
    fbo_height = 0

    
    try:
        # Quad with texture coordinates for displaying the rendered texture
        quad_vertices = [
            # positions   # tex coords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ]
        quad_vertices = (GLfloat * len(quad_vertices))(*quad_vertices)
        
        display_vao = glGenVertexArrays(1)
        glBindVertexArray(display_vao)
        display_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, display_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(quad_vertices) * 4, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)  # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))  # tex coord
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
    except Exception as e:
        print(f"Warning: Could not create display shader: {e}")
        print("Falling back to direct rendering (resolution scale may not work correctly)")
    
    # Simple shader for displaying texture
    display_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
    """
    
    display_fragment_shader = """
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D renderTexture;
    uniform int isAccumulation;

    void main() {
        vec4 tex = texture(renderTexture, TexCoord);

        if (isAccumulation == 1) {
            FragColor = vec4(tex.rgb, 1.0);
        } else {
            FragColor = vec4(tex.rgb, 1.0);
        }
    }
    """
    
    display_shader = None
    display_vao = None
    display_vbo = None
    
    try:
        display_shader = compileProgram(
            compileShader(display_vertex_shader, GL_VERTEX_SHADER),
            compileShader(display_fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Quad with texture coordinates for displaying the rendered texture
        quad_vertices = [
            # positions   # tex coords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ]
        quad_vertices = (GLfloat * len(quad_vertices))(*quad_vertices)
        
        display_vao = glGenVertexArrays(1)
        glBindVertexArray(display_vao)
        display_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, display_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(quad_vertices) * 4, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)  # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))  # tex coord
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
    except Exception as e:
        print(f"Warning: Could not create display shader: {e}")
        print("Falling back to direct rendering (resolution scale may not work correctly)")
        display_shader = None


    # --- Accumulation Buffer Setup ---
    accumulation_fbo = None
    accumulation_texture = None
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

    global UI_SCALE
    UI_SCALE = UConfig["UIScale"]
    rebuild_imgui_fonts(impl, "assets/fonts/Roboto-Medium.ttf", 16.0, UI_SCALE)


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
        io = imgui.get_io()

        ShortCuts = {
            "Rename" : (glfw.KEY_F2),
            "Add" : (glfw.KEY_A, "CTRL"), 
            "Delete" : (glfw.KEY_DELETE),
            "Compile" : (glfw.KEY_B, "CTRL"),
            "Undo" : (glfw.KEY_Z, "CTRL"),
            "Redo" : (glfw.KEY_Z, "CTRL", "SHIFT"),
            "Redo2" : (glfw.KEY_Y, "CTRL"),
            "Move" : (glfw.KEY_G),
            "Rotate" : (glfw.KEY_R),
            "X" : (glfw.KEY_X),
            "Y" : (glfw.KEY_Y),
            "Z" : (glfw.KEY_Z),
            "Open" : (glfw.KEY_O, "CTRL"),
            "Save" : (glfw.KEY_S, "CTRL"),
            "Duplicate": (glfw.KEY_D, "CTRL"),
        }


        def input_handle(action : str) -> bool:
            # Helper function to get the live state of a modifier ID
            def get_live_modifier_state(modifier_id):
                if modifier_id == "CTRL":
                    return io.key_ctrl
                if modifier_id == "SHIFT":
                    return io.key_shift
                return False
            
            keys_required = ShortCuts.get(action)
            
            if keys_required is None:
                return False

            # Ensure keys_required is always iterable (a tuple)
            if not isinstance(keys_required, tuple):
                keys_required = (keys_required,)

            # --- STEP 1: Check if ALL conditions are met (Is the combination currently held?) ---
            all_keys_down_this_frame = True
            main_key_code = None # Store the main key code for debouncing later

            for key_check in keys_required:
                
                if isinstance(key_check, int):
                    # Standard key code: Must be currently down
                    if not io.keys_down[key_check]:
                        all_keys_down_this_frame = False
                        break
                    # Store this as the potential main key to check for initial press
                    main_key_code = key_check
                        
                elif isinstance(key_check, str):
                    # Modifier: Must be currently down
                    if not get_live_modifier_state(key_check):
                        all_keys_down_this_frame = False
                        break

            if not all_keys_down_this_frame:
                return False # Combo is not active right now

            # --- STEP 2: Debounce (Did the key press START this frame?) ---
            
            # Case A: Single Key (like F2)
            if len(keys_required) == 1 and main_key_code is not None:
                if io.keys_down[main_key_code]:
                    return True
            
            # Case B: Combination Key (like Ctrl+A)
            elif len(keys_required) > 1 and main_key_code is not None:
                if io.keys_down[main_key_code]:
                    return True
                    
            # If we reach here, the combination is held, but the trigger key wasn't *newly* pressed this frame.
            return False


                
        # Check Ctrl+A for add window 
        if input_handle("Add"):
            if not last_key_a_pressed:
                show_selection_window = True
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
        if input_handle("Delete") and selected_item_id is not None:
            if not last_key_delete_pressed:
                if scene_builder.delete_item(selected_item_id):
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    selected_item_id = None
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = None
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


        # --- Interpolate camera angles ---
        cam_yaw += (target_yaw - cam_yaw) * (CAMERA_LERP_FACTOR*delta_time)
        cam_pitch += (target_pitch - cam_pitch) * (CAMERA_LERP_FACTOR*delta_time)

        # --- Interpolate camera Pan ---
        cam_pan_y += (target_pan_y - cam_pan_y) * (CAMERA_LERP_FACTOR*delta_time)
        cam_pan_x -= (target_pan_x + cam_pan_x) * (CAMERA_LERP_FACTOR*delta_time)

        # --- Camera vectors ---

        forward_x = math.cos(cam_pitch) * math.sin(cam_yaw)
        forward_y = math.sin(cam_pitch)
        forward_z = math.cos(cam_pitch) * math.cos(cam_yaw)


        right_x = math.cos(cam_yaw)
        right_y = 0
        right_z = -math.sin(cam_yaw)


        up_x = forward_y * right_z - forward_z * right_y
        up_y = forward_z * right_x - forward_x * right_z
        up_z = forward_x * right_y - forward_y * right_x


        orbit_center_offset_x = cam_pan_x * right_x + cam_pan_y * up_x
        orbit_center_offset_y = cam_pan_x * right_y + cam_pan_y * up_y
        orbit_center_offset_z = cam_pan_x * right_z + cam_pan_y * up_z

        cam_orbit = (
            orbit_center_offset_z, # Yoow! (Correctly)
            orbit_center_offset_y,
            orbit_center_offset_x
        )

        # -----

        if io.keys_down[glfw.KEY_HOME]:
            cam_pan_x = cam_pan_y = target_pan_x = target_pan_y = 0.0
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

                bind_sprite_textures(uniform_locs)
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
            bind_sprite_textures(uniform_locs)
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
                bind_sprite_textures(uniform_locs)
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
                    show_selection_window = True
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
                        if sid in scene_builder.id_to_index:
                            itype, idx = scene_builder.id_to_index[sid]
                            if itype == 'primitive':
                                prim = scene_builder.primitives[idx][1]
                                # Recreate with same properties (preserve kwargs)
                                new_id = scene_builder.add_primitive(
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
                elif selected_item_id and selected_item_id in scene_builder.id_to_index:
                    itype, idx = scene_builder.id_to_index[selected_item_id]
                    if itype == 'primitive':
                        prim = scene_builder.primitives[idx][1]
                        new_id = scene_builder.add_primitive(
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
                    selection_mode = 'primitive'
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

                if dragging_op_id and dragging_op_id in scene_builder.id_to_index:
                    item_type, idx = scene_builder.id_to_index[dragging_op_id]
                    if item_type == 'primitive':
                        prim = scene_builder.primitives[idx][1]
                        # Copy the primitive start position so later comparisons/undo work
                        drag_start_pos = prim.position[:]
                        # Reset accumulated movement
                        drag_accum = [0.0, 0.0, 0.0]
                        # Record starting mouse cursor (independent of camera last_x/last_y)
                        drag_last_x, drag_last_y = glfw.get_cursor_pos(window)
                    else:
                        # nothing valid to drag
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
                # Stop dragging: commit final position (register undo/redo)
                if dragging_op_id and dragging_op_id in scene_builder.id_to_index:
                    item_type, idx = scene_builder.id_to_index[dragging_op_id]
                    if item_type == 'primitive':
                        prim = scene_builder.primitives[idx][1]
                        final_pos = prim.position
                        # Register only if changed
                        if drag_start_pos is not None and final_pos != drag_start_pos:
                            scene_builder.modify_primitive_property(dragging_op_id, 'position', drag_start_pos, final_pos)
                            # Optionally recompile if scene/code generation depends on selection
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                # clear drag state
                dragging_op_id = None
                drag_start_pos = None
                drag_accum = [0.0, 0.0, 0.0]
                axis_toggled_gx = axis_toggled_gy = axis_toggled_gz = False

        # Always update last_key_g_pressed for proper edge detection next frame
        last_key_g_pressed = key_g_is_down

        # Handle axis toggles (Blender-style): toggle on key press, update debounced state every frame
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

        # Update the "last key" flags for X/Y/Z so we only toggle once per press
        last_key_gx_pressed = key_x_is_down
        last_key_gy_pressed = key_y_is_down
        last_key_gz_pressed = key_z_is_down

        # Determine active axis (None => free drag)
        active_axis = None
        if axis_toggled_gx:
            active_axis = 0
        elif axis_toggled_gy:
            active_axis = 1
        elif axis_toggled_gz:
            active_axis = 2

        # Per-frame drag movement (this must run every frame while dragging)
        if dragging and dragging_op_id and dragging_op_id in scene_builder.id_to_index:
            # read current mouse and compute delta since last frame
            current_x, current_y = glfw.get_cursor_pos(window)
            dx = current_x - drag_last_x
            dy = current_y - drag_last_y
            # store for next frame
            drag_last_x, drag_last_y = current_x, current_y

            # convert to mouse-space movement (invert Y so screen-up => world up)
            mouse_delta_x = dx * DRAG_SENSITIVITY
            mouse_delta_y = -dy * DRAG_SENSITIVITY

            if np.linalg.norm(np.array([mouse_delta_x, mouse_delta_y])) > 0.01:
                frame_count = 0
                clear_accumulation_fbos(accumulation_fbos,scaled_rendering_width, scaled_rendering_height)

            # transform mouse deltas into world-space using camera right/up vectors
            # right_x, right_y, right_z and up_x, up_y, up_z must be computed earlier (they are in your code)
            move_delta_x = mouse_delta_x * right_x + mouse_delta_y * up_x
            move_delta_y = mouse_delta_x * right_y + mouse_delta_y * up_y
            move_delta_z = mouse_delta_x * right_z + mouse_delta_y * up_z

            # axis constraints (lock movement to a single world axis)
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

            # accumulate world movement since drag started
            drag_accum[0] += move_delta_z
            drag_accum[1] += move_delta_y
            drag_accum[2] += move_delta_x

            # compute new primitive position from saved start pos + accumulated movement
            item_type, idx = scene_builder.id_to_index[dragging_op_id]
            prim = scene_builder.primitives[idx][1]
            if drag_start_pos is None:
                drag_start_pos = prim.position.copy()

            new_pos = [
                drag_start_pos[0] + drag_accum[0],
                drag_start_pos[1] + drag_accum[1],
                drag_start_pos[2] + drag_accum[2],
            ]

            # apply live position (no historical entry yet — recorded on drag end)
            prim.position = new_pos
            drag_position = new_pos.copy()

        else:
            # When not dragging keep shader uniform aligned with selection or zero
            if selected_item_id and selected_item_id in scene_builder.id_to_index:
                itype, idx = scene_builder.id_to_index[selected_item_id]
                if itype == 'primitive':
                    prim = scene_builder.primitives[idx][1]
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

                if R_dragging_op_id and R_dragging_op_id in scene_builder.id_to_index:
                    item_type, idx = scene_builder.id_to_index[R_dragging_op_id]
                    if item_type == 'primitive':
                        prim = scene_builder.primitives[idx][1]
                        # Save starting rotation (copy)
                        R_drag_start_pos = prim.rotation.copy()
                        R_drag_accum = [0.0, 0.0, 0.0]
                        # Save last mouse position for delta
                        R_drag_last_x, R_drag_last_y = glfw.get_cursor_pos(window)
                    else:
                        # nothing valid to rotate
                        R_dragging_op_id = None
                        R_drag_start_pos = None
                        R_drag_accum = [0.0, 0.0, 0.0]
                else:
                    R_dragging_op_id = None
                    R_drag_start_pos = None
                    R_drag_accum = [0.0, 0.0, 0.0]

                # Reset rotation-specific axis toggles when starting a new rotate
                axis_toggled_rx = axis_toggled_ry = axis_toggled_rz = False

            else:
                # Stop rotation: commit final rotation (register undo/redo)
                if R_dragging_op_id and R_dragging_op_id in scene_builder.id_to_index:
                    item_type, idx = scene_builder.id_to_index[R_dragging_op_id]
                    if item_type == 'primitive':
                        prim = scene_builder.primitives[idx][1]
                        final_rot = prim.rotation
                        # Register only if changed
                        if R_drag_start_pos is not None and final_rot != R_drag_start_pos:
                            scene_builder.modify_primitive_property(R_dragging_op_id, 'rotation', R_drag_start_pos, final_rot)
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                # clear rotation state
                R_dragging_op_id = None
                R_drag_start_pos = None
                R_drag_accum = [0.0, 0.0, 0.0]
                axis_toggled_rx = axis_toggled_ry = axis_toggled_rz = False

        # Update the R debounce flag
        last_key_r_pressed = key_r_is_down

        # Rotation axis toggles (Blender-style) — use rotation-specific debounced keys
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

        # Update per-key last flags for rotation keys so we toggle once per press
        last_key_rx_pressed = key_x_is_down
        last_key_ry_pressed = key_y_is_down
        last_key_rz_pressed = key_z_is_down

        # Determine active rotation axis (None => free rotation mapping: vertical->X, horizontal->Y)
        active_rot_axis = None
        if axis_toggled_rx:
            active_rot_axis = 0
        elif axis_toggled_ry:
            active_rot_axis = 1
        elif axis_toggled_rz:
            active_rot_axis = 2

        # Per-frame rotation update while R_dragging is active
        if R_dragging and R_dragging_op_id and R_dragging_op_id in scene_builder.id_to_index:
            current_x, current_y = glfw.get_cursor_pos(window)
            dx = current_x - R_drag_last_x
            dy = current_y - R_drag_last_y
            R_drag_last_x, R_drag_last_y = current_x, current_y

            # Sensitivity in radians per pixel
            R_ROT_SENSITIVITY = 0.005

            # Default free-rotation mapping
            rot_delta_x = -dy * R_ROT_SENSITIVITY   # vertical mouse -> rotation around X
            rot_delta_y = -dx * R_ROT_SENSITIVITY   # horizontal mouse -> rotation around Y
            rot_delta_z = 0.0

            # If axis-locked, map mouse motion to rotation around that world axis only
            if active_rot_axis is not None:
                if active_rot_axis == 0:
                    # X-axis lock: use vertical mouse movement for X rotation only
                    rot_delta_x = -dy * R_ROT_SENSITIVITY
                    rot_delta_y = 0.0
                    rot_delta_z = 0.0
                elif active_rot_axis == 1:
                    # Y-axis lock: use horizontal mouse movement for Y rotation only
                    rot_delta_x = 0.0
                    rot_delta_y = -dx * R_ROT_SENSITIVITY
                    rot_delta_z = 0.0
                elif active_rot_axis == 2:
                    # Z-axis lock: use horizontal mouse movement for Z rotation
                    rot_delta_x = 0.0
                    rot_delta_y = 0.0
                    rot_delta_z = -dx * R_ROT_SENSITIVITY

            # reset accumulation if movement is non-trivial (so accumulation-based renderer refreshes)
            if abs(rot_delta_x) + abs(rot_delta_y) + abs(rot_delta_z) > 1e-5:
                frame_count = 0
                clear_accumulation_fbos(accumulation_fbos, scaled_rendering_width, scaled_rendering_height)

            # accumulate
            R_drag_accum[0] += rot_delta_x
            R_drag_accum[1] += rot_delta_y
            R_drag_accum[2] += rot_delta_z

            # Apply live rotation to primitive (no history until release)
            item_type, idx = scene_builder.id_to_index[R_dragging_op_id]
            prim = scene_builder.primitives[idx][1]
            if R_drag_start_pos is None:
                R_drag_start_pos = prim.rotation.copy()

            new_rot = [
                R_drag_start_pos[0] + R_drag_accum[0],
                R_drag_start_pos[1] + R_drag_accum[1],
                R_drag_start_pos[2] + R_drag_accum[2],
            ]

            prim.rotation = new_rot
            # Keep shader MoveRot uniform in sync for live interaction
            drag_rot_position = new_rot.copy()

        else:
            # When not rotating keep shader MoveRot aligned with selection (or zero)
            if selected_item_id and selected_item_id in scene_builder.id_to_index:
                itype, idx = scene_builder.id_to_index[selected_item_id]
                if itype == 'primitive':
                    prim = scene_builder.primitives[idx][1]
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

                bind_sprite_textures(uniform_locs)


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
                    bind_sprite_textures(uniform_locs)
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
                    bind_sprite_textures(uniform_locs)
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

            global UI_SCALE
            changed, UI_SCALE = input_float("Interface Scale", UI_SCALE, 0.05, 100)
            imgui.text_colored("For the changes to take effect, restart the application.", 1.0,0.9,0.7)
            if imgui.button("RESTART"):
                restart()

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
                    # We use a child window/group to hold the content that changes based on the selection.
                    
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
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 110)
            imgui.set_next_window_size(300, 220)
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

            changed, vox_quality = input_float("Voxelization Quality", vox_quality, 0.25, 100)

            imgui.separator()
            imgui.spacing()

            file_preview_size = sdfexp.calculate_sdf_file_size(grid_size, vox_quality)
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
                comp_bin = sdfexp.compute_sdf_3d(grid_size, vox_quality, code, additional_scene_code, window)
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

            imgui.separator()
            imgui.spacing()

            if imgui.button("Cancel", 135,30):
                show_export_obj_window = False

            imgui.same_line(150)

            if imgui.button("Export", 135,30):
                code = scene_builder.generate_raymarch_code()
                comp_bin = sdfexp.compute_sdf_3d(grid_size, vox_quality, code, additional_scene_code, window)
                elvl = np.interp(export_level, [0,1], [comp_bin.min(), comp_bin.max()])
                success, message = save_sdfobj_dialog(sdfexp, comp_bin, export_z_up, elvl)
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
                config = {"Theme": theme, "UIScale": UI_SCALE}
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
                config = {"Theme": theme, "UIScale": UI_SCALE}
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

        # --- LEFT PANEL: Scene Tree ---
        # Offset panels below menu bar (menu_bar_height already calculated above)
        imgui.set_next_window_position(0, menu_bar_height)
        imgui.set_next_window_size(panel_width, height - menu_bar_height)
        imgui.begin("Scene Tree", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        
        wbn = 16
        hbn = 20

        def format_label(name, op_id, max_chars=16):
            # Truncate name to max_chars, preserving full ID
            if len(name) > max_chars:
                truncated_name = name[:max_chars - 3] + "..."
            else:
                truncated_name = name
            return f"{truncated_name} ({op_id})"
        

    
        imgui.text("Primitives:")
        io_local = imgui.get_io()

        # Render groups first as parent nodes (showing members as children)
        for op_id, primitive in scene_builder.primitives:
            if primitive.primitive_type != 'group':
                continue

            label = format_label(primitive.ui_name, op_id)
            # Tree node for the group
            flags = 0
            if selected_item_id == op_id:
                flags |= imgui.TREE_NODE_SELECTED

            node_open = imgui.tree_node(label, flags)

            # Clicking the group selects it
            if imgui.is_item_clicked():
                selected_item_id = op_id
                scene_builder.update_selected_item_id(selected_item_id)
                selection_mode = 'primitive'
                renaming_item_id = None
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            if node_open:
                # Render members as indented child items
                members = primitive.kwargs.get('members', [])
                for mid in members:
                    if mid not in scene_builder.id_to_index:
                        continue
                    m_type, m_idx = scene_builder.id_to_index[mid]
                    if m_type != 'primitive':
                        continue
                    member_prim = scene_builder.primitives[m_idx][1]
                    mlabel = format_label(member_prim.ui_name, mid)
                    # draw a selectable / leaf for the member
                    flags_m = imgui.TREE_NODE_LEAF
                    if selected_item_id == mid or mid in selected_items:
                        flags_m |= imgui.TREE_NODE_SELECTED

                    # Use same visual layout as before (arrow buttons left)
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (1, 1))
                    if imgui.arrow_button(f"##up_{mid}", 2):
                        idx = scene_builder.id_to_index[mid][1]
                        if idx > 0:
                            scene_builder.move_item(mid, idx - 1)
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                    imgui.same_line()
                    if imgui.arrow_button(f"##down_{mid}", 3):
                        idx = scene_builder.id_to_index[mid][1]
                        if idx < len(scene_builder.primitives) - 1:
                            scene_builder.move_item(mid, idx + 1)
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                    imgui.pop_style_var(1)

                    imgui.same_line()
                    node_open_m = imgui.tree_node(mlabel, flags_m)
                    if imgui.is_item_clicked():
                        # respect Ctrl multi-select
                        if io_local.key_ctrl:
                            if mid in selected_items:
                                selected_items.remove(mid)
                            else:
                                selected_items.add(mid)
                            selected_item_id = None
                            scene_builder.update_selected_item_id(selected_item_id)
                            selection_mode = None
                        else:
                            selected_items.clear()
                            selected_item_id = mid
                            scene_builder.update_selected_item_id(selected_item_id)
                            selection_mode = 'primitive'
                            renaming_item_id = None
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                    if node_open_m:
                        imgui.tree_pop()

                imgui.tree_pop()

        # Render remaining top-level primitives that are not grouped
        for op_id, primitive in scene_builder.primitives:
            # Skip groups and primitives marked as grouped (they're shown as children above)
            if primitive.primitive_type == 'group' or primitive.kwargs.get('grouped', None) is not None:
                continue

            label = format_label(primitive.ui_name, op_id)
            flags = imgui.TREE_NODE_LEAF
            if selected_item_id == op_id or op_id in selected_items:
                flags |= imgui.TREE_NODE_SELECTED

            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (1, 1))
            if imgui.arrow_button(f"##up_{op_id}", 2):
                idx = scene_builder.id_to_index[op_id][1]
                if idx > 0:
                    scene_builder.move_item(op_id, idx - 1)
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

            imgui.same_line()

            if imgui.arrow_button(f"##down_{op_id}", 3):
                idx = scene_builder.id_to_index[op_id][1]
                if idx < len(scene_builder.primitives) - 1:
                    scene_builder.move_item(op_id, idx + 1)
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
            imgui.pop_style_var(1)

            imgui.same_line()

            node_open = imgui.tree_node(label, flags)

            if imgui.is_item_clicked():
                # If CTRL is held, toggle multi-select
                if io_local.key_ctrl:
                    if op_id in selected_items:
                        selected_items.remove(op_id)
                    else:
                        selected_items.add(op_id)
                    # clear single selection if multiple selected
                    selected_item_id = None
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = None
                else:
                    # regular single select
                    selected_items.clear()
                    selected_item_id = op_id
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = 'primitive'
                    renaming_item_id = None

                    # Recompile shader for live selection-based uniforms
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

            if node_open:
                imgui.tree_pop()

        imgui.spacing()
        imgui.text("Operations:")
        for op_id, operation in scene_builder.operations:
            label = format_label(operation.ui_name, op_id)
            flags = imgui.TREE_NODE_LEAF
            if selected_item_id == op_id:
                flags |= imgui.TREE_NODE_SELECTED

            # First create the buttons (they'll appear on the left)
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (1, 1))
            if imgui.arrow_button(f"##upop_{op_id}", 2):
                idx = scene_builder.id_to_index[op_id][1]
                if idx > 0:
                    scene_builder.move_item(op_id, idx - 1)
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
            

            imgui.same_line()  # Keep buttons on the same line

            if imgui.arrow_button(f"##downop_{op_id}", 3):
                idx = scene_builder.id_to_index[op_id][1]
                if idx < len(scene_builder.operations) - 1:
                    scene_builder.move_item(op_id, idx + 1)
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
            imgui.pop_style_var(1)

            imgui.same_line()  # Keep label on the same line as buttons

            # Then create the tree node (label will appear after buttons)
            node_open = imgui.tree_node(label, flags)

            # Handle selection when the node is clicked
            if imgui.is_item_clicked():
                selected_item_id = op_id
                scene_builder.update_selected_item_id(selected_item_id)
                selection_mode = 'operation'
                renaming_item_id = None

                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            if node_open:
                imgui.tree_pop()

        imgui.spacing()
        if imgui.button("Add (Ctrl+A)", -1):
            show_selection_window = True
        
        if imgui.button("Compile (Ctrl+B)", -1):
            success, new_uniforms = recompile_shader()
            if success:
                uniform_locs = new_uniforms
        
        # Multi-selection -> group creation UI
        if len(selected_items) > 1:
            imgui.separator()
            imgui.text(f"{len(selected_items)} selected")
            # A small buffer for the group name
            if 'group_name_buffer' not in globals():
                group_name_buffer = "Group"
            changed, group_name_buffer = imgui.input_text("Group name", group_name_buffer, 128)
            if imgui.button("Create Group", -1):
                members = list(selected_items)
                new_group_id = scene_builder.add_group(members, position=(0.0,0.0,0.0), ui_name=group_name_buffer)
                # Clear multi selection and select the new group
                selected_items.clear()
                selected_item_id = new_group_id
                scene_builder.update_selected_item_id(selected_item_id)
                selection_mode = 'primitive'
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms
        
        imgui.spacing()
        imgui.text_colored("Press Delete to remove", 1.0, 1.0, 0.0, 1.0)

        imgui.end()

        # --- RIGHT PANEL: Properties/Inspector ---
        imgui.set_next_window_position(width - panel_width, menu_bar_height)
        imgui.set_next_window_size(panel_width, height - menu_bar_height)
        imgui.begin("Inspector", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        if renaming_item_id is not None:
            imgui.text("Renaming:")
            changed, rename_text = imgui.input_text("##rename", rename_text, 256)
            
            if imgui.button("OK", width / 5):
                if renaming_item_id in scene_builder.id_to_index:
                    item_type, index = scene_builder.id_to_index[renaming_item_id]
                    if item_type == 'primitive':
                        scene_builder.primitives[index][1].ui_name = rename_text
                    else:
                        scene_builder.operations[index][1].ui_name = rename_text
                renaming_item_id = None
            
            imgui.same_line()
            if imgui.button("Cancel", width / 5):
                renaming_item_id = None
            
            imgui.separator()

        if selected_item_id is not None:
            imgui.text(f"Selected: {scene_builder.get_item_name(selected_item_id)}")
            imgui.separator()

            if selection_mode == 'primitive':
                # Find the primitive
                for op_id, primitive in scene_builder.primitives:
                    if op_id == selected_item_id:
                        imgui.text(f"Type: {primitive.primitive_type}")
                    
                        
                        if primitive.primitive_type == "sprite":
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

                        # Size/Radius - varies by primitive type
                        # I have python 3.8, "match", get out.
                        if primitive.primitive_type == "sphere":
                            changed, primitive.size_or_radius[0] = input_float(
                                "Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                        elif primitive.primitive_type == "torus":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Major Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Minor Radius", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "hex_prism":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Hex Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Height", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "vertical_capsule":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Height", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Radius", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "capped_cylinder":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Height", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "rounded_cylinder":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Radius A", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Radius B", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        else:
                            if primitive.primitive_type not in ["cone", "plane", "rounded_cylinder", "pointer", "sprite", "curve"]:
                                changed, primitive.size_or_radius = input_vec3(
                                    "Size", primitive.size_or_radius, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                                )
                        if primitive.primitive_type not in ["pointer", "sprite", "curve"]:
                            if changed:
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # Special parameters for specific primitives
                        if primitive.primitive_type == "cone":
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
                        
                        elif primitive.primitive_type == "plane":
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
                        
                        elif primitive.primitive_type == "rounded_cylinder":
                            height = primitive.kwargs.get('height', 1.0)
                            changed, height = input_float("Height", height, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                            if changed:
                                primitive.kwargs['height'] = height
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # --- Inspector: add UI to edit pointer function selection (inside the primitive inspector branch) ---
                        if primitive.primitive_type == "pointer":
                            old_pos = primitive.position
                            changed_pos, primitive.position = input_vec3(
                                "Position", primitive.position, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                            )
                            if changed_pos:
                                scene_builder.modify_primitive_property(op_id, 'position', old_pos, primitive.position)
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
                                old_func = current_func
                                primitive.kwargs['func'] = new_func
                                # Record change in history for undo/redo
                                scene_builder.modify_primitive_property(op_id, "kwargs.func", old_func, new_func)
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            imgui.separator()
                            imgui.text("Pointer functions mutate \nthe raymarch point `p` \nfor subsequent primitives.")
                            imgui.text_colored("Place a pointer earlier in \nthe tree to affect later objects.", 0.9, 0.8, 0.2, 1.0)

                        elif primitive.primitive_type == "sprite":
                            pass # Skip Transforms and Color

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
                            elif primitive.primitive_type == "curve":
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
                                

                            imgui.begin_group()

                            imgui.spacing()
                            imgui.separator()
                            imgui.dummy((panel_width/4)-8, 0)
                            imgui.same_line()
                            imgui.text_colored("Transform", 1.0,0.7,0.5,1.0)
                            imgui.spacing()
                
                            imgui.end_group()


                            # Position
                            old_pos = primitive.position
                            changed, primitive.position = input_vec3(
                                "Position", primitive.position, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                            )
                            if changed:
                                scene_builder.modify_primitive_property(op_id, 'position', old_pos, primitive.position)
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms


                            # Show rotation as degrees
                            current_degrees = [math.degrees(a) for a in primitive.rotation]
                            changed, degs = input_vec3(
                                "Rotation °", current_degrees, STEP_VARIABLE_ROTATION, panel_elem_width_vec3
                            )
                            if changed:
                                primitive.rotation = [math.radians(a) for a in degs]
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            # Scale
                            changed, primitive.scale = input_vec3(
                                "Scale", primitive.scale, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                            )
                            if changed:
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                    
                            
                            # Color picker
                            imgui.begin_group()

                            imgui.spacing()
                            imgui.separator()
                            imgui.dummy((panel_width/3)-16, 0)
                            imgui.same_line()
                            imgui.text_colored("Color", 1.0,0.7,0.5,1.0)
                            imgui.spacing()
                
                            imgui.end_group()

                            # Color edit - imgui automatically shows a picker button
                            old_color = primitive.color.copy()
                            color_changed, color_rgba = imgui.color_edit3("Color##color", *primitive.color)
                            if color_changed:
                                primitive.color = list(color_rgba[: 3])
                                scene_builder.modify_primitive_property(op_id, 'color', old_color, primitive.color)
                                success, new_uniforms = recompile_shader()
                                if success: 
                                    uniform_locs = new_uniforms
                            
                            # Show color preview button
                            imgui.same_line()
                            # color_button takes: label, r, g, b, flags, size_x, size_y
                            #imgui.color_button("Preview##color_preview", primitive.color[0], primitive.color[1], primitive.color[2], 0, 20, 20)
                            
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
                            
                            break

            elif selection_mode == 'operation':
                # Find the operation
                for op_id, operation in scene_builder.operations:
                    if op_id == selected_item_id:
                        imgui.text(f"Type: {operation.operation_type}")
                        
                        # Get valid operands (only those declared before this operation)
                        valid_operands = scene_builder.get_valid_operands(selected_item_id)
                        
                        # Determine if this is a single-operand or two-operand operation
                        is_single_operand = operation.operation_type in ['round', 'onion', 'invert', 'snoiseDisp']
                        num_operands = 1 if is_single_operand else 2
                        
                        if len(valid_operands) == 0:
                            imgui.text_colored("No valid operands available!", 1.0, 0.0, 0.0, 1.0)
                        else:
                            for i in range(num_operands):
                                operand_label = "Operand" if is_single_operand else ("Operand A" if i == 0 else "Operand B")
                                
                                # Create a combo box for selecting operands
                                current_operand = operation.args[i]
                                current_index = 0
                                
                                # Find current operand in valid list
                                for idx, (item_id, _) in enumerate(valid_operands):
                                    if item_id == current_operand:
                                        current_index = idx
                                        break
                                
                                clicked, new_index = imgui.combo(
                                    f"##operand_{i}",
                                    current_index,
                                    [scene_builder.get_item_name(item_id) for item_id, _ in valid_operands]
                                )
                                                                
                                if clicked:
                                    old_operand = operation.args[i]
                                    new_operand = valid_operands[new_index][0]
                                    # Use scene_builder API so the change is recorded in history
                                    scene_builder.modify_operation_parameter(op_id, f"args[{i}]", old_operand, new_operand)
                                    success, new_uniforms = recompile_shader()
                                    if success:
                                        uniform_locs = new_uniforms
                        
                        imgui.separator()
                        
                        # Show float parameter for single-operand operations with parameters
                        if hasattr(operation, 'float_param') and operation.float_param is not None:
                            changed, operation.float_param = input_float("Parameter", operation.float_param, 0.01, panel_elem_width_float)
                            if changed:
                                # Update the operation with new parameter
                                if len(operation.args) >= 2:
                                    operation.args[1] = operation.float_param
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # Show smoothing factor for smooth operations
                        elif operation.smooth_k is not None:
                            changed, operation.smooth_k = input_float("Smoothing Factor (k)", operation.smooth_k, 0.01, panel_elem_width_float)
                            if changed:
                                if len(operation.args) >= 3:
                                    operation.args[2] = operation.smooth_k
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        break
        else:
            imgui.text("Select an item to edit")

        imgui.end()

        # --- SELECTION WINDOW ---
        if show_selection_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 150)
            imgui.set_next_window_size(300, 400)
            # Prevent window from being collapsed - use WINDOW_NO_COLLAPSE flag
            is_open, show_selection_window = imgui.begin("Add Primitive/Operation", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                show_selection_window = False

            imgui.text("Select a Primitive:")
            
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
                ("Rounded Cylinder", "rounded_cylinder", (0.3, 0.1)),
                ("Pointer", "pointer", None),
                ("Sprite", "sprite", None),
                ("Curve", "curve", None)
            ]

            for label, prim_type, size_radius in primitives_list:
                if imgui.button(f"  {label}", -1):
                    if prim_type == "box":
                        new_id = scene_builder.add_box((0.0, 0.0, 0.0), size_radius, ui_name=label)
                    elif prim_type == "round_box":
                        new_id = scene_builder.add_roundbox((0.0, 0.0, 0.0), size_radius, 0.1, ui_name=label)
                    elif prim_type == "sphere":
                        new_id = scene_builder.add_sphere((0.0, 0.0, 0.0), size_radius, ui_name=label)
                    elif prim_type == "torus":
                        new_id = scene_builder.add_torus((0.0, 0.0, 0.0), size_radius[0], size_radius[1], ui_name=label)
                    elif prim_type == "cone":
                        new_id = scene_builder.add_cone((0.0, 0.0, 0.0), 0.5, 0.866, 1.0, ui_name=label)
                    elif prim_type == "plane":
                        new_id = scene_builder.add_plane((0.0, 0.0, 0.0), [0.0, 1.0, 0.0], 0.0, ui_name=label)
                    elif prim_type == "hex_prism":
                        new_id = scene_builder.add_hex_prism((0.0, 0.0, 0.0), size_radius[0], size_radius[1], ui_name=label)
                    elif prim_type == "vertical_capsule":
                        new_id = scene_builder.add_vertical_capsule((0.0, 0.0, 0.0), size_radius[0], size_radius[1], ui_name=label)
                    elif prim_type == "capped_cylinder":
                        new_id = scene_builder.add_capped_cylinder((0.0, 0.0, 0.0), size_radius[0], size_radius[1], ui_name=label)
                    elif prim_type == "rounded_cylinder":
                        new_id = scene_builder.add_rounded_cylinder((0.0, 0.0, 0.0), size_radius[0], size_radius[1], 1.0, ui_name=label)
                    elif prim_type == "pointer":
                        # default pointer function
                        new_id = scene_builder.add_pointer((0.0, 0.0, 0.0), func='pointer_identity', ui_name=label)
                    elif prim_type == "sprite":
                        # Create a Sprite object and append to the global sprites_array.
                        # Default plane is centered in front of camera/origin; uvSize default 1x1.
                        new_spr = Sprite(
                        planePoint=(0.0, 0.0, 0.0),
                        planeNormal=(0.0, 0.0, 1.0),
                        planeWidth=2.0,
                        planeHeight=2.0,
                        SprTexture=f"sprTex{len(sprites_array)}",
                        uvSize=(1.0, 1.0),
                        Alpha=1.0,
                        LOD=0.0
                        )
                        sprites_array.append(new_spr)
                        # Create a SDF primitive that references the sprite index so it shows in the tree
                        new_id = scene_builder.add_primitive("sprite", (0.0, 0.0, 0.0), [0.0,0.0,0.0], ui_name=label, color=[1.0,1.0,1.0], sprite_index=len(sprites_array)-1)
                    if prim_type == "curve":
                        new_id = scene_builder.add_primitive(
                            "curve",
                            position=(0.0, 0.0, 0.0),
                            size_or_radius=None,
                            ui_name=label,
                            points=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]],
                            thickness=0.1,
                            color=(0.5, 0.7, 1.0)
                        )
                    
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    
                    selected_item_id = new_id
                    scene_builder.update_selected_item_id(selected_item_id)
                    selection_mode = 'primitive'
                    show_selection_window = False

            imgui.separator()
            imgui.text("Select an Operation:")

            # Need at least 1 primitive/operation for most operations
            all_items = scene_builder.get_all_items()
            
            if len(all_items) >= 1:
                operations_list = [
                    ("Union", "union"),
                    ("Subtraction", "sub"),
                    ("Intersection", "inter"),
                    ("Smooth Union", "sunion"),
                    ("Smooth Subtraction", "ssub"),
                    ("Smooth Intersection", "sinter"),
                    ("Mix", "mix"),
                    ("XOR", "xor"),
                    ("Invert", "invert"),
                    ("Round", "round"),
                    ("Onion", "onion"),
                    ("snoiseDisp", "snoiseDisp")
                ]

                for label, op_type in operations_list:
                    # Single-operand operations (invert, round, onion)
                    is_single_operand = op_type in ['invert', 'round', 'onion', 'snoiseDisp']
                    min_operands = 1
                    available_operands = len(all_items)
                    
                    if available_operands >= min_operands:
                        if imgui.button(f"  {label}", -1):
                            # For single operand operations
                            if is_single_operand:
                                if op_type == "invert": 
                                    new_id = scene_builder.invert(all_items[-1][0], ui_name=label)
                                elif op_type == "round":
                                    new_id = scene_builder.round(all_items[-1][0], 0.1, ui_name=label)
                                elif op_type == "onion":
                                    new_id = scene_builder.onion(all_items[-1][0], 0.05, ui_name=label)
                                elif op_type == "snoiseDisp":
                                    new_id = scene_builder.snoiseDisp(all_items[-1][0], 0.05, ui_name=label)
                            elif op_type in ["sunion", "ssub", "sinter", "mix"]:
                                if len(all_items) >= 2:
                                    new_id = getattr(scene_builder, op_type)(all_items[-2][0], all_items[-1][0], 
                                        (0.5 if op_type == 'mix' else 0.05), ui_name=label)
                                else:
                                    new_id = getattr(scene_builder, op_type)(all_items[-1][0], all_items[-1][0], 
                                        (0.5 if op_type == 'mix' else 0.05), ui_name=label)
                            else:
                                if len(all_items) >= 2:
                                    new_id = getattr(scene_builder, op_type)(all_items[-2][0], all_items[-1][0], ui_name=label)
                                else:
                                    new_id = getattr(scene_builder, op_type)(all_items[-1][0], all_items[-1][0], ui_name=label)
                            
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                            
                            selected_item_id = new_id
                            scene_builder.update_selected_item_id(selected_item_id)
                            selection_mode = 'operation'
                            show_selection_window = False
            else:
                imgui.text("Add at least 1 primitive to use operations")

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