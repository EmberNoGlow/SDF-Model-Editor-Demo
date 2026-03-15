from dataclasses import dataclass, field
from typing import List, Optional, Any
import time

@dataclass
class st:
    """
    Container for application state variables that change during runtime.
    Groups variables by functional areas with detailed comments.
    """
    start_time = 0.0
    prev_time  = 0.0


    selected_item_id = None
    
    # Moved variables
    drag_position = [0,0,0]
    drag_rot_position = [0,0,0]

    
    # Multi-selection support (CTRL+click to toggle)
    selected_items = set()
    selected_item_id = None


    # --- Accumulation Buffer Setup ---
    accumulation_width = 0
    accumulation_height = 0
    max_frames = 128
    accumulation_textures = [None, None]  # Double buffer
    accumulation_fbos = [None, None]
    current_accum_index = 0  # Which one to write to

    scaled_rendering_width = 0
    scaled_rendering_height = 0

    scaled_rendering_width = 0.0
    scaled_rendering_height = 0.0

    # --- Camera State ---
    target_yaw: float = 0.0  # Target yaw angle (horizontal rotation) in radians
    target_pitch: float = 0.0  # Target pitch angle (vertical rotation) in radians
    target_pan_y: float = 0.0  # Target vertical pan offset
    target_pan_x: float = 0.0  # Target horizontal pan offset
    target_radius: float = 5.0  # Target distance from camera to target
    cam_yaw: float = 0.0  # Current camera yaw angle
    cam_pitch: float = 0.0  # Current camera pitch angle
    last_x: float = 0.0  # Last mouse X coordinate
    last_y: float = 0.0  # Last mouse Y coordinate
    last_pan_x: float = 0.0  # Separate tracking for horizontal panning
    last_pan_y: float = 0.0  # Separate tracking for vertical panning
    cam_radius: float = 5.0  # Current camera distance from target
    cam_orbit = [0.0, 0.0, 0.0]  # Camera orbit position [x, y, z]

    # --- Mouse State ---
    is_mmb_pressed: bool = False  # Middle mouse button pressed state
    is_shift_mmb_pressed: bool = False  # Shift + Middle mouse button state

    # --- Save/Load Messages ---
    save_load_message: Optional[str] = None  # Current save/load status message
    save_load_message_time: Optional[float] = None  # Timestamp when message was shown
    export_obj_message: Optional[str] = None  # Export OBJ status message
    export_obj_message_time: Optional[float] = None  # Timestamp for export message

    # --- Keyboard State ---
    last_key_s_pressed: bool = False
    last_key_o_pressed: bool = False
    last_key_z_pressed: bool = False
    last_key_y_pressed: bool = False
    last_key_g_pressed: bool = False
    axis_toggled_gx: bool = False  # X-axis toggle for G-movement
    axis_toggled_gy: bool = False  # Y-axis toggle for G-movement
    axis_toggled_gz: bool = False  # Z-axis toggle for G-movement
    last_key_gx_pressed: bool = False
    last_key_gy_pressed: bool = False
    last_key_gz_pressed: bool = False
    last_key_r_pressed: bool = False
    axis_toggled_rx: bool = False  # X-axis toggle for rotation
    axis_toggled_ry: bool = False  # Y-axis toggle for rotation
    axis_toggled_rz: bool = False  # Z-axis toggle for rotation
    last_key_rx_pressed: bool = False
    last_key_ry_pressed: bool = False
    last_key_rz_pressed: bool = False
    last_key_f10_pressed: bool = False  # F10 key press state
    last_key_d_pressed: bool = False  # D key press state (duplicate key debounce)

    # --- Dragging State ---
    dragging: bool = False  # Whether dragging operation is active
    dragging_op_id: Optional[Any] = None  # op_id of the item currently being dragged
    drag_last_x: float = 0.0  # Last mouse X position during dragging
    drag_last_y: float = 0.0  # Last mouse Y position during dragging
    drag_start_pos: Optional[List[float]] = None  # Original primitive position at drag start
    drag_accum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Accumulated world-space movement since drag start

    # --- Rotation Dragging State ---
    R_dragging: bool = False  # Whether rotation dragging is active
    R_dragging_op_id: Optional[Any] = None  # op_id of item being rotated
    R_drag_last_x: float = 0.0  # Last mouse X during rotation drag
    R_drag_last_y: float = 0.0  # Last mouse Y during rotation drag
    R_drag_start_pos: Optional[List[float]] = None  # Original position at rotation drag start
    R_drag_accum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Accumulated rotation movement

    # --- Timing ---
    delta_time: float = 0.0  # Time delta between frames (seconds)

    # --- UI State ---
    show_operation_selection_window: bool = False
    show_primitive_selection_window: bool = False
    show_settings_window: bool = False
    show_add_change_window: bool = False
    pending_change_node_id: Optional[Any] = None
    property_change_node_id: Optional[Any] = None
    show_property_change_window: bool = False
    show_reparent_window: bool = False
    reparent_node_id: Optional[Any] = None
    reparent_target_parent: Optional[Any] = None  # Selected new parent
    reparent_child_to_replace: Optional[Any] = None  # Child to delete if parent is full
    show_editor_settings_window: bool = False
    current_settings_tab: str = "Themes"  # Active settings tab
    show_export_vol_window: bool = False
    show_export_obj_window: bool = False
    show_about_window: bool = False
    show_exit_window: bool = False
    show_restart_window: bool = False
    selection_mode: Optional[str] = None  # 'primitive' or 'operation'
    renaming_item_id: Optional[Any] = None  # Item being renamed
    rename_text: str = ""  # Text being entered during rename
    last_key_a_pressed: bool = False  # Ctrl+A key press state
    last_key_f2_pressed: bool = False  # F2 key press state
    last_key_delete_pressed: bool = False  # Delete key press state
    last_key_compile_pressed: bool = False  # Ctrl+B key press state

    # --- Theme and UI ---
    theme: Any = None  # Current UI theme (will be set later)

    # --- Shader State ---
    shader_choice: int = 0  # 0 = template, 1 = cycles
    shader_names = [
            "shaders/fragment/template.glsl",
            "shaders/fragment/cycles.glsl"
        ]


    # Sky shaders uniforms (cycles)
    sky_top_color= [0.7, 0.8, 1.0]
    sky_bottom_color= [0.1, 0.15, 0.25]

    # Grid (template)
    GridEnabled: bool = True

    # Light
    LightDir = [0.5, 1.0, 0.7]

    # --- Settings ---
    resolution_scale: float = 1.0  # 1.0 = normal, 2.0 = oversampling, <1.0 = low res for performance

    # Export Config
    grid_size: int = 16
    vox_quality: float = 1.0
    export_z_up: bool = True
    export_level: float = 0.0
    exp_use_color: bool = True

    # Sprites
    sprites_array = []


    # --- FPS tracking ---
    fps_clock: float = time.time()
    fps_frames: int = 0
    fps_value: int = 0

    frame_count = 0


    # --- Shader compilation and error tracking ---
    shader_compile_error: Any = None
    shader_cache = {}  # Cache for compiled shaders: {hash: (shader_program, uniforms)}

    additional_scene_code: str = ""
    