import math

# Consts
cn = {
    # Screen and field of view settings
    'SCREEN_SIZE': (1280, 720),  # Width and height in pixels
    'FOV_ANGLE': math.radians(75),  # Field of View angle in radians - used for ray direction calculation in rendering
    'STEP_VARIABLE_FLOAT': 0.1,  # Step size for floating movement (e.g., forward/backward motion)
    'STEP_VARIABLE_ROTATION': 5.0,  # Rotation step in degrees - used when rotating the view or object

    # UI (User Interface) constants
    'PANEL_WIDTH_RATIO': 0.2,  # Ratio of left/right panel width relative to total window width (e.g., 20 % of screen width)
    'FPS_WINDOW_OFFSET': 25,  # Vertical offset from the top edge of the screen for the FPS display window
    'FPS_WINDOW_WIDTH': 140,  # Width of the FPS display window in pixels
    'FPS_WINDOW_HEIGHT': 30,  # Height of the FPS display window in pixels

    'ORI_WINDOW_OFFSET': 60,  # Vertical offset from the top edge for the Orientation display window
    'ORI_WINDOW_WIDTH': 70,  # Width of the Orientation display window in pixels
    'ORI_WINDOW_HEIGHT': 110,  # Height of the Orientation display window in pixels

    # Camera control constants
    'MOUSE_SENSITIVITY': 0.005,  # Sensitivity factor for mouse movement - affects how fast the camera rotates with mouse input
    'PAN_SENSITIVITY': 0.1,  # Sensitivity for panning movements (e.g., dragging the view horizontally/vertically)
    'CAMERA_LERP_FACTOR': 7.5,  # Interpolation factor for smooth camera movement (LERP = Linear Interpolation)
    'ZOOM_SENSITIVITY': 0.5,  # Sensitivity for zooming in/out - controls how much zoom changes per input unit
    'MIN_RADIUS': 1.0,  # Minimum distance (radius) from camera to target point
    'MAX_RADIUS': 100.0,  # Maximum distance (radius) from camera to target point
    'MIN_PITCH': -math.radians(90),  # Minimum allowed pitch angle in radians (−90° - looking straight down)
    'MAX_PITCH': math.radians(90)  # Maximum allowed pitch angle in radians (+90° - looking straight up)
}
