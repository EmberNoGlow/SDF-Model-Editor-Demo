import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from skimage import measure
import os
import glfw  # Using GLFW for context management
import time

# Vertex shader: pass-through for full-screen quad
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

# Fragment shader: compute 3D SDF at gl_FragCoord.xy
fragment_shader_template = """
#version 330 core
out float fragDistance;

uniform vec3 worldMin;
uniform vec3 worldMax;
uniform float zCoord;  // Slice z in [0,1]
uniform vec2 viewportSize;

vec3 mixColorSmooth(vec3 colA, vec3 colB, float dA, float dB, float k) {
    k *= 4.0;
    float h = max(k - abs(dA - dB), 0.0) / k;
    float t = clamp(0.5 + 0.5 * (dB - dA) / k, 0.0, 1.0);
    vec3 blended = mix(colA, colB, t);
    vec3 closer = (dA < dB) ? colA : colB;
    return mix(closer, blended, h);
}


{SDF_LIBRARY}


vec4 map(vec3 p) {{
{SCENE_CODE}
}}

void main() {{
    // Normalize screen coordinates to [0,1]
    vec2 uv = gl_FragCoord.xy / viewportSize;
    
    // Interpolate 3D point: uv.x, uv.y → X,Y; zCoord → Z
    vec3 p = mix(worldMin, worldMax, vec3(uv.x, uv.y, zCoord));
    
    fragDistance = map(p).w;
}}
"""

def initialize_headless_context(width, height):
    """Initialize an OpenGL context using GLFW that is NOT visible."""
    
    # 1. Initialize GLFW (If the main script hasn't done so, this is necessary)
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed.")
        
    # 2. Set context hints for headless operation
    # Crucial: This prevents GLFW from trying to show a window on screen.
    glfw.window_hint(glfw.VISIBLE, False)
    
    # Request an OpenGL version compatible with your shaders (330 core)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    # 3. Create Window (This creates the context bound to this window handle)
    window = glfw.create_window(width, height, "Headless SDF Renderer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window/context creation failed.")
    
    # 4. Bind the context to the current thread (Crucial step)
    glfw.make_context_current(window)
    
    # --- VAO/VBO Setup (Full-screen quad) ---
    
    # Full-screen quad vertices
    vertices = np.array([
        -1.0, -1.0, 0.0,
         1.0, -1.0, 0.0,
        -1.0,  1.0, 0.0,
         1.0,  1.0, 0.0
    ], dtype=np.float32)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    return VAO, VBO, window

def cleanup_context(VAO, shader, VBO, window):
    """Safely clean up resources and destroy the temporary GLFW context."""
    
    # Ensure all commands are executed before tearing down
    glFinish()
    
    # Delete OpenGL Resources
    glDeleteVertexArrays(1, [VAO])
    glDeleteProgram(shader)
    glDeleteBuffers(1, [VBO])
    
    # Destroy the context window
    glfw.destroy_window(window)
    
    # WARNING: Do NOT call glfw.terminate() here, as the main application relies on it.

def compute_sdf_3d(grid_size=32, quality = 1.0, scene_code="return vec4(vec3(0.0), 100.0);", 
                   main_window_handle=None, sdf_library_path="shaders/sdf_library.glsl"):
    
    # 1. Load Library Code (Assuming external file reading is acceptable here)
    try:
        with open(sdf_library_path, 'r') as f:
            sdf_library_code = f.read()
    except FileNotFoundError:
        print(f"Warning: Could not find {sdf_library_path}. Using dummy content.")
        sdf_library_code = "// Dummy SDF Library Content"
        
    # Inject scene code into the template
    final_fragment_shader = fragment_shader_template.replace("{SDF_LIBRARY}", sdf_library_code)
    final_fragment_shader = final_fragment_shader.replace("{SCENE_CODE}", scene_code)

    # World bounds setup (Context Independent)
    hgs_base = grid_size // 2
    world_min = (-hgs_base, -hgs_base, -hgs_base)
    world_max = ( hgs_base,  hgs_base,  hgs_base)

    render_dim = int(grid_size * quality)
    final_grid_size = render_dim
    
    VAO, VBO, temp_window = None, None, None # Renamed temporary window to avoid confusion
    shader = None
    
    try:
        # 2. Initialize Headless Context (Context B)
        VAO, VBO, temp_window = initialize_headless_context(render_dim, render_dim)
        
        # Compile shaders using the context we just bound
        shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(final_fragment_shader, GL_FRAGMENT_SHADER)
        )
        glUseProgram(shader)

        # 3. Set Uniforms
        world_min_loc = glGetUniformLocation(shader, "worldMin")
        world_max_loc = glGetUniformLocation(shader, "worldMax")
        viewport_size_loc = glGetUniformLocation(shader, "viewportSize")
        
        glUniform3f(world_min_loc, *world_min)
        glUniform3f(world_max_loc, *world_max)
        glUniform2f(viewport_size_loc, float(render_dim), float(render_dim))

        distance_array = np.zeros((final_grid_size, final_grid_size, final_grid_size), dtype=np.float32)
        z_coord_loc = glGetUniformLocation(shader, "zCoord")

        # 4. Loop over Z-slices (All drawing happens within Context B)
        for z_idx in range(final_grid_size):
            
            # FBO and Texture Setup for this slice
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, render_dim, render_dim, 0, GL_RED, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("FBO is not complete!")

            # Set z-coordinate
            z_coord = (z_idx / (final_grid_size - 1)) if final_grid_size > 1 else 0.5
            glUniform1f(z_coord_loc, z_coord)

            # Render
            glViewport(0, 0, render_dim, render_dim)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Draw the full-screen quad (Uses VAO/VBO set up during context creation)
            glBindVertexArray(VAO) 
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            # Read pixel data (Reads from the texture attached to the FBO)
            high_res_data = glReadPixels(0, 0, render_dim, render_dim, GL_RED, GL_FLOAT)
            high_res_slice = np.frombuffer(high_res_data, dtype=np.float32).reshape((render_dim, render_dim))

            distance_array[:, :, z_idx] = high_res_slice

            # Clean up FBO/Texture for the next slice
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [texture])
            
    except Exception as e:
        print(f"An error occurred during headless computation: {e}")
        raise
        
    finally:
        # 5. Cleanup Context B
        if temp_window is not None:
            # Destroy context B resources
            cleanup_context(VAO, shader, VBO, temp_window)
            
        # 6. CRITICAL STEP: Re-bind Context A (the main application context)
        if main_window_handle:
            # We use the handle from the main application to make its context current again.
            # We assume the main application's context object is still valid.
            glfw.make_context_current(main_window_handle)
            glFinish() # Ensure synchronization after context switch
            
    return distance_array


def save_3d_texture(array, filename="sdf_texture.bin"):
    """Save 3D numpy array as a binary file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    # Save as raw binary data
    with open(filename, 'wb') as f:
        f.write(array.tobytes())

    print(f"Saved 3D texture to {filename}")
    print(f"Shape: {array.shape}, dtype: {array.dtype}")
    print(f"Size: {array.nbytes / 1024:.2f} KB")


def export_to_obj(sdf_array: np.ndarray, filename: str, level: float = 0.0, scale: float = 1.0, offset: tuple = (0.0, 0.0, 0.0)):

    if sdf_array.dtype != np.float32:
        print(f"Warning: Input array is not float32. Converting from {sdf_array.dtype}.")
        sdf_array = sdf_array.astype(np.float32)

    grid_size = sdf_array.shape[0]
    
    print(f"Starting marching cubes extraction on array shape: {sdf_array.shape} at level {level}...")

    try:
        # Extract vertices and faces using marching cubes
        vertices, faces, normals, values = measure.marching_cubes(
            sdf_array, 
            level=level, 
            spacing=(1.0, 1.0, 1.0)
        )
        
        print(f"Marching cubes generated {len(vertices)} vertices and {len(faces)} faces.")

    except ValueError as e:
        print(f"Error during marching cubes execution: {e}")
        return

    # --- Apply transformations ---

    # 1. Centering: Shift vertices so that the center of the grid (N-1)/2 becomes (0, 0, 0)
    center_shift = (grid_size - 1) / 2.0
    vertices[:, 0] -= center_shift
    vertices[:, 1] -= center_shift
    vertices[:, 2] -= center_shift

    # 2. Apply scaling
    if scale != 1.0:
        vertices *= scale

    # 3. Apply final offset
    if offset != (0.0, 0.0, 0.0):
        offset_array = np.array(offset, dtype=np.float32)
        vertices += offset_array

    # --- Write to OBJ File ---
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write("# OBJ file generated from SDF marching cubes\n")
        
        # Write Vertices (v)
        for v in vertices:
            # OBJ format: v x y z
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write Normals (vn)
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
        # Write Faces (f)
        for face in faces:
            v1_idx, v2_idx, v3_idx = face + 1
            f.write(f"f {v1_idx}//{v1_idx} {v2_idx}//{v2_idx} {v3_idx}//{v3_idx}\n")

    print(f"Successfully exported mesh to {filename}")


# Helper function for previewing the size of the resulting bin file
def calculate_sdf_file_size(grid_size=32, quality=1.0):
    final_grid_size = int(grid_size * quality)
    total_voxels = final_grid_size ** 3
    bytes_per_voxel = 4  # float32
    total_size_bytes = total_voxels * bytes_per_voxel

    # Format
    total_size_kb = total_size_bytes / 1024
    total_size_mb = total_size_kb / 1024
    
    return total_size_kb, total_size_mb