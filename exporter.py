import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileProgram, compileShader
from skimage import measure
import os


# Vertex shader: pass-through for full-screen quad
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

# Fragment shader: compute 3D SDF at gl_FragCoord.xy
fragment_shader = """
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


vec4 map(vec3 p) {
{SCENE_CODE}
}

void main() {
    // Normalize screen coordinates to [0,1]
    vec2 uv = gl_FragCoord.xy / viewportSize;
    
    // Interpolate 3D point: uv.x, uv.y → X,Y; zCoord → Z
    vec3 p = mix(worldMin, worldMax, vec3(uv.x, uv.y, zCoord));
    
    fragDistance = map(p).w;
}
"""

def init_opengl(width, height):
    """Initialize OpenGL context and shader program."""
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutCreateWindow(b"3D SDF Calculation")

    # Compile shaders and link program
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader)

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

    return VAO, shader


def compute_sdf_3d(grid_size=32, quality = 1.0, scene_code="return vec4(vec3(0.0), 100.0);"):
    # World bounds
    hgs_base = grid_size // 2
    world_min = (-hgs_base, -hgs_base, -hgs_base)
    world_max = ( hgs_base,  hgs_base,  hgs_base)

    # 2. Define Render Dimensions (Viewport Size) based on quality
    # If quality=1, render 32x32. If quality=2, render 64x64.
    render_dim = int(grid_size * quality)
    
    # 3. Define the Voxel Indexing Grid (This is the size of the output array)
    final_grid_size = render_dim 

    # Load and inject shader components
    from main import load_shader_code
    sdf_library = load_shader_code("shaders/sdf_library.glsl")
    
    global fragment_shader
    fragment_shader = fragment_shader.replace("{SDF_LIBRARY}", sdf_library)
    fragment_shader = fragment_shader.replace("{SCENE_CODE}", scene_code)

    # Initialize OpenGL using the high-resolution render dimensions
    VAO, shader = init_opengl(render_dim, render_dim)

    # Set world bounds (They are now fixed, not scaled by quality)
    world_min_loc = glGetUniformLocation(shader, "worldMin")
    world_max_loc = glGetUniformLocation(shader, "worldMax")
    viewport_size_loc = glGetUniformLocation(shader, "viewportSize")
    # We no longer need to pass 'quality' as a uniform affecting the distance map calculation
    
    glUniform3f(world_min_loc, *world_min)
    glUniform3f(world_max_loc, *world_max)
    glUniform2f(viewport_size_loc, float(render_dim), float(render_dim))

    # Create 3D array to store results (This should match the final desired storage size)
    distance_array = np.zeros((final_grid_size, final_grid_size, final_grid_size), dtype=np.float32)

    # Loop over Z-slices (We iterate based on the final storage size)
    for z_idx in range(final_grid_size):
        # Create FBO
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
        z_coord_loc = glGetUniformLocation(shader, "zCoord")
        glUniform1f(z_coord_loc, z_coord)


        # Read pixel data
        glViewport(0, 0, render_dim, render_dim)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # Read pixel data (This now correctly reads from the render_dim x render_dim texture)
        high_res_data = glReadPixels(0, 0, render_dim, render_dim, GL_RED, GL_FLOAT)
        high_res_slice = np.frombuffer(high_res_data, dtype=np.float32).reshape((render_dim, render_dim))

        if render_dim <= final_grid_size:
            downsampled_slice = high_res_slice
            
        distance_array[:, :, z_idx] = downsampled_slice # Stores render_dim x render_dim slice


        # Clean up
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [texture])

    # Clean up OpenGL objects
    glDeleteVertexArrays(1, [VAO])
    glDeleteProgram(shader)

    return distance_array



def save_3d_texture(array, filename="sdf_texture.bin"):
    """Save 3D numpy array as a binary file."""
    # Ensure directory exists

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
        # spacing=(1.0, 1.0, 1.0) assumes voxel dimensions are 1x1x1
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
            # Write normals directly. Dark shading usually implies normals are missing 
            # or incorrectly oriented relative to the face winding.
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
        # Write Faces (f)
        # Format: f v1//vn1 v2//vn2 v3//vn3 (We use v/vt/vn, assuming vt=empty)
        for face in faces:
            # Face indices are 0-indexed from marching_cubes, need +1 for OBJ format
            v1_idx, v2_idx, v3_idx = face + 1
            
            # Since marching_cubes outputs corresponding normals for vertices, we reuse the vertex index for the normal index
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
