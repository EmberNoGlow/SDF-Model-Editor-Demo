import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileProgram, compileShader
import os
import argparse

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


{SDF_LIBRARY}

float map(vec3 p) {
{SCENE_CODE}
}

void main() {
    // Normalize screen coordinates to [0,1]
    vec2 uv = gl_FragCoord.xy / viewportSize;
    
    // Interpolate 3D point: uv.x, uv.y → X,Y; zCoord → Z
    vec3 p = mix(worldMin, worldMax, vec3(uv.x, uv.y, zCoord));
    
    fragDistance = map(p);
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


def compute_sdf_3d(grid_size=32, quality = 1.0, scene_code="return 100.0;"):
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




distance_array = compute_sdf_3d(32, 2.0, "return min(sdBox( p, vec3(1.0) ), sdSphere(p, 1.0) );")
save_3d_texture(distance_array)