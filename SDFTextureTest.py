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

// SDF for a sphere
float sphereSDF(vec3 p, vec3 center, float radius) {
    return length(p - center) - radius;
}

// SDF for an axis-aligned box
float boxSDF(vec3 p, vec3 boxMin, vec3 boxMax) {
    vec3 d = max(boxMin - p, p - boxMax);
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

uniform vec3 worldMin;
uniform vec3 worldMax;
uniform float zCoord;
uniform vec2 viewportSize;

void main() {
    // Map gl_FragCoord to 3D space
    vec2 uv = gl_FragCoord.xy / viewportSize;
    vec3 p = mix(worldMin, worldMax, vec3(uv, zCoord));

    // Define primitives
    float d1 = sphereSDF(p, vec3(0.0, 0.0, 0.0), 5.0);      // Sphere at origin, r=5
    float d2 = boxSDF(p, vec3(-4.0, -4.0, -4.0), vec3(4.0, 4.0, 4.0)); // Box from -4 to +4

    fragDistance = min(d1, d2);  // Combined SDF
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

def compute_sdf_3d(grid_size=32, world_min=(-16.0, -16.0, -16.0), world_max=(16.0, 16.0, 16.0)):
    """
    Compute SDF on a 3D grid of size grid_size^3.
    Args:
        grid_size: Size of the grid in each dimension (grid_size x grid_size x grid_size)
        world_min: Minimum world coordinates (x, y, z)
        world_max: Maximum world coordinates (x, y, z)
    Returns:
        3D numpy array of distance values
    """
    VAO, shader = init_opengl(grid_size, grid_size)

    # Set world bounds in shader
    world_min_loc = glGetUniformLocation(shader, "worldMin")
    world_max_loc = glGetUniformLocation(shader, "worldMax")
    viewport_size_loc = glGetUniformLocation(shader, "viewportSize")
    glUniform3f(world_min_loc, *world_min)
    glUniform3f(world_max_loc, *world_max)
    glUniform2f(viewport_size_loc, float(grid_size), float(grid_size))

    # Create 3D array to store results
    distance_array = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # Loop over z-slices
    for z_idx in range(grid_size):
        # Create FBO
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, grid_size, grid_size, 0, GL_RED, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("FBO is not complete!")

        # Set z-coordinate
        z_coord = z_idx / (grid_size - 1) if grid_size > 1 else 0.5
        z_coord_loc = glGetUniformLocation(shader, "zCoord")
        glUniform1f(z_coord_loc, z_coord)

        # Render
        glViewport(0, 0, grid_size, grid_size)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # Read pixel data
        data = glReadPixels(0, 0, grid_size, grid_size, GL_RED, GL_FLOAT)
        distance_array[:, :, z_idx] = np.frombuffer(data, dtype=np.float32).reshape((grid_size, grid_size))

        # Clean up
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [texture])

    # Clean up OpenGL objects
    glDeleteVertexArrays(1, [VAO])
    glDeleteProgram(shader)

    return distance_array

def save_3d_texture(array, filename):
    """Save 3D numpy array as a binary file."""
    # Ensure directory exists

    # Save as raw binary data
    with open(filename, 'wb') as f:
        f.write(array.tobytes())

    print(f"Saved 3D texture to {filename}")
    print(f"Shape: {array.shape}, dtype: {array.dtype}")
    print(f"Size: {array.nbytes / 1024:.2f} KB")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute 3D Signed Distance Fields")
    parser.add_argument("--grid_size", type=int, default=32,
                       help="Size of the grid in each dimension (default: 32)")
    parser.add_argument("--world_min", type=float, nargs=3, default=[-16.0, -16.0, -16.0],
                       help="Minimum world coordinates (x y z) (default: -16 -16 -16)")
    parser.add_argument("--world_max", type=float, nargs=3, default=[16.0, 16.0, 16.0],
                       help="Maximum world coordinates (x y z) (default: 16 16 16)")
    parser.add_argument("--output", type=str, default="sdf_texture.bin",
                       help="Output filename for the 3D texture (default: sdf_texture.bin)")
    parser.add_argument("--print_samples", action="store_true",
                       help="Print sample distance values")

    args = parser.parse_args()

    print(f"Computing SDF on {args.grid_size}x{args.grid_size}x{args.grid_size} grid...")
    print(f"World bounds: from {args.world_min} to {args.world_max}")

    # Compute SDF
    distance_array = compute_sdf_3d(
        grid_size=args.grid_size,
        world_min=args.world_min,
        world_max=args.world_max
    )

    # Save to disk
    save_3d_texture(distance_array, args.output)

    # Print samples if requested
    if args.print_samples:
        print("\nSample distance values:")
        step = max(1, args.grid_size // 4)
        for z in range(0, args.grid_size, step):
            for y in range(0, args.grid_size, step):
                for x in range(0, args.grid_size, step):
                    world_x = args.world_min[0] + (args.world_max[0] - args.world_min[0]) * (x / (args.grid_size - 1))
                    world_y = args.world_min[1] + (args.world_max[1] - args.world_min[1]) * (y / (args.grid_size - 1))
                    world_z = args.world_min[2] + (args.world_max[2] - args.world_min[2]) * (z / (args.grid_size - 1))
                    print(f"({world_x:6.2f}, {world_y:6.2f}, {world_z:6.2f}): {distance_array[x, y, z]:8.4f}")

    print(f"\nTotal points computed: {args.grid_size**3}")

if __name__ == "__main__":
    main()