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
# --- Replace the fragment_shader_template with this updated template ---
fragment_shader_template = """
#version 330 core
out vec4 fragColor;      // RGBA output: rgb = color, a = distance (or pack distance in .a)
uniform vec3 worldMin;
uniform vec3 worldMax;
uniform float zCoord;  // Slice z in [0,1]
uniform vec2 viewportSize;
uniform bool useColor; // toggle color output

vec3 mixColorSmooth(vec3 colA, vec3 colB, float dA, float dB, float k) {
    k *= 4.0;
    float h = max(k - abs(dA - dB), 0.0) / k;
    float t = clamp(0.5 + 0.5 * (dB - dA) / k, 0.0, 1.0);
    vec3 blended = mix(colA, colB, t);
    vec3 closer = (dA < dB) ? colA : colB;
    return mix(closer, blended, h);
}

{SDF_LIBRARY}

vec4 getSceneDist(vec3 p)
{
    {SCENE_CODE}
}

vec4 map(vec3 p) {
    vec4 sceneRes = getSceneDist(p);
    {ADDITIONAL_SCENE_CODE}
    return sceneRes;
}

void main() {
    // Normalize screen coordinates to [0,1]
    vec2 uv = gl_FragCoord.xy / viewportSize;
    // Interpolate 3D point: uv.x, uv.y → X,Y; zCoord → Z
    vec3 p = mix(worldMin, worldMax, vec3(uv.x, uv.y, zCoord));
    vec4 res = map(p); // res.xyz = color (if provided), res.w = distance

    // If color is requested and scene provides color in res.xyz, output it.
    if (useColor) {
        // Convert distance to grayscale fallback if color is zero
        vec3 col = (length(res.xyz) > 0.0) ? res.xyz : vec3(clamp(1.0 - res.w * 0.01, 0.0, 1.0));
        fragColor = vec4(col, res.w);
    } else {
        // Output grayscale in rgb channels and distance in alpha
        float g = clamp(1.0 - res.w * 0.01, 0.0, 1.0);
        fragColor = vec4(vec3(g), res.w);
    }
}
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
    

def compute_sdf_3d(
    grid_size=32, quality = 1.0,
    scene_code="return vec4(vec3(0.0), 100.0);",
    additional_scene_code="",
    use_color: bool = False, main_window_handle=None,
    sdf_library_path="shaders/sdf_library.glsl" 
):
    # Load sdf library as before...
    try:
        with open(sdf_library_path, 'r') as f:
            sdf_library_code = f.read()
    except FileNotFoundError:
        print(f"Warning: Could not find {sdf_library_path}. Using dummy content.")
        sdf_library_code = "// Dummy SDF Library Content"

    # Inject scene code into the template
    final_fragment_shader = fragment_shader_template.replace("{SDF_LIBRARY}", sdf_library_code)
    final_fragment_shader = final_fragment_shader.replace("{SCENE_CODE}", scene_code)
    final_fragment_shader = final_fragment_shader.replace("{ADDITIONAL_SCENE_CODE}", additional_scene_code)

    # World bounds setup
    hgs_base = grid_size // 2
    world_min = (-hgs_base, -hgs_base, -hgs_base)
    world_max = ( hgs_base,  hgs_base,  hgs_base)

    render_dim = int(grid_size * quality)
    final_grid_size = render_dim

    VAO, VBO, temp_window = None, None, None
    shader = None

    try:
        VAO, VBO, temp_window = initialize_headless_context(render_dim, render_dim)

        # Compile shaders
        shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(final_fragment_shader, GL_FRAGMENT_SHADER)
        )
        glUseProgram(shader)

        # Uniforms
        world_min_loc = glGetUniformLocation(shader, "worldMin")
        world_max_loc = glGetUniformLocation(shader, "worldMax")
        viewport_size_loc = glGetUniformLocation(shader, "viewportSize")
        use_color_loc = glGetUniformLocation(shader, "useColor")

        glUniform3f(world_min_loc, *world_min)
        glUniform3f(world_max_loc, *world_max)
        glUniform2f(viewport_size_loc, float(render_dim), float(render_dim))
        glUniform1i(use_color_loc, int(use_color))

        distance_array = np.zeros((final_grid_size, final_grid_size, final_grid_size), dtype=np.float32)
        color_volume = None
        if use_color:
            color_volume = np.zeros((final_grid_size, final_grid_size, final_grid_size, 3), dtype=np.float32)

        z_coord_loc = glGetUniformLocation(shader, "zCoord")

        for z_idx in range(final_grid_size):
            # Create texture format depending on color flag
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            if use_color:
                # RGBA float texture: rgb=color, a=distance
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, render_dim, render_dim, 0, GL_RGBA, GL_FLOAT, None)
            else:
                # Keep single-channel path compatible with previous behavior by still using RGBA32F
                # but we will read only the red channel (grayscale)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, render_dim, render_dim, 0, GL_RGBA, GL_FLOAT, None)

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
            glBindVertexArray(VAO)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

            # Read RGBA float pixels
            raw = glReadPixels(0, 0, render_dim, render_dim, GL_RGBA, GL_FLOAT)
            slice_rgba = np.frombuffer(raw, dtype=np.float32).reshape((render_dim, render_dim, 4))

            # Extract distance (alpha channel) and store
            distance_slice = slice_rgba[:, :, 3]  # alpha stores distance in our shader
            distance_array[:, :, z_idx] = distance_slice

            if use_color:
                # Extract rgb and store (flip Y if needed depending on glReadPixels orientation)
                color_slice = slice_rgba[:, :, :3]
                color_volume[:, :, z_idx, :] = color_slice

            # Cleanup
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [texture])

    except Exception as e:
        print(f"An error occurred during headless computation: {e}")
        raise

    finally:
        if temp_window is not None:
            cleanup_context(VAO, shader, VBO, temp_window)

        if main_window_handle:
            glfw.make_context_current(main_window_handle)
            glFinish()

    if use_color:
        return distance_array, color_volume
    else:
        return distance_array



def save_3d_texture(array, filename="sdf_texture.bin"):
    """Save 3D numpy array as a binary file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    # If array is turple(dist, color)
    if isinstance(array, tuple):
        distance_array, color_volume = array # unpack

        # Add distance as 4th channel 
        distance_expanded = distance_array[..., None] # (N, N, N, 1)
        rgba_volume = np.concatenate([color_volume, distance_expanded], axis=3) # (N, N, N, 4)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(rgba_volume.astype(np.float32).tobytes())

        print(f"Saved 3D texture to {filename}")
        print(f"Size: {rgba_volume.nbytes / 1024:.2f} KB")

        return
        

    # Save as raw binary data
    with open(filename, 'wb') as f:
        f.write(array.tobytes())

    print(f"Saved 3D texture to {filename}")
    print(f"Shape: {array.shape}, dtype: {array.dtype}")
    print(f"Size: {array.nbytes / 1024:.2f} KB")



import os
from typing import List, Tuple, Optional

def save_obj_with_mtl(
    obj_path: str,
    verts: List[Tuple[float, float, float]],
    faces: List[Tuple[int, int, int]],
    face_colors: Optional[List[Tuple[float, float, float]]] = None,
    vertex_colors: Optional[List[Tuple[float, float, float]]] = None,
    write_vertex_color_extension: bool = False
) -> Tuple[bool, str]:
    """
    Save an OBJ file with an accompanying MTL file to preserve colors.

    Parameters
    - obj_path: full path to .obj file to write (will create .mtl alongside it)
    - verts: list of (x,y,z) vertex tuples
    - faces: list of (i0,i1,i2) indices (0-based)
    - face_colors: optional list of (r,g,b) per-face colors (same length as faces).
                   Colors can be in 0..1 or 0..255 range.
    - vertex_colors: optional list of (r,g,b) per-vertex colors (same length as verts).
                      If both face_colors and vertex_colors are provided, face_colors takes precedence.
    - write_vertex_color_extension: if True and vertex_colors provided, write vertex lines as:
                      v x y z r g b  (non-standard but supported by many tools)

    Returns (success: bool, message: str)
    """
    try:
        if not obj_path.lower().endswith('.obj'):
            return False, "obj_path must end with .obj"

        base_dir = os.path.dirname(obj_path) or '.'
        base_name = os.path.splitext(os.path.basename(obj_path))[0]
        mtl_name = base_name + '.mtl'
        mtl_path = os.path.join(base_dir, mtl_name)

        # Normalize color helper
        def normalize_color(c):
            r, g, b = c
            # If values appear in 0..255, convert to 0..1
            if max(r, g, b) > 1.0:
                return (r / 255.0, g / 255.0, b / 255.0)
            return (float(r), float(g), float(b))

        # Build mapping from color -> material name
        color_to_mat = {}
        mat_list = []  # list of (mat_name, (r,g,b))

        # If face_colors provided, ensure length matches faces
        if face_colors is not None:
            if len(face_colors) != len(faces):
                return False, "face_colors length must match faces length"
            # Map unique colors
            for col in face_colors:
                coln = normalize_color(col)
                if coln not in color_to_mat:
                    mat_name = f"mat{len(color_to_mat)}"
                    color_to_mat[coln] = mat_name
                    mat_list.append((mat_name, coln))

        # If only vertex colors provided and user wants materials, derive per-face color by averaging vertex colors
        elif vertex_colors is not None and len(vertex_colors) == len(verts):
            # compute per-face average color
            face_colors = []
            for f in faces:
                c0 = normalize_color(vertex_colors[f[0]])
                c1 = normalize_color(vertex_colors[f[1]])
                c2 = normalize_color(vertex_colors[f[2]])
                avg = ((c0[0]+c1[0]+c2[0]) / 3.0, (c0[1]+c1[1]+c2[1]) / 3.0, (c0[2]+c1[2]+c2[2]) / 3.0)
                face_colors.append(avg)
            for coln in face_colors:
                if coln not in color_to_mat:
                    mat_name = f"mat{len(color_to_mat)}"
                    color_to_mat[coln] = mat_name
                    mat_list.append((mat_name, coln))

        # Write MTL file (if we have materials)
        if mat_list:
            with open(mtl_path, 'w', encoding='utf-8') as mtl_f:
                for mat_name, (r, g, b) in mat_list:
                    mtl_f.write(f"newmtl {mat_name}\n")
                    mtl_f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
                    mtl_f.write("Ka 0.000000 0.000000 0.000000\n")
                    mtl_f.write("Ks 0.000000 0.000000 0.000000\n")
                    mtl_f.write("d 1.0\n\n")

        # Write OBJ file
        with open(obj_path, 'w', encoding='utf-8') as obj_f:
            if mat_list:
                obj_f.write(f"mtllib {mtl_name}\n")

            # Write vertices (optionally with vertex colors)
            if vertex_colors and write_vertex_color_extension:
                # Normalize vertex colors
                for v, vc in zip(verts, vertex_colors):
                    r, g, b = normalize_color(vc)
                    obj_f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            else:
                for v in verts:
                    obj_f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Optionally write vertex normals or texcoords here if available

            # If we have materials, write faces grouped with usemtl
            if mat_list and face_colors is not None:
                # Iterate faces and write usemtl before each face if material changes
                last_mat = None
                for fi, f in enumerate(faces):
                    coln = normalize_color(face_colors[fi])
                    mat_name = color_to_mat.get(coln)
                    if mat_name != last_mat:
                        obj_f.write(f"usemtl {mat_name}\n")
                        last_mat = mat_name
                    # OBJ indices are 1-based
                    obj_f.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
            else:
                # No materials: just write faces
                for f in faces:
                    obj_f.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

        return True, f"Wrote OBJ: {obj_path} and MTL: {mtl_path if mat_list else '(no mtl)'}"

    except Exception as e:
        return False, f"Error writing OBJ/MTL: {e}"



def _trilinear_sample(volume: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Trilinear sample `volume` at fractional voxel coordinates `points`.
    - volume: shape (N, N, N) or (N, N, N, C)
    - points: shape (M, 3) with coordinates in voxel index space (0..N-1)
    Returns: sampled values shape (M, C) or (M,) for scalar volume.
    """
    if points.size == 0:
        return np.zeros((0, volume.shape[3] if volume.ndim == 4 else 1))

    N0, N1, N2 = volume.shape[0], volume.shape[1], volume.shape[2]
    pts = np.asarray(points, dtype=np.float32)

    # Clamp coordinates to valid range [0, N-1]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, N0 - 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, N1 - 1.0)
    pts[:, 2] = np.clip(pts[:, 2], 0.0, N2 - 1.0)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, N0 - 1)
    y1 = np.clip(y0 + 1, 0, N1 - 1)
    z1 = np.clip(z0 + 1, 0, N2 - 1)

    xd = (x - x0).reshape(-1, 1)
    yd = (y - y0).reshape(-1, 1)
    zd = (z - z0).reshape(-1, 1)

    # Helper to gather values
    def gather(ix, iy, iz):
        if volume.ndim == 3:
            return volume[ix, iy, iz].reshape(-1, 1)
        else:
            # volume.ndim == 4, channels last
            return volume[ix, iy, iz, :]

    c000 = gather(x0, y0, z0)
    c100 = gather(x1, y0, z0)
    c010 = gather(x0, y1, z0)
    c110 = gather(x1, y1, z0)
    c001 = gather(x0, y0, z1)
    c101 = gather(x1, y0, z1)
    c011 = gather(x0, y1, z1)
    c111 = gather(x1, y1, z1)

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c  # shape (M,1) or (M,C)


def export_to_obj(
    sdf_array: np.ndarray,
    color_volume: np.ndarray,
    filename: str,
    Z_UP: bool = True,
    level: float = 0.0,
    use_color: bool = False,
    scale: float = 1.0,
    offset: tuple = (0.0, 0.0, 0.0),
) -> Tuple[bool, str]:
    """
    Export mesh from SDF volume using marching cubes. Optionally export colors.

    Parameters:
    - sdf_array: 3D numpy float32 SDF volume (shape N,N,N)
    - filename: output .obj path
    - Z_UP: whether to output Z-up coordinates (True) or apply alternate swap
    - level: iso-level for marching cubes
    - use_color: if True, attempt to produce colors and call save_obj_with_mtl
    - scale: uniform scale applied to vertices after centering
    - offset: final translation applied to vertices after scaling
    - color_volume: optional color volume with shape (N,N,N,3) or (N,N,N) to sample colors from.
                    If None and use_color=True, grayscale colors are derived from sdf_array values.

    Returns (success: bool, message: str)
    """
    try:
        if sdf_array.dtype != np.float32:
            print(f"Warning: Input array is not float32. Converting from {sdf_array.dtype}.")
            sdf_array = sdf_array.astype(np.float32)

        grid_size = sdf_array.shape[0]
        print(f"Starting marching cubes extraction on array shape: {sdf_array.shape} at level {level}...")

        # Extract vertices and faces using marching cubes
        vertices, faces, normals, values = measure.marching_cubes(
            sdf_array,
            level=level,
            spacing=(1.0, 1.0, 1.0)
        )

        print(f"Marching cubes generated {len(vertices)} vertices and {len(faces)} faces.")

    except ValueError as e:
        print(f"Error during marching cubes execution: {e}")
        min_val = sdf_array.min()
        max_val = sdf_array.max()
        print(f"volume data min: {min_val}, max: {max_val}")
        return False, f"Error: {e}"

    # Keep a copy of raw vertex positions in voxel index space for sampling colors
    raw_vertices = vertices.copy()  # shape (M,3) in voxel coordinates

    # --- Apply transformations (center, scale, axis swap, offset) ---
    center_shift = (grid_size - 1) / 2.0

    # Centering (apply to vertices and normals)
    vertices[:, 0] -= center_shift
    vertices[:, 1] -= center_shift
    vertices[:, 2] -= center_shift

    # Apply scaling
    if scale != 1.0:
        vertices *= scale

    # Axis reorientation
    if not Z_UP:
        # Convert to Y-Up or other convention as in your original code
        vertices = vertices[:, [2, 1, 0]]
        normals = normals[:, [2, 1, 0]]
    else:
        vertices = vertices[:, [1, 0, 2]]
        normals = normals[:, [1, 0, 2]]

    # Apply final offset
    if offset != (0.0, 0.0, 0.0):
        offset_array = np.array(offset, dtype=np.float32)
        vertices += offset_array

    # --- Write to OBJ File (base) ---
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    # If use_color is requested, prepare colors before writing OBJ and call save_obj_with_mtl
    face_colors = None
    vertex_colors = None

    if use_color:
        # Determine sampling source: color_volume if provided, else derive from sdf_array
        if color_volume is not None:
            vol = color_volume
            # Validate shape
            if vol.shape[0:3] != sdf_array.shape:
                print("Warning: color_volume shape does not match sdf_array shape. Ignoring color_volume.")
                vol = None
        else:
            vol = None

        # If no explicit color volume, use sdf_array to derive grayscale colors
        if vol is None:
            # Normalize sdf values to 0..1 for coloring (clamp)
            vmin = float(np.min(sdf_array))
            vmax = float(np.max(sdf_array))
            if vmax - vmin < 1e-6:
                norm_sdf = np.zeros_like(sdf_array, dtype=np.float32)
            else:
                norm_sdf = (sdf_array - vmin) / (vmax - vmin)
            # Use normalized SDF as single-channel volume for sampling
            vol = norm_sdf  # shape (N,N,N)

        # Sample colors at raw vertex positions (raw_vertices are in voxel index space)
        # marching_cubes returns coordinates in voxel index space already, so use raw_vertices directly.
        sampled = _trilinear_sample(vol, raw_vertices)  # shape (M,1) or (M,3)

        # Normalize sampled to RGB tuples in 0..1
        if sampled.ndim == 2 and sampled.shape[1] == 1:
            # grayscale -> replicate to RGB
            sampled_rgb = np.repeat(sampled, 3, axis=1)
        else:
            sampled_rgb = sampled.astype(np.float32)
            if sampled_rgb.shape[1] == 4:
                # If color volume had alpha, drop it
                sampled_rgb = sampled_rgb[:, :3]

        # Clamp to 0..1
        sampled_rgb = np.clip(sampled_rgb, 0.0, 1.0)

        # Save per-vertex colors (in same order as vertices)
        vertex_colors = [tuple(map(float, sampled_rgb[i])) for i in range(sampled_rgb.shape[0])]

        # Compute per-face colors by averaging the three vertex colors for each face
        face_colors = []
        for f in faces:
            c0 = sampled_rgb[f[0]]
            c1 = sampled_rgb[f[1]]
            c2 = sampled_rgb[f[2]]
            avg = ((c0[0] + c1[0] + c2[0]) / 3.0, (c0[1] + c1[1] + c2[1]) / 3.0, (c0[2] + c1[2] + c2[2]) / 3.0)
            face_colors.append(avg)

        # Now call save_obj_with_mtl to write OBJ + MTL grouped by color
        try:
            ok, msg = save_obj_with_mtl(
                obj_path=filename,
                verts=[(float(v[0]), float(v[1]), float(v[2])) for v in vertices],
                faces=[(int(f[0]), int(f[1]), int(f[2])) for f in faces],
                face_colors=face_colors,
                vertex_colors=vertex_colors,
                write_vertex_color_extension=False
            )
            if ok:
                print(f"Successfully exported mesh with colors to {filename}")
                return True, msg
            else:
                # Fall back to plain OBJ if MTL writer failed
                print(f"save_obj_with_mtl failed: {msg}. Falling back to plain OBJ.")
        except Exception as e:
            print(f"Exception while writing OBJ+MTL: {e}. Falling back to plain OBJ.")

    # --- Plain OBJ writer (no colors or fallback) ---
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# OBJ file generated from SDF marching cubes\n")

        # Write Vertices (v)
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write Normals (vn)
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # Write Faces (f) with vertex//normal indices
        for face in faces:
            v1_idx, v2_idx, v3_idx = face + 1
            f.write(f"f {v1_idx}//{v1_idx} {v2_idx}//{v2_idx} {v3_idx}//{v3_idx}\n")

    print(f"Successfully exported mesh to {filename}")
    return True, f"File saved successfully!"





# Helper function for previewing the size of the resulting bin file
def calculate_sdf_file_size(grid_size=32, quality=1.0, use_color=False):
    final_grid_size = int(grid_size * quality)
    total_voxels = final_grid_size ** 3

    # 1 float32 per voxel (distance) or 4 float32 per voxel (RGBA)
    channels = 4 if use_color else 1
    bytes_per_voxel = 4 * channels  # float32 = 4 bytes

    total_size_bytes = total_voxels * bytes_per_voxel
    total_size_kb = total_size_bytes / 1024
    total_size_mb = total_size_kb / 1024

    return total_size_kb, total_size_mb
