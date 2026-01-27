import numpy as np
from skimage.measure import marching_cubes
import os
from typing import Tuple, Optional

def load_sdf_volume(filename: str, grid_size: int = 32) -> np.ndarray:
    """
    Loads a 3D SDF volume from a binary .bin file.
    
    Args:
        filename (str): Path to the binary file (e.g., "sdf_texture.bin")
        grid_size (int): Resolution of the 3D grid (assumed cubic: grid_size^3)
    
    Returns:
        np.ndarray: 3D array of shape (grid_size, grid_size, grid_size) with dtype float32
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If file size does not match expected bytes
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    expected_bytes = grid_size ** 3 * 4  # 4 bytes per float32
    file_size = os.path.getsize(filename)
    
    if file_size != expected_bytes:
        raise ValueError(
            f"File size mismatch: expected {expected_bytes} bytes, got {file_size} bytes. "
            "Check grid_size or file integrity."
        )

    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape((grid_size, grid_size, grid_size))



def save_isosurface_to_obj(
    volume: np.ndarray,
    level: float = 0.0,
    world_min: Tuple[float, float, float] = (-16.0, -16.0, -16.0),
    world_max: Tuple[float, float, float] = (16.0, 16.0, 16.0),
    output_path: str = "sdf_surface.obj"
) -> None:
    """
    Extracts an isosurface from the SDF volume using Marching Cubes and saves it as an OBJ file.
    
    Args:
        volume (np.ndarray): 3D SDF array of shape (N, N, N)
        level (float): Isovalue to extract (typically 0.0 for surface)
        world_min (tuple): Min bounds (x, y, z) of the volume in world coordinates
        world_max (tuple): Max bounds (x, y, z) of the volume in world coordinates
        output_path (str): Path to save the .obj file
    
    Raises:
        RuntimeError: If Marching Cubes fails (e.g., no surface at given level)
    """
    grid_size = volume.shape[0]

    # Compute spacing between voxels in each dimension
    spacing = (
        (world_max[0] - world_min[0]) / (grid_size - 1),
        (world_max[1] - world_min[1]) / (grid_size - 1),
        (world_max[2] - world_min[2]) / (grid_size - 1)
    )

    try:
        # Run Marching Cubes to extract triangles
        vertices, faces, _, _ = marching_cubes(volume, level=level, spacing=spacing)
    except ValueError as e:
        raise RuntimeError(
            f"Marching Cubes failed at level {level}. "
            "Possible reasons: no surface at this level, or invalid volume."
        ) from e

    # Convert voxel indices to world coordinates
    vertices[:, 0] += world_min[0]
    vertices[:, 1] += world_min[1]
    vertices[:, 2] += world_min[2]

    # Write to OBJ file
    with open(output_path, "w") as f:
        # Write vertices (OBJ uses "v x y z")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ uses 1-based indexing; "f v1 v2 v3")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"✅ Isosurface saved to: {output_path}")
    print(f"   - Isovalue: {level}")
    print(f"   - Vertex count: {len(vertices)}")
    print(f"   - Face count: {len(faces)}")

def main():
    """
    Main function to load SDF volume and export isosurface as OBJ.
    Configure parameters here.
    """
    # Input/Output parameters
    INPUT_FILE = "sdf_texture.bin"
    GRID_SIZE = 32
    WORLD_MIN = (-16.0, -16.0, -16.0)
    WORLD_MAX = (16.0, 16.0, 16.0)
    ISO_LEVEL = 0.0  # SDF=0 defines the surface
    OUTPUT_OBJ = "sdf_surface.obj"

    try:
        print("Loading SDF volume...")
        sdf_volume = load_sdf_volume(INPUT_FILE, GRID_SIZE)

        print("Extracting isosurface...")
        save_isosurface_to_obj(
            volume=sdf_volume,
            level=ISO_LEVEL,
            world_min=WORLD_MIN,
            world_max=WORLD_MAX,
            output_path=OUTPUT_OBJ
        )

    except Exception as e:
        print("f!❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
