import imgui
from PIL import Image
import numpy as np
from OpenGL.GL import *
import os

# Assume load_texture_from_image is defined as before (it's functional)
def load_texture_from_image(image_path):
    """Loads an image into an OpenGL texture and returns its ID, width, and height."""
    try:
        image = Image.open(image_path).convert("RGBA")
        width, height = image.size
        image_data = np.array(list(image.getdata()), np.uint8)

        # Create OpenGL texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Set wrapping/filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Upload data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        
        return texture_id, width, height
    
    except FileNotFoundError:
        print(f"Error: Texture file not found at {image_path}")
        return None, 0, 0
    except Exception as e:
        print(f"Error loading texture {image_path}: {e}")
        return None, 0, 0


def load_all_textures():
    """
    Loads all specified textures, handles duplicate filenames by appending a counter,
    and returns a dictionary mapping the unique ImageName to (texture_id, width, height).
    """
    # Use a list of paths that you want to load
    paths_to_load = [] # Add path to icons (in png format)
    
    loaded_textures = {}
    filename_counts = {} # To track duplicates

    for path in paths_to_load:
        # 1. Extract filename (ImageName) securely using os.path
        base_filename = os.path.basename(path).replace(".png", "") # Remove Extension
        
        # 2. Handle potential filename collisions
        if base_filename in filename_counts:
            filename_counts[base_filename] += 1
            # Create a unique key, e.g., "apple" -> "apple_1"
            name_key = f"{os.path.splitext(base_filename)[0]}_{filename_counts[base_filename]}{os.path.splitext(base_filename)[1]}"
        else:
            filename_counts[base_filename] = 0
            name_key = base_filename
            
        # 3. Load the texture (more secure by wrapping the load call)
        texture_info = load_texture_from_image(path)
        texture_id, width, height = texture_info

        if texture_id is not None:
            # 4. Store in the desired format: {ImageName: (texture_id, width, height)}
            loaded_textures[name_key] = (texture_id, width, height)
        else:
            print(f"Skipped loading texture from {path} due to previous error.")

    return loaded_textures
