import tkinter as tk
from tkinter import filedialog, messagebox


def save_scene_dialog(scene_builder, parent_window=None):
    # Open a save dialog and save the scene to JSON.
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="scene.json",
        )

        if not filepath:
            return False, "Save cancelled"
        
        success, message = scene_builder.save_to_file(filepath)
        if not success:
            return False, f"Failed to save: {message}"
        return True, f"Scene saved to {filepath}"

    except Exception as e:
        error_msg = f"Error during save: {e}"
        return False, error_msg
    finally:
        root.destroy()


def load_scene_dialog(scene_builder):
    # Open a load dialog and load a scene from JSON.
    try:
        root = tk.Tk()
        root.withdraw()

        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not filepath:
            return False, "Load cancelled"

        success, message = scene_builder.load_from_file(filepath)
        if not success:
            return False, f"Failed to load: {message}"
        return True, f"Scene loaded from {filepath}"

    except Exception as e:
        error_msg = f"Error during load: {e}"
        return False, error_msg
    finally:
        root.destroy()


def save_sdfvol_dialog(sdfexp, data, parent_window=None):
    # Open a save dialog and save the scene to JSON.
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".bin",
            filetypes=[("binary files", "*.bin"), ("All files", "*.*")],
            initialfile="scene.bin",
        )
        
        if not filepath:
            return False, "Save cancelled"
        
        sdfexp.save_3d_texture(data, filepath)
        return True
    except Exception as e:
        error_msg = f"Error during saving: {e}"
        return False, error_msg
    finally:
        root.destroy()


def save_sdfobj_dialog(sdfexp, dist_sdf, color_sdf, export_z_up, export_level, exp_use_color, parent_window=None):
    # Open a save dialog and save the scene to JSON.
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".obj",
            filetypes=[("wavefront obj", "*.obj"), ("All files", "*.*")],
            initialfile="scene.obj",
        )
        
        
        if not filepath:
            return False, "Filepath is not valid"
        
        success, message = sdfexp.export_to_obj(dist_sdf, color_sdf, filepath, export_z_up, export_level, exp_use_color)
        return success, message

    except Exception as e:
        error_msg = f"Error during saving: {e}"
        return False, error_msg
    finally:
        root.destroy()


import glfw
import numpy as np

from OpenGL.GL import glReadBuffer, glReadPixels, GL_FRONT, GL_RGB, GL_UNSIGNED_BYTE
from PIL import Image

def take_screenshot(window):
    # Get window dimensions
    width, height = glfw.get_framebuffer_size(window)

    # Read pixels from the framebuffer
    glReadBuffer(GL_FRONT)  # Read from the front buffer
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    image = np.flipud(image)

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG Images", "*.png"),
            ("JPEG Images", "*.jpg;*.jpeg"),
            ("BMP Images", "*.bmp"),
            ("All Files", "*.*")
        ],
        initialfile="Screenshot.jpg"
    )

    root.destroy()

    # Save using Pillow
    if filepath:
        img = Image.fromarray(image, 'RGB')
        img.save(filepath)
