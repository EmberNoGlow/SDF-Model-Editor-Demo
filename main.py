import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import math
import hashlib
import ctypes
import imgui
import imgui.core
import gui.themes
import exporter as sdfexp

from imgui.integrations.glfw import GlfwRenderer
from PIL import Image

import numpy as np
import math
import copy


def load_shader_code(file_path):
    """
    Load shader code from a file and return it as a string.

    Args:
        file_path (str): Path to the shader code file.
    Returns:
        str: Contents of the shader code file.
    Raises:
        FileNotFoundError: If the shader file cannot be found.
        IOError: If the shader file cannot be read.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Shader file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading shader file {file_path}: {e}")


# --- SaveLoad funcions
import tkinter as tk
from tkinter import filedialog, messagebox

def save_scene_dialog(scene_builder, parent_window=None):
    """Open a save dialog and save the scene to JSON."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    filepath = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile="scene.json"
    )
    
    root.destroy()
    
    if filepath:
        return scene_builder.save_to_json(filepath)
    return False, "Save cancelled"


def load_scene_dialog(scene_builder, parent_window=None):
    """Open a load dialog and load a scene from JSON."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    filepath = filedialog.askopenfilename(
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    root.destroy()
    
    if filepath:
        return scene_builder.load_from_json(filepath)
    return False, "Load cancelled"


def save_sdfvol_dialog(data, parent_window=None):
    """Open a save dialog and save the scene to JSON."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    filepath = filedialog.asksaveasfilename(
        defaultextension=".bin",
        filetypes=[("binary files", "*.bin"), ("All files", "*.*")],
        initialfile="scene.bin"
    )
    
    root.destroy()
    
    if filepath:
        sdfexp.save_3d_texture(data, filepath)
        return True
    return False


def save_sdfobj_dialog(data, export_z_up, export_level = 0.0, parent_window=None):
    """Open a save dialog and save the scene to JSON."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    filepath = filedialog.asksaveasfilename(
        defaultextension=".obj",
        filetypes=[("wavefront obj", "*.obj"), ("All files", "*.*")],
        initialfile="scene.obj"
    )
    
    root.destroy()
    
    if filepath:
        sdfexp.export_to_obj(data, filepath, export_z_up, export_level)
        return True
    return False



# --- Configuration ---
SCREEN_SIZE = (1200, 720)
FOV_ANGLE = math.radians(75)  # Field of View - Used for ray direction calculation
STEP_VARIABLE_FLOAT = 0.1
STEP_VARIABLE_ROTATION = 5.0


# UI Constants
PANEL_WIDTH_RATIO = 0.2  # Left and right panel width as ratio of window width
FPS_WINDOW_OFFSET = 25  # Offset from top for FPS window
FPS_WINDOW_WIDTH = 140
FPS_WINDOW_HEIGHT = 30

# Camera Constants
MOUSE_SENSITIVITY = 0.005
PAN_SENSITIVITY = 0.1
CAMERA_LERP_FACTOR = 7.5
ZOOM_SENSITIVITY = 0.5
MIN_RADIUS = 1.0
MAX_RADIUS = 100.0
MIN_PITCH = -math.radians(90)
MAX_PITCH = math.radians(90)

# Moved variables
drag_position = [0,0,0] # Track calculation result
selected_item_id = None  # Track which item is selected in the tree



# Load shader files with error handling
try:
    # Vertex shader source code
    vertex_shader = load_shader_code("shaders/vertex_shader.glsl")
    
    # SDF Library
    sdf_library = load_shader_code("shaders/sdf_library.glsl")
    
    # Fragment shader template
    fragment_shader_template = load_shader_code("shaders/fragment/template.glsl")
except (FileNotFoundError, IOError) as e:
    print(f"Error loading shader files: {e}")
    print("Please ensure all shader files are present in the project directory.")
    exit(1)



class History:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def add(self, undo_func, redo_func, undo_args=None, redo_args=None, undo_kwargs=None, redo_kwargs=None):
        if undo_args is None: 
            undo_args = ()
        if redo_args is None:
            redo_args = ()
        if undo_kwargs is None:
            undo_kwargs = {}
        if redo_kwargs is None:
            redo_kwargs = {}

        self.undo_stack.append((undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs))
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return False

        undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs = self.undo_stack.pop()
        undo_func(*undo_args, **undo_kwargs)
        self.redo_stack.append((undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs))
        return True

    def redo(self):
        if not self.redo_stack:
            return False

        undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs = self.redo_stack.pop()
        redo_func(*redo_args, **redo_kwargs)
        self.undo_stack.append((undo_func, redo_func, undo_args, redo_args, undo_kwargs, redo_kwargs))
        return True


glob_history = History()

start_drag = False
end_drag = False


class SDFPrimitive:
    def __init__(self, primitive_type, position, size_or_radius, rotation=None, scale=None, ui_name=None, color=None, **kwargs):
        self.primitive_type = primitive_type
        self.position = list(position)
        self.size_or_radius = size_or_radius if isinstance(size_or_radius, (list, tuple)) else [size_or_radius]
        if not isinstance(self.size_or_radius, list):
            self.size_or_radius = list(self.size_or_radius)
        # Always initialize as 3D vectors
        self.rotation = list(rotation) if rotation else [0.0, 0.0, 0.0]
        self.scale = list(scale) if scale else [1.0, 1.0, 1.0]
        self.color = list(color) if color else [0.8, 0.6, 0.4]
        self.kwargs = kwargs
        self.ui_name = ui_name or primitive_type


    def generate_transform_code(self, op_id):
        """
        Generate the GLSL transform code for this primitive.
        If this primitive is currently selected, the shader will subtract
        the MovePos uniform (so the C-side can move the primitive interactively).
        New: supports 'pointer' primitive which mutates the global 'p' variable.
        """
        global selected_item_id

        # If the selected item is this primitive, use the MovePos uniform in GLSL.
        if selected_item_id is not None and selected_item_id == op_id:
            new_position = ["MovePos.x", "MovePos.y", "MovePos.z"]
        else:
            # Use literal numeric components
            new_position = [self.position[0], self.position[1], self.position[2]]

        # Pointer primitives mutate the global `p` and do NOT create p{op_id}
        if self.primitive_type == "pointer":
            # pointer function name stored in kwargs['func'] (default identity)
            func_name = self.kwargs.get('func', 'pointer_identity')
            # Optionally pass extra params stored in kwargs['params'] (not used by default)
            # We pass position as second argument so pointer functions can be local around a point
            pos_arg = f"vec3({new_position[0]}, {new_position[1]}, {new_position[2]})"
            return f"    p = {func_name}(p, {pos_arg});"

        # For normal primitives generate the usual transform that works on a local p{op_id}
        transform_code = f"vec3 p{op_id} = p;"
        transform_code += f"\n    p{op_id} -= vec3({new_position[0]}, {new_position[1]}, {new_position[2]});"

        if self.rotation:
            transform_code += f"\n    p{op_id} = rotateZ({self.rotation[2]}) * rotateX({self.rotation[0]}) * rotateY({self.rotation[1]}) * p{op_id};"

        if self.scale:
            transform_code += f"\n    p{op_id} = scale(p{op_id}, vec3({self.scale[0]}, {self.scale[1]}, {self.scale[2]}));"

        return transform_code


    def generate_sdf_code(self, op_id):
        # Pointer primitives do not emit SDF distance/color—they only mutate p.
        if self.primitive_type == "pointer" or self.primitive_type == "sprite":
            return ""  # no distance/color for pointers
        
        color_vec = f"vec3({self.color[0]}, {self.color[1]}, {self.color[2]})"
        if self.primitive_type == "box":
            return f"float {op_id} = sdBox(p{op_id}, vec3({self.size_or_radius[0]}, {self.size_or_radius[1]}, {self.size_or_radius[2]}));\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "round_box":
            radius = self.kwargs.get('radius', 0.1)
            return f"float {op_id} = sdRoundBox(p{op_id}, vec3({self.size_or_radius[0]}, {self.size_or_radius[1]}, {self.size_or_radius[2]}), {radius});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "sphere":
            return f"float {op_id} = sdSphere(p{op_id}, {self.size_or_radius[0]});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "torus":
            # size_or_radius[0] = major radius, size_or_radius[1] = minor radius
            return f"float {op_id} = sdTorus(p{op_id}, vec2({self.size_or_radius[0]}, {self.size_or_radius[1]}));\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "cone":
            # size_or_radius[0] = sin(angle), size_or_radius[1] = cos(angle), kwargs['height'] = height
            c_sin = self.kwargs.get('c_sin', 0.5)
            c_cos = self.kwargs.get('c_cos', 0.866)
            height = self.kwargs.get('height', 1.0)
            return f"float {op_id} = sdCone(p{op_id}, vec2({c_sin}, {c_cos}), {height});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "plane":
            # kwargs['normal'] = normal vector, kwargs['h'] = offset
            normal = self.kwargs.get('normal', [0.0, 1.0, 0.0])
            h = self.kwargs.get('h', 0.0)
            return f"float {op_id} = sdPlane(p{op_id}, vec3({normal[0]}, {normal[1]}, {normal[2]}), {h});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "hex_prism":
            # size_or_radius[0] = hex radius, size_or_radius[1] = height
            return f"float {op_id} = sdHexPrism(p{op_id}, vec2({self.size_or_radius[0]}, {self.size_or_radius[1]}));\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "vertical_capsule":
            # size_or_radius[0] = height, size_or_radius[1] = radius
            return f"float {op_id} = sdVerticalCapsule(p{op_id}, {self.size_or_radius[0]}, {self.size_or_radius[1]});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "capped_cylinder":
            # size_or_radius[0] = radius, size_or_radius[1] = height
            return f"float {op_id} = sdCappedCylinder(p{op_id}, {self.size_or_radius[0]}, {self.size_or_radius[1]});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "rounded_cylinder":
            # size_or_radius[0] = radius a, size_or_radius[1] = radius b, kwargs['height'] = height
            height = self.kwargs.get('height', 1.0)
            return f"float {op_id} = sdRoundedCylinder(p{op_id}, {self.size_or_radius[0]}, {self.size_or_radius[1]}, {height});\n    vec3 col{op_id} = {color_vec};"
        else:
            raise ValueError(f"Unknown primitive type: {self.primitive_type}")


    def to_dict(self):
        """Convert primitive to a dictionary for JSON serialization."""
        return {
            "type": "primitive",
            "primitive_type": self.primitive_type,
            "position": self.position,
            "size_or_radius": self.size_or_radius,
            "rotation": self.rotation,
            "scale": self.scale,
            "color": self.color,
            "ui_name": self.ui_name,
            "kwargs": self.kwargs
        }



class Sprite:
    """
    A structure to hold the parameters defining a sprite, 
    including its projection plane and texture information.
    """
    def __init__(self,
        # Plane
        planePoint, planeNormal, 
        planeWidth: float, planeHeight: float,

        # Texture
        SprTexture, uvSize,
        Alpha: float = 1.0, LOD: float = 0.0
    ):
        # Store the data as instance attributes
        self.planePoint = list(planePoint)
        self.planeNormal = list(planeNormal)
        self.planeWidth = float(planeWidth)
        self.planeHeight = float(planeHeight)

        # Sampler name (used in shader code). Set a default unique name if none given
        self.SprTexture = SprTexture if SprTexture else f"sprTex_{id(self)}"
        self.uvSize = list(uvSize)
        self.Alpha = float(Alpha)
        self.LOD = float(LOD)

        # GL texture handle (created when loading image from disk). None => not loaded
        self.texture_id = None
        self.tex_size = (0, 0)

    def generate_spr_code(self):
        # NOTE: This injects literal values into the shader. The sampler is passed
        # as the identifier self.SprTexture (must match the uniform declared).
        code = (
            f"col = Sprite("
            f"ro,rd,"
            f"vec3({self.planePoint[0]},{self.planePoint[1]},{self.planePoint[2]}),"
            f"vec3({self.planeNormal[0]},{self.planeNormal[1]},{self.planeNormal[2]}),"
            f"{self.planeWidth:.6f},"
            f"-{self.planeHeight:.6f},"
            f"col, d,"
            f"{self.SprTexture},"  # sampler uniform name (no quotes)
            f"vec2({self.uvSize[0]:.6f},{self.uvSize[1]:.6f}),"
            f"{self.Alpha:.6f},"
            f"{self.LOD:.6f}"
            f");\n"
        )

        return code

    def generate_uniforms_code(self):
        # Return a sampler declaration using the sampler name
        return f"uniform sampler2D {self.SprTexture};\n"

    def load_texture_from_file(self, filepath):
        """Load an image from disk and upload to GL as an RGBA texture. Returns True on success."""
        try:
            img = Image.open(filepath).convert("RGBA")
            w, h = img.size
            img_data = img.tobytes("raw", "RGBA", 0, -1)
            


            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            # Upload
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)

            # If an old texture existed, delete it
            if self.texture_id:
                try:
                    glDeleteTextures(1, [self.texture_id])
                except Exception:
                    pass

            self.texture_id = tex
            self.tex_size = (w, h)
            return True
        except Exception as e:
            print(f"Failed to load sprite texture '{filepath}': {e}")
            return False

    def free_texture(self):
        if self.texture_id:
            try:
                glDeleteTextures(1, [self.texture_id])
            except Exception:
                pass
            self.texture_id = None
            self.tex_size = (0, 0)



def generate_postproc_code(Sprites):
    uni_code = ""
    code = ""
    for spr in Sprites:
        code += spr.generate_spr_code()
        uni_code += spr.generate_uniforms_code()
    return code, uni_code





class SDFOperation:
    def __init__(self, operation_type, *args, ui_name=None):
        self.operation_type = operation_type
        self.args = list(args)  # Store as list for mutability

        # For smooth operations and mix, track the smoothing factor k
        if operation_type in ['sunion', 'ssub', 'sinter', 'mix']:
            self.smooth_k = args[2] if len(args) > 2 else (0.5 if operation_type == 'mix' else 0.05)
        # For single-operand operations with a float parameter (round, onion)
        elif operation_type in ['round', 'onion']:
            self.float_param = args[1] if len(args) > 1 else (0.1 if operation_type == 'round' else 0.05)
            self.smooth_k = None
        else:
            self.smooth_k = None
            self.float_param = None

        self.ui_name = ui_name or operation_type

    def generate_code(self, op_id):
        OPERATION_TEMPLATES = {
            "sunion": {
                "dist_template": "float {op_id} = SmoothUnion({d_a}, {d_b}, {k});",
                "color_template": "vec3 col{op_id} = mixColorSmooth({col_a_name}, {col_b_name}, {d_a}, {d_b}, {k});",
                "unpack": lambda args: (args[0], args[1], args[2]),
            },
            "ssub": {
                "dist_template": "float {op_id} = SmoothSubtraction({d_a}, {d_b}, {k});",
                "color_template": "vec3 col{op_id} = mixColorSmooth({col_a_name}, {col_b_name}, {d_a}, {d_b}, {k});",
                "unpack": lambda args: (args[0], args[1], args[2]),
            },
            "sinter": {
                "dist_template": "float {op_id} = SmoothIntersection({d_a}, {d_b}, {k});",
                "color_template": "vec3 col{op_id} = mixColorSmooth({col_a_name}, {col_b_name}, {d_a}, {d_b}, {k});",
                "unpack": lambda args: (args[0], args[1], args[2]),
            },
            "mix": {
                "dist_template": "float {op_id} = Mix({d_a}, {d_b}, {k});",
                "color_template": "vec3 col{op_id} = mixColorSmooth({col_a_name}, {col_b_name}, {d_a}, {d_b}, {k});",
                "unpack": lambda args: (args[0], args[1], args[2]),
            },
            "invert": {
                "dist_template": "float {op_id} = invert({d_a});",
                "color_template": "vec3 col{op_id} = {col_a_name};",
                "unpack": lambda args: (args[0],),
            },
            "sub": {
                "dist_template": "float {op_id} = Subtraction({d_a}, {d_b});",
                "color_template": "vec3 col{op_id} = {col_a_name};",
                "unpack": lambda args: (args[0], args[1]),
            },
            "union": {
                "dist_template": "float {op_id} = Union({d_a}, {d_b});",
                "color_template": "vec3 col{op_id} = ({d_a} < {d_b}) ? {col_a_name} : {col_b_name};",
                "unpack": lambda args: (args[0], args[1]),
            },
            "inter": {
                "dist_template": "float {op_id} = Intersection({d_a}, {d_b});",
                "color_template": "vec3 col{op_id} = ({d_a} > {d_b}) ? {col_a_name} : {col_b_name};",
                "unpack": lambda args: (args[0], args[1]),
            },
            "xor": {
                "dist_template": "float {op_id} = Xor({d_a}, {d_b});",
                "color_template": "vec3 col{op_id} = (abs({d_a}) < abs({d_b})) ? {col_a_name} : {col_b_name};",
                "unpack": lambda args: (args[0], args[1]),
            },
            "round": {
                "dist_template": "float {op_id} = Round({d_a}, {param});",
                "color_template": "vec3 col{op_id} = {col_a_name};",
                "unpack": lambda args: (args[0], args[1]),
            },
            "onion": {
                "dist_template": "float {op_id} = Onion({d_a}, {param});",
                "color_template": "vec3 col{op_id} = {col_a_name};",
                "unpack": lambda args: (args[0], args[1]),
            },
        }

        if self.operation_type not in OPERATION_TEMPLATES:
            raise ValueError(f"Unknown operation type: {self.operation_type}")

        template_info = OPERATION_TEMPLATES[self.operation_type]
    
        try:
            unpacked_args = template_info['unpack'](self.args)
        except IndexError:
            raise ValueError(f"Not enough arguments for operation {self.operation_type}.")

        context = {'op_id': op_id}
    
        num_args = len(unpacked_args)
    
        if num_args >= 1:
            context['d_a'] = unpacked_args[0]
            context['col_a_name'] = f'col{unpacked_args[0]}'
        if num_args >= 2:
            context['d_b'] = unpacked_args[1]
            context['col_b_name'] = f'col{unpacked_args[1]}'
            context['param'] = unpacked_args[1]  # For single-operand ops, second arg is the parameter
        if num_args >= 3:
            context['k'] = unpacked_args[2]
        
        dist_code = template_info['dist_template'].format(**context)
        color_code = template_info['color_template'].format(**context)
    
        return f"    {dist_code}\n    {color_code}"

    def to_dict(self):
        """Convert operation to a dictionary for JSON serialization."""
        return {
            "type": "operation",
            "operation_type": self.operation_type,
            "args": self.args,
            "smooth_k": self.smooth_k,
            "ui_name": self.ui_name
        }


 # A variable to track what we recompiled the shader
 # in cycles mode for later updating the fbo
monitor = False


def MonitorChanges(func):
    def wrapper(*args, **kwargs):
        global monitor; monitor = True
        result = func(*args, **kwargs)
        return result
    return wrapper
        
class SDFSceneBuilder:
    def __init__(self):
        self.primitives = []
        self.operations = []
        self.next_id = 0
        self.id_to_index = {}
        self.deleted_items_cache = {}  # Cache for restoring deleted items

    def _save_item_state(self, op_id):
        """Save the complete state of an item for undo/redo."""
        if op_id not in self.id_to_index:
            return None

        item_type, index = self.id_to_index[op_id]

        if item_type == 'primitive':
            primitive = self.primitives[index][1]
            return {
                'type': 'primitive',
                'op_id': op_id,
                'index': index,
                'data': primitive.to_dict()
            }
        else:
            operation = self.operations[index][1]
            return {
                'type': 'operation',
                'op_id': op_id,
                'index': index,
                'data': operation.to_dict()
            }

    def _get_all_dependent_items(self, op_id):
        """Get all operations that depend on this item (directly or indirectly)."""
        dependent = []

        def get_dependents(item_id):
            for op_id_check, operation in self.operations:
                # operation.args may include references to other op ids
                if item_id in operation.args and op_id_check not in dependent:
                    dependent.append(op_id_check)
                    get_dependents(op_id_check)  # Recursively get dependents of dependents

        get_dependents(op_id)
        return dependent

    def add_primitive(self, primitive_type, position, size_or_radius,
                      rotation=None, scale=None, ui_name=None, color=None,
                      forced_op_id=None, **kwargs):

        op_id = forced_op_id or f"d{self.next_id}"

        # Ensure uniqueness
        self._ensure_op_id_unique(op_id)

        primitive = SDFPrimitive(primitive_type, position, size_or_radius, rotation, scale, ui_name, color, **kwargs)
        self.primitives.append((op_id, primitive))
        self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)

        # Always increment next_id if not forced
        if not forced_op_id:
            self.next_id += 1

        # Register undo/redo
        # redo should restore the same op_id; pass it via redo_kwargs
        redo_kwargs = copy.deepcopy(kwargs) if kwargs else {}
        redo_kwargs['forced_op_id'] = op_id

        glob_history.add(
            self.delete_item,
            self.add_primitive,
            (op_id,),
            (primitive_type, copy.deepcopy(position), copy.deepcopy(size_or_radius),
             copy.deepcopy(rotation), copy.deepcopy(scale), ui_name, copy.deepcopy(color)),
            {},
            redo_kwargs
        )

        return op_id

    def add_box(self, position, size, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("box", position, size, rotation, scale, ui_name, color)

    def add_roundbox(self, position, size, radius, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("round_box", position, size, rotation, scale, ui_name, color, radius=radius)

    def add_sphere(self, position, radius, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("sphere", position, radius, rotation, scale, ui_name, color)

    def add_torus(self, position, major_radius, minor_radius, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("torus", position, [major_radius, minor_radius], rotation, scale, ui_name, color)

    def add_cone(self, position, c_sin, c_cos, height, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("cone", position, [0.0], rotation, scale, ui_name, color, c_sin=c_sin, c_cos=c_cos, height=height)

    def add_plane(self, position, normal, h, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("plane", position, [0.0], rotation, scale, ui_name, color, normal=normal, h=h)

    def add_hex_prism(self, position, hex_radius, height, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("hex_prism", position, [hex_radius, height], rotation, scale, ui_name, color)

    def add_vertical_capsule(self, position, height, radius, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("vertical_capsule", position, [height, radius], rotation, scale, ui_name, color)

    def add_capped_cylinder(self, position, radius, height, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("capped_cylinder", position, [radius, height], rotation, scale, ui_name, color)

    def add_rounded_cylinder(self, position, radius_a, radius_b, height, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("rounded_cylinder", position, [radius_a, radius_b], rotation, scale, ui_name, color, height=height)

    def add_operation(self, operation_type, *args, ui_name=None, forced_op_id=None):
        """
        Add an operation. Accepts forced_op_id so undo/redo can recreate the same id.
        """
        op_id = forced_op_id or f"d{self.next_id}"

        # Ensure uniqueness before adding
        self._ensure_op_id_unique(op_id)

        operation = SDFOperation(operation_type, *args, ui_name=ui_name)
        self.operations.append((op_id, operation))
        self.id_to_index[op_id] = ('operation', len(self.operations) - 1)

        if not forced_op_id:
            self.next_id += 1

        # Register undo/redo for operations
        redo_kwargs = {'forced_op_id': op_id}
        glob_history.add(
            self._undo_operation_delete,
            self._redo_operation_add,
            (op_id, operation_type, copy.deepcopy(args), copy.deepcopy(ui_name)),
            (copy.deepcopy(operation_type), copy.deepcopy(args), copy.deepcopy(ui_name)),
            {},
            redo_kwargs
        )

        return op_id

    def sunion(self, d_a, d_b, k=0.05, ui_name=None):
        return self.add_operation("sunion", d_a, d_b, k, ui_name=ui_name)

    def ssub(self, d_a, d_b, k=0.05, ui_name=None):
        return self.add_operation("ssub", d_a, d_b, k, ui_name=ui_name)

    def sinter(self, d_a, d_b, k=0.05, ui_name=None):
        return self.add_operation("sinter", d_a, d_b, k, ui_name=ui_name)

    def mix(self, d_a, d_b, k=0.5, ui_name=None):
        return self.add_operation("mix", d_a, d_b, k, ui_name=ui_name)

    def invert(self, d_a, ui_name=None):
        return self.add_operation("invert", d_a, ui_name=ui_name)

    def sub(self, d_a, d_b, ui_name=None):
        return self.add_operation("sub", d_a, d_b, ui_name=ui_name)

    def union(self, d_a, d_b, ui_name=None):
        return self.add_operation("union", d_a, d_b, ui_name=ui_name)

    def inter(self, d_a, d_b, ui_name=None):
        return self.add_operation("inter", d_a, d_b, ui_name=ui_name)

    def xor(self, d_a, d_b, ui_name=None):
        return self.add_operation("xor", d_a, d_b, ui_name=ui_name)

    def round(self, d_a, radius, ui_name=None):
        return self.add_operation("round", d_a, radius, ui_name=ui_name)

    def onion(self, d_a, thickness, ui_name=None):
        return self.add_operation("onion", d_a, thickness, ui_name=ui_name)

    def _ensure_op_id_unique(self, op_id):
        """Remove any duplicate op_id from primitives or operations before adding new one."""
        # Remove from primitives
        self.primitives = [(pid, prim) for pid, prim in self.primitives if pid != op_id]

        # Remove from operations
        self.operations = [(oid, op) for oid, op in self.operations if oid != op_id]

        # Remove from mapping
        if op_id in self.id_to_index:
            del self.id_to_index[op_id]

        # Update all indices after removal
        for i, (pid, _) in enumerate(self.primitives):
            self.id_to_index[pid] = ('primitive', i)
        for i, (oid, _) in enumerate(self.operations):
            self.id_to_index[oid] = ('operation', i)

    def add_pointer(self, position=(0.0, 0.0, 0.0), func='pointer_identity', ui_name=None, color=None, forced_op_id=None, **kwargs):
        """
        Add a pointer primitive. `func` is the name of a GLSL function in the sdf library
        that takes (vec3 p, vec3 pos) and returns vec3 p (transformed).
        """
        # Store the chosen function name in kwargs so it will be serialized
        kwargs = dict(kwargs) if kwargs else {}
        kwargs['func'] = func
        op_id = self.add_primitive("pointer", position, [0.0, 0.0, 0.0], rotation=None, scale=None, ui_name=ui_name or "Pointer", color=color, forced_op_id=forced_op_id, **kwargs)
        return op_id



    def _undo_operation_delete(self, op_id, operation_type, args, ui_name):
        """Helper to restore a deleted operation (used by history)."""
        # Make sure this op_id will be unique (remove any current duplicates)
        self._ensure_op_id_unique(op_id)
        self.operations.append((op_id, SDFOperation(operation_type, *args, ui_name=ui_name)))
        self.id_to_index[op_id] = ('operation', len(self.operations) - 1)

    def _redo_operation_add(self, operation_type, args, ui_name, forced_op_id=None):
        """Helper to add an operation for redo (preserve op id if provided)."""
        return self.add_operation(operation_type, *args, ui_name=ui_name, forced_op_id=forced_op_id)

    def delete_item(self, op_id):
        """Delete a primitive or operation by its ID, with full undo support."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]

        # Save state of the item being deleted
        deleted_item_state = self._save_item_state(op_id)

        # Get all dependent items BEFORE deletion
        dependent_ops = self._get_all_dependent_items(op_id)
        # Save dependent states, skipping any that can't be saved
        dependent_states = [self._save_item_state(dep_id) for dep_id in dependent_ops]
        dependent_states = [s for s in dependent_states if s is not None]

        if item_type == 'primitive':
            # Remove the primitive
            del self.primitives[index]
            # Update indices for all primitives after this one
            for i in range(index, len(self.primitives)):
                prim_op_id = self.primitives[i][0]
                self.id_to_index[prim_op_id] = ('primitive', i)
        else:
            # Remove the operation
            del self.operations[index]
            # Update indices for all operations after this one
            for i in range(index, len(self.operations)):
                op_op_id = self.operations[i][0]
                self.id_to_index[op_op_id] = ('operation', i)

        # Remove from mapping
        if op_id in self.id_to_index:
            del self.id_to_index[op_id]

        # Remove any operations that depend on this deleted item
        # We must be careful because indices change as we delete; use id lookup each time
        for dep_id in dependent_ops:
            if dep_id in self.id_to_index:
                dep_item_type, dep_index = self.id_to_index[dep_id]
                if dep_item_type == 'operation':
                    # Find exact tuple index in operations list for this dep_id
                    for i, (oid, _) in enumerate(self.operations):
                        if oid == dep_id:
                            del self.operations[i]
                            # Update indices
                            for j in range(i, len(self.operations)):
                                op_op_id = self.operations[j][0]
                                self.id_to_index[op_op_id] = ('operation', j)
                            break
                if dep_id in self.id_to_index:
                    del self.id_to_index[dep_id]

        # Register undo/redo for the deletion
        glob_history.add(
            self._undo_delete_with_dependents,
            self._redo_delete_with_dependents,
            (deleted_item_state, dependent_states),
            (op_id,),
            {},
            {}
        )

        return True



    def _insert_primitive_at(self, index, op_id, primitive):
        # clamp index
        if index < 0:
            index = 0
        if index > len(self.primitives):
            index = len(self.primitives)
        self.primitives.insert(index, (op_id, primitive))
        # update id mapping for primitives
        for i, (pid, _) in enumerate(self.primitives):
            self.id_to_index[pid] = ('primitive', i)

    def _insert_operation_at(self, index, op_id, operation):
        # clamp index
        if index < 0:
            index = 0
        if index > len(self.operations):
            index = len(self.operations)
        self.operations.insert(index, (op_id, operation))
        # update id mapping for operations
        for i, (oid, _) in enumerate(self.operations):
            self.id_to_index[oid] = ('operation', i)

    def _undo_delete_with_dependents(self, deleted_item_state, dependent_states):
        """Restore a deleted item and all its dependent operations at their original indices."""
        if deleted_item_state is None:
            return

        # Restore the main item at its original index (if present)
        item = deleted_item_state
        op_id = item['op_id']
        original_index = item.get('index', None)

        # Ensure uniqueness before restoring
        self._ensure_op_id_unique(op_id)

        if item['type'] == 'primitive':
            prim_dict = item['data']
            primitive = SDFPrimitive(
                primitive_type=prim_dict["primitive_type"],
                position=prim_dict["position"],
                size_or_radius=prim_dict["size_or_radius"],
                rotation=prim_dict.get("rotation", [0.0, 0.0, 0.0]),
                scale=prim_dict.get("scale", [1.0, 1.0, 1.0]),
                ui_name=prim_dict.get("ui_name"),
                color=prim_dict.get("color", [0.8, 0.6, 0.4]),
                **prim_dict.get("kwargs", {})
            )
            # insert at saved index if available
            if original_index is None:
                self.primitives.append((op_id, primitive))
                self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)
            else:
                self._insert_primitive_at(original_index, op_id, primitive)
        else:
            op_dict = item['data']
            operation = SDFOperation(
                op_dict["operation_type"],
                *op_dict["args"],
                ui_name=op_dict.get("ui_name")
            )
            if op_dict.get("smooth_k") is not None:
                operation.smooth_k = op_dict["smooth_k"]
            if original_index is None:
                self.operations.append((op_id, operation))
                self.id_to_index[op_id] = ('operation', len(self.operations) - 1)
            else:
                self._insert_operation_at(original_index, op_id, operation)

        # Restore dependent operations at their saved indices.
        # Skip invalid/null dependent states; sort by index ascending so insertion doesn't invalidate later indices.
        valid_dep_states = [s for s in (dependent_states or []) if s]
        # Filter for operation-type states only (dependents are operations)
        valid_dep_states = [s for s in valid_dep_states if s.get('type') == 'operation']
        # sort by their original index (missing index -> large number -> appended at end)
        def dep_index_key(s):
            try:
                return s.get('index', 10**9)
            except Exception:
                return 10**9
        valid_dep_states.sort(key=dep_index_key)

        for dep_state in valid_dep_states:
            dep_id = dep_state['op_id']
            # ensure uniqueness
            self._ensure_op_id_unique(dep_id)
            op_dict = dep_state['data']
            operation = SDFOperation(
                op_dict["operation_type"],
                *op_dict["args"],
                ui_name=op_dict.get("ui_name")
            )
            if op_dict.get("smooth_k") is not None:
                operation.smooth_k = op_dict["smooth_k"]
            dep_index = dep_state.get('index', None)
            if dep_index is None:
                # append at end
                self.operations.append((dep_id, operation))
                self.id_to_index[dep_id] = ('operation', len(self.operations) - 1)
            else:
                self._insert_operation_at(dep_index, dep_id, operation)

        # Recompute next_id to avoid future duplicates
        all_ids = [int(op_id[1:]) for op_id, _ in (self.primitives + self.operations) if op_id.startswith('d')]
        if all_ids:
            self.next_id = max(all_ids) + 1

    def _redo_delete_with_dependents(self, op_id):
        """Redo deletion of an item and all its dependents."""
        self.delete_item(op_id)

    # ---- Property change helpers ----
    def _set_primitive_property(self, op_id, property_name, value):
        """Set primitive property without recording history (used by undo/redo)."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]
        if item_type != 'primitive':
            return False

        primitive = self.primitives[index][1]

        if property_name == 'position':
            primitive.position = list(value)
        elif property_name == 'size_or_radius':
            primitive.size_or_radius = list(value) if isinstance(value, (list, tuple)) else [value]
        elif property_name == 'rotation':
            primitive.rotation = list(value)
        elif property_name == 'scale':
            primitive.scale = list(value)
        elif property_name == 'color':
            primitive.color = list(value)
        elif property_name.startswith('kwargs.'):
            kwarg_name = property_name[7:]
            primitive.kwargs[kwarg_name] = value
        return True

    def modify_primitive_property(self, op_id, property_name, old_value, new_value):
        """Track modifications to primitive properties for undo/redo."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]
        if item_type != 'primitive':
            return False

        # Register the modification in history
        glob_history.add(
            self._undo_property_change,
            self._redo_property_change,
            (op_id, property_name, copy.deepcopy(old_value)),
            (op_id, property_name, copy.deepcopy(new_value)),
            {},
            {}
        )

        # Apply the new value without creating another history entry
        return self._set_primitive_property(op_id, property_name, new_value)

    def _undo_property_change(self, op_id, property_name, old_value):
        """Restore old property value (without creating history)."""
        self._set_primitive_property(op_id, property_name, old_value)

    def _redo_property_change(self, op_id, property_name, new_value):
        """Reapply property change (without creating history)."""
        self._set_primitive_property(op_id, property_name, new_value)

    def _set_operation_parameter(self, op_id, param_name, value):
        """Set operation parameter without recording history (used by undo/redo)."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]
        if item_type != 'operation':
            return False

        operation = self.operations[index][1]

        if param_name == 'smooth_k':
            operation.smooth_k = value
            if len(operation.args) >= 3:
                operation.args[2] = value
        elif param_name == 'float_param':
            operation.float_param = value
            if len(operation.args) >= 2:
                operation.args[1] = value
        elif param_name.startswith('args['):
            # Handle args like "args[0]", "args[1]", etc.
            arg_index = int(param_name.split('[')[1].split(']')[0])
            if arg_index < len(operation.args):
                operation.args[arg_index] = value
        return True

    def modify_operation_parameter(self, op_id, param_name, old_value, new_value):
        """Track modifications to operation parameters for undo/redo."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]
        if item_type != 'operation':
            return False

        glob_history.add(
            self._undo_op_param_change,
            self._redo_op_param_change,
            (op_id, param_name, copy.deepcopy(old_value)),
            (op_id, param_name, copy.deepcopy(new_value)),
            {},
            {}
        )

        # Apply the new value without creating another history entry
        return self._set_operation_parameter(op_id, param_name, new_value)

    def _undo_op_param_change(self, op_id, param_name, old_value):
        """Restore old operation parameter value (without creating history)."""
        self._set_operation_parameter(op_id, param_name, old_value)

    def _redo_op_param_change(self, op_id, param_name, new_value):
        """Reapply operation parameter change (without creating history)."""
        self._set_operation_parameter(op_id, param_name, new_value)



    def _move_item_no_history(self, op_id, new_index):
        """Move an existing item to new_index within its list without creating a history entry.

        For primitives the behavior is unchanged.

        For operations we allow the move but then sanitize operation arguments:
        - After the move we ensure every operation's operand references only items
        that come earlier in the combined order (primitives then operations).
        - If an operand would refer to an item that comes later (i.e. becomes invalid),
        we replace it with the nearest higher-level item (the nearest item that
        appears before the operation in the combined ordering). If none exists we
        leave the argument unchanged (usually only happens in degenerate scenes).
        """
        if op_id not in self.id_to_index:
            return False

        item_type, old_index = self.id_to_index[op_id]

        if item_type == 'primitive':
            item = self.primitives.pop(old_index)
            # clamp
            new_index = max(0, min(new_index, len(self.primitives)))
            self.primitives.insert(new_index, item)
            # update indices
            for i, (pid, _) in enumerate(self.primitives):
                self.id_to_index[pid] = ('primitive', i)
            return True

        # --- Operation move ---
        # Allow insertion positions from 0..len(self.operations)
        desired_new_index = max(0, min(new_index, len(self.operations)))

        # Remove the item from the list
        item = self.operations.pop(old_index)

        # Adjust insertion index because the list is now shorter if removing an earlier element
        insert_index = desired_new_index
        if insert_index > old_index:
            insert_index -= 1

        # Clamp final insertion index
        insert_index = max(0, min(insert_index, len(self.operations)))
        # Insert
        self.operations.insert(insert_index, item)

        # Update id mapping for operations (and keep primitives mapping as-is)
        for i, (oid, _) in enumerate(self.operations):
            self.id_to_index[oid] = ('operation', i)

        # --- Sanitize operands so every operation only references items declared earlier ---
        # Build combined order: primitives first, then operations (their current order)
        combined = []
        for pid, _ in self.primitives:
            combined.append(pid)
        for oid, _ in self.operations:
            combined.append(oid)

        combined_index = {opid: idx for idx, opid in enumerate(combined)}

        # Helper: find nearest valid prior item id (< limit_idx), returns None if none
        def find_nearest_prior(limit_idx):
            for k in range(limit_idx - 1, -1, -1):
                return combined[k]
            return None

        # Iterate through operations and fix arguments that point to items that come
        # at or after the operation itself (invalid).
        for op_idx, (cur_op_id, cur_op) in enumerate(self.operations):
            # compute combined index of this operation
            if cur_op_id not in combined_index:
                continue
            cur_combined_idx = combined_index[cur_op_id]

            new_args = []
            changed = False
            for arg in cur_op.args:
                # Only adjust string references that exist in combined_index
                if isinstance(arg, str) and arg in combined_index:
                    arg_combined_idx = combined_index[arg]
                    if arg_combined_idx >= cur_combined_idx:
                        # invalid reference: pick the nearest prior item
                        replacement = find_nearest_prior(cur_combined_idx)
                        if replacement is not None and replacement != arg:
                            new_args.append(replacement)
                            changed = True
                            continue
                        # if no valid prior found, fall through and keep original arg (degenerate case)
                new_args.append(arg)

            if changed:
                # apply sanitized args (no history recorded here)
                cur_op.args = new_args

        # After possibly changing args, done. id_to_index already updated.
        return True

    def move_item(self, op_id, new_index):
        """
        Public API to move an item within its section (primitives or operations).
        Records undo/redo so the action can be reverted.
        """
        if op_id not in self.id_to_index:
            return False

        item_type, old_index = self.id_to_index[op_id]

        # Record undo/redo entries that call the same low-level move (no-history).
        glob_history.add(
            self._move_item_no_history,  # undo: move back
            self._move_item_no_history,  # redo: move to new_index again
            (op_id, old_index),
            (op_id, new_index),
            {},
            {}
        )

        # Apply the move
        return self._move_item_no_history(op_id, new_index)






    # --- Remaining methods unchanged (but included for completeness) ---
    def get_all_items(self):
        """Get all items in order: primitives then operations."""
        return self.primitives + self.operations

    def get_valid_operands(self, current_op_id):
        """Get all valid operands for an operation (excluding itself and operations that reference it)."""
        all_items = self.get_all_items()
        valid_items = []

        # Find the index of current operation
        current_index = -1
        for idx, (item_id, _) in enumerate(all_items):
            if item_id == current_op_id:
                current_index = idx
                break

        # Only allow items that come before the current operation
        for idx, item in enumerate(all_items):
            if idx < current_index:
                valid_items.append(item)

        return valid_items

    def get_item_name(self, op_id):
        """Get the display name of an item."""
        if op_id not in self.id_to_index:
            return op_id

        item_type, index = self.id_to_index[op_id]
        if item_type == 'primitive':
            return self.primitives[index][1].ui_name
        else:
            return self.operations[index][1].ui_name

    def generate_raymarch_code(self):
        scene_code = []

        for op_id, primitive in self.primitives:
            transform_code = primitive.generate_transform_code(op_id)
            sdf_code = primitive.generate_sdf_code(op_id)
            scene_code.append(transform_code)
            scene_code.append(sdf_code)

        for op_id, operation in self.operations:
            code = operation.generate_code(op_id)
            scene_code.append(code)

        if scene_code:
            scene_code = "\n    ".join(scene_code)
            # Return the last operation if there are any, otherwise the last primitive
            if self.operations:
                last_op_id = self.operations[-1][0]
                last_col_id = f"col{last_op_id}"
            elif self.primitives:
                last_op_id = self.primitives[-1][0]
                last_col_id = f"col{last_op_id}"
            else:
                return "return vec4(1000.0, 0.0, 0.0, 0.0);"
            scene_code += f"\n    return vec4({last_col_id}, {last_op_id});"
        else:
            scene_code = "return vec4(0.0, 0.0, 0.0, 1000.0);"

        return scene_code

    def to_dict(self):
        """Convert the entire scene to a dictionary for JSON serialization."""
        scene_dict = {
            "primitives": [],
            "operations": [],
            "sprites": []
        }

        # Serialize primitives
        for op_id, primitive in self.primitives:
            prim_dict = primitive.to_dict()
            prim_dict["op_id"] = op_id
            scene_dict["primitives"].append(prim_dict)

        # Serialize operations
        for op_id, operation in self.operations:
            op_dict = operation.to_dict()
            op_dict["op_id"] = op_id
            scene_dict["operations"].append(op_dict)

        # Serialize sprites if the global sprites_array exists
        # (we keep texture_id out of the JSON; textures must be reloaded by the user)
        sprs = globals().get("sprites_array", None)
        if sprs:
            for spr in sprs:
                scene_dict["sprites"].append({
                    "planePoint": spr.planePoint,
                    "planeNormal": spr.planeNormal,
                    "planeWidth": spr.planeWidth,
                    "planeHeight": spr.planeHeight,
                    "SprTexture": spr.SprTexture,
                    "uvSize": spr.uvSize,
                    "Alpha": spr.Alpha,
                    "LOD": spr.LOD
                })

        return scene_dict


    def from_dict(self, scene_dict):
        """Load a scene from a dictionary (inverse of to_dict)."""
        # Clear current scene
        self.primitives.clear()
        self.operations.clear()
        self.id_to_index.clear()
        self.next_id = 0

        # Rebuild sprites_array first so sprite_index references in primitives are valid
        # This creates module-level sprites_array used by the rest of the application/UI
        global sprites_array
        sprites_array = []
        for s in scene_dict.get("sprites", []):
            spr = Sprite(
                planePoint=tuple(s.get("planePoint", (0.0, 0.0, 0.0))),
                planeNormal=tuple(s.get("planeNormal", (0.0, 0.0, 1.0))),
                planeWidth=float(s.get("planeWidth", 1.0)),
                planeHeight=float(s.get("planeHeight", 1.0)),
                SprTexture=s.get("SprTexture", f"sprTex{len(sprites_array)}"),
                uvSize=tuple(s.get("uvSize", (1.0, 1.0))),
                Alpha=float(s.get("Alpha", 1.0)),
                LOD=float(s.get("LOD", 0.0))
            )
            # Note: texture_id remains None — user must load textures again (that's expected)
            sprites_array.append(spr)

        # Load primitives
        for prim_dict in scene_dict.get("primitives", []):
            op_id = prim_dict["op_id"]

            primitive = SDFPrimitive(
                primitive_type=prim_dict["primitive_type"],
                position=prim_dict["position"],
                size_or_radius=prim_dict["size_or_radius"],
                rotation=prim_dict.get("rotation", [0.0, 0.0, 0.0]),
                scale=prim_dict.get("scale", [1.0, 1.0, 1.0]),
                ui_name=prim_dict.get("ui_name"),
                color=prim_dict.get("color", [0.8, 0.6, 0.4]),
                **prim_dict.get("kwargs", {})
            )

            self.primitives.append((op_id, primitive))
            self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)

            # Update next_id
            try:
                prim_num = int(op_id[1:])  # Extract number from "d0", "d1", etc.
                self.next_id = max(self.next_id, prim_num + 1)
            except Exception:
                pass

        # Load operations
        for op_dict in scene_dict.get("operations", []):
            op_id = op_dict["op_id"]

            operation = SDFOperation(
                op_dict["operation_type"],     # Pass as positional first
                *op_dict["args"],              # Then unpack the rest of the positional args
                ui_name=op_dict.get("ui_name") # Finally, keyword arguments
            )

            # Restore smooth_k if it was set
            if op_dict.get("smooth_k") is not None:
                operation.smooth_k = op_dict["smooth_k"]

            self.operations.append((op_id, operation))
            self.id_to_index[op_id] = ('operation', len(self.operations) - 1)

            # Update next_id
            try:
                op_num = int(op_id[1:])
                self.next_id = max(self.next_id, op_num + 1)
            except Exception:
                pass

    def save_to_json(self, filepath):
        """Save the scene to a JSON file."""
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True, f"Scene saved to {filepath}"
        except Exception as e:
            return False, f"Error saving scene: {str(e)}"

    def load_from_json(self, filepath):
        """Load a scene from a JSON file."""
        import json
        try:
            with open(filepath, 'r') as f:
                scene_dict = json.load(f)
            self.from_dict(scene_dict)
            return True, f"Scene loaded from {filepath}"
        except FileNotFoundError:
            return False, f"File not found: {filepath}"
        except json.JSONDecodeError:
            return False, f"Invalid JSON file: {filepath}"
        except Exception as e:
            return False, f"Error loading scene: {str(e)}"




def orbital_to_cartesian(_yaw, _pitch, _radius):
    yaw_rad = _yaw
    pitch_rad = _pitch

    x = _radius * math.cos(pitch_rad) * math.cos(yaw_rad)
    y = _radius * math.sin(pitch_rad)                    
    z = _radius * math.cos(pitch_rad) * math.sin(yaw_rad)

    return (x, y, z)



def clear_accumulation_fbos(accumulation_fbos,scaled_rendering_width,scaled_rendering_height):
    # Reset accumulation buffers so no stale data is read later
    if accumulation_fbos[0] is not None and accumulation_fbos[1] is not None:
        # store current viewport to restore later if you need; here we assume you will set proper viewport when drawing
        glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[0])
        glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[1])
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


# Projection 3D to 2D
# TODO: Unsuccessful attempt
def proj_3d22d(points, azim_deg=45, elev_deg=30, invert_axes=True):
    """
    Projects 3D points onto a 2D plane with specified viewing angles.

    Parameters:
    - points: numpy.ndarray (N, 3) - array of 3D points
    - azim_deg: float - azimuth angle in degrees (default 45)
    - elev_deg: float - elevation angle in degrees (default 30)
    - invert_axes: bool - invert axes to match the view (default True)

    Returns:
    - numpy.ndarray (N, 2) - array of 2D points
    """
    # Convert angles to radians
    azim = azim_deg * np.pi / 180
    elev = elev_deg * np.pi / 180

    # Azimuth direction vector
    a_vec = np.array([np.cos(azim), np.sin(azim), 0])
    # Normal vector of the projection plane
    normal = np.cos(elev) * a_vec + np.array([0, 0, np.sin(elev)])

    # Reference vector (Z-axis)
    z_vec = np.array([0, 0, 1])
    # Projection of Z onto the plane, orthogonal to normal
    y_comp = z_vec - (z_vec @ normal) * normal
    y_comp = y_comp / np.sqrt(np.sum(y_comp**2))  # normalization

    # X-axis as perpendicular to Y and normal
    x_comp = np.cross(y_comp, normal)

    # Projection matrix (2×3)
    proj_mat = np.vstack([x_comp, y_comp])

    if invert_axes:
        proj_mat = -proj_mat  # invert axes to match view

    # Apply projection: (N,3) @ (3,2) → (N,2)
    return points @ proj_mat.T



# --- Font ---
def rebuild_imgui_fonts(renderer, base_font_path="path/to/your/font.ttf", base_font_size=16.0):
    # base_font_size is in logical points; multiply by framebuffer scale for pixel-perfect atlas
    io = imgui.get_io()
    fb_scale_x, fb_scale_y = io.display_fb_scale

    # clear existing fonts and add scaled font
    io.fonts.clear()
    pixel_size = base_font_size * max(fb_scale_x, fb_scale_y)
    io.fonts.add_font_from_file_ttf(base_font_path, pixel_size)

    # rebuild texture and let the renderer upload it
    renderer.refresh_font_texture()

    # force nearest filtering if you want crisp text at integer scales
    tex_id = io.fonts.texture_id
    if tex_id:
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)




def HSpinner(value, value_step, name, width=16, height=16, under=True, repeat_delay=0.25, repeat_rate=0.1):
    # Combined spinner with input field above buttons with auto-repeat on hold
    imgui.begin_group()

    # Input field at top
    imgui.push_item_width(width)
    input_changed, value = imgui.input_float(f"##input_{name}", value, 0, 0, "%.3f")
    imgui.pop_item_width()
    if under==False: imgui.same_line()

    btn_changed = False
    width_mul = 2.0 if under else 5.0

    # Create unique IDs for buttons
    minus_btn_id = f"##btn_{name}_minus"
    plus_btn_id = f"##btn_{name}_plus"
    
    # Track button state and timing
    if not hasattr(HSpinner, "button_states"):
        HSpinner.button_states = {}
    
    # Minus button (left)
    if imgui.button(f"-{minus_btn_id}", (width/width_mul)-1, height):
        value -= value_step
        btn_changed = True
    
    # Check if minus button is held down
    if imgui.is_item_active() and imgui.is_mouse_down(0):  # Left mouse button
        current_time = time.time()
        btn_key = minus_btn_id
        
        # Initialize button state if not exists
        if btn_key not in HSpinner.button_states:
            HSpinner.button_states[btn_key] = {
                'first_press_time': current_time,
                'last_repeat_time': current_time,
                'has_repeated': False
            }
        
        state = HSpinner.button_states[btn_key]
        
        # Check if enough time has passed for first repeat
        if not state['has_repeated'] and (current_time - state['first_press_time']) >= repeat_delay:
            value -= value_step
            btn_changed = True
            state['has_repeated'] = True
            state['last_repeat_time'] = current_time
        # Check if enough time has passed for subsequent repeats
        elif state['has_repeated'] and (current_time - state['last_repeat_time']) >= repeat_rate:
            value -= value_step
            btn_changed = True
            state['last_repeat_time'] = current_time
    else:
        # Reset button state when not pressed
        minus_btn_key = minus_btn_id
        if minus_btn_key in HSpinner.button_states:
            del HSpinner.button_states[minus_btn_key]

    imgui.same_line(0, 2)

    # Plus button (right)
    if imgui.button(f"+{plus_btn_id}", (width/width_mul)-1, height):
        value += value_step
        btn_changed = True
    
    # Check if plus button is held down
    if imgui.is_item_active() and imgui.is_mouse_down(0):  # Left mouse button
        current_time = time.time()
        btn_key = plus_btn_id
        
        # Initialize button state if not exists
        if btn_key not in HSpinner.button_states:
            HSpinner.button_states[btn_key] = {
                'first_press_time': current_time,
                'last_repeat_time': current_time,
                'has_repeated': False
            }
        
        state = HSpinner.button_states[btn_key]
        
        # Check if enough time has passed for first repeat
        if not state['has_repeated'] and (current_time - state['first_press_time']) >= repeat_delay:
            value += value_step
            btn_changed = True
            state['has_repeated'] = True
            state['last_repeat_time'] = current_time
        # Check if enough time has passed for subsequent repeats
        elif state['has_repeated'] and (current_time - state['last_repeat_time']) >= repeat_rate:
            value += value_step
            btn_changed = True
            state['last_repeat_time'] = current_time
    else:
        # Reset button state when not pressed
        plus_btn_key = plus_btn_id
        if plus_btn_key in HSpinner.button_states:
            del HSpinner.button_states[plus_btn_key]

    imgui.end_group()
    return input_changed or btn_changed, value

def input_vec3(name, vector, value_step=0.1, item_width=60):
    # Handles a 3D vector input with separate HSpinners for each component
    imgui.begin_group()
    changed = False
    for i, axis in enumerate(['x', 'y', 'z']):
        c, vector[i] = HSpinner(vector[i], value_step, f"{name}_{axis}", item_width)
        changed = changed or c
        if i < 2:
            imgui.same_line()
    imgui.end_group()

    imgui.same_line()
    imgui.text(name)
    return changed, vector


def input_vec2(name, vector, value_step=0.1, item_width=60):
    # Handles a 3D vector input with separate HSpinners for each component
    imgui.begin_group()
    changed = False
    for i, axis in enumerate(['x', 'y']):
        c, vector[i] = HSpinner(vector[i], value_step, f"{name}_{axis}", item_width)
        changed = changed or c
        if i < 1:
            imgui.same_line()
    imgui.end_group()

    imgui.same_line()
    imgui.text(name)
    return changed, vector


def input_float(name, value, value_step=0.1, item_width=60):
    imgui.begin_group()
    changed, value = HSpinner(value, value_step, f"{name}_f", item_width, 20, False)
    imgui.same_line()
    imgui.end_group()

    imgui.same_line()
    imgui.text(name)
    return changed, value




def main():
    # Globals
    global start_drag, end_drag, dragging, selected_item_id, drag_position

    # Initialize GLFW
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(SCREEN_SIZE[0], SCREEN_SIZE[1], "Viewport", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Initialize ImGui
    imgui.create_context()
    impl = GlfwRenderer(window)
    rebuild_imgui_fonts(impl, "gui/fonts/Roboto-Medium.ttf", 16.0)


    # --- Camera State ---
    target_yaw = 0.0
    target_pitch = 0.0
    target_pan_y = 0.0
    target_pan_x = 0.0
    target_radius = 5.0
    cam_yaw = 0.0
    cam_pitch = 0.0
    cam_pan_y = 0.0
    cam_pan_x = 0.0
    last_x, last_y = 0.0, 0.0
    last_pan_x, last_pan_y = 0.0, 0.0  # Separate tracking for panning
    cam_radius = 5.0
    cam_orbit = [0.0, 0.0, 0.0]
    PAN_SENSITIVITY = 0.01  # Adjust this to control pan speed
    DRAG_SENSITIVITY = 0.01 # Adjust this to control drag (primitive) speed
    last_x, last_y = 0.0, 0.0

    is_mmb_pressed = False
    is_shift_mmb_pressed = False

    # --- SaveLoad ---
    save_load_message = None
    save_load_message_time = None

    # --- Keys ---
    last_key_s_pressed = False
    last_key_o_pressed = False
    last_key_z_pressed = False
    last_key_y_pressed = False
    last_key_g_pressed = False
    axis_toggled_gx = False
    axis_toggled_gy = False
    axis_toggled_gz = False
    last_key_gx_pressed = False
    last_key_gy_pressed = False
    last_key_gz_pressed = False

    last_key_f10_pressed = False  # Add this if not present

    # --- Draging ---
    dragging = False
    dragging_op_id = None           # op_id of the item currently being dragged
    drag_last_x = 0.0               # last mouse x while dragging (separate from camera last_x/last_y)
    drag_last_y = 0.0
    drag_start_pos = None           # original primitive position at drag start (copied list)
    drag_accum = [0.0, 0.0, 0.0]    # accumulated world-space movement since drag start
    DRAG_SENSITIVITY = 0.01         # adjust for speed; consider scaling with cam_radius for consistent feel
    
    # Helper: safely set MovePos uniform (call this wherever you were directly doing glUniform3f for MovePos)
    def set_move_pos_uniform(shader_program, uniform_locs, pos):
        """
        Safely set the MovePos uniform. If the cached uniform location is missing (-1 or None),
        query it dynamically and cache it. Only call glUniform if the location exists.
        """
        if uniform_locs is None or shader_program is None:
            return
        move_key = 'move_pos'
        loc = uniform_locs.get(move_key, None)
        if loc is None or loc == -1:
            # Query the active program for the location (this is safe and will return -1 if not declared)
            loc = glGetUniformLocation(shader_program, "MovePos")
            uniform_locs[move_key] = loc
        if loc != -1:
            glUniform3f(loc, float(pos[0]), float(pos[1]), float(pos[2]))


    def bind_sprite_textures(uniforms):
        """
        Bind loaded sprite textures to texture units and upload the sampler uniform indices.
        Assumes texture unit 0 may be used for accumulation/render targets, so start at unit 1.
        """
        base_unit = 1
        for i, spr in enumerate(sprites_array):
            loc = uniforms.get(spr.SprTexture, -1) if uniforms else -1
            unit = base_unit + i
            if spr.texture_id is not None and loc is not None and loc != -1:
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, spr.texture_id)
                # Tell shader which texture unit to sample from
                glUniform1i(loc, unit)
            else:
                # If texture not loaded, bind 0 to keep behavior stable
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, 0)
                if loc is not None and loc != -1:
                    glUniform1i(loc, unit)
        # restore active texture to 0
        glActiveTexture(GL_TEXTURE0)


    # --- Delta time --- 
    delta_time = 0.0 


    # --- Scene Definition ---
    scene_builder = SDFSceneBuilder()

    box = scene_builder.add_box((0.0, -0.5+2.0, 0.0), (0.5, 0.5, 0.5), ui_name="Box 1", color=[0.8, 0.2, 0.2])
    sphere_id = scene_builder.add_sphere((0.0, -0.75+2.0, 0.0), 0.5, ui_name="Sphere 1", color=[0.2, 0.8, 0.2])
    sphere_id = scene_builder.ssub(sphere_id, box, 0.05, ui_name="Subtract 1")
    box_id = scene_builder.add_roundbox((0.0, -2.0+2.0, 0.0), (3.0, 1.0, 3.0), 0.1, ui_name="Round Box 1", color=[0.4, 0.4, 0.8])
    box2_id = scene_builder.add_box((0.0, -1.5+2.0, 0.0), (2.0, 1.0, 2.0), ui_name="Box 2", color=[0.8, 0.8, 0.4])
    final_id = scene_builder.ssub(box2_id, box_id, 0.05, ui_name="Subtract 2")
    final_id = scene_builder.sunion(final_id, sphere_id, 0.05, ui_name="Union 1")
    

    # --- UI State ---
    show_selection_window = False
    show_settings_window = False
    show_export_vol_window = False
    show_export_obj_window = False
    show_about_window = False
    selection_mode = None  # 'primitive' or 'operation'
    renaming_item_id = None  # Item being renamed
    rename_text = ""
    last_key_a_pressed = False  # Track if Ctrl+A was pressed
    last_key_f2_pressed = False  # Track if F2 was pressed
    last_key_delete_pressed = False  # Track if Delete was pressed
    last_key_compile_pressed = False  # Track if Ctrl+B was pressed
    
    # Shader selection
    shader_choice = 0  # 0 = template, 1 = cycles
    shader_names = ["shaders/fragment/template.glsl", "shaders/fragment/cycles.glsl"]

    # Sky shaders uniforms (cycles)
    sky_top_color = [0.7, 0.8, 1.0]
    sky_bottom_color = [0.1, 0.15, 0.25]

    # Grid (template)
    GridEnabled = True

    # Light
    LightDir = [0.5, 0.5, -1.0]

    # --- Settings ---
    resolution_scale = 1.0  # 1.0 = normal, 2.0 = oversampling, <1.0 = low res for performance

    # Export Config
    grid_size = 16
    vox_quality = 1.0
    export_z_up = True
    export_level = 0.0

    # Sprites
    sprites_array = []


    # --- FPS tracking ---
    fps_clock = time.time()
    fps_frames = 0
    fps_value = 0


    # --- Shader compilation and error tracking ---
    shader_compile_error = None
    shader_cache = {}  # Cache for compiled shaders: {hash: (shader_program, uniforms)}
    
    def get_shader_hash():
        """Generate a hash of the current shader code for caching."""
        scene_code = scene_builder.generate_raymarch_code()
        postproc_code = generate_postproc_code(sprites_array)
        selected_fragment_shader = load_shader_code(shader_names[shader_choice])
        fragment_shader = selected_fragment_shader.replace("{SDF_LIBRARY}", sdf_library)
        fragment_shader = fragment_shader.replace("{SCENE_CODE}", scene_code)
        fragment_shader = fragment_shader.replace("{FOV_ANGLE_VAL}", str(FOV_ANGLE))
        fragment_shader = fragment_shader.replace("{POSTPROC}", postproc_code[0])
        fragment_shader = fragment_shader.replace("{ADDITIONAL_UNIFORMS}", postproc_code[1])
        
        # Create hash of the complete shader code (including shader choice)
        shader_code = f"{vertex_shader}\n{fragment_shader}\n{shader_names[shader_choice]}"
        return hashlib.md5(shader_code.encode('utf-8')).hexdigest()
    


    def compile_shader():
        """Compile the shader program from the current scene.  Uses caching."""
        nonlocal shader_compile_error
        
        # Check cache first
        shader_hash = get_shader_hash()
        if shader_hash in shader_cache:
            cached_shader, cached_uniforms = shader_cache[shader_hash]
            shader_compile_error = None
            return cached_shader, cached_uniforms
        
        # Not in cache, compile new shader
        try:
            scene_code = scene_builder.generate_raymarch_code()
            # Use selected shader
            postproc_code = generate_postproc_code(sprites_array)
            selected_fragment_shader = load_shader_code(shader_names[shader_choice])
            fragment_shader = selected_fragment_shader.replace("{SDF_LIBRARY}", sdf_library)
            fragment_shader = fragment_shader.replace("{SCENE_CODE}", scene_code)
            fragment_shader = fragment_shader.replace("{FOV_ANGLE_VAL}", str(FOV_ANGLE))
            fragment_shader = fragment_shader.replace("{POSTPROC}", postproc_code[0])
            fragment_shader = fragment_shader.replace("{ADDITIONAL_UNIFORMS}", postproc_code[1])
            
            shader_program = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
            
            # Get uniform locations
            uniforms = get_uniform_locations(shader_program)
            
            # Cache the compiled shader
            shader_cache[shader_hash] = (shader_program, uniforms)
            
            shader_compile_error = None
            return shader_program, uniforms
        except Exception as e:
            shader_compile_error = str(e)
            print(f"Shader compilation error:  {e}")
            return None, None
    
    def get_uniform_locations(shader_program):
        # Get all uniform locations for the shader program.
        uniforms = {
            'time'                 :       glGetUniformLocation(shader_program, "time"),
            'resolution'           :       glGetUniformLocation(shader_program, "resolution"),
            'viewportOffset'       :       glGetUniformLocation(shader_program, "viewportOffset"),
            'camYaw'               :       glGetUniformLocation(shader_program, "camYaw"),
            'camPitch'             :       glGetUniformLocation(shader_program, "camPitch"),
            'radius'               :       glGetUniformLocation(shader_program, "radius"),
            'CamOrbit'             :       glGetUniformLocation(shader_program, "CamOrbit"),
            'frameIndex'           :       glGetUniformLocation(shader_program, "frameIndex"),
            'accumulationTexture'  :       glGetUniformLocation(shader_program, "accumulationTexture"),
            'useAccumulation'      :       glGetUniformLocation(shader_program, "useAccumulation"),
            'col_sky_top'          :       glGetUniformLocation(shader_program, "SkyColorTop"),
            'col_sky_bottom'       :       glGetUniformLocation(shader_program, "SkyColorBottom"),
            'grid_enabled'         :       glGetUniformLocation(shader_program, "GridEnabled"),
            'move_pos'             :       glGetUniformLocation(shader_program, "MovePos"),
            'maxFrames'            :       glGetUniformLocation(shader_program, "MaxFrames"),
            'LightDir'             :       glGetUniformLocation(shader_program, "LightDir")
        }

        # Register sprite sampler uniforms (dynamic)
        # sprites_array is in outer scope; it's the list of Sprite objects used for postprocessing
        try:
            for spr in sprites_array:
                # Use sampler name string as key, store location (may be -1 if unused)
                uniforms[spr.SprTexture] = glGetUniformLocation(shader_program, spr.SprTexture)
        except Exception:
            # If sprites_array is not defined yet, skip (defensive)
            pass

        return uniforms


    @MonitorChanges
    def recompile_shader():
        """Recompile shader and update uniform locations.  Returns (success, uniforms_dict). Uses caching."""
        nonlocal shader, uniform_locs
        
        new_shader, new_uniforms = compile_shader()
        if new_shader is None:
            return False, None
        
        if shader is not None and shader != new_shader:
            old_hash = None
            for cached_hash, (cached_shader, _) in shader_cache.items():
                if cached_shader == shader:
                    old_hash = cached_hash
                    break
            
            if old_hash is None:
                glDeleteProgram(shader)
        
        shader = new_shader
        uniform_locs = new_uniforms
        return True, new_uniforms

    shader, uniform_locs = compile_shader()
    if shader is None:
        print("Failed to compile initial shader. Exiting.")
        impl.shutdown()
        glfw.terminate()
        return

    # --- OpenGL Setup (Quad VAO/VBO) ---
    vertices = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    vertices = (GLfloat * len(vertices))(*vertices)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    
    # --- Framebuffer Setup for Resolution Scaling ---
    fbo = None
    render_texture = None
    fbo_width = 0
    fbo_height = 0
    
    def setup_framebuffer(width, height):
        """Create or update framebuffer for rendering at scaled resolution."""
        nonlocal fbo, render_texture, fbo_width, fbo_height
        
        # Only recreate if size changed
        if fbo is None or fbo_width != width or fbo_height != height:
            # Delete old framebuffer if it exists
            if fbo is not None:
                glDeleteFramebuffers(1, [fbo])
                glDeleteTextures(1, [render_texture])
            
            # Create framebuffer
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            
            # Create texture to render to
            render_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, render_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Attach texture to framebuffer
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_texture, 0)
            
            # Check framebuffer completeness
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("Error: Framebuffer is not complete!")
                return False
            
            fbo_width = width
            fbo_height = height
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return True
        return True

    
    
    try:
        # Quad with texture coordinates for displaying the rendered texture
        quad_vertices = [
            # positions   # tex coords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ]
        quad_vertices = (GLfloat * len(quad_vertices))(*quad_vertices)
        
        display_vao = glGenVertexArrays(1)
        glBindVertexArray(display_vao)
        display_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, display_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(quad_vertices) * 4, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)  # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))  # tex coord
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
    except Exception as e:
        print(f"Warning: Could not create display shader: {e}")
        print("Falling back to direct rendering (resolution scale may not work correctly)")
    
    # --- Framebuffer Setup for Resolution Scaling ---
    fbo = None
    render_texture = None
    fbo_width = 0
    fbo_height = 0
    
    def setup_framebuffer(width, height):
        """Create or update framebuffer for rendering at scaled resolution."""
        nonlocal fbo, render_texture, fbo_width, fbo_height
        
        # Only recreate if size changed
        if fbo is None or fbo_width != width or fbo_height != height:
            # Delete old framebuffer if it exists
            if fbo is not None:
                glDeleteFramebuffers(1, [fbo])
                glDeleteTextures(1, [render_texture])
            
            # Create framebuffer
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            
            # Create texture to render to
            render_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, render_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Attach texture to framebuffer
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_texture, 0)
            
            # Check framebuffer completeness
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("Error: Framebuffer is not complete!")
                return False
            
            fbo_width = width
            fbo_height = height
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return True
        return True
    
    # Simple shader for displaying texture
    display_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
    """
    
    display_fragment_shader = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D renderTexture;
uniform int isAccumulation;

void main() {
    vec4 tex = texture(renderTexture, TexCoord);

    if (isAccumulation == 1) {
        FragColor = vec4(tex.rgb, 1.0);
    } else {
        FragColor = vec4(tex.rgb, 1.0);
    }
}
    """
    
    display_shader = None
    display_vao = None
    display_vbo = None
    
    try:
        display_shader = compileProgram(
            compileShader(display_vertex_shader, GL_VERTEX_SHADER),
            compileShader(display_fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Quad with texture coordinates for displaying the rendered texture
        quad_vertices = [
            # positions   # tex coords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ]
        quad_vertices = (GLfloat * len(quad_vertices))(*quad_vertices)
        
        display_vao = glGenVertexArrays(1)
        glBindVertexArray(display_vao)
        display_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, display_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(quad_vertices) * 4, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)  # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))  # tex coord
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
    except Exception as e:
        print(f"Warning: Could not create display shader: {e}")
        print("Falling back to direct rendering (resolution scale may not work correctly)")
        display_shader = None


    # --- Accumulation Buffer Setup ---
    accumulation_fbo = None
    accumulation_texture = None
    accumulation_width = 0
    accumulation_height = 0
    frame_count = 0
    max_frames = 128
    accumulation_textures = [None, None]  # Double buffer
    accumulation_fbos = [None, None]
    current_accum_index = 0  # Which one to write to


    def setup_accumulation_buffer(width, height):
        """Create or update accumulation buffers (double-buffered) for temporal filtering."""
        nonlocal accumulation_fbos, accumulation_textures, accumulation_width, accumulation_height

        # If already set up for this size and both buffers exist, nothing to do.
        if (accumulation_width == width and accumulation_height == height and
                accumulation_fbos[0] is not None and accumulation_fbos[1] is not None and
                accumulation_textures[0] is not None and accumulation_textures[1] is not None):
            return True

        # Delete old buffers/textures if they exist
        for i in range(2):
            if accumulation_fbos[i] is not None:
                try:
                    glDeleteFramebuffers(1, [accumulation_fbos[i]])
                except Exception:
                    pass
                accumulation_fbos[i] = None
            if accumulation_textures[i] is not None:
                try:
                    glDeleteTextures(1, [accumulation_textures[i]])
                except Exception:
                    pass
                accumulation_textures[i] = None

        # Create two FBO/texture pairs
        for i in range(2):
            fbo_i = glGenFramebuffers(1)
            tex_i = glGenTextures(1)

            glBindFramebuffer(GL_FRAMEBUFFER, fbo_i)
            glBindTexture(GL_TEXTURE_2D, tex_i)

            # Allocate floating point RGBA texture for accumulation
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

            # Attach texture to the framebuffer
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_i, 0)

            # Check framebuffer completeness
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print(f"Error: Accumulation framebuffer {i} is not complete!")
                # Clean up what we created so far
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                for j in range(2):
                    if accumulation_fbos[j] is not None:
                        try:
                            glDeleteFramebuffers(1, [accumulation_fbos[j]])
                        except Exception:
                            pass
                        accumulation_fbos[j] = None
                    if accumulation_textures[j] is not None:
                        try:
                            glDeleteTextures(1, [accumulation_textures[j]])
                        except Exception:
                            pass
                        accumulation_textures[j] = None
                return False

            # Store handles
            accumulation_fbos[i] = fbo_i
            accumulation_textures[i] = tex_i

        # Update size bookkeeping and unbind framebuffer
        accumulation_width = width
        accumulation_height = height
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return True




    # --- Main Loop ---
    start_time = time.time()
    prev_time = time.time() 


    while not glfw.window_should_close(window):
        # calc Delta time 
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        gui.themes.setup_dark_red_theme()



        # --- FPS calculation ---
        fps_frames += 1
        current_time = time.time()
        if current_time - fps_clock >= 1.0:
            fps_value = fps_frames
            fps_frames = 0
            fps_clock = current_time

        # --- Handle keyboard input ---
        io = imgui.get_io()
        
        # Check Ctrl+A for add window (with debouncing)
        if io.keys_down[glfw.KEY_A] and io.key_ctrl:
            if not last_key_a_pressed:
                show_selection_window = True
                last_key_a_pressed = True
        else:
            last_key_a_pressed = False
        
        # Check F2 for rename (with debouncing)
        if io.keys_down[glfw.KEY_F2] and selected_item_id is not None and renaming_item_id is None:
            if not last_key_f2_pressed:
                renaming_item_id = selected_item_id
                rename_text = scene_builder.get_item_name(selected_item_id)
                last_key_f2_pressed = True
        else:
            last_key_f2_pressed = False
        
        # Check Delete key for deletion (with debouncing)
        if io.keys_down[glfw.KEY_DELETE] and selected_item_id is not None:
            if not last_key_delete_pressed:
                if scene_builder.delete_item(selected_item_id):
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    selected_item_id = None
                    selection_mode = None
                last_key_delete_pressed = True
        else:
            last_key_delete_pressed = False
        
        # Check Ctrl+B for compile (with debouncing)
        if io.keys_down[glfw.KEY_B] and io.key_ctrl:
            if not last_key_compile_pressed:
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms
                last_key_compile_pressed = True
        else:
            last_key_compile_pressed = False



        # Increment frame counter only when using cycles shader
        if shader_choice == 1:   # cycles_fragment_shader.glsl
            frame_count = min(frame_count + 1, max_frames)
        else: 
            frame_count = 0  # Reset accumulation when switching shaders
        
        # Get window and rendering dimensions
        width, height = glfw.get_framebuffer_size(window)
        menu_bar_height = int(imgui.get_frame_height())
        panel_width = int(width * PANEL_WIDTH_RATIO)
        rendering_width = width - 2 * panel_width
        rendering_height = height - menu_bar_height
        panel_elem_width_vec3 = (panel_width/4)-14
        panel_elem_width_float = (panel_width/2)-14

        
        scaled_rendering_width = int(rendering_width * resolution_scale)
        scaled_rendering_height = int(rendering_height * resolution_scale)







        # Get the current window size
        width, height = glfw.get_framebuffer_size(window)
        # Get menu bar height (needed for calculations) - convert to int for glViewport
        menu_bar_height = int(imgui.get_frame_height())
        panel_width = int(width * PANEL_WIDTH_RATIO)
        rendering_width = width - 2 * panel_width
        rendering_height = height - menu_bar_height
        
        # Apply resolution scale
        scaled_rendering_width = int(rendering_width * resolution_scale)
        scaled_rendering_height = int(rendering_height * resolution_scale)


        # If we recompiled the shader, we will update the fbo
        global monitor
        if monitor == True and shader_choice == 1:
            monitor = False
            frame_count = 0
            clear_accumulation_fbos(accumulation_fbos,scaled_rendering_width, scaled_rendering_height)
            current_accum_index = 0



        # Handle MMB press and release for camera control
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            shift_pressed = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
            
            if not is_mmb_pressed:
                is_mmb_pressed = True
                is_shift_mmb_pressed = shift_pressed
                last_x, last_y = glfw.get_cursor_pos(window)
                if shift_pressed:
                    last_pan_x, last_pan_y = last_x, last_y
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.RELEASE:
            if is_mmb_pressed:
                is_mmb_pressed = False
                is_shift_mmb_pressed = False

        if is_mmb_pressed or dragging:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        else:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)


        prev_cam_yaw = cam_yaw
        prev_cam_pitch = cam_pitch
        prev_cam_radius = cam_radius
        prev_cam_orbit = cam_orbit


        # Handle mouse wheel input for camera zoom
        if io.mouse_wheel != 0:
            target_radius -= io.mouse_wheel * ZOOM_SENSITIVITY
            target_radius = max(MIN_RADIUS, min(MAX_RADIUS, target_radius))

        cam_radius += (target_radius - cam_radius) * (CAMERA_LERP_FACTOR * delta_time)

        # Only update target camera angles if MMB is pressed
        if is_mmb_pressed:
            current_x, current_y = glfw.get_cursor_pos(window)
            if is_shift_mmb_pressed:
                # Panning mode: Shift + MMB
                dx = current_x - last_pan_x
                dy = current_y - last_pan_y
                last_pan_x, last_pan_y = current_x, current_y
                target_pan_x += dx * PAN_SENSITIVITY
                target_pan_y += dy * PAN_SENSITIVITY
            else:
                # Rotation mode: MMB only
                dx = current_x - last_x
                dy = current_y - last_y
                last_x, last_y = current_x, current_y
                target_yaw -= dx * MOUSE_SENSITIVITY
                target_pitch += dy * MOUSE_SENSITIVITY
                target_pitch = max(MIN_PITCH, min(MAX_PITCH, target_pitch))


        # --- Interpolate camera angles ---
        cam_yaw += (target_yaw - cam_yaw) * (CAMERA_LERP_FACTOR*delta_time)
        cam_pitch += (target_pitch - cam_pitch) * (CAMERA_LERP_FACTOR*delta_time)

        # --- Interpolate camera Pan ---
        cam_pan_y += (target_pan_y - cam_pan_y) * (CAMERA_LERP_FACTOR*delta_time)
        cam_pan_x -= (target_pan_x + cam_pan_x) * (CAMERA_LERP_FACTOR*delta_time)

        # --- Camera vectors ---

        forward_x = math.cos(cam_pitch) * math.sin(cam_yaw)
        forward_y = math.sin(cam_pitch)
        forward_z = math.cos(cam_pitch) * math.cos(cam_yaw)


        right_x = math.cos(cam_yaw)
        right_y = 0
        right_z = -math.sin(cam_yaw)


        up_x = forward_y * right_z - forward_z * right_y
        up_y = forward_z * right_x - forward_x * right_z
        up_z = forward_x * right_y - forward_y * right_x


        orbit_center_offset_x = cam_pan_x * right_x + cam_pan_y * up_x
        orbit_center_offset_y = cam_pan_x * right_y + cam_pan_y * up_y
        orbit_center_offset_z = cam_pan_x * right_z + cam_pan_y * up_z

        cam_orbit = (
            orbit_center_offset_z, # Yoow! (Correctly)
            orbit_center_offset_y,
            orbit_center_offset_x
        )

        # -----

        if io.keys_down[glfw.KEY_HOME]:
            cam_pan_x = cam_pan_y = target_pan_x = target_pan_y = 0.0
            cam_orbit = [0.0,0.0,0.0]



        elip = 0.0001
        if (abs(cam_yaw - prev_cam_yaw) > elip or 
            abs(cam_pitch - prev_cam_pitch) > elip or
            abs(cam_radius - prev_cam_radius) > elip or
            any(abs(cam_orbit[i] - prev_cam_orbit[i]) > elip for i in range(3))):

            # Reset accumulation buffers so no stale data is read later
            frame_count = 0
            clear_accumulation_fbos(accumulation_fbos,scaled_rendering_width, scaled_rendering_height)
            current_accum_index = 0


        prev_cam_yaw = cam_yaw
        prev_cam_pitch = cam_pitch
        prev_cam_radius = cam_radius
        prev_cam_orbit = cam_orbit


        # TODO: Unsuccessful attempt
        #circle_points = proj_3d22d(np.array([[0.0, 0.0, 100.0]]), cam_yaw, cam_pitch)
        #print(circle_points)
        
        #bg_draw_list = imgui.get_background_draw_list()
        
        #bg_draw_list.add_circle_filled(
        #    circle_points[0][0]+(width//2),
        #    circle_points[0][1]+(height//2),
        #    25, 
        #    imgui.get_color_u32_rgba(1, 0, 0, 1)
        #)


        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)
        
        
        # --- Setup accumulation buffer if using cycles shader ---
        use_accumulation = 0
        if shader_choice == 1:  # cycles.glsl
            if setup_accumulation_buffer(scaled_rendering_width, scaled_rendering_height):
                use_accumulation = 1

        # --- RENDER TO ACCUMULATION BUFFER ---
        if shader is not None and shader_choice == 1 and use_accumulation == 1:
            write_buffer = current_accum_index
            read_buffer = 1 - current_accum_index
            glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[write_buffer])
            glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)

            if frame_count == 0:
                glClear(GL_COLOR_BUFFER_BIT)

            if frame_count < max_frames:
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], scaled_rendering_width, scaled_rendering_height)
                    glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                    glUniform1f(uniform_locs['camYaw'], cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], cam_pitch)
                    glUniform1f(uniform_locs['radius'], cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                    glUniform1i(uniform_locs['frameIndex'], frame_count)
                    glUniform1i(uniform_locs['maxFrames'], max_frames)
                    set_move_pos_uniform(shader, uniform_locs, drag_position)

                    # Bind accumulation texture for reading
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, accumulation_textures[read_buffer])
                    glUniform1i(uniform_locs['accumulationTexture'], 0)
                    glUniform1i(uniform_locs['useAccumulation'], 1)

                    glUniform3f(uniform_locs['col_sky_top'], sky_top_color[0], sky_top_color[1], sky_top_color[2])
                    glUniform3f(uniform_locs['col_sky_bottom'], sky_bottom_color[0], sky_bottom_color[1], sky_bottom_color[2])
                    
                    glUniform3f(uniform_locs['LightDir'], LightDir[0], LightDir[1], LightDir[2])

                bind_sprite_textures(uniform_locs)
                glBindVertexArray(vao)
                glDrawArrays(GL_QUADS, 0, 4)

            # Switch back to default framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, width, height)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, accumulation_textures[write_buffer])

            # Display accumulated result
            glUseProgram(display_shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, accumulation_textures[write_buffer])
            glUniform1i(glGetUniformLocation(display_shader, "renderTexture"), 0)
            

            # Set isAccumulation to 1 if rendering is complete
            if frame_count >= max_frames:
                glUniform1i(glGetUniformLocation(display_shader, "isAccumulation"), 1)
            else:
                glUniform1i(glGetUniformLocation(display_shader, "isAccumulation"), 0)
            #print(glGetUniformLocation(display_shader,"isAccumulation"))

            glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
            bind_sprite_textures(uniform_locs)
            glBindVertexArray(display_vao)
            glDrawArrays(GL_QUADS, 0, 4)
            glBindVertexArray(0)

            glViewport(0, 0, width, height)
            current_accum_index = 1 - current_accum_index
        
        # --- RENDER DIRECTLY (if NOT using cycles or accumulation disabled) ---
        elif shader is not None: 
            glUseProgram(shader)
            if uniform_locs is not None:
                current_time_uniform = time.time() - start_time
                glUniform1f(uniform_locs['time'], current_time_uniform)
                glUniform2f(uniform_locs['resolution'], rendering_width, rendering_height)
                # When rendering directly into the screen viewport we must subtract the panel/menu offset
                glUniform2f(uniform_locs['viewportOffset'], float(panel_width), float(menu_bar_height))
                glUniform1f(uniform_locs['camYaw'], cam_yaw)
                glUniform1f(uniform_locs['camPitch'], cam_pitch)
                glUniform1f(uniform_locs['radius'], cam_radius)
                glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                glUniform1i(uniform_locs['frameIndex'], 0)
                glUniform1i(uniform_locs['useAccumulation'], 0)
                set_move_pos_uniform(shader, uniform_locs, drag_position)

                glUniform3f(uniform_locs['col_sky_top'], sky_top_color[0], sky_top_color[1], sky_top_color[2])
                glUniform3f(uniform_locs['col_sky_bottom'], sky_bottom_color[0], sky_bottom_color[1], sky_bottom_color[2])

                glUniform1i(uniform_locs['grid_enabled'], GridEnabled)
                glUniform3f(uniform_locs['LightDir'], LightDir[0], LightDir[1], LightDir[2])


            # Check if viewport is minimized
            if rendering_width > 0 and rendering_height > 0:
                glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
                glBindVertexArray(vao)
                bind_sprite_textures(uniform_locs)
                glDrawArrays(GL_QUADS, 0, 4)

            glViewport(0, 0, width, height)



        # --- TOP MENU BAR ---
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                if imgui.menu_item("Save Scene", "Ctrl+S")[0]:
                    # Trigger save dialog
                    success, message = save_scene_dialog(scene_builder)
                    save_load_message = message
                    save_load_message_time = time.time()
        
                if imgui.menu_item("Load Scene", "Ctrl+O")[0]:
                    # Trigger load dialog
                    success, message = load_scene_dialog(scene_builder)
                    save_load_message = message
                    save_load_message_time = time.time()
                    if success:
                        glob_history.undo_stack.clear()
                        glob_history.redo_stack.clear() 
                        selected_item_id = None
                        selection_mode = None
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms

                imgui.separator()
                imgui.spacing()
                if imgui.begin_menu("Export..."):
                    if imgui.menu_item("As Volume")[0]:
                        show_export_vol_window = True
                    if imgui.menu_item("To OBJ")[0]:
                        show_export_obj_window = True
                    imgui.end_menu()

                imgui.spacing()

                imgui.separator()
                imgui.spacing()
                if imgui.menu_item("Exit", "Alt+F4")[0]:
                    glfw.set_window_should_close(window, True)

                imgui.end_menu()

            if imgui.begin_menu("Edit", True):
                if imgui.menu_item("Add Primitive/Operation", "Ctrl+A")[0]:
                    show_selection_window = True
                if imgui.menu_item("Compile Shader", "Ctrl+B")[0]:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                imgui.end_menu()
    
            if imgui.begin_menu("View", True):
                if imgui.menu_item("Settings", "F10")[0]:
                    show_settings_window = True
                imgui.end_menu()
    
            if imgui.begin_menu("About", True):
                if imgui.menu_item("Information")[0]:
                    show_about_window = True
                imgui.end_menu()


            # --- Fast Change Rendering mode ---
            cursor_pos = imgui.get_cursor_pos()
            window_width = imgui.get_window_width()
            remaining_width = window_width - cursor_pos.x

            # Calculate positions for centered buttons
            button_width = 100
            spacing = 20
            total_buttons_width = 2 * button_width + spacing
            start_x = (cursor_pos.x + (remaining_width - total_buttons_width)) / 2


            imgui.set_cursor_pos_x(start_x)
            if imgui.button("Template", button_width):
                shader_choice = 0
                # Recompile with new shader
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.set_cursor_pos_x(start_x + button_width + spacing)
            if imgui.button("Cycles", button_width):
                shader_choice = 1
                # Recompile with new shader
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms



            imgui.end_main_menu_bar()
        

        # Check Ctrl + S/O
        if io.keys_down[glfw.KEY_O] and io.key_ctrl:
            if not last_key_o_pressed: 
                success, message = load_scene_dialog(scene_builder)
                save_load_message = message
                save_load_message_time = time.time()
                if success:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    selected_item_id = None
                    selection_mode = None
                last_key_o_pressed = True
        else:
            last_key_o_pressed = False

        #####

        if io.keys_down[glfw.KEY_S] and io.key_ctrl:
            if not last_key_s_pressed: 
                success, message = save_scene_dialog(scene_builder)
                save_load_message = message
                save_load_message_time = time.time()
                if success:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    selected_item_id = None
                    selection_mode = None
                last_key_s_pressed = True
        else:
            last_key_s_pressed = False


        # Check Undo/Redo keys Ctrl+Z/Y
        if io.keys_down[glfw.KEY_Z] and io.key_ctrl and not io.key_shift:
            if not last_key_z_pressed: 
                undo_success = glob_history.undo()
                if undo_success:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                last_key_z_pressed = True
        else:
            last_key_z_pressed = False


        if (io.keys_down[glfw.KEY_Y] and io.key_ctrl) or \
           (io.keys_down[glfw.KEY_Z] and io.key_ctrl and io.key_shift):
            if not last_key_y_pressed: 
                undo_success = glob_history.redo()
                if undo_success:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                last_key_y_pressed = True
        else:
            last_key_y_pressed = False



        # Drag on G
        # Read raw key states using GLFW so ImGui doesn't interfere with toggles
        key_g_is_down = glfw.get_key(window, glfw.KEY_G) == glfw.PRESS
        key_x_is_down = glfw.get_key(window, glfw.KEY_X) == glfw.PRESS
        key_y_is_down = glfw.get_key(window, glfw.KEY_Y) == glfw.PRESS
        key_z_is_down = glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS

        # Toggle dragging on G press (edge detect)
        if key_g_is_down and not last_key_g_pressed:
            # Toggle dragging state
            dragging = not dragging

            if dragging:
                # Start dragging: capture which item and initialize drag state
                dragging_op_id = selected_item_id

                if dragging_op_id and dragging_op_id in scene_builder.id_to_index:
                    item_type, idx = scene_builder.id_to_index[dragging_op_id]
                    if item_type == 'primitive':
                        prim = scene_builder.primitives[idx][1]
                        # Copy the primitive start position so later comparisons/undo work
                        drag_start_pos = prim.position[:]
                        # Reset accumulated movement
                        drag_accum = [0.0, 0.0, 0.0]
                        # Record starting mouse cursor (independent of camera last_x/last_y)
                        drag_last_x, drag_last_y = glfw.get_cursor_pos(window)
                    else:
                        # nothing valid to drag
                        dragging_op_id = None
                        drag_start_pos = None
                        drag_accum = [0.0, 0.0, 0.0]
                else:
                    dragging_op_id = None
                    drag_start_pos = None
                    drag_accum = [0.0, 0.0, 0.0]

                # Reset axis toggles when starting a new drag
                axis_toggled_gx = axis_toggled_gy = axis_toggled_gz = False

            else:
                # Stop dragging: commit final position (register undo/redo)
                if dragging_op_id and dragging_op_id in scene_builder.id_to_index:
                    item_type, idx = scene_builder.id_to_index[dragging_op_id]
                    if item_type == 'primitive':
                        prim = scene_builder.primitives[idx][1]
                        final_pos = prim.position
                        # Register only if changed
                        if drag_start_pos is not None and final_pos != drag_start_pos:
                            scene_builder.modify_primitive_property(dragging_op_id, 'position', drag_start_pos, final_pos)
                            # Optionally recompile if scene/code generation depends on selection
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                # clear drag state
                dragging_op_id = None
                drag_start_pos = None
                drag_accum = [0.0, 0.0, 0.0]
                axis_toggled_gx = axis_toggled_gy = axis_toggled_gz = False

        # Always update last_key_g_pressed for proper edge detection next frame
        last_key_g_pressed = key_g_is_down

        # Handle axis toggles (Blender-style): toggle on key press, update debounced state every frame
        if dragging:
            if key_x_is_down and not last_key_gx_pressed:
                state = not axis_toggled_gx
                axis_toggled_gx, axis_toggled_gy, axis_toggled_gz = state, False, False

            if key_y_is_down and not last_key_gy_pressed:
                state = not axis_toggled_gy
                axis_toggled_gx, axis_toggled_gy, axis_toggled_gz = False, state, False

            if key_z_is_down and not last_key_gz_pressed:
                state = not axis_toggled_gz
                axis_toggled_gx, axis_toggled_gy, axis_toggled_gz = False, False, state

        # Update the "last key" flags for X/Y/Z so we only toggle once per press
        last_key_gx_pressed = key_x_is_down
        last_key_gy_pressed = key_y_is_down
        last_key_gz_pressed = key_z_is_down

        # Determine active axis (None => free drag)
        active_axis = None
        if axis_toggled_gx:
            active_axis = 0
        elif axis_toggled_gy:
            active_axis = 1
        elif axis_toggled_gz:
            active_axis = 2

        # Per-frame drag movement (this must run every frame while dragging)
        if dragging and dragging_op_id and dragging_op_id in scene_builder.id_to_index:
            # read current mouse and compute delta since last frame
            current_x, current_y = glfw.get_cursor_pos(window)
            dx = current_x - drag_last_x
            dy = current_y - drag_last_y
            # store for next frame
            drag_last_x, drag_last_y = current_x, current_y

            # convert to mouse-space movement (invert Y so screen-up => world up)
            mouse_delta_x = dx * DRAG_SENSITIVITY
            mouse_delta_y = -dy * DRAG_SENSITIVITY

            if np.linalg.norm(np.array([mouse_delta_x, mouse_delta_y])) > 0.01:
                frame_count = 0
                clear_accumulation_fbos(accumulation_fbos,scaled_rendering_width, scaled_rendering_height)

            # transform mouse deltas into world-space using camera right/up vectors
            # right_x, right_y, right_z and up_x, up_y, up_z must be computed earlier (they are in your code)
            move_delta_x = mouse_delta_x * right_x + mouse_delta_y * up_x
            move_delta_y = mouse_delta_x * right_y + mouse_delta_y * up_y
            move_delta_z = mouse_delta_x * right_z + mouse_delta_y * up_z

            # axis constraints (lock movement to a single world axis)
            if active_axis is not None:
                if active_axis == 0:
                    move_delta_y = 0.0
                    move_delta_z = 0.0
                elif active_axis == 1:
                    move_delta_x = 0.0
                    move_delta_z = 0.0
                elif active_axis == 2:
                    move_delta_x = 0.0
                    move_delta_y = 0.0

            # accumulate world movement since drag started
            drag_accum[0] += move_delta_z
            drag_accum[1] += move_delta_y
            drag_accum[2] += move_delta_x

            # compute new primitive position from saved start pos + accumulated movement
            item_type, idx = scene_builder.id_to_index[dragging_op_id]
            prim = scene_builder.primitives[idx][1]
            if drag_start_pos is None:
                drag_start_pos = prim.position.copy()

            new_pos = [
                drag_start_pos[0] + drag_accum[0],
                drag_start_pos[1] + drag_accum[1],
                drag_start_pos[2] + drag_accum[2],
            ]

            # apply live position (no historical entry yet — recorded on drag end)
            prim.position = new_pos
            drag_position = new_pos.copy()

        else:
            # When not dragging keep shader uniform aligned with selection or zero
            if selected_item_id and selected_item_id in scene_builder.id_to_index:
                itype, idx = scene_builder.id_to_index[selected_item_id]
                if itype == 'primitive':
                    prim = scene_builder.primitives[idx][1]
                    drag_position = prim.position
            else:
                drag_position = [0.0, 0.0, 0.0]


        # Check F10 for settings (with debouncing)
        if io.keys_down[glfw.KEY_F10]:
            if not last_key_f10_pressed:
                show_settings_window = True
                last_key_f10_pressed = True
        else:
            last_key_f10_pressed = False
        
        # --- RENDER TO FRAMEBUFFER AT SCALED RESOLUTION ---
        # If we've already rendered & displayed the accumulation buffer above (cycles shader),
        # skip the further framebuffer / direct rendering to avoid double-draw and viewport offset.
        if shader is not None and shader_choice == 1 and use_accumulation == 1:
            # accumulation rendering & display already handled above
            pass

        elif shader is not None and display_shader is not None and resolution_scale != 1.0:
            # Setup framebuffer
            if setup_framebuffer(scaled_rendering_width, scaled_rendering_height):
                # Render to framebuffer
                glBindFramebuffer(GL_FRAMEBUFFER, fbo)
                glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)
                glClear(GL_COLOR_BUFFER_BIT)
                
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], scaled_rendering_width, scaled_rendering_height)
                    glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                    glUniform1f(uniform_locs['camYaw'], cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], cam_pitch)
                    glUniform1f(uniform_locs['radius'], cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                    set_move_pos_uniform(shader, uniform_locs, drag_position)

                glBindVertexArray(vao)
                bind_sprite_textures(uniform_locs)
                glDrawArrays(GL_QUADS, 0, 4)
                
                # Switch back to default framebuffer
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glViewport(0, 0, width, height)
                
                # Display the texture stretched to the viewport
                glUseProgram(display_shader)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, render_texture)
                glUniform1i(glGetUniformLocation(display_shader, "renderTexture"), 0)
                
                # Set viewport to the rendering area (accounting for menu bar)
                glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
                glBindVertexArray(display_vao)
                bind_sprite_textures(uniform_locs)
                glDrawArrays(GL_QUADS, 0, 4)
                glBindVertexArray(0)
                
                # Reset viewport
                glViewport(0, 0, width, height)
            else:
                # Fallback to direct rendering if framebuffer fails
                if shader is not None:
                    glUseProgram(shader)
                    if uniform_locs is not None:
                        current_time_uniform = time.time() - start_time
                        glUniform1f(uniform_locs['time'], current_time_uniform)
                        glUniform2f(uniform_locs['resolution'], scaled_rendering_width, scaled_rendering_height)
                        glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                        glUniform1f(uniform_locs['camYaw'], cam_yaw)
                        glUniform1f(uniform_locs['camPitch'], cam_pitch)
                        glUniform1f(uniform_locs['radius'], cam_radius)
                        glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                        set_move_pos_uniform(shader, uniform_locs, drag_position)

                    glViewport(panel_width, menu_bar_height, scaled_rendering_width, scaled_rendering_height)
                    glBindVertexArray(vao)
                    bind_sprite_textures(uniform_locs)
                    glDrawArrays(GL_QUADS, 0, 4)
                    glViewport(0, 0, width, height)
        else:
            # Direct rendering when scale is 1.0 or display shader not available
            # Skip if accumulation handled above (see guard at top)
            if shader is not None:
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], rendering_width, rendering_height)
                    # Default framebuffer viewport is offset by the left panel and menu bar
                    glUniform2f(uniform_locs['viewportOffset'], float(panel_width), float(menu_bar_height))
                    glUniform1f(uniform_locs['camYaw'], cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], cam_pitch)
                    glUniform1f(uniform_locs['radius'], cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                    set_move_pos_uniform(shader, uniform_locs, drag_position)

                # Check if viewport is minimized
                if rendering_width > 0 and rendering_height > 0:
                    glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
                    glBindVertexArray(vao)
                    bind_sprite_textures(uniform_locs)
                    glDrawArrays(GL_QUADS, 0, 4)

                glViewport(0, 0, width, height)
        

        # --- SETTINGS WINDOW ---
        if show_settings_window:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 150)
            imgui.set_next_window_size(400, 300)  # Increased height
            is_open, show_settings_window = imgui.begin("Settings", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                show_settings_window = False
            
            imgui.text("Rendering Settings")
            imgui.separator()
            
            # Shader Selection
            imgui.text("Fragment Shader:")
            clicked, shader_choice = imgui.combo(
                "##shader_select",
                shader_choice,
                [name.replace("shaders/fragment/", "") for name in shader_names]
            )
            if clicked:
                # Recompile with new shader
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms
            
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            # Resolution Scale
            imgui.text("Resolution Scale:")
            imgui.same_line()
            imgui.text(f"{resolution_scale:.2f}x")
            
            changed, resolution_scale = imgui.slider_float("##resolution_scale", resolution_scale, 0.25, 2.0, "%.2f")
            if changed:
                frame_count = 0


            imgui.spacing()
            imgui.text_colored("1.0 = Normal resolution", 0.7, 0.7, 0.7, 1.0)
            imgui.text_colored("2.0 = Oversampling (better quality)", 0.7, 0.7, 0.7, 1.0)
            imgui.text_colored("<1.0 = Low resolution (better performance)", 0.7, 0.7, 0.7, 1.0)
            
            imgui.spacing()
            imgui.separator()


            # Show Sky colors
            imgui.text("Sky Top Color:")
            top_color_changed, top_color_rgba = imgui.color_edit3("SkyTopColor##color", sky_top_color[0], sky_top_color[1], sky_top_color[2])
            if top_color_changed:
                sky_top_color = list(top_color_rgba[:3])  # Only use RGB, ignore alpha
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.text("Sky Bottom Color:")
            bottom_color_changed, bottom_color_rgba = imgui.color_edit3("SkyBottomColor##color", sky_bottom_color[0], sky_bottom_color[1], sky_bottom_color[2])
            if bottom_color_changed:
                sky_bottom_color = list(bottom_color_rgba[:3])  # Only use RGB, ignore alpha
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            if shader_choice == 0:
                imgui.text("Grid Enabled:")
                changed, GridEnabled = imgui.checkbox("", GridEnabled)
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

                imgui.spacing()
                imgui.separator()

            elif shader_choice == 1:
                imgui.text("Max Samples count:")
                changed, max_frames = imgui.input_int("", max_frames)
                if changed:
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

                imgui.spacing()
                imgui.separator()
            

            imgui.text("Sun:")
            changed, LightDir = input_vec3("Sun Direction", LightDir)
            if changed:
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            imgui.spacing()
            imgui.separator()


            # Calculate scaled size for display
            scaled_w = int(rendering_width * resolution_scale)
            scaled_h = int(rendering_height * resolution_scale)
            imgui.text(f"Current render size: {scaled_w}x{scaled_h}")
            imgui.text(f"Base size: {rendering_width}x{rendering_height}")
            
            imgui.spacing()
            if imgui.button("Close", -1):
                show_settings_window = False
            
            
            imgui.end()


        if show_export_vol_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 110)
            imgui.set_next_window_size(300, 220)
            is_open, show_export_vol_window = imgui.begin("Export as Volume", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                show_export_vol_window = False

            imgui.text("Grid Size:")
            changed, grid_size = imgui.input_int("##GridSize", grid_size, 8)
            imgui.text_colored(
                "Note that its dimensions range \nfrom -GridSize/2 to +GridSize/2.",
                0.56, 0.93, 0.56
            )

            imgui.spacing()

            changed, vox_quality = input_float("Voxelization Quality", vox_quality, 0.25, 100)

            imgui.separator()
            imgui.spacing()

            file_preview_size = sdfexp.calculate_sdf_file_size(grid_size, vox_quality)
            if file_preview_size[1]>1:
                imgui.text(f"File size = {file_preview_size[1]:.2f} mb")
            else:
                imgui.text(f"File size = {file_preview_size[0]:.2f} kb")

            imgui.spacing()
            imgui.spacing()

            if imgui.button("Cancel", 135,30):
                show_export_vol_window = False

            imgui.same_line(150)

            if imgui.button("Export", 135,30):
                code = scene_builder.generate_raymarch_code()
                comp_bin = sdfexp.compute_sdf_3d(grid_size, vox_quality, code, window)
                save_sdfvol_dialog(comp_bin)

                show_export_vol_window = False

            imgui.end()

        if show_export_obj_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 130)
            imgui.set_next_window_size(300, 260)
            is_open, show_export_obj_window = imgui.begin("Export to OBJ", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                show_export_obj_window = False

            imgui.text("Grid Size:")
            changed, grid_size = imgui.input_int("##GridSize", grid_size, 8)
            imgui.text_colored(
                "Note that its dimensions range \nfrom -GridSize/2 to +GridSize/2.",
                0.56, 0.93, 0.56
            )

            imgui.spacing()

            changed, vox_quality = input_float("Voxelization Quality", vox_quality, 0.25, 100)

            imgui.separator()
            imgui.spacing() 

            changed, export_level = input_float("Level", export_level, 0.05, 100)

            imgui.spacing()

            changed, export_z_up = imgui.checkbox("Z up", export_z_up)

            imgui.separator()
            imgui.spacing()

            if imgui.button("Cancel", 135,30):
                show_export_obj_window = False

            imgui.same_line(150)

            if imgui.button("Export", 135,30):
                code = scene_builder.generate_raymarch_code()
                comp_bin = sdfexp.compute_sdf_3d(grid_size, vox_quality, code, window)
                elvl = np.interp(export_level, [0,1], [comp_bin.min(), comp_bin.max()])
                save_sdfobj_dialog(comp_bin, export_z_up, elvl)

                show_export_obj_window = False

            imgui.end()


        if show_about_window:
            imgui.set_next_window_position(width // 2 - 250, height // 2 - 200)
            imgui.set_next_window_size(500, 400)  # Increased height
            is_open, show_about_window = imgui.begin("About", True, imgui.WINDOW_NO_COLLAPSE)
            
            if not is_open:
                show_about_window = False
            
            about_text = """
MIT License

Copyright (c) 2025-present EmberNoGlow

------------------

This is a project in which I created rendering and full interaction with sdf primitives. Using Python, GLSL, Imgui, glfw, pyopengl.

------------------

Thank you for using this project! If you liked the project, give it a star on github.

You can also support the project by reporting an error, or by suggesting an improvement by opening a Pull Request (PR).
            """
            imgui.begin_child("LicenseText", width=490, height=300, border=True)
            imgui.text_wrapped(about_text)
            imgui.end_child()
            
            imgui.spacing()

            # --- Github project page URL ---
            import webbrowser
            
            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.6, 0.4, 0.1, 0.5)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 0.5, 1.0)
            if imgui.selectable("Visit project page in Github (Double click)", False):
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    webbrowser.open("https://github.com/EmberNoGlow/SDF-Model-Editor-Demo")
            imgui.pop_style_color()
            imgui.pop_style_color(3)

            imgui.spacing()
            if imgui.button("Close", -1):
                show_about_window = False

            imgui.end()

        # --- FPS OVERLAY (Top Right, above right panel) ---
        fps_x = width - panel_width - FPS_WINDOW_WIDTH - FPS_WINDOW_OFFSET
        imgui.set_next_window_position(fps_x, FPS_WINDOW_OFFSET)
        imgui.set_next_window_size(FPS_WINDOW_WIDTH, FPS_WINDOW_HEIGHT)
        imgui.begin("FPS", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        if shader_choice == 0:
            imgui.text_colored("FPS: " + str(fps_value), 0.0, 1.0, 0.0, 1.0)
        elif shader_choice == 1:
            imgui.text_colored("Sample: " + str(frame_count), 1.0, 1.0, 0.0, 1.0)

        imgui.end()
        

        # Display save/load status message
        if save_load_message is not None:
            # Show message for 3 seconds
            if time.time() - save_load_message_time < 3.0:
                imgui.set_next_window_position(width // 2 - 150, 100)
                imgui.begin("Status", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)

                # Color based on success
                is_success = "saved" in save_load_message.lower() or "loaded" in save_load_message.lower()
                color = (0.0, 1.0, 0.0, 1.0) if is_success else (1.0, 0.0, 0.0, 1.0)
                imgui.text_colored(save_load_message, *color)

                imgui.end()
            else:
                save_load_message = None


        # --- Error Display (if shader compilation failed) ---
        if shader_compile_error:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 50)
            imgui.set_next_window_size(400, 100)
            imgui.begin("Shader Compilation Error", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
            imgui.text_colored("Error:", 1.0, 0.0, 0.0, 1.0)
            imgui.same_line()
            imgui.text_wrapped(shader_compile_error)
            if imgui.button("Dismiss"):
                shader_compile_error = None
            imgui.end()

        # --- LEFT PANEL: Scene Tree ---
        # Offset panels below menu bar (menu_bar_height already calculated above)
        imgui.set_next_window_position(0, menu_bar_height)
        imgui.set_next_window_size(panel_width, height - menu_bar_height)
        imgui.begin("Scene Tree", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        
        wbn = 16
        hbn = 20

        def format_label(name, op_id, max_chars=16):
            # Truncate name to max_chars, preserving full ID
            if len(name) > max_chars:
                truncated_name = name[:max_chars - 3] + "..."
            else:
                truncated_name = name
            return f"{truncated_name} ({op_id})"
        
        imgui.text("Primitives:")
        for op_id, primitive in scene_builder.primitives:
            label = format_label(primitive.ui_name, op_id)
            flags = imgui.TREE_NODE_LEAF
            if selected_item_id == op_id:
                flags |= imgui.TREE_NODE_SELECTED

            # Create the buttons
            if imgui.button(f"^##up_{op_id}", wbn, hbn):
                idx = scene_builder.id_to_index[op_id][1]
                if idx > 0:
                    scene_builder.move_item(op_id, idx - 1)
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

            imgui.same_line()

            if imgui.button(f"v##down_{op_id}", wbn, hbn):
                idx = scene_builder.id_to_index[op_id][1]
                if idx < len(scene_builder.primitives) - 1:
                    scene_builder.move_item(op_id, idx + 1)
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

            imgui.same_line()

            # Then create the tree node (label will appear after buttons)
            node_open = imgui.tree_node(label, flags)

            # Handle selection when the node is clicked
            if imgui.is_item_clicked():
                selected_item_id = op_id
                selection_mode = 'primitive'
                renaming_item_id = None

                # Recompile shader
                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            if node_open:
                imgui.tree_pop()

        imgui.spacing()
        imgui.text("Operations:")
        for op_id, operation in scene_builder.operations:
            label = format_label(operation.ui_name, op_id)
            flags = imgui.TREE_NODE_LEAF
            if selected_item_id == op_id:
                flags |= imgui.TREE_NODE_SELECTED

            # First create the buttons (they'll appear on the left)
            if imgui.button(f"^##upop_{op_id}", wbn, hbn):
                idx = scene_builder.id_to_index[op_id][1]
                if idx > 0:
                    scene_builder.move_item(op_id, idx - 1)
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

            imgui.same_line()  # Keep buttons on the same line

            if imgui.button(f"v##downop_{op_id}", wbn, hbn):
                idx = scene_builder.id_to_index[op_id][1]
                if idx < len(scene_builder.operations) - 1:
                    scene_builder.move_item(op_id, idx + 1)
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms

            imgui.same_line()  # Keep label on the same line as buttons

            # Then create the tree node (label will appear after buttons)
            node_open = imgui.tree_node(label, flags)

            # Handle selection when the node is clicked
            if imgui.is_item_clicked():
                selected_item_id = op_id
                selection_mode = 'operation'
                renaming_item_id = None

                success, new_uniforms = recompile_shader()
                if success:
                    uniform_locs = new_uniforms

            if node_open:
                imgui.tree_pop()

        imgui.spacing()
        if imgui.button("Add (Ctrl+A)", -1):
            show_selection_window = True
        
        if imgui.button("Compile (Ctrl+B)", -1):
            success, new_uniforms = recompile_shader()
            if success:
                uniform_locs = new_uniforms
        
        imgui.spacing()
        imgui.text_colored("Press Delete to remove", 1.0, 1.0, 0.0, 1.0)

        imgui.end()

        # --- RIGHT PANEL: Properties/Inspector ---
        imgui.set_next_window_position(width - panel_width, menu_bar_height)
        imgui.set_next_window_size(panel_width, height - menu_bar_height)
        imgui.begin("Inspector", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        if renaming_item_id is not None:
            imgui.text("Renaming:")
            changed, rename_text = imgui.input_text("##rename", rename_text, 256)
            
            if imgui.button("OK", width / 5):
                if renaming_item_id in scene_builder.id_to_index:
                    item_type, index = scene_builder.id_to_index[renaming_item_id]
                    if item_type == 'primitive':
                        scene_builder.primitives[index][1].ui_name = rename_text
                    else:
                        scene_builder.operations[index][1].ui_name = rename_text
                renaming_item_id = None
            
            imgui.same_line()
            if imgui.button("Cancel", width / 5):
                renaming_item_id = None
            
            imgui.separator()

        if selected_item_id is not None:
            imgui.text(f"Selected: {scene_builder.get_item_name(selected_item_id)}")
            imgui.separator()

            if selection_mode == 'primitive':
                # Find the primitive
                for op_id, primitive in scene_builder.primitives:
                    if op_id == selected_item_id:
                        imgui.text(f"Type: {primitive.primitive_type}")
                    
                        
                        if primitive.primitive_type == "sprite":
                            # sprite_index is stored in primitive.kwargs at creation time
                            sprite_idx = primitive.kwargs.get('sprite_index', None)
                            if sprite_idx is None or sprite_idx >= len(sprites_array):
                                imgui.text_colored("Sprite data missing or corrupted", 1.0, 0.0, 0.0, 1.0)
                            else:
                                spr = sprites_array[sprite_idx]
                                imgui.text("Plane parameters:")
                                changed, primitive.position = input_vec3("Point", primitive.position, STEP_VARIABLE_FLOAT, panel_elem_width_vec3)
                                changed2, spr.planeNormal = input_vec3("Normal", spr.planeNormal, STEP_VARIABLE_FLOAT, panel_elem_width_vec3)
                                changed3, spr.planeWidth = input_float("Width", spr.planeWidth, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                                changed4, spr.planeHeight = input_float("Height", spr.planeHeight, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                                spr.planePoint = primitive.position
                                if changed or changed2 or changed3 or changed4:
                                    success, new_uniforms = recompile_shader()
                                    if success:
                                        uniform_locs = new_uniforms

                                imgui.separator()
                                imgui.text("Mapping:")
                                uv2 = spr.uvSize
                                changed_uv, uv2 = input_vec2("UV Size", uv2, 0.1, panel_elem_width_vec3)
                                spr.uvSize[0], spr.uvSize[1] = uv2[0], uv2[1]
                                changed_alpha, spr.Alpha = input_float("Alpha", spr.Alpha, 0.01, panel_elem_width_float)
                                changed_lod, spr.LOD = input_float("LOD", spr.LOD, 0.1, panel_elem_width_float)

                                if changed_uv or changed_alpha or changed_lod:
                                    success, new_uniforms = recompile_shader()
                                    if success:
                                        uniform_locs = new_uniforms

                                # Show texture status and "Load Texture" button
                                if spr.texture_id:
                                    imgui.text(f"Texture loaded: {spr.tex_size[0]}x{spr.tex_size[1]}")
                                else:
                                    imgui.text_colored("No texture loaded", 0.9, 0.3, 0.3, 1.0)

                                imgui.spacing()
                                if imgui.button("Load Texture", -1):
                                    # Use tkinter filedialog (as in other parts of the code)
                                    root = tk.Tk()
                                    root.withdraw()
                                    filetypes = [("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tga")), ("All files", "*.*")]
                                    filepath = filedialog.askopenfilename(filetypes=filetypes)
                                    root.destroy()
                                    if filepath:
                                        ok = spr.load_texture_from_file(filepath)
                                        if ok:
                                            # Ensure sampler name is unique and recompile so the sampler uniform is declared/located
                                            spr.SprTexture = f"sprTex{sprite_idx}"
                                            success, new_uniforms = recompile_shader()
                                            if success:
                                                uniform_locs = new_uniforms

                        # Size/Radius - varies by primitive type
                        # I have python 3.8, "match", get out.
                        if primitive.primitive_type == "sphere":
                            changed, primitive.size_or_radius[0] = input_float(
                                "Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                        elif primitive.primitive_type == "torus":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Major Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Minor Radius", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "hex_prism":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Hex Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Height", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "vertical_capsule":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Height", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Radius", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "capped_cylinder":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Radius", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Height", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "rounded_cylinder":
                            changed1, primitive.size_or_radius[0] = input_float(
                                "Radius A", primitive.size_or_radius[0], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, primitive.size_or_radius[1] = input_float(
                                "Radius B", primitive.size_or_radius[1], 
                                STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed = changed1 or changed2
                        else:
                            if primitive.primitive_type not in ["cone", "plane", "rounded_cylinder", "pointer", "sprite"]:
                                changed, primitive.size_or_radius = input_vec3(
                                    "Size", primitive.size_or_radius, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                                )
                        if primitive.primitive_type not in ["pointer", "sprite"]: # HACK
                            if changed:
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # Special parameters for specific primitives
                        if primitive.primitive_type == "cone":
                            c_sin = primitive.kwargs.get('c_sin', 0.5)
                            c_cos = primitive.kwargs.get('c_cos', 0.866)
                            height = primitive.kwargs.get('height', 1.0)
                            changed1, c_sin = input_float(
                                "Sin(Angle)", c_sin, STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed2, c_cos = input_float(
                                "Cos(Angle)", c_cos, STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            changed3, height = input_float(
                                "Height", height, STEP_VARIABLE_FLOAT, panel_elem_width_float
                            )
                            if changed1 or changed2 or changed3:
                                primitive.kwargs['c_sin'] = c_sin
                                primitive.kwargs['c_cos'] = c_cos
                                primitive.kwargs['height'] = height
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        elif primitive.primitive_type == "plane":
                            normal = primitive.kwargs.get('normal', [0.0, 1.0, 0.0])
                            h = primitive.kwargs.get('h', 0.0)
                            changed1, normal = input_vec3("Normal", normal, STEP_VARIABLE_FLOAT, panel_elem_width_vec3)
                            changed2, h = input_float("Offset (h)", h, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                            if changed1 or changed2:
                                # Normalize the normal vector
                                norm_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
                                if norm_len > 0.001:
                                    normal = [normal[0]/norm_len, normal[1]/norm_len, normal[2]/norm_len]
                                primitive.kwargs['normal'] = normal
                                primitive.kwargs['h'] = h
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        elif primitive.primitive_type == "rounded_cylinder":
                            height = primitive.kwargs.get('height', 1.0)
                            changed, height = input_float("Height", height, STEP_VARIABLE_FLOAT, panel_elem_width_float)
                            if changed:
                                primitive.kwargs['height'] = height
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # --- Inspector: add UI to edit pointer function selection (inside the primitive inspector branch) ---
                        if primitive.primitive_type == "pointer":
                            old_pos = primitive.position
                            changed_pos, primitive.position = input_vec3(
                                "Position", primitive.position, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                            )
                            if changed_pos:
                                scene_builder.modify_primitive_property(op_id, 'position', old_pos, primitive.position)
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            # List of available pointer functions (must exist in sdf_library.glsl)
                            pointer_funcs = [
                                "pointer_identity",
                                "pointer_symmetry_x",
                                "pointer_symmetry_y",
                                "pointer_symmetry_z",
                                # add your custom pointer function names here...
                            ]
                            current_func = primitive.kwargs.get('func', 'pointer_identity')
                            try:
                                current_index = pointer_funcs.index(current_func)
                            except ValueError:
                                pointer_funcs.append(current_func)
                                current_index = len(pointer_funcs)-1

                            clicked, new_index = imgui.combo("Function", current_index, pointer_funcs)
                            if clicked:
                                new_func = pointer_funcs[new_index]
                                old_func = current_func
                                primitive.kwargs['func'] = new_func
                                # Record change in history for undo/redo
                                scene_builder.modify_primitive_property(op_id, "kwargs.func", old_func, new_func)
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            imgui.separator()
                            imgui.text("Pointer functions mutate \nthe raymarch point `p` \nfor subsequent primitives.")
                            imgui.text_colored("Place a pointer earlier in \nthe tree to affect later objects.", 0.9, 0.8, 0.2, 1.0)

                        elif primitive.primitive_type == "sprite":
                            pass # Skip Transforms and Color

                        else:
                            # Special parameters for specific primitives
                            if primitive.primitive_type == "round_box":
                                imgui.spacing()
                                changed, primitive.kwargs['radius'] = input_float(
                                    "Radius", primitive.kwargs.get('radius', 0.1),STEP_VARIABLE_FLOAT, panel_elem_width_float
                                    )
                                if changed:
                                    success, new_uniforms = recompile_shader()
                                    if success:
                                        uniform_locs = new_uniforms


                            imgui.begin_group()

                            imgui.spacing()
                            imgui.separator()
                            imgui.dummy((panel_width/4)-8, 0)
                            imgui.same_line()
                            imgui.text_colored("Transform", 1.0,0.7,0.5,1.0)
                            imgui.spacing()
                
                            imgui.end_group()


                            # Position
                            old_pos = primitive.position
                            changed, primitive.position = input_vec3(
                                "Position", primitive.position, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                            )
                            if changed:
                                scene_builder.modify_primitive_property(op_id, 'position', old_pos, primitive.position)
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms


                            # Show rotation as degrees
                            current_degrees = [math.degrees(a) for a in primitive.rotation]
                            changed, degs = input_vec3(
                                "Rotation °", current_degrees, STEP_VARIABLE_ROTATION, panel_elem_width_vec3
                            )
                            if changed:
                                primitive.rotation = [math.radians(a) for a in degs]
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms

                            # Scale
                            changed, primitive.scale = input_vec3(
                                "Scale", primitive.scale, STEP_VARIABLE_FLOAT, panel_elem_width_vec3
                            )
                            if changed:
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                    
                            
                            # Color picker
                            imgui.begin_group()

                            imgui.spacing()
                            imgui.separator()
                            imgui.dummy((panel_width/3)-16, 0)
                            imgui.same_line()
                            imgui.text_colored("Color", 1.0,0.7,0.5,1.0)
                            imgui.spacing()
                
                            imgui.end_group()

                            # Color edit - imgui automatically shows a picker button
                            old_color = primitive.color.copy()
                            color_changed, color_rgba = imgui.color_edit3("Color##color", *primitive.color)
                            if color_changed:
                                primitive.color = list(color_rgba[: 3])
                                scene_builder.modify_primitive_property(op_id, 'color', old_color, primitive.color)
                                success, new_uniforms = recompile_shader()
                                if success: 
                                    uniform_locs = new_uniforms
                            
                            # Show color preview button
                            imgui.same_line()
                            # color_button takes: label, r, g, b, flags, size_x, size_y
                            #imgui.color_button("Preview##color_preview", primitive.color[0], primitive.color[1], primitive.color[2], 0, 20, 20)
                            
                            # Alternative: RGB sliders for fine control
                            imgui.spacing()
                            imgui.text("RGB Sliders:")
                            r_changed, primitive.color[0] = imgui.slider_float("R##color_r", primitive.color[0], 0.0, 1.0)
                            g_changed, primitive.color[1] = imgui.slider_float("G##color_g", primitive.color[1], 0.0, 1.0)
                            b_changed, primitive.color[2] = imgui.slider_float("B##color_b", primitive.color[2], 0.0, 1.0)
                            if r_changed or g_changed or b_changed:
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                            
                            break

            elif selection_mode == 'operation':
                # Find the operation
                for op_id, operation in scene_builder.operations:
                    if op_id == selected_item_id:
                        imgui.text(f"Type: {operation.operation_type}")
                        
                        # Get valid operands (only those declared before this operation)
                        valid_operands = scene_builder.get_valid_operands(selected_item_id)
                        
                        # Determine if this is a single-operand or two-operand operation
                        is_single_operand = operation.operation_type in ['round', 'onion', 'invert']
                        num_operands = 1 if is_single_operand else 2
                        
                        if len(valid_operands) == 0:
                            imgui.text_colored("No valid operands available!", 1.0, 0.0, 0.0, 1.0)
                        else:
                            for i in range(num_operands):
                                operand_label = "Operand" if is_single_operand else ("Operand A" if i == 0 else "Operand B")
                                
                                # Create a combo box for selecting operands
                                current_operand = operation.args[i]
                                current_index = 0
                                
                                # Find current operand in valid list
                                for idx, (item_id, _) in enumerate(valid_operands):
                                    if item_id == current_operand:
                                        current_index = idx
                                        break
                                
                                clicked, new_index = imgui.combo(
                                    f"##operand_{i}",
                                    current_index,
                                    [scene_builder.get_item_name(item_id) for item_id, _ in valid_operands]
                                )
                                                                
                                if clicked:
                                    old_operand = operation.args[i]
                                    new_operand = valid_operands[new_index][0]
                                    # Use scene_builder API so the change is recorded in history
                                    scene_builder.modify_operation_parameter(op_id, f"args[{i}]", old_operand, new_operand)
                                    success, new_uniforms = recompile_shader()
                                    if success:
                                        uniform_locs = new_uniforms
                        
                        imgui.separator()
                        
                        # Show float parameter for single-operand operations with parameters
                        if hasattr(operation, 'float_param') and operation.float_param is not None:
                            changed, operation.float_param = input_float("Parameter", operation.float_param, 0.01, panel_elem_width_float)
                            if changed:
                                # Update the operation with new parameter
                                if len(operation.args) >= 2:
                                    operation.args[1] = operation.float_param
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # Show smoothing factor for smooth operations
                        elif operation.smooth_k is not None:
                            changed, operation.smooth_k = input_float("Smoothing Factor (k)", operation.smooth_k, 0.01, panel_elem_width_float)
                            if changed:
                                if len(operation.args) >= 3:
                                    operation.args[2] = operation.smooth_k
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        break
        else:
            imgui.text("Select an item to edit")

        imgui.end()

        # --- SELECTION WINDOW ---
        if show_selection_window:
            imgui.set_next_window_position(width // 2 - 150, height // 2 - 150)
            imgui.set_next_window_size(300, 400)
            # Prevent window from being collapsed - use WINDOW_NO_COLLAPSE flag
            is_open, show_selection_window = imgui.begin("Add Primitive/Operation", True, imgui.WINDOW_NO_COLLAPSE)

            if not is_open:
                show_selection_window = False

            imgui.text("Select a Primitive:")
            
            primitives_list = [
                ("Box", "box", (0.5, 0.5, 0.5)),
                ("Sphere", "sphere", 0.5),
                ("Round Box", "round_box", (0.5, 0.5, 0.5)),
                ("Torus", "torus", (0.5, 0.25)),
                ("Cone", "cone", None),
                ("Plane", "plane", None),
                ("Hex Prism", "hex_prism", (0.5, 0.5)),
                ("Vertical Capsule", "vertical_capsule", (1.0, 0.3)),
                ("Capped Cylinder", "capped_cylinder", (0.3, 1.0)),
                ("Rounded Cylinder", "rounded_cylinder", (0.3, 0.1)),
                ("Pointer", "pointer", None),
                ("Sprite", "sprite", None)
            ]

            for label, prim_type, size_radius in primitives_list:
                if imgui.button(f"  {label}", -1):
                    if prim_type == "round_box":
                        new_id = scene_builder.add_roundbox((0.0, 0.0, 0.0), size_radius, 0.1, ui_name=label)
                    elif prim_type == "sphere":
                        new_id = scene_builder.add_sphere((0.0, 0.0, 0.0), size_radius, ui_name=label)
                    elif prim_type == "torus":
                        new_id = scene_builder.add_torus((0.0, 0.0, 0.0), size_radius[0], size_radius[1], ui_name=label)
                    elif prim_type == "cone":
                        new_id = scene_builder.add_cone((0.0, 0.0, 0.0), 0.5, 0.866, 1.0, ui_name=label)
                    elif prim_type == "plane":
                        new_id = scene_builder.add_plane((0.0, 0.0, 0.0), [0.0, 1.0, 0.0], 0.0, ui_name=label)
                    elif prim_type == "hex_prism":
                        new_id = scene_builder.add_hex_prism((0.0, 0.0, 0.0), size_radius[0], size_radius[1], ui_name=label)
                    elif prim_type == "vertical_capsule":
                        new_id = scene_builder.add_vertical_capsule((0.0, 0.0, 0.0), size_radius[0], size_radius[1], ui_name=label)
                    elif prim_type == "capped_cylinder":
                        new_id = scene_builder.add_capped_cylinder((0.0, 0.0, 0.0), size_radius[0], size_radius[1], ui_name=label)
                    elif prim_type == "rounded_cylinder":
                        new_id = scene_builder.add_rounded_cylinder((0.0, 0.0, 0.0), size_radius[0], size_radius[1], 1.0, ui_name=label)
                    elif prim_type == "pointer":
                        # default pointer function
                        new_id = scene_builder.add_pointer((0.0, 0.0, 0.0), func='pointer_identity', ui_name=label)
                    elif prim_type == "sprite":
                        # Create a Sprite object and append to the global sprites_array.
                        # Default plane is centered in front of camera/origin; uvSize default 1x1.
                        new_spr = Sprite(
                        planePoint=(0.0, 0.0, 0.0),
                        planeNormal=(0.0, 0.0, 1.0),
                        planeWidth=2.0,
                        planeHeight=2.0,
                        SprTexture=f"sprTex{len(sprites_array)}",
                        uvSize=(1.0, 1.0),
                        Alpha=1.0,
                        LOD=0.0
                        )
                        sprites_array.append(new_spr)
                        # Create a SDF primitive that references the sprite index so it shows in the tree
                        new_id = scene_builder.add_primitive("sprite", (0.0, 0.0, 0.0), [0.0,0.0,0.0], ui_name=label, color=[1.0,1.0,1.0], sprite_index=len(sprites_array)-1)
                    
                    else:
                        new_id = scene_builder.add_box((0.0, 0.0, 0.0), size_radius, ui_name=label)
                    
                    success, new_uniforms = recompile_shader()
                    if success:
                        uniform_locs = new_uniforms
                    
                    selected_item_id = new_id
                    selection_mode = 'primitive'
                    show_selection_window = False

            imgui.separator()
            imgui.text("Select an Operation:")

            # Need at least 1 primitive/operation for most operations
            all_items = scene_builder.get_all_items()
            
            if len(all_items) >= 1:
                operations_list = [
                    ("Union", "union"),
                    ("Subtraction", "sub"),
                    ("Intersection", "inter"),
                    ("Smooth Union", "sunion"),
                    ("Smooth Subtraction", "ssub"),
                    ("Smooth Intersection", "sinter"),
                    ("Mix", "mix"),
                    ("XOR", "xor"),
                    ("Invert", "invert"),
                    ("Round", "round"),
                    ("Onion", "onion")
                ]

                for label, op_type in operations_list:
                    # Single-operand operations (invert, round, onion)
                    is_single_operand = op_type in ['invert', 'round', 'onion']
                    min_operands = 1
                    available_operands = len(all_items)
                    
                    if available_operands >= min_operands:
                        if imgui.button(f"  {label}", -1):
                            # For single operand operations
                            if is_single_operand:
                                if op_type == "invert": 
                                    new_id = scene_builder.invert(all_items[-1][0], ui_name=label)
                                elif op_type == "round":
                                    new_id = scene_builder.round(all_items[-1][0], 0.1, ui_name=label)
                                elif op_type == "onion":
                                    new_id = scene_builder.onion(all_items[-1][0], 0.05, ui_name=label)
                            elif op_type in ["sunion", "ssub", "sinter", "mix"]:
                                if len(all_items) >= 2:
                                    new_id = getattr(scene_builder, op_type)(all_items[-2][0], all_items[-1][0], 
                                        (0.5 if op_type == 'mix' else 0.05), ui_name=label)
                                else:
                                    new_id = getattr(scene_builder, op_type)(all_items[-1][0], all_items[-1][0], 
                                        (0.5 if op_type == 'mix' else 0.05), ui_name=label)
                            else:
                                if len(all_items) >= 2:
                                    new_id = getattr(scene_builder, op_type)(all_items[-2][0], all_items[-1][0], ui_name=label)
                                else:
                                    new_id = getattr(scene_builder, op_type)(all_items[-1][0], all_items[-1][0], ui_name=label)
                            
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                            
                            selected_item_id = new_id
                            selection_mode = 'operation'
                            show_selection_window = False
            else:
                imgui.text("Add at least 1 primitive to use operations")

            imgui.end()

        # Render ImGui
        imgui.render()
        impl.render(imgui.get_draw_data())

        # Swap front and back buffers
        glfw.swap_buffers(window)

    for i in range(2):
        if accumulation_fbos[i] is not None:
            try:
                glDeleteFramebuffers(1, [accumulation_fbos[i]])
            except Exception:
                pass
            accumulation_fbos[i] = None
        if accumulation_textures[i] is not None:
            try:
                glDeleteTextures(1, [accumulation_textures[i]])
            except Exception:
                pass
            accumulation_textures[i] = None

    # Clean up
    # Delete all cached shaders
    for cached_shader, _ in shader_cache.values():
        if cached_shader is not None:
            glDeleteProgram(cached_shader)
    shader_cache.clear()
    
    # Clean up framebuffer
    if fbo is not None:
        glDeleteFramebuffers(1, [fbo])
    if render_texture is not None:
        glDeleteTextures(1, [render_texture])
    
    # Clean up display shader
    if display_shader is not None:
        glDeleteProgram(display_shader)
    if display_vao is not None:
        glDeleteVertexArrays(1, [display_vao])
    if display_vbo is not None:
        glDeleteBuffers(1, [display_vbo])
    
    impl.shutdown()
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    if shader is not None:
        glDeleteProgram(shader)
    glfw.terminate()




if __name__ == "__main__":
    main()