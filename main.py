import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time
import math
import hashlib
import ctypes
import imgui
import imgui.core
from imgui.integrations.glfw import GlfwRenderer

import numpy as np
import math


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
        filetypes=[("JSON files", "*. json"), ("All files", "*.*")]
    )
    
    root.destroy()
    
    if filepath:
        return scene_builder.load_from_json(filepath)
    return False, "Load cancelled"






# --- Configuration ---
SCREEN_SIZE = (800, 600)
FOV_ANGLE = math.radians(75)  # Field of View - Used for ray direction calculation

# UI Constants
PANEL_WIDTH_RATIO = 0.2  # Left and right panel width as ratio of window width
FPS_WINDOW_OFFSET = 25  # Offset from top for FPS window
FPS_WINDOW_WIDTH = 140
FPS_WINDOW_HEIGHT = 30

# Camera Constants
MOUSE_SENSITIVITY = 0.005
PAN_SENSITIVITY = 0.1
CAMERA_LERP_FACTOR = 0.075
ZOOM_SENSITIVITY = 0.5
MIN_RADIUS = 1.0
MAX_RADIUS = 100.0
MIN_PITCH = -math.radians(90)
MAX_PITCH = math.radians(90)

# Load shader files with error handling
try:
    # Vertex shader source code
    vertex_shader = load_shader_code("vertex_shader.glsl")
    
    # SDF Library
    sdf_library = load_shader_code("sdf_library.glsl")
    
    # Fragment shader template
    fragment_shader_template = load_shader_code("fragment_shader_template.glsl")
except (FileNotFoundError, IOError) as e:
    print(f"Error loading shader files: {e}")
    print("Please ensure all shader files are present in the project directory.")
    exit(1)

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
        transform_code = f"vec3 p{op_id} = p;"
        transform_code += f"\np{op_id} -= vec3({self.position[0]}, {self.position[1]}, {self.position[2]});"

        if self.rotation:
            transform_code += f"\np{op_id} = rotateZ({self.rotation[2]}) * rotateX({self.rotation[0]}) * rotateY({self.rotation[1]}) * p{op_id};"

        if self.scale:
            transform_code += f"\np{op_id} = scale(p{op_id}, vec3({self.scale[0]}, {self.scale[1]}, {self.scale[2]}));"

        return transform_code

    def generate_sdf_code(self, op_id):
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


class SDFSceneBuilder:
    def __init__(self):
        self.primitives = []
        self.operations = []
        self.next_id = 0
        self.id_to_index = {}  # Map op_id to (type, index) for quick lookup

    def add_primitive(self, primitive_type, position, size_or_radius, rotation=None, scale=None, ui_name=None, color=None, **kwargs):
        op_id = f"d{self.next_id}"
        primitive = SDFPrimitive(primitive_type, position, size_or_radius, rotation, scale, ui_name, color, **kwargs)
        self.primitives.append((op_id, primitive))
        self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)
        self.next_id += 1
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

    def add_operation(self, operation_type, *args, ui_name=None):
        op_id = f"d{self.next_id}"
        operation = SDFOperation(operation_type, *args, ui_name=ui_name)
        self.operations.append((op_id, operation))
        self.id_to_index[op_id] = ('operation', len(self.operations) - 1)
        self.next_id += 1
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


    def delete_item(self, op_id):
        """Delete a primitive or operation by its ID."""
        if op_id not in self.id_to_index:
            return False
        
        item_type, index = self.id_to_index[op_id]
        
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
        del self.id_to_index[op_id]
        
        # Remove any operations that depend on this deleted item
        self._cleanup_dependencies(op_id)
        
        return True

    def _cleanup_dependencies(self, deleted_id):
        """Remove operations that depend on a deleted primitive/operation."""
        to_delete = []
        
        for op_id, operation in self.operations:
            if deleted_id in operation.args:
                to_delete.append(op_id)
        
        for op_id in to_delete:
            self.delete_item(op_id)

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
            "operations": []
        }
    
        # Serialize primitives
        for op_id, primitive in self.primitives:
            prim_dict = primitive.to_dict()
            prim_dict["op_id"] = op_id
            scene_dict["primitives"]. append(prim_dict)
    
        # Serialize operations
        for op_id, operation in self.operations:
            op_dict = operation.to_dict()
            op_dict["op_id"] = op_id
            scene_dict["operations"].append(op_dict)
    
        return scene_dict

    def from_dict(self, scene_dict):
        """Load a scene from a dictionary (inverse of to_dict)."""
        # Clear current scene
        self.primitives.clear()
        self.operations.clear()
        self.id_to_index.clear()
        self.next_id = 0
    
        # Load primitives
        for prim_dict in scene_dict. get("primitives", []):
            op_id = prim_dict["op_id"]

            primitive = SDFPrimitive(
                primitive_type=prim_dict["primitive_type"],
                position=prim_dict["position"],
                size_or_radius=prim_dict["size_or_radius"],
                rotation=prim_dict. get("rotation", [0.0, 0.0, 0.0]),
                scale=prim_dict.get("scale", [1.0, 1.0, 1.0]),
                ui_name=prim_dict. get("ui_name"),
                color=prim_dict.get("color", [0.8, 0.6, 0.4]),
                **prim_dict.get("kwargs", {})
            )
        
            self.primitives.append((op_id, primitive))
            self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)
        
            # Update next_id
            prim_num = int(op_id[1:])  # Extract number from "d0", "d1", etc.
            self.next_id = max(self.next_id, prim_num + 1)
    
        # Load operations
        for op_dict in scene_dict. get("operations", []):
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
            op_num = int(op_id[1:])
            self.next_id = max(self.next_id, op_num + 1)


   
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

def main():
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
    last_x, last_y = 0.0, 0.0

    is_mmb_pressed = False
    is_shift_mmb_pressed = False

    # --- SaveLoad ---
    save_load_message = None
    save_load_message_time = None
    last_key_s_pressed = False
    last_key_o_pressed = False
    last_key_f10_pressed = False  # Add this if not present


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
    selected_item_id = None  # Track which item is selected in the tree
    selection_mode = None  # 'primitive' or 'operation'
    renaming_item_id = None  # Item being renamed
    rename_text = ""
    last_key_a_pressed = False  # Track if Ctrl+A was pressed
    last_key_f2_pressed = False  # Track if F2 was pressed
    last_key_delete_pressed = False  # Track if Delete was pressed
    last_key_compile_pressed = False  # Track if Ctrl+B was pressed
    
    # Shader selection
    shader_choice = 0  # 0 = template, 1 = cycles
    shader_names = ["fragment_shader_template.glsl", "cycles_fragment_shader.glsl"]

    # --- Settings ---
    resolution_scale = 1.0  # 1.0 = normal, 2.0 = oversampling, <1.0 = low res for performance

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
        selected_fragment_shader = load_shader_code(shader_names[shader_choice])
        fragment_shader = selected_fragment_shader.replace("{SDF_LIBRARY}", sdf_library)
        fragment_shader = fragment_shader.replace("{SCENE_CODE}", scene_code)
        fragment_shader = fragment_shader.replace("{FOV_ANGLE_VAL}", str(FOV_ANGLE))
        
        # Create hash of the complete shader code (including shader choice)
        shader_code = vertex_shader + fragment_shader + shader_names[shader_choice]
        return hashlib.md5(shader_code.encode()).hexdigest()
    
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
            selected_fragment_shader = load_shader_code(shader_names[shader_choice])
            fragment_shader = selected_fragment_shader. replace("{SDF_LIBRARY}", sdf_library)
            fragment_shader = fragment_shader.replace("{SCENE_CODE}", scene_code)
            fragment_shader = fragment_shader.replace("{FOV_ANGLE_VAL}", str(FOV_ANGLE))
            
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
        """Get all uniform locations for the shader program."""
        return {
            'time': glGetUniformLocation(shader_program, "time"),
            'resolution': glGetUniformLocation(shader_program, "resolution"),
            'camYaw': glGetUniformLocation(shader_program, "camYaw"),
            'camPitch': glGetUniformLocation(shader_program, "camPitch"),
            'radius': glGetUniformLocation(shader_program, "radius"),
            'CamOrbit': glGetUniformLocation(shader_program, "CamOrbit"),
            'frameIndex':  glGetUniformLocation(shader_program, "frameIndex"),
            'accumulationTexture': glGetUniformLocation(shader_program, "accumulationTexture"),
            'useAccumulation': glGetUniformLocation(shader_program, "useAccumulation"),
        }   




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
    
    # Simple shader for displaying texture
    display_vertex_shader = """
    #version 330 bindings
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
    uniform int isAccumulation; // 1 = accumulation (linear), 0 = already-tonemapped
    void main() {
        vec4 tex = texture(renderTexture, TexCoord);
        vec3 color = tex.rgb;
        if (isAccumulation == 1) {
            // display accumulation buffer: apply tonemapping & gamma
            // clamp to avoid negative / NaN
            vec3 mapped = pow(clamp(color, 0.0, 10000.0), vec3(0.4545));
            FragColor = vec4(mapped, 1.0);
        } else {
            // already-tonemapped render targets (pass-through)
            FragColor = vec4(color, 1.0);
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
    void main() {
        FragColor = texture(renderTexture, TexCoord);
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

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

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
        if shader_choice == 1:   # cycles_fragment_shader. glsl
            frame_count += 1
        else: 
            frame_count = 0  # Reset accumulation when switching shaders
        
        # Get window and rendering dimensions
        width, height = glfw.get_framebuffer_size(window)
        menu_bar_height = int(imgui.get_frame_height())
        panel_width = int(width * PANEL_WIDTH_RATIO)
        rendering_width = width - 2 * panel_width
        rendering_height = height - menu_bar_height
        
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

        # Handle MMB press and release for camera control
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            shift_pressed = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            
            if not is_mmb_pressed and not shift_pressed:
                is_mmb_pressed = True
                last_x, last_y = glfw.get_cursor_pos(window)
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
            elif not is_mmb_pressed and shift_pressed:
                is_mmb_pressed = True
                is_shift_mmb_pressed = True
                last_x, last_y = glfw.get_cursor_pos(window)
                last_pan_x, last_pan_y = last_x, last_y
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.RELEASE:
            if is_mmb_pressed:
                is_mmb_pressed = False
                is_shift_mmb_pressed = False
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)


        prev_cam_yaw = cam_yaw
        prev_cam_pitch = cam_pitch
        prev_cam_radius = cam_radius
        prev_cam_orbit = cam_orbit


        # Handle mouse wheel input for camera zoom
        if io.mouse_wheel != 0:
            target_radius -= io.mouse_wheel * ZOOM_SENSITIVITY
            target_radius = max(MIN_RADIUS, min(MAX_RADIUS, target_radius))

        cam_radius += (target_radius - cam_radius) * CAMERA_LERP_FACTOR

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
        cam_yaw += (target_yaw - cam_yaw) * CAMERA_LERP_FACTOR
        cam_pitch += (target_pitch - cam_pitch) * CAMERA_LERP_FACTOR

        # --- Interpolate camera Pan ---
        cam_pan_y += (target_pan_y - cam_pan_y) * CAMERA_LERP_FACTOR
        cam_pan_x -= (target_pan_x + cam_pan_x) * CAMERA_LERP_FACTOR

        #####
        forward_x = math.cos(cam_pitch) * math.sin(cam_yaw)
        forward_y = math.sin(cam_pitch)
        forward_z = math.cos(cam_pitch) * math.cos(cam_yaw)


        right_x = math.cos(cam_yaw)
        right_y = 0
        right_z = -math.sin(cam_yaw)


        up_x = forward_y * right_z - forward_z * right_y
        up_y = forward_z * right_x - forward_x * right_z
        up_z = forward_x * right_y - forward_y * right_x
        #####


        orbit_center_offset_x = cam_pan_x * right_x + cam_pan_y * up_x
        orbit_center_offset_y = cam_pan_x * right_y + cam_pan_y * up_y
        orbit_center_offset_z = cam_pan_x * right_z + cam_pan_y * up_z

        cam_orbit = (
            orbit_center_offset_z, # Yoow! (Correctly)
            orbit_center_offset_y,
            orbit_center_offset_x
        )



        if io.keys_down[glfw.KEY_HOME]:
            cam_pan_x = cam_pan_y = target_pan_x = target_pan_y = 0.0
            cam_orbit = [0.0,0.0,0.0]



        elip = 0.005
        if (abs(cam_yaw - prev_cam_yaw) > elip or 
            abs(cam_pitch - prev_cam_pitch) > elip or
            abs(cam_radius - prev_cam_radius) > elip or
            any(abs(cam_orbit[i] - prev_cam_orbit[i]) > elip for i in range(3))):
            frame_count = 0
            # Reset accumulation buffers so no stale data is read later
            if accumulation_fbos[0] is not None and accumulation_fbos[1] is not None:
                # store current viewport to restore later if you need; here we assume you will set proper viewport when drawing
                glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[0])
                glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)
                glClearColor(0.0, 0.0, 0.0, 0.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[1])
                glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)
                glClearColor(0.0, 0.0, 0.0, 0.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
            current_accum_index = 0


        print(frame_count)

        prev_cam_yaw = cam_yaw
        prev_cam_pitch = cam_pitch
        prev_cam_radius = cam_radius
        prev_cam_orbit = cam_orbit


        #bg_draw_list = imgui.get_background_draw_list()
        
        #bg_draw_list.add_circle_filled(
        #    400, 
        #    300, 
        #    25, 
        #    imgui.get_color_u32_rgba(1, 0, 0, 1)
        #)


        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)
        
        
        # --- Setup accumulation buffer if using cycles shader ---
        use_accumulation = 0
        if shader_choice == 1:  # cycles_fragment_shader.glsl
            if setup_accumulation_buffer(scaled_rendering_width, scaled_rendering_height):
                use_accumulation = 1

        # --- RENDER TO ACCUMULATION BUFFER (if using cycles) ---
        if shader is not None and shader_choice == 1 and use_accumulation == 1:
            write_buffer = current_accum_index
            read_buffer = 1 - current_accum_index

            glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[write_buffer])
            glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)

            if frame_count == 0:
                glClear(GL_COLOR_BUFFER_BIT)

            glUseProgram(shader)
            if uniform_locs is not None:
                current_time_uniform = time. time() - start_time
                glUniform1f(uniform_locs['time'], current_time_uniform)
                glUniform2f(uniform_locs['resolution'], scaled_rendering_width, scaled_rendering_height)
                glUniform1f(uniform_locs['camYaw'], cam_yaw)
                glUniform1f(uniform_locs['camPitch'], cam_pitch)
                glUniform1f(uniform_locs['radius'], cam_radius)
                glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                glUniform1i(uniform_locs['frameIndex'], frame_count)
                
                # Bind accumulation texture for reading
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, accumulation_textures[read_buffer])
                glUniform1i(uniform_locs['accumulationTexture'], 0)
                glUniform1i(uniform_locs['useAccumulation'], 1)

            glBindVertexArray(vao)
            glDrawArrays(GL_QUADS, 0, 4)
            
            # Switch back to default framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, width, height)
            
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, accumulation_textures[write_buffer])  # <- use the correct texture handle
            glUniform1i(uniform_locs['accumulationTexture'], 0)

            # Display accumulated result
            glUseProgram(display_shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, accumulation_textures[write_buffer])  # <- use the same texture for display
            glUniform1i(glGetUniformLocation(display_shader, "renderTexture"), 0)
            glUniform1i(glGetUniformLocation(display_shader, "isAccumulation"), 0)

            glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
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
                glUniform1f(uniform_locs['camYaw'], cam_yaw)
                glUniform1f(uniform_locs['camPitch'], cam_pitch)
                glUniform1f(uniform_locs['radius'], cam_radius)
                glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])
                glUniform1i(uniform_locs['frameIndex'], 0)
                glUniform1i(uniform_locs['useAccumulation'], 0)

            glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
            glBindVertexArray(vao)
            glDrawArrays(GL_QUADS, 0, 4)
            glViewport(0, 0, width, height)



        # --- TOP MENU BAR (Render first so it's on top) ---
        if imgui. begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                if imgui.menu_item("Save Scene", "Ctrl+S")[0]:
                    # Trigger save dialog
                    success, message = save_scene_dialog(scene_builder)
                    save_load_message = message
                    save_load_message_time = time.time()
                    #if success and shader is not None:
        
                if imgui. menu_item("Load Scene", "Ctrl+O")[0]:
                    # Trigger load dialog
                    success, message = load_scene_dialog(scene_builder)
                    save_load_message = message
                    save_load_message_time = time.time()
                    if success: 
                        selected_item_id = None
                        selection_mode = None
                        # Recompile shader after loading
                        success, new_uniforms = recompile_shader()
                        if success:
                            uniform_locs = new_uniforms


                imgui.separator()
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
    
            imgui.end_main_menu_bar()
        

        # Check Ctrl + S/O
        if io.keys_down[glfw. KEY_O] and io.key_ctrl:
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

        if io.keys_down[glfw. KEY_S] and io.key_ctrl:
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


        # Check F10 for settings (with debouncing)
        if io.keys_down[glfw.KEY_F10]:
            if not last_key_f10_pressed:
                show_settings_window = True
                last_key_f10_pressed = True
        else:
            last_key_f10_pressed = False
        
        # --- RENDER TO FRAMEBUFFER AT SCALED RESOLUTION ---
        if shader is not None and display_shader is not None and resolution_scale != 1.0:
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
                    glUniform1f(uniform_locs['camYaw'], cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], cam_pitch)
                    glUniform1f(uniform_locs['radius'], cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])

                glBindVertexArray(vao)
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
                        glUniform1f(uniform_locs['camYaw'], cam_yaw)
                        glUniform1f(uniform_locs['camPitch'], cam_pitch)
                        glUniform1f(uniform_locs['radius'], cam_radius)
                        glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])

                    glViewport(panel_width, menu_bar_height, scaled_rendering_width, scaled_rendering_height)
                    glBindVertexArray(vao)
                    glDrawArrays(GL_QUADS, 0, 4)
                    glViewport(0, 0, width, height)
        else:
            # Direct rendering when scale is 1.0 or display shader not available
            if shader is not None:
                glUseProgram(shader)
                if uniform_locs is not None:
                    current_time_uniform = time.time() - start_time
                    glUniform1f(uniform_locs['time'], current_time_uniform)
                    glUniform2f(uniform_locs['resolution'], rendering_width, rendering_height)
                    glUniform1f(uniform_locs['camYaw'], cam_yaw)
                    glUniform1f(uniform_locs['camPitch'], cam_pitch)
                    glUniform1f(uniform_locs['radius'], cam_radius)
                    glUniform3f(uniform_locs['CamOrbit'], cam_orbit[0], cam_orbit[1], cam_orbit[2])

                glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
                glBindVertexArray(vao)
                glDrawArrays(GL_QUADS, 0, 4)
                glViewport(0, 0, width, height)

        # --- SETTINGS WINDOW ---
        if show_settings_window:
            imgui.set_next_window_position(width // 2 - 200, height // 2 - 150)
            imgui.set_next_window_size(400, 300)  # Increased height
            is_open, show_settings_window = imgui. begin("Settings", True, imgui. WINDOW_NO_COLLAPSE)
            
            if not is_open:
                show_settings_window = False
            
            imgui.text("Rendering Settings")
            imgui.separator()
            
            # Shader Selection
            imgui.text("Fragment Shader:")
            clicked, shader_choice = imgui.combo(
                "##shader_select",
                shader_choice,
                shader_names
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
            
            imgui.spacing()
            imgui.text_colored("1.0 = Normal resolution", 0.7, 0.7, 0.7, 1.0)
            imgui.text_colored("2.0 = Oversampling (better quality)", 0.7, 0.7, 0.7, 1.0)
            imgui.text_colored("<1.0 = Low resolution (better performance)", 0.7, 0.7, 0.7, 1.0)
            
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

            

        # --- FPS OVERLAY (Top Right, above right panel) ---
        fps_x = width - panel_width - FPS_WINDOW_WIDTH - FPS_WINDOW_OFFSET
        imgui.set_next_window_position(fps_x, FPS_WINDOW_OFFSET)
        imgui.set_next_window_size(FPS_WINDOW_WIDTH, FPS_WINDOW_HEIGHT)
        imgui.begin("FPS", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        imgui.text_colored("FPS: " + str(fps_value), 0.0, 1.0, 0.0, 1.0)
        imgui.end()
        

        # Display save/load status message
        if save_load_message is not None:
            # Show message for 3 seconds
            if time.time() - save_load_message_time < 3.0:
                imgui.set_next_window_position(width // 2 - 150, 100)
                imgui.begin("Status", False, imgui. WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)

                # Color based on success
                is_success = "saved" in save_load_message. lower() or "loaded" in save_load_message.lower()
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

        imgui.text("Primitives:")
        for op_id, primitive in scene_builder.primitives:
            label = f"{primitive.ui_name} ({op_id})"
            flags = imgui.TREE_NODE_LEAF
            if selected_item_id == op_id:
                flags |= imgui.TREE_NODE_SELECTED
            
            imgui.tree_node(label, flags)
            
            # Left-click for selection
            if imgui.is_item_clicked():
                selected_item_id = op_id
                selection_mode = 'primitive'
                renaming_item_id = None
            
            imgui.tree_pop()

        imgui.spacing()
        imgui.text("Operations:")
        for op_id, operation in scene_builder.operations:
            label = f"{operation.ui_name} ({op_id})"
            flags = imgui.TREE_NODE_LEAF
            if selected_item_id == op_id:
                flags |= imgui.TREE_NODE_SELECTED
            
            imgui.tree_node(label, flags)
            
            # Left-click for selection
            if imgui.is_item_clicked():
                selected_item_id = op_id
                selection_mode = 'operation'
                renaming_item_id = None
            
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
                        
                        # Position
                        changed, primitive.position = imgui.input_float3("Position##pos", *primitive.position)
                        if changed:
                            success, new_uniforms = recompile_shader() # Fixed: UnboundLocalError
                            if success:
                                uniform_locs = new_uniforms
                        
                        # Size/Radius - varies by primitive type
                        # I have python 3.8, "match", get out.
                        if primitive.primitive_type == "sphere":
                            changed, primitive.size_or_radius[0] = imgui.input_float("Radius", primitive.size_or_radius[0])
                        elif primitive.primitive_type == "torus":
                            changed1, primitive.size_or_radius[0] = imgui.input_float("Major Radius", primitive.size_or_radius[0])
                            changed2, primitive.size_or_radius[1] = imgui.input_float("Minor Radius", primitive.size_or_radius[1])
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "hex_prism":
                            changed1, primitive.size_or_radius[0] = imgui.input_float("Hex Radius", primitive.size_or_radius[0])
                            changed2, primitive.size_or_radius[1] = imgui.input_float("Height", primitive.size_or_radius[1])
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "vertical_capsule":
                            changed1, primitive.size_or_radius[0] = imgui.input_float("Height", primitive.size_or_radius[0])
                            changed2, primitive.size_or_radius[1] = imgui.input_float("Radius", primitive.size_or_radius[1])
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "capped_cylinder":
                            changed1, primitive.size_or_radius[0] = imgui.input_float("Radius", primitive.size_or_radius[0])
                            changed2, primitive.size_or_radius[1] = imgui.input_float("Height", primitive.size_or_radius[1])
                            changed = changed1 or changed2
                        elif primitive.primitive_type == "rounded_cylinder":
                            changed1, primitive.size_or_radius[0] = imgui.input_float("Radius A", primitive.size_or_radius[0])
                            changed2, primitive.size_or_radius[1] = imgui.input_float("Radius B", primitive.size_or_radius[1])
                            changed = changed1 or changed2
                        else:
                            if primitive.primitive_type not in ["cone", "plane", "rounded_cylinder"]:
                                changed, primitive.size_or_radius = imgui.input_float3("Size##size", *primitive.size_or_radius)
                        if changed:
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                        
                        # Special parameters for specific primitives
                        if primitive.primitive_type == "cone":
                            c_sin = primitive.kwargs.get('c_sin', 0.5)
                            c_cos = primitive.kwargs.get('c_cos', 0.866)
                            height = primitive.kwargs.get('height', 1.0)
                            changed1, c_sin = imgui.input_float("Sin(Angle)", c_sin)
                            changed2, c_cos = imgui.input_float("Cos(Angle)", c_cos)
                            changed3, height = imgui.input_float("Height", height)
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
                            changed1, normal = imgui.input_float3("Normal", *normal)
                            changed2, h = imgui.input_float("Offset (h)", h)
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
                            changed, height = imgui.input_float("Height", height)
                            if changed:
                                primitive.kwargs['height'] = height
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # Show rotation as degrees
                        current_degrees = [math.degrees(a) for a in primitive.rotation]
                        changed, degs = imgui.input_float3("Rotation °", *current_degrees)
                        if changed:
                            primitive.rotation = [math.radians(a) for a in degs]
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms

                        # Scale stays as-is
                        changed, primitive.scale = imgui. input_float3("Scale", *primitive.scale)
                        if changed:
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                        

                        # Special parameters for specific primitives
                        if primitive.primitive_type == "round_box":
                            changed, primitive.kwargs['radius'] = imgui.input_float("Corner Radius", primitive.kwargs.get('radius', 0.1))
                            if changed:
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # Color picker
                        imgui.separator()
                        imgui.text("Color:")
                        # Color edit - imgui automatically shows a picker button
                        color_changed, color_rgba = imgui.color_edit3("Color##color", *primitive.color)
                        if color_changed:
                            primitive.color = list(color_rgba[:3])  # Only use RGB, ignore alpha
                            success, new_uniforms = recompile_shader()
                            if success:
                                uniform_locs = new_uniforms
                        
                        # Show color preview button
                        imgui.same_line()
                        # color_button takes: label, r, g, b, flags, size_x, size_y
                        imgui.color_button("Preview##color_preview", primitive.color[0], primitive.color[1], primitive.color[2], 0, 20, 20)
                        
                        # Alternative: RGB sliders for fine control
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
                                    operation.args[i] = valid_operands[new_index][0]
                                    success, new_uniforms = recompile_shader()
                                    if success:
                                        uniform_locs = new_uniforms
                        
                        imgui.separator()
                        
                        # Show float parameter for single-operand operations with parameters
                        if hasattr(operation, 'float_param') and operation.float_param is not None:
                            changed, operation.float_param = imgui.input_float("Parameter", operation.float_param, 0.01, 0.1)
                            if changed:
                                # Update the operation with new parameter
                                if len(operation.args) >= 2:
                                    operation. args[1] = operation.float_param
                                success, new_uniforms = recompile_shader()
                                if success:
                                    uniform_locs = new_uniforms
                        
                        # Show smoothing factor for smooth operations
                        elif operation.smooth_k is not None:
                            changed, operation.smooth_k = imgui.input_float("Smoothing Factor (k)", operation.smooth_k, 0.01, 0.1)
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
                                    new_id = scene_builder. onion(all_items[-1][0], 0.05, ui_name=label)
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
