class SDFPrimitive:
    def __init__(
        self,
        selected_item_id,
        primitive_type,
        position,
        size_or_radius,
        rotation=None,
        scale=None,
        ui_name=None,
        color=None,
        **kwargs,
    ):
        self.primitive_type = primitive_type
        self.position = list(position)

        # Normalize size_or_radius into a list (accept scalar or sequence)
        if isinstance(size_or_radius, (list, tuple)):
            self.size_or_radius = list(size_or_radius)
        else:
            self.size_or_radius = [size_or_radius]

        # Ensure size_or_radius has the expected length for this primitive.
        # For a scalar input we repeat the last element to fill required components
        expected_len = None
        if primitive_type in ("box", "round_box"):
            expected_len = 3
        elif primitive_type in (
            "torus",
            "hex_prism",
            "rounded_cylinder",
            "capped_cylinder",
            "vertical_capsule",
        ):
            expected_len = 2
        elif primitive_type in ("sphere", "pointer", "sprite", "curve"):
            expected_len = 1

        if expected_len is not None and len(self.size_or_radius) < expected_len:
            last = self.size_or_radius[-1] if self.size_or_radius else 0.0
            self.size_or_radius += [last] * (expected_len - len(self.size_or_radius))

        # Always initialize as 3D vectors for rotation/scale/color defaults
        self.rotation = list(rotation) if rotation else [0.0, 0.0, 0.0]
        self.scale = list(scale) if scale else [1.0, 1.0, 1.0]
        self.color = list(color) if color else [0.8, 0.6, 0.4]
        self.kwargs = kwargs
        self.ui_name = ui_name or primitive_type
        self.selected_item_id = selected_item_id
        self.properties = {}

    def update_selected_item_id(self, new_value):
        self.selected_item_id = new_value

    # Working with primitive properties - symmetry, etc.
    def update_property(self, name: str, new_value):
        self.properties[name] = new_value

    def delete_property(self, name: str):
        if name in self.properties:
            self.properties.pop(name)

    # Generate Code
    def generate_transform_code(self, op_id):
        """
        Generate the GLSL transform code for this primitive.
        If this primitive is currently selected, the shader will subtract
        the MovePos uniform (so the C-side can move the primitive interactively).
        """

        # If the selected item is this primitive, use the MovePos uniform in GLSL.
        if self.selected_item_id is not None and self.selected_item_id == op_id:
            new_position = ["MovePos.x", "MovePos.y", "MovePos.z"]
            # Use MoveRot.z (was incorrect MoveRot.y twice)
            new_rotation = ["MoveRot.x", "MoveRot.y", "MoveRot.z"]
        else:
            # Use literal numeric components
            new_position = [self.position[0], self.position[1], self.position[2]]
            new_rotation = [self.rotation[0], self.rotation[1], self.rotation[2]]

        # Pointer primitives mutate the global `p` and do NOT create p{op_id}
        if self.primitive_type == "pointer":
            # pointer function name stored in kwargs['func'] (default identity)
            func_name = self.kwargs.get("func", "pointer_identity")
            # Optionally pass extra params stored in kwargs['params'] (not used by default)
            # We pass position as second argument so pointer functions can be local around a point
            pos_arg = f"vec3({new_position[0]}, {new_position[1]}, {new_position[2]})"
            return f"    p = {func_name}(p, {pos_arg});"

        # For normal primitives generate the usual transform that works on a local p{op_id}
        transform_code = f"vec3 p{op_id} = p;"

        # Aplly Symmetry property
        sym = self.properties.get("symmetry")  # List (x,y,z : bool)
        if isinstance(sym, list) and len(sym) >= 3:
            if sym[0]:
                transform_code += f"p{op_id}.x = abs(p.x);"
            if sym[1]:
                transform_code += f"p{op_id}.y = abs(p.y);"
            if sym[2]:
                transform_code += f"p{op_id}.z = abs(p.z);"

        transform_code += f"\n    p{op_id} -= vec3({new_position[0]}, {new_position[1]}, {new_position[2]});"

        if self.rotation:
            transform_code += f"\n    p{op_id} = rotateZ({new_rotation[2]}) * rotateX({new_rotation[0]}) * rotateY({new_rotation[1]}) * p{op_id};"

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
            radius = self.kwargs.get("radius", 0.1)
            return f"float {op_id} = sdRoundBox(p{op_id}, vec3({self.size_or_radius[0]}, {self.size_or_radius[1]}, {self.size_or_radius[2]}), {radius});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "sphere":
            radius = (
                self.size_or_radius
                if isinstance(self.size_or_radius, list)
                else [self.size_or_radius]
            )
            return f"float {op_id} = sdSphere(p{op_id}, {radius[0]});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "torus":
            # size_or_radius[0] = major radius, size_or_radius[1] = minor radius
            return f"float {op_id} = sdTorus(p{op_id}, vec2({self.size_or_radius[0]}, {self.size_or_radius[1]}));\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "cone":
            # size_or_radius[0] = sin(angle), size_or_radius[1] = cos(angle), kwargs['height'] = height
            c_sin = self.kwargs.get("c_sin", 0.5)
            c_cos = self.kwargs.get("c_cos", 0.866)
            height = self.kwargs.get("height", 1.0)
            return f"float {op_id} = sdCone(p{op_id}, vec2({c_sin}, {c_cos}), {height});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "plane":
            # kwargs['normal'] = normal vector, kwargs['h'] = offset
            normal = self.kwargs.get("normal", [0.0, 1.0, 0.0])
            h = self.kwargs.get("h", 0.0)
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
            height = self.kwargs.get("height", 1.0)
            return f"float {op_id} = sdRoundedCylinder(p{op_id}, {self.size_or_radius[0]}, {self.size_or_radius[1]}, {height});\n    vec3 col{op_id} = {color_vec};"
        elif self.primitive_type == "curve":
            points = self.kwargs.get("points", [[0, 0, 0], [1, 1, 1]])
            thickness = self.kwargs.get("thickness", 0.1)
            n_pts = len(points)

            if len(points) < 8:
                for i in range(0, 8 - len(points)):
                    points.append([0, 0, 0])

            # Generate point array code
            pt_strs = [f"vec3({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})" for p in points]
            pt_array = "{" + ", ".join(pt_strs) + "}"

            return (
                f"vec3 curve_pts_{op_id}[{n_pts}] = {pt_array};\n"
                f"    float {op_id} = sdfCurve(p{op_id}, curve_pts_{op_id}, {n_pts}, {thickness});\n"
                f"    vec3 col{op_id} = {color_vec};"
            )
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
            "kwargs": self.kwargs,
            "properties": self.properties,
        }


class SDFOperation:
    def __init__(self, operation_type, *args, ui_name=None):
        self.operation_type = operation_type
        self.args = list(args)  # Store as list for mutability

        # For smooth operations and mix, track the smoothing factor k
        if operation_type in ["sunion", "ssub", "sinter", "mix"]:
            self.smooth_k = (
                args[2] if len(args) > 2 else (0.5 if operation_type == "mix" else 0.05)
            )
        # For single-operand operations with a float parameter (round, onion)
        elif operation_type in ["round", "onion", "snoiseDisp"]:
            self.float_param = (
                args[1]
                if len(args) > 1
                else (0.1 if operation_type == "round" else 0.05)
            )
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
            "snoiseDisp": {
                "dist_template": "float {op_id} = snoiseDisplace({d_a}, {param}, p);",
                "color_template": "vec3 col{op_id} = {col_a_name};",
                "unpack": lambda args: (args[0], args[1]),
            },
        }

        if self.operation_type not in OPERATION_TEMPLATES:
            raise ValueError(f"Unknown operation type: {self.operation_type}")

        template_info = OPERATION_TEMPLATES[self.operation_type]

        try:
            unpacked_args = template_info["unpack"](self.args)
        except IndexError:
            raise ValueError(
                f"Not enough arguments for operation {self.operation_type}."
            )

        context = {"op_id": op_id}

        num_args = len(unpacked_args)

        if num_args >= 1:
            context["d_a"] = unpacked_args[0]
            context["col_a_name"] = f"col{unpacked_args[0]}"
        if num_args >= 2:
            context["d_b"] = unpacked_args[1]
            context["col_b_name"] = f"col{unpacked_args[1]}"
            context["param"] = unpacked_args[
                1
            ]  # For single-operand ops, second arg is the parameter
        if num_args >= 3:
            context["k"] = unpacked_args[2]

        dist_code = template_info["dist_template"].format(**context)
        color_code = template_info["color_template"].format(**context)

        return f"    {dist_code}\n    {color_code}"

    def to_dict(self):
        """Convert operation to a dictionary for JSON serialization."""
        return {
            "type": "operation",
            "operation_type": self.operation_type,
            "args": self.args,
            "smooth_k": self.smooth_k,
            "ui_name": self.ui_name,
        }
