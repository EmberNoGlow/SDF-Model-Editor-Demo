"""Left and right panel rendering."""
import imgui
import math
from src.app.data.states import st
from src.app.data.consts import cn
from src.ui.helpers import input_vec3, input_vec2, input_float
from src.rendering.shader_compiler import recompile_shader


def render_scene_tree_panel(width, height, menu_bar_height, panel_width, scene_builder):
    """Render the left panel with scene hierarchy."""
    imgui.set_next_window_position(0, menu_bar_height)
    imgui.set_next_window_size(panel_width, height - menu_bar_height)
    imgui.begin(
        "Scene Tree",
        False,
        imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
    )

    imgui.text("Scene Hierarchy:")
    imgui.separator()

    for root_id in scene_builder.root_children:
        _render_scene_node_recursive(root_id, scene_builder, panel_width, 0)

    imgui.spacing()
    imgui.separator()

    if imgui.button(f"Add (Ctrl+A)", -1):
        st.show_add_change_window = True
        st.pending_change_node_id = None

    imgui.end()


def _render_scene_node_recursive(node_id, scene_builder, panel_width, depth):
    """Recursively render scene hierarchy nodes."""
    node = scene_builder.get_node(node_id)
    if not node:
        return

    item_data = node.item_data
    children = node.children
    is_leaf = len(children) == 0

    label = _format_node_label(item_data.ui_name, node_id)

    flags = 0
    if not is_leaf:
        flags |= imgui.TREE_NODE_DEFAULT_OPEN
    else:
        flags |= imgui.TREE_NODE_LEAF

    if st.selected_item_id == node_id:
        flags |= imgui.TREE_NODE_SELECTED

    if node.parent_id is None and node_id in scene_builder.root_children:
        _render_root_node_movement_buttons(node_id, scene_builder)

    imgui.push_id(f"delete_{node_id}")
    clicked_delete = imgui.button("X", 20, 20)
    imgui.pop_id()

    if clicked_delete:
        scene_builder.delete_node(node_id)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms
        return

    imgui.same_line()

    node_open = imgui.tree_node(label, flags)

    if imgui.is_item_clicked():
        _handle_scene_node_selection(node_id, scene_builder)

    _render_node_context_menu(node_id, node, scene_builder)

    if children:
        for child_id in list(children):
            _render_scene_node_recursive(
                child_id, scene_builder, panel_width, depth + 1
            )

    if node_open:
        imgui.tree_pop()


def _format_node_label(name, op_id, max_chars=16):
    """Format node label with truncation."""
    if len(name) > max_chars:
        truncated_name = name[: max_chars - 3] + "..."
    else:
        truncated_name = name
    return f"{truncated_name} ({op_id})"


def _render_root_node_movement_buttons(node_id, scene_builder):
    """Render up/down buttons for root-level nodes."""
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (1, 1))
    root_idx = scene_builder.root_children.index(node_id)

    if imgui.arrow_button(f"##up_{node_id}", 2):
        if root_idx > 0:
            scene_builder.move_root_node(node_id, root_idx - 1)
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

    imgui.same_line()

    if imgui.arrow_button(f"##down_{node_id}", 3):
        if root_idx < len(scene_builder.root_children) - 1:
            scene_builder.move_root_node(node_id, root_idx + 1)
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

    imgui.pop_style_var(1)
    imgui.same_line()


def _handle_scene_node_selection(node_id, scene_builder):
    """Handle left-click node selection with Ctrl for multi-select."""
    io_local = imgui.get_io()
    if io_local.key_ctrl:
        if node_id in st.selected_items:
            st.selected_items.remove(node_id)
        else:
            st.selected_items.add(node_id)
        if len(st.selected_items) > 0:
            st.selected_item_id = None
    else:
        st.selected_items.clear()
        st.selected_item_id = node_id
        scene_builder.update_selected_item_id(st.selected_item_id)
        st.selection_mode = "node"
        st.renaming_item_id = None
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_node_context_menu(node_id, node, scene_builder):
    """Render right-click context menu for nodes."""
    popup_id = f"node_ctx_{node_id}"
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
        imgui.open_popup(popup_id)

    if imgui.begin_popup(popup_id):
        if node.node_type == "operation":
            if imgui.menu_item("Change Operation Type")[0]:
                st.pending_change_node_id = node_id
                st.show_add_change_window = True
                imgui.close_current_popup()
        else:
            if imgui.menu_item("Change Type")[0]:
                st.pending_change_node_id = node_id
                st.show_add_change_window = True
                imgui.close_current_popup()

        imgui.separator()

        if imgui.menu_item("Change Properties")[0]:
            st.property_change_node_id = node_id
            st.show_property_change_window = True
            imgui.close_current_popup()

        if imgui.menu_item("Reparent")[0]:
            st.reparent_node_id = node_id
            st.show_reparent_window = True
            imgui.close_current_popup()

        imgui.end_popup()


def render_inspector_panel(width, height, menu_bar_height, panel_width, scene_builder):
    """Render the right panel with property inspector."""
    imgui.set_next_window_position(width - panel_width, menu_bar_height)
    imgui.set_next_window_size(panel_width, height - menu_bar_height)
    imgui.begin(
        "Inspector",
        False,
        imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE,
    )

    if (
        st.selected_item_id is not None
        and st.selected_item_id in scene_builder.id_to_node
    ):
        node = scene_builder.get_node(st.selected_item_id)
        if node:
            _render_node_inspector(node, scene_builder, panel_width)
    else:
        imgui.text("No node selected")
        imgui.text("Click on a node in the Scene Tree")

    imgui.end()


def _render_node_inspector(node, scene_builder, panel_width):
    """Render inspector content for a selected node."""
    item_data = node.item_data

    imgui.text(f"Type: {node.node_type}")
    if node.node_type == "operation":
        imgui.text(f"Operation: {item_data.operation_type}")
    else:
        imgui.text(f"Primitive: {item_data.primitive_type}")

    imgui.separator()
    imgui.text(f"Selected: {item_data.ui_name}")

    _render_node_rename_ui(node, scene_builder)
    imgui.separator()

    if node.node_type == "primitive":
        _render_primitive_inspector(node, scene_builder, item_data, panel_width)
    elif node.node_type == "operation":
        _render_operation_inspector(node, scene_builder, item_data)


def _render_node_rename_ui(node, scene_builder):
    """Render rename input field."""
    if imgui.button("Rename"):
        st.renaming_item_id = st.selected_item_id
        st.rename_text = node.item_data.ui_name

    if st.renaming_item_id == st.selected_item_id:
        changed, st.rename_text = imgui.input_text("##rename", st.rename_text, 256)

        if imgui.button("OK", 60):
            scene_builder.rename_node(st.selected_item_id, st.rename_text)
            st.renaming_item_id = None
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

        imgui.same_line()
        if imgui.button("Cancel", 60):
            st.renaming_item_id = None


def _render_primitive_inspector(node, scene_builder, item_data, panel_width):
    """Render inspector for primitive nodes."""
    panel_elem_width_vec3 = (panel_width / 4) - 14
    panel_elem_width_float = (panel_width / 2) - 14

    primitive = node.item_data

    _render_primitive_type_properties(
        primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float
    )

    imgui.begin_group()
    imgui.spacing()
    imgui.separator()
    imgui.dummy((panel_width / 4) - 8, 0)
    imgui.same_line()
    imgui.text_colored("Transform", 1.0, 0.7, 0.5, 1.0)
    imgui.spacing()
    imgui.end_group()

    changed, item_data.position = input_vec3(
        "Position", item_data.position, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    changed, item_data.rotation = input_vec3(
        "Rotation", item_data.rotation, cn["STEP_VARIABLE_ANGLE"], panel_elem_width_vec3
    )
    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    changed, item_data.scale = input_vec3(
        "Scale", item_data.scale, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.begin_group()
    imgui.spacing()
    imgui.separator()
    imgui.dummy((panel_width / 3) - 12, 0)
    imgui.same_line()
    imgui.text_colored("Color", 1.0, 0.7, 0.5, 1.0)
    imgui.spacing()
    imgui.end_group()

    color_changed, color_rgba = imgui.color_edit3("Color##color", *primitive.color)
    if color_changed:
        primitive.color = list(color_rgba[:3])
        scene_builder.modify_primitive_property(node.item_id, "color", primitive.color)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.spacing()
    imgui.text("RGB Sliders:")
    r_changed, primitive.color[0] = imgui.slider_float(
        "R##color_r", primitive.color[0], 0.0, 1.0
    )
    g_changed, primitive.color[1] = imgui.slider_float(
        "G##color_g", primitive.color[1], 0.0, 1.0
    )
    b_changed, primitive.color[2] = imgui.slider_float(
        "B##color_b", primitive.color[2], 0.0, 1.0
    )

    if r_changed or g_changed or b_changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_primitive_type_properties(primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float):
    """Render type-specific primitive properties."""
    prim_type = primitive.primitive_type

    if prim_type == "sprite":
        _render_sprite_properties(primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float)
    elif prim_type == "cone":
        _render_cone_properties(primitive, scene_builder, panel_elem_width_float)
    elif prim_type == "plane":
        _render_plane_properties(primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float)
    elif prim_type == "pointer":
        _render_pointer_properties(primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float)
    elif prim_type == "curve":
        _render_curve_properties(primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float)
    else:
        _render_standard_primitive_properties(primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float)

    if prim_type == "round_box":
        imgui.spacing()
        changed, primitive.kwargs["radius"] = input_float(
            "Radius",
            primitive.kwargs.get("radius", 0.1),
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        if changed:
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms


def _render_sprite_properties(primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float):
    """Render sprite-specific properties."""
    import tkinter as tk
    from tkinter import filedialog
    
    sprite_idx = primitive.kwargs.get("sprite_index", None)
    if sprite_idx is None or sprite_idx >= len(st.sprites_array):
        imgui.text_colored("Sprite data missing or corrupted", 1.0, 0.0, 0.0, 1.0)
        return

    spr = st.sprites_array[sprite_idx]
    imgui.text("Plane parameters:")

    changed, primitive.position = input_vec3(
        "Point", primitive.position, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    changed2, spr.planeNormal = input_vec3(
        "Normal", spr.planeNormal, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    changed3, spr.planeWidth = input_float(
        "Width", spr.planeWidth, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    changed4, spr.planeHeight = input_float(
        "Height", spr.planeHeight, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    spr.planePoint = primitive.position

    if changed or changed2 or changed3 or changed4:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.separator()
    imgui.text("Mapping:")

    uv2 = spr.uvSize
    changed_uv, uv2 = input_vec2("UV Size", uv2, 0.1, panel_elem_width_vec3)
    spr.uvSize[0], spr.uvSize[1] = uv2[0], uv2[1]

    changed_alpha, spr.Alpha = input_float("Alpha", spr.Alpha, 0.01, panel_elem_width_float)
    changed_lod, spr.LOD = input_float("LOD", spr.LOD, 0.1, panel_elem_width_float)

    if changed_uv or changed_alpha or changed_lod:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    if spr.texture_id:
        imgui.text(f"Texture loaded: {spr.tex_size[0]}x{spr.tex_size[1]}")
    else:
        imgui.text_colored("No texture loaded", 0.9, 0.3, 0.3, 1.0)

    imgui.spacing()
    if imgui.button("Load Texture", -1):
        root = tk.Tk()
        root.withdraw()
        filetypes = [
            ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tga")),
            ("All files", "*.*"),
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        root.destroy()

        if filepath:
            ok = spr.load_texture_from_file(filepath)
            if ok:
                spr.SprTexture = f"sprTex{sprite_idx}"
                success, new_uniforms = recompile_shader(scene_builder)
                if success:
                    st.uniform_locs = new_uniforms


def _render_cone_properties(primitive, scene_builder, panel_elem_width_float):
    """Render cone-specific properties."""
    c_sin = primitive.kwargs.get("c_sin", 0.5)
    c_cos = primitive.kwargs.get("c_cos", 0.866)
    height = primitive.kwargs.get("height", 1.0)

    changed1, c_sin = input_float(
        "Sin(Angle)", c_sin, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    changed2, c_cos = input_float(
        "Cos(Angle)", c_cos, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    changed3, height = input_float(
        "Height", height, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )

    if changed1 or changed2 or changed3:
        primitive.kwargs["c_sin"] = c_sin
        primitive.kwargs["c_cos"] = c_cos
        primitive.kwargs["height"] = height
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_plane_properties(primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float):
    """Render plane-specific properties."""
    normal = primitive.kwargs.get("normal", [0.0, 1.0, 0.0])
    h = primitive.kwargs.get("h", 0.0)

    changed1, normal = input_vec3(
        "Normal", normal, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )
    changed2, h = input_float(
        "Offset (h)", h, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )

    if changed1 or changed2:
        norm_len = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        if norm_len > 0.001:
            normal = [normal[0] / norm_len, normal[1] / norm_len, normal[2] / norm_len]

        primitive.kwargs["normal"] = normal
        primitive.kwargs["h"] = h
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_pointer_properties(primitive, node, scene_builder, panel_elem_width_vec3, panel_elem_width_float):
    """Render pointer function properties."""
    changed_pos, primitive.position = input_vec3(
        "Position", primitive.position, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
    )

    if changed_pos:
        scene_builder.modify_primitive_property(node.item_id, "position", primitive.position)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    pointer_funcs = [
        "pointer_identity",
        "pointer_symmetry_x",
        "pointer_symmetry_y",
        "pointer_symmetry_z",
    ]

    current_func = primitive.kwargs.get("func", "pointer_identity")
    try:
        current_index = pointer_funcs.index(current_func)
    except ValueError:
        pointer_funcs.append(current_func)
        current_index = len(pointer_funcs) - 1

    clicked, new_index = imgui.combo("Function", current_index, pointer_funcs)
    if clicked:
        new_func = pointer_funcs[new_index]
        primitive.kwargs["func"] = new_func
        scene_builder.modify_primitive_property(node.item_id, "kwargs.func", new_func)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.separator()
    imgui.text(
        "Pointer functions mutate \nthe raymarch point `p` \nfor subsequent primitives."
    )
    imgui.text_colored(
        "Place a pointer earlier in \nthe tree to affect later objects.",
        0.9,
        0.8,
        0.2,
        1.0,
    )


def _render_curve_properties(primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float):
    """Render curve-specific properties."""
    imgui.spacing()

    points = primitive.kwargs.get("points", [[0, 0, 0], [1, 1, 1]])
    imgui.text("Curve Points:")

    points_to_remove = None
    for i, pt in enumerate(points):
        changed, new_pt = input_vec3(
            f"Point {i}", list(pt), cn["STEP_VARIABLE_FLOAT"], panel_elem_width_vec3
        )
        if changed:
            points[i] = new_pt
            primitive.kwargs["points"] = points
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms

        imgui.same_line()
        if imgui.button(f"Remove##pt{i}", width=60):
            points_to_remove = i

    if points_to_remove is not None and len(points) > 2:
        points.pop(points_to_remove)
        primitive.kwargs["points"] = points
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    if imgui.button("Add Point", width=-1):
        points.append([0.0, 0.0, 0.0])
        primitive.kwargs["points"] = points
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    imgui.spacing()

    thickness = primitive.kwargs.get("thickness", 0.1)
    changed, thickness = input_float(
        "Thickness", thickness, cn["STEP_VARIABLE_FLOAT"], panel_elem_width_float
    )
    if changed:
        primitive.kwargs["thickness"] = thickness
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_standard_primitive_properties(primitive, scene_builder, panel_elem_width_vec3, panel_elem_width_float):
    """Render standard size/radius properties for most primitives."""
    primitive.size_or_radius = (
        list(primitive.size_or_radius)
        if isinstance(primitive.size_or_radius, tuple)
        else primitive.size_or_radius
    )
    primitive.size_or_radius = (
        [primitive.size_or_radius]
        if isinstance(primitive.size_or_radius, float)
        else primitive.size_or_radius
    )

    changed = False
    prim_type = primitive.primitive_type

    if prim_type == "sphere":
        changed, primitive.size_or_radius[0] = input_float(
            "Radius",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )

    elif prim_type == "torus":
        changed1, primitive.size_or_radius[0] = input_float(
            "Major Radius",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Minor Radius",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2

    elif prim_type == "hex_prism":
        changed1, primitive.size_or_radius[0] = input_float(
            "Hex Radius",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Height",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2

    elif prim_type == "vertical_capsule":
        changed1, primitive.size_or_radius[0] = input_float(
            "Height",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Radius",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2

    elif prim_type == "capped_cylinder":
        changed1, primitive.size_or_radius[0] = input_float(
            "Radius",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Height",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2

    elif prim_type == "rounded_cylinder":
        changed1, primitive.size_or_radius[0] = input_float(
            "Radius A",
            primitive.size_or_radius[0],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed2, primitive.size_or_radius[1] = input_float(
            "Radius B",
            primitive.size_or_radius[1],
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed3, primitive.kwargs["height"] = input_float(
            "Height",
            primitive.kwargs.get("height", 1.0),
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_float,
        )
        changed = changed1 or changed2 or changed3

    elif prim_type not in ["cone", "plane", "pointer", "sprite", "curve"]:
        changed, primitive.size_or_radius = input_vec3(
            "Size",
            primitive.size_or_radius,
            cn["STEP_VARIABLE_FLOAT"],
            panel_elem_width_vec3,
        )

    if changed:
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms


def _render_operation_inspector(node, scene_builder, item_data):
    """Render inspector for operation nodes."""
    imgui.text(f"Operation Type: {item_data.operation_type}")

    imgui.text("Operands:")
    for i, operand_id in enumerate(node.children):
        operand_node = scene_builder.get_node(operand_id)
        if operand_node:
            imgui.text(f"  {i+1}. {operand_node.item_data.ui_name} ({operand_id})")

    if hasattr(item_data, "smooth_k") and item_data.smooth_k is not None:
        changed, new_k = imgui.slider_float("Smooth K", item_data.smooth_k, 0.0, 1.0)
        if changed:
            item_data.smooth_k = new_k
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms