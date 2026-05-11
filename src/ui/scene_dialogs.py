"""Scene add/change dialogs."""
import imgui
from src.app.data.states import st
from src.rendering.shader_compiler import recompile_shader


def render_add_change_window(width, height, scene_builder):
    """Render the combined add/change type dialog."""
    if not st.show_add_change_window:
        return

    imgui.set_next_window_position(width // 2 - 300, height // 2 - 235)
    imgui.set_next_window_size(600, 470)
    is_open, st.show_add_change_window = imgui.begin(
        "Add / Change Type", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_add_change_window = False
        st.pending_change_node_id = None
        imgui.end()
        return

    primitives_list = _get_primitives_list()
    operations_list = _get_operations_list()

    imgui.columns(2, "add_change_cols", border=True)
    imgui.set_column_width(0, 290)

    imgui.text("Primitives")
    imgui.separator()

    for label, prim_type, size in primitives_list:
        if imgui.button(label, -1, 24):
            _handle_primitive_selection(prim_type, label, size, scene_builder)

    imgui.next_column()
    imgui.text("Operations")
    imgui.separator()

    for label, op_type, operand_count, description in operations_list:
        if imgui.button(label, -1, 24):
            _handle_operation_selection(op_type, label, scene_builder)

        if imgui.is_item_hovered():
            imgui.set_tooltip(description)

    imgui.columns(1)
    imgui.separator()
    imgui.spacing()
    imgui.same_line(20, 0)

    if imgui.button("Cancel", 265, 28):
        st.show_add_change_window = False
        st.pending_change_node_id = None

    imgui.same_line(305, 0)

    if imgui.button("Close", 265, 28):
        st.show_add_change_window = False
        st.pending_change_node_id = None

    imgui.end()


def render_property_change_window(width, height, scene_builder):
    """Render the property change window for symmetry and other properties."""
    if not st.show_property_change_window:
        return

    imgui.set_next_window_position(width // 2 - 150, height // 2 - 125)
    imgui.set_next_window_size(300, 250)
    is_open, st.show_property_change_window = imgui.begin(
        "Change Properties", True, imgui.WINDOW_NO_COLLAPSE
    )

    if not is_open:
        st.show_property_change_window = False
        st.property_change_node_id = None
        imgui.end()
        return

    node = scene_builder.get_node(st.property_change_node_id)
    if not node:
        imgui.end()
        return

    prim = node.item_data
    sym = prim.properties.get("symmetry")
    if sym is None:
        sym = [False, False, False]
        prim.update_property("symmetry", sym)

    imgui.spacing()
    imgui.text("Symmetry:")
    imgui.same_line()
    changed_x, sym[0] = imgui.checkbox("X", sym[0])
    imgui.same_line()
    changed_y, sym[1] = imgui.checkbox("Y", sym[1])
    imgui.same_line()
    changed_z, sym[2] = imgui.checkbox("Z", sym[2])
    imgui.spacing()

    if changed_x or changed_y or changed_z:
        prim.update_property("symmetry", sym)
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

    if imgui.button("Close", -1):
        st.show_property_change_window = False
        st.property_change_node_id = None

    imgui.end()


def render_reparent_window(width, height, scene_builder):
    """Render the reparent node window."""
    if not st.show_reparent_window or not st.reparent_node_id:
        return

    imgui.set_next_window_size(400, 500)
    st.show_reparent_window, _ = imgui.begin("Reparent Node", True)

    if not st.show_reparent_window:
        imgui.end()
        return

    reparent_node = scene_builder.get_node(st.reparent_node_id)
    if not reparent_node:
        imgui.end()
        return

    imgui.text(f"Reparenting: {reparent_node.item_data.ui_name}")
    imgui.separator()
    imgui.text("Select new parent operation:")

    all_descendants = scene_builder.get_all_children_recursive(st.reparent_node_id)
    all_descendants.append(st.reparent_node_id)

    parent_selected = False
    for root_id in scene_builder.root_children:
        parent_selected |= _render_reparent_node_list(
            root_id, st.reparent_node_id, all_descendants, scene_builder, "  "
        )

    if parent_selected and st.reparent_target_parent:
        _render_reparent_operand_selection(st.reparent_target_parent, scene_builder)

    imgui.spacing()
    imgui.separator()

    if imgui.button("Cancel", 100, 30):
        st.show_reparent_window = False
        st.reparent_node_id = None
        st.reparent_target_parent = None
        st.reparent_child_to_replace = None

    imgui.same_line(150)

    can_reparent = _can_reparent_node(st.reparent_target_parent, scene_builder)
    if imgui.button("Reparent", 100, 30):
        if can_reparent:
            if scene_builder.reparent_node(
                st.reparent_node_id,
                st.reparent_target_parent,
                st.reparent_child_to_replace,
            ):
                success, new_uniforms = recompile_shader(scene_builder)
                if success:
                    st.uniform_locs = new_uniforms

            st.show_reparent_window = False
            st.reparent_node_id = None
            st.reparent_target_parent = None
            st.reparent_child_to_replace = None

    imgui.end()


def _render_reparent_node_list(node_id, exclude_node_id, exclude_descendants, scene_builder, indent=""):
    """Recursively render selectable nodes for reparent window."""
    if node_id in exclude_descendants or node_id == exclude_node_id:
        return False

    node = scene_builder.get_node(node_id)
    if not node or node.node_type != "operation":
        return False

    label = f"{indent}{node.item_data.ui_name} ({node_id})"
    clicked, _ = imgui.selectable(label, False)

    if clicked:
        st.reparent_target_parent = node_id
        return True

    result = imgui.is_item_clicked()

    for child_id in node.children:
        child_node = scene_builder.get_node(child_id)
        if child_node and child_node.node_type == "operation":
            result |= _render_reparent_node_list(
                child_id,
                exclude_node_id,
                exclude_descendants,
                scene_builder,
                indent + "  ",
            )

    return result


def _render_reparent_operand_selection(parent_id, scene_builder):
    """Render operand selection if parent is at capacity."""
    parent_node = scene_builder.get_node(parent_id)
    if not parent_node or parent_node.node_type != "operation":
        return

    required_operands = scene_builder._get_operand_count(
        parent_node.item_data.operation_type
    )
    current_operands = len(parent_node.children)

    if current_operands >= required_operands:
        imgui.separator()
        imgui.text(f"Parent is full ({current_operands}/{required_operands} operands)")
        imgui.text("Select child to replace:")

        for i, child_id in enumerate(parent_node.children):
            child_node = scene_builder.get_node(child_id)
            if child_node:
                if imgui.selectable(
                    f"{child_node.item_data.ui_name} ({child_id})",
                    st.reparent_child_to_replace == child_id,
                )[0]:
                    st.reparent_child_to_replace = child_id


def _can_reparent_node(target_parent_id, scene_builder):
    """Check if reparenting to target parent is valid."""
    if target_parent_id is None:
        return False

    parent_node = scene_builder.get_node(target_parent_id)
    if not parent_node or parent_node.node_type != "operation":
        return False

    required_operands = scene_builder._get_operand_count(
        parent_node.item_data.operation_type
    )
    current_operands = len(parent_node.children)

    if current_operands >= required_operands and st.reparent_child_to_replace is None:
        return False

    return True


def _get_primitives_list():
    """Return list of available primitives."""
    return [
        ("Box", "box", (0.5, 0.5, 0.5)),
        ("Sphere", "sphere", 0.5),
        ("Round Box", "round_box", (0.5, 0.5, 0.5)),
        ("Torus", "torus", (0.5, 0.25)),
        ("Cone", "cone", None),
        ("Plane", "plane", None),
        ("Hex Prism", "hex_prism", (0.5, 0.5)),
        ("Vertical Capsule", "vertical_capsule", (1.0, 0.3)),
        ("Capped Cylinder", "capped_cylinder", (0.3, 1.0)),
        ("Rounded Cylinder", "rounded_cylinder", (0.3, 0.3)),
    ]


def _get_operations_list():
    """Return list of available operations."""
    return [
        ("Union", "union", 2, "Combines two shapes (minimum distance)"),
        ("Subtraction", "sub", 2, "Subtracts second from first"),
        ("Intersection", "inter", 2, "Keeps only overlapping parts"),
        ("Smooth Union", "sunion", 2, "Union with smooth blending"),
        ("Smooth Subtraction", "ssub", 2, "Subtraction with smooth blending"),
        ("Smooth Intersection", "sinter", 2, "Intersection with smooth blending"),
        ("Mix", "mix", 2, "Blends between two distances"),
        ("Invert", "invert", 1, "Inverts the shape"),
        ("Round", "round", 1, "Rounds the shape"),
        ("Onion", "onion", 1, "Creates a shell effect"),
        ("XOR", "xor", 2, "Exclusive OR operation"),
        ("snoiseDisp", "snoiseDisp", 1, "Noise displacement"),
    ]


def _handle_primitive_selection(prim_type, label, size, scene_builder):
    """Handle primitive type selection in add/change dialog."""
    if st.pending_change_node_id is None:
        new_id = scene_builder.add_standalone_primitive(
            prim_type,
            position=[0.0, 0.0, 0.0],
            size_or_radius=size if size is not None else 0.5,
            ui_name=label,
        )
        if new_id:
            st.selected_items.clear()
            st.selected_item_id = new_id
            scene_builder.update_selected_item_id(st.selected_item_id)
            st.selection_mode = "node"
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms
    else:
        scene_builder.change_node_to_primitive(
            st.pending_change_node_id,
            prim_type,
            position=None,
            size_or_radius=(size if size is not None else 0.5),
        )
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

        st.pending_change_node_id = None

    st.show_add_change_window = False


def _handle_operation_selection(op_type, label, scene_builder):
    """Handle operation type selection in add/change dialog."""
    if st.pending_change_node_id is None:
        new_op_id = scene_builder.add_operation_with_auto_primitives(
            op_type, auto_primitive_type="box", ui_name=label
        )
        if new_op_id:
            st.selected_items.clear()
            st.selected_item_id = new_op_id
            scene_builder.update_selected_item_id(st.selected_item_id)
            st.selection_mode = "node"
            success, new_uniforms = recompile_shader(scene_builder)
            if success:
                st.uniform_locs = new_uniforms
    else:
        scene_builder.change_node_to_operation(
            st.pending_change_node_id, op_type, auto_primitive_type="box"
        )
        success, new_uniforms = recompile_shader(scene_builder)
        if success:
            st.uniform_locs = new_uniforms

        st.pending_change_node_id = None

    st.show_add_change_window = False