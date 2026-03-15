from .SDFObjects import SDFOperation


def generate_raymarch_code(scene_builder) -> str:
    """
    Generate GLSL code from the hierarchical scene tree.

    The natural tree structure ensures dependencies are satisfied:
    - Children (operands) are emitted before parent (operation)
    - No need to track "valid operands" or ordering constraints

    Returns:
        GLSL code as string
    """
    scene_lines = []
    emitted_ids = set()
    last_emitted_id = None

    def emit_node_code(node_id: str) -> str:
        """
        Recursively emit code for a node and its children.

        Returns:
            The variable ID that represents this node's output
        """
        nonlocal last_emitted_id

        if node_id in emitted_ids:
            return node_id

        node = scene_builder.get_node(node_id)
        if not node:
            return None

        # Emit all children first (they are operands)
        child_ids = []
        for child_id in node.children:
            child_id_out = emit_node_code(child_id)
            if child_id_out:
                child_ids.append(child_id_out)

        # Now emit this node
        if node.node_type == "primitive":
            primitive = node.item_data
            primitive.update_selected_item_id(scene_builder.selected_item_id)

            # Emit transform code
            transform_code = primitive.generate_transform_code(node_id)
            if transform_code:
                scene_lines.append(transform_code)

            # Emit SDF code
            sdf_code = primitive.generate_sdf_code(node_id)
            if sdf_code:
                scene_lines.append(sdf_code)

            last_emitted_id = node_id

        elif node.node_type == "operation":
            operation = node.item_data

            # Use child IDs as operands (they're already emitted)
            # Determine minimum operand count (1 or 2) and skip if not enough children
            required_ops = scene_builder._get_operand_count(operation.operation_type)
            if len(child_ids) < required_ops:
                # Missing operands (e.g. child was deleted) — skip emitting this operation
                emitted_ids.add(node_id)
                last_emitted_id = node_id
                return node_id

            # Build op_args: start with emitted child IDs (in order), then append any literal args
            op_args = list(child_ids)
            for arg in operation.args:
                if isinstance(arg, str):
                    # If arg matches one of the child IDs, it's already included
                    if arg in child_ids:
                        continue
                    else:
                        op_args.append(arg)
                else:
                    op_args.append(arg)

            # Some operations require an extra numeric parameter (smooth k, round param, etc.).
            # Ensure those are present; prefer stored attributes like smooth_k if available.
            if operation.operation_type in {"sunion", "ssub", "sinter", "mix"}:
                # These templates expect (d_a, d_b, k)
                if len(op_args) < 3:
                    default_k = getattr(operation, "smooth_k", 0.1)
                    op_args.append(default_k)
            elif operation.operation_type in {"round", "onion", "snoiseDisp"}:
                # These templates expect (d_a, param)
                if len(op_args) < 2:
                    default_param = getattr(operation, "param", 0.1)
                    op_args.append(default_param)

            # Reconstruct operation with resolved arguments (so generate_code sees correct ids/params)
            op_copy = SDFOperation(
                operation.operation_type, *op_args, ui_name=operation.ui_name
            )
            # Preserve smooth_k if present on the original operation object
            if hasattr(operation, "smooth_k"):
                op_copy.smooth_k = operation.smooth_k

            # Emit operation code
            try:
                op_code = op_copy.generate_code(node_id)
                if op_code:
                    scene_lines.append(op_code)
            except Exception as e:
                # Defensive: if generation fails, skip this node but mark it emitted so we don't loop
                print(f"Warning: failed to generate code for operation {node_id}: {e}")

            last_emitted_id = node_id
            emitted_ids.add(node_id)
            return node_id

        emitted_ids.add(node_id)
        return node_id

    # Emit all root nodes
    for root_id in scene_builder.root_children:
        emit_node_code(root_id)

    # Build final shader code
    if scene_lines:
        scene_code = "\n    ".join(scene_lines)

        # Find last emitted ID for return statement
        if last_emitted_id:
            scene_code += f"\n    return vec4(col{last_emitted_id}, {last_emitted_id});"
        else:
            scene_code += "\n    return vec4(0.0, 0.0, 0.0, 1000.0);"

        return scene_code
    else:
        return "return vec4(0.0, 0.0, 0.0, 1000.0);"
