from src.core.SDFObjects import SDFOperation, SDFPrimitive
from src.core.classes.scene_tree.SceneNode import SceneNode

from typing import Dict, List, Any, Optional, Tuple

import copy


def add_operation_with_auto_primitives(
    builder,
    operation_type: str,
    ui_name: Optional[str] = None,
    auto_primitive_type: str = "box",
    forced_op_id: Optional[str] = None,
) -> str:
    # Determine operand count based on operation type
    operand_count = builder._get_operand_count(operation_type)

    # Create unique operation ID
    op_id = forced_op_id or f"d{builder.next_id}"
    builder._ensure_op_id_unique(op_id)

    # Pre-allocate operand IDs
    operand_ids = [f"d{builder.next_id + 1 + i}" for i in range(operand_count)]

    # Create operation with operand IDs as arguments
    operation = SDFOperation(
        operation_type, *operand_ids, ui_name=ui_name or operation_type
    )

    # Create operation node
    operation_node = SceneNode("operation", op_id, operation, parent_id=None)
    builder.scene_nodes[op_id] = operation_node
    builder.id_to_node[op_id] = operation_node

    # Add to root level
    builder.root_children.append(op_id)

    # Auto-create primitive children
    for i, operand_id in enumerate(operand_ids):
        # Offset each primitive so they're visible
        position = [1.0 * i, 0.0, 0.0]

        # Create primitive
        primitive = SDFPrimitive(
            builder.selected_item_id,
            auto_primitive_type,
            position,
            0.5,  # Default radius/size
            ui_name=f"{auto_primitive_type.title()} {i + 1}",
        )

        # Create primitive node as child of operation
        prim_node = SceneNode("primitive", operand_id, primitive, parent_id=op_id)
        builder.scene_nodes[operand_id] = prim_node
        builder.id_to_node[operand_id] = prim_node

        # Add to operation's children
        operation_node.add_child(operand_id)

        builder.next_id += 1

    builder.next_id += 1  # Increment for next item

    # Register undo/redo
    redo_kwargs = {"forced_op_id": op_id}
    builder.glob_history.add(
        builder.delete_node,  # undo: delete
        builder.add_operation_with_auto_primitives,  # redo: recreate
        (op_id,),  # undo args
        (operation_type, ui_name, auto_primitive_type),  # redo args
        {},
        redo_kwargs,
    )

    builder.invalidate_cache()
    return op_id


def get_item_name(builder, node_id: str) -> str:
    """Get the display name of a node (for compatibility)."""
    node = builder.get_node(node_id)
    if node:
        return node.item_data.ui_name
    return node_id


def modify_primitive_property(builder, node_id: str, property_name: str, new_value):
    """Compatibility method for modifying primitive properties."""
    node = builder.get_node(node_id)
    if not node or node.node_type != "primitive":
        return False

    prim = node.item_data

    if property_name == "position":
        prim.position = list(new_value)
    elif property_name == "rotation":
        prim.rotation = list(new_value)
    elif property_name == "scale":
        prim.scale = list(new_value)
    elif property_name == "color":
        prim.color = list(new_value)

    builder.invalidate_cache()
    return True


def delete_item(builder, node_id: str) -> bool:
    """Compatibility method for old delete_item calls."""
    return builder.delete_node(node_id)


def add_child_primitive(
    builder,
    parent_op_id: str,
    primitive_type: str = "box",
    position: List[float] = None,
    size_or_radius=0.5,
    rotation: Optional[List[float]] = None,
    scale: Optional[List[float]] = None,
    ui_name: Optional[str] = None,
    color: Optional[List[float]] = None,
    forced_op_id: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """
    Add a primitive as a child of an operation node.

    Returns the new primitive node id, or None on failure.
    """
    if position is None:
        position = [0.0, 0.0, 0.0]

    parent = builder.get_node(parent_op_id)
    if not parent or parent.node_type != "operation":
        return None

    # Check capacity
    required = builder._get_operand_count(parent.item_data.operation_type)
    if len(parent.children) >= required:
        return None

    node_id = forced_op_id or f"d{builder.next_id}"
    builder._ensure_op_id_unique(node_id)

    primitive = SDFPrimitive(
        builder.selected_item_id,
        primitive_type,
        position,
        size_or_radius,
        rotation,
        scale,
        ui_name or primitive_type,
        color,
        **(kwargs or {}),
    )

    prim_node = SceneNode("primitive", node_id, primitive, parent_id=parent_op_id)
    builder.scene_nodes[node_id] = prim_node
    builder.id_to_node[node_id] = prim_node

    # Attach to parent node
    parent.add_child(node_id)

    # Update operation args (append the child id)
    op = parent.item_data
    if hasattr(op, "args"):
        try:
            # prefer list
            if isinstance(op.args, tuple):
                op.args = list(op.args) + [node_id]
            else:
                op.args.append(node_id)
        except Exception:
            # Fallback: set args to single list
            op.args = getattr(op, "args", []) + [node_id]
    else:
        op.args = [node_id]

    if not forced_op_id:
        builder.next_id += 1

    # Register undo/redo: undo deletes node, redo re-creates as child with forced id
    redo_kwargs = copy.deepcopy(kwargs) if kwargs else {}
    redo_kwargs["forced_op_id"] = node_id
    builder.glob_history.add(
        builder.delete_node,
        builder.add_child_primitive,
        (node_id,),
        (
            parent_op_id,
            primitive_type,
            copy.deepcopy(position),
            copy.deepcopy(size_or_radius),
            copy.deepcopy(rotation),
            copy.deepcopy(scale),
            ui_name,
            copy.deepcopy(color),
        ),
        {},
        redo_kwargs,
    )

    builder.invalidate_cache()
    return node_id


def add_child_operation(
    builder,
    parent_op_id: str,
    operation_type: str,
    ui_name: Optional[str] = None,
    auto_primitive_type: str = "box",
    forced_op_id: Optional[str] = None,
) -> Optional[str]:
    """
    Add an operation node as a child of an existing operation. The newly-added
    operation will be created with its auto-primitives (same behavior as
    add_operation_with_auto_primitives) and then attached as a child to the parent.

    Returns the new operation id or None on failure.
    """
    parent = builder.get_node(parent_op_id)
    if not parent or parent.node_type != "operation":
        return None

    # Check capacity of parent
    required = builder._get_operand_count(parent.item_data.operation_type)
    if len(parent.children) >= required:
        return None

    # Create op id
    op_id = forced_op_id or f"d{builder.next_id}"
    builder._ensure_op_id_unique(op_id)

    # Determine operand count for the new child operation
    operand_count = builder._get_operand_count(operation_type)
    operand_ids = [f"d{builder.next_id + 1 + i}" for i in range(operand_count)]

    # Build the child operation (but parented to parent_op_id)
    operation = SDFOperation(
        operation_type, *operand_ids, ui_name=ui_name or operation_type
    )

    op_node = SceneNode("operation", op_id, operation, parent_id=parent_op_id)
    builder.scene_nodes[op_id] = op_node
    builder.id_to_node[op_id] = op_node

    # Attach to parent
    parent.add_child(op_id)

    # Ensure parent's args reference the op_id
    op_parent = parent.item_data
    if hasattr(op_parent, "args"):
        if isinstance(op_parent.args, tuple):
            op_parent.args = list(op_parent.args) + [op_id]
        else:
            op_parent.args.append(op_id)
    else:
        op_parent.args = [op_id]

    # Create the operand primitives for the newly-created child operation
    for i, operand_id in enumerate(operand_ids):
        position = [1.0 * i, 0.0, 0.0]
        primitive = SDFPrimitive(
            builder.selected_item_id,
            auto_primitive_type,
            position,
            0.5,
            ui_name=f"{auto_primitive_type.title()} {i + 1}",
        )
        prim_node = SceneNode("primitive", operand_id, primitive, parent_id=op_id)
        builder.scene_nodes[operand_id] = prim_node
        builder.id_to_node[operand_id] = prim_node
        op_node.add_child(operand_id)
        builder.next_id += 1

    builder.next_id += 1

    # Register undo/redo
    redo_kwargs = {"forced_op_id": op_id}
    builder.glob_history.add(
        builder.delete_node,
        builder.add_child_operation,
        (op_id,),
        (parent_op_id, operation_type, ui_name, auto_primitive_type),
        {},
        redo_kwargs,
    )

    builder.invalidate_cache()
    return op_id


def add_standalone_primitive(
    builder,
    primitive_type: str,
    position: List[float],
    size_or_radius,
    rotation: Optional[List[float]] = None,
    scale: Optional[List[float]] = None,
    ui_name: Optional[str] = None,
    color: Optional[List[float]] = None,
    forced_op_id: Optional[str] = None,
    **kwargs,
) -> str:
    op_id = forced_op_id or f"d{builder.next_id}"
    builder._ensure_op_id_unique(op_id)

    # Create primitive
    primitive = SDFPrimitive(
        builder.selected_item_id,
        primitive_type,
        position,
        size_or_radius,
        rotation,
        scale,
        ui_name or primitive_type,
        color,
        **kwargs,
    )

    # Create node
    prim_node = SceneNode("primitive", op_id, primitive, parent_id=None)
    builder.scene_nodes[op_id] = prim_node
    builder.id_to_node[op_id] = prim_node

    # Add to root
    builder.root_children.append(op_id)

    if not forced_op_id:
        builder.next_id += 1

    # Register undo/redo
    redo_kwargs = copy.deepcopy(kwargs) if kwargs else {}
    redo_kwargs["forced_op_id"] = op_id

    builder.glob_history.add(
        builder.delete_node,
        builder.add_standalone_primitive,
        (op_id,),
        (
            primitive_type,
            copy.deepcopy(position),
            copy.deepcopy(size_or_radius),
            copy.deepcopy(rotation),
            copy.deepcopy(scale),
            ui_name,
            copy.deepcopy(color),
        ),
        {},
        redo_kwargs,
    )

    builder.invalidate_cache()
    return op_id
