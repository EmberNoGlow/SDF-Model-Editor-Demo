from src.core.SDFObjects import SDFOperation, SDFPrimitive
from src.core.classes.scene_tree.SceneNode import SceneNode

from typing import Optional


def rename_node(builder, node_id: str, new_name: str) -> bool:
    """
    Rename a node (update ui_name).

    Args:
        node_id: ID of node to rename
        new_name: New display name

    Returns:
        True if successful
    """
    node = builder.get_node(node_id)
    if not node:
        return False

    old_name = node.item_data.ui_name
    node.item_data.ui_name = new_name

    # Register undo/redo
    builder.glob_history.add(
        lambda nid, old: builder.rename_node(nid, old),
        lambda nid, new: builder.rename_node(nid, new),
        (node_id, old_name),
        (node_id, new_name),
        {},
        {},
    )

    builder.invalidate_cache()
    return True


def move_root_node(builder, node_id: str, new_index: int) -> bool:
    """
    Move a root-level node to a different position in root list.

    Args:
        node_id: ID of root node to move
        new_index: New position in root_children list

    Returns:
        True if successful
    """
    if node_id not in builder.root_children:
        return False

    old_index = builder.root_children.index(node_id)

    # Clamp new_index
    new_index = max(0, min(new_index, len(builder.root_children) - 1))

    if new_index == old_index:
        return True

    # Move
    builder.root_children.pop(old_index)
    builder.root_children.insert(new_index, node_id)

    # Register undo/redo
    builder.glob_history.add(
        builder.move_root_node,
        builder.move_root_node,
        (node_id, old_index),
        (node_id, new_index),
        {},
        {},
    )

    builder.invalidate_cache()
    return True


def apply_primitive_state(builder, node_id: str, state: dict) -> bool:
    """
    Internal helper to apply a saved primitive state to a node WITHOUT registering undo.
    Used as both the undo and redo callback so reapplying doesn't add nested history.
    """
    node = builder.get_node(node_id)
    if not node or node.node_type != "primitive":
        return False
    p = node.item_data
    p.primitive_type = state.get("primitive_type", p.primitive_type)
    p.position = list(state.get("position", p.position))
    p.size_or_radius = list(state.get("size_or_radius", p.size_or_radius))
    p.rotation = list(state.get("rotation", p.rotation))
    p.scale = list(state.get("scale", p.scale))
    p.color = list(state.get("color", p.color))
    p.kwargs = dict(state.get("kwargs", p.kwargs))
    return True


def change_primitive_type(
    builder, node_id: str, new_type: str, new_size_or_radius=None, **kwargs
) -> bool:
    """
    Change a primitive's type in-place (keeps the node_id stable).
    Registers an undo/redo entry that restores the previous state and re-applies the new state.
    """
    node = builder.get_node(node_id)
    if not node or node.node_type != "primitive":
        return False

    prim = node.item_data
    # Save old state snapshot
    old_state = prim.to_dict()

    # Compute new state (apply changes to a copy)
    new_state = old_state.copy()
    new_state["primitive_type"] = new_type

    # Normalize size_or_radius into list/appropriate form
    if new_size_or_radius is not None:
        new_state["size_or_radius"] = new_size_or_radius
    else:
        # sensible defaults for common types
        default_map = {
            "box": [0.5, 0.5, 0.5],
            "round_box": [0.5, 0.5, 0.5],
            "sphere": [0.5],
            "torus": [0.5, 0.25],
            "cone": [0.5],
            "plane": [1.0],
            "hex_prism": [0.5, 0.5],
            "vertical_capsule": [1.0, 0.3],
            "capped_cylinder": [0.3, 1.0],
            "rounded_cylinder": [0.3, 0.3],
        }
        new_state["size_or_radius"] = default_map.get(
            new_type, old_state.get("size_or_radius", [])
        )

    # Update kwargs if provided
    new_kwargs = dict(old_state.get("kwargs", {}))
    if kwargs:
        new_kwargs.update(kwargs)
    new_state["kwargs"] = new_kwargs

    # Immediately apply the new state (live)
    applied = builder._apply_primitive_state(node_id, new_state)
    if not applied:
        return False

    # Register undo/redo using the internal apply helper (so undo/redo won't add additional history)
    builder.glob_history.add(
        builder._apply_primitive_state,  # undo: apply old state
        builder._apply_primitive_state,  # redo: apply new state
        (node_id, old_state),
        (node_id, new_state),
        {},
        {},
    )

    builder.invalidate_cache()
    return True


def alloc_id(builder) -> str:
    """Allocate a new unique ID like d0, d1, ... and increment next_id."""
    op_id = f"d{builder.next_id}"
    builder.next_id += 1
    while op_id in builder.scene_nodes:
        op_id = f"d{builder.next_id}"
        builder.next_id += 1
    return op_id


def ensure_op_id_unique(builder, op_id: str) -> str:
    """
    Ensure a requested op_id is unique. If it's already present, return a fresh id.
    Do NOT delete existing nodes here.
    """
    if op_id in builder.scene_nodes:
        return builder._alloc_id()
    return op_id


def delete_subtree_no_history(builder, node_id: str):
    """Delete a node and all descendants without recording history (internal helper)."""
    if node_id not in builder.scene_nodes:
        return
    all_to_delete = [node_id] + builder.get_all_children_recursive(node_id)
    for cid in all_to_delete:
        if cid in builder.scene_nodes:
            # remove reference from parent if present
            parent = builder.get_parent(cid)
            if parent and cid in parent.children:
                parent.remove_child(cid)
            if cid in builder.root_children:
                try:
                    builder.root_children.remove(cid)
                except ValueError:
                    pass
            # delete maps
            if cid in builder.scene_nodes:
                del builder.scene_nodes[cid]
            if cid in builder.id_to_node:
                del builder.id_to_node[cid]


def change_node_to_operation(
    builder, node_id: str, operation_type: str, auto_primitive_type: str = "box"
) -> bool:
    """
    Convert a primitive node into an operation node IN-PLACE (keeps node_id).
    The original primitive becomes the first operand (moved into a newly-created child).
    Additional operand primitives are created automatically.
    Registers undo/redo by saving the subtree before/after conversion.
    """
    node = builder.get_node(node_id)
    if not node:
        return False

    # Save state for undo
    old_state = builder._save_node_tree_state(node_id)

    # If node is already an operation, simply update its type & return
    if node.node_type == "operation":
        node.item_data.operation_type = operation_type
        builder.invalidate_cache()
        return True

    # node is primitive: preserve its SDFPrimitive as first child
    if node.node_type != "primitive":
        return False

    old_prim = node.item_data

    # Create operand ids (keep count according to operation type)
    operand_count = builder._get_operand_count(operation_type)
    operand_ids = [builder._alloc_id() for _ in range(operand_count)]

    # Create new child nodes: move old primitive into operand_ids[0]
    child_nodes = []

    for i, oid in enumerate(operand_ids):
        if i == 0:
            # reuse the existing primitive object as child
            prim = old_prim
        else:
            # create default primitives for remaining operands
            prim = SDFPrimitive(
                builder.selected_item_id,
                auto_primitive_type,
                [1.0 * i, 0.0, 0.0],
                0.5,
                ui_name=f"{auto_primitive_type.title()} {i+1}",
            )
        prim_node = SceneNode("primitive", oid, prim, parent_id=node_id)
        builder.scene_nodes[oid] = prim_node
        builder.id_to_node[oid] = prim_node
        child_nodes.append(oid)

    # Replace the current node to be an operation
    operation = SDFOperation(operation_type, *operand_ids, ui_name=operation_type)
    node.node_type = "operation"
    node.item_data = operation
    node.children = child_nodes
    # ensure old primitive is no longer referenced as a root child
    if node_id in builder.root_children:
        # node remains root; children are nested under it
        pass

    new_state = builder._save_node_tree_state(node_id)

    # Register undo/redo: on undo restore old_state, on redo restore new_state
    def _do_restore(state):
        nid = state["node_id"]
        # remove existing subtree
        builder._delete_subtree_no_history(nid)
        builder._restore_node_tree(state)

    builder.glob_history.add(
        _do_restore,  # undo: restore old_state
        _do_restore,  # redo: restore new_state (we will pass new_state as redo args)
        (old_state,),
        (new_state,),
        {},
        {},
    )

    builder.invalidate_cache()
    return True


def change_node_to_primitive(
    builder,
    node_id: str,
    primitive_type: str = "box",
    position=None,
    size_or_radius=None,
    rotation=None,
    scale=None,
    color=None,
    **kwargs,
) -> bool:
    """
    Convert an operation node (and its children) into a single primitive node IN-PLACE.
    The operation's subtree is deleted; the node becomes a primitive with supplied parameters.
    Undo/redo is registered by saving the old subtree and the new subtree.
    """
    node = builder.get_node(node_id)
    if not node:
        return False

    old_state = builder._save_node_tree_state(node_id)

    # If already primitive, update its type/params
    if node.node_type == "primitive":
        prim = node.item_data
        prim.primitive_type = primitive_type
        if position is not None:
            prim.position = list(position)
        if size_or_radius is not None:
            prim.size_or_radius = size_or_radius
        if rotation is not None:
            prim.rotation = rotation
        if scale is not None:
            prim.scale = scale
        if color is not None:
            prim.color = color
        if kwargs:
            prim.kwargs.update(kwargs)

        new_state = builder._save_node_tree_state(node_id)

        def _do_restore(state):
            nid = state["node_id"]
            builder._delete_subtree_no_history(nid)
            builder._restore_node_tree(state)

        builder.glob_history.add(
            _do_restore, _do_restore, (old_state,), (new_state,), {}, {}
        )
        builder.invalidate_cache()
        return True

    # node is operation: delete its children and replace with a primitive
    # Remove children nodes
    child_ids = list(node.children)
    for cid in child_ids:
        builder._delete_subtree_no_history(cid)

    # Replace with new primitive
    pos = position or [0.0, 0.0, 0.0]
    s_or_r = (
        size_or_radius
        if size_or_radius is not None
        else (0.5 if primitive_type != "box" else [0.5, 0.5, 0.5])
    )
    prim = SDFPrimitive(
        builder.selected_item_id,
        primitive_type,
        pos,
        s_or_r,
        rotation or [0.0, 0.0, 0.0],
        scale or [1.0, 1.0, 1.0],
        ui_name=primitive_type,
        color=color or [0.8, 0.6, 0.4],
        **(kwargs or {}),
    )
    node.node_type = "primitive"
    node.item_data = prim
    node.children = []

    new_state = builder._save_node_tree_state(node_id)

    def _do_restore(state):
        nid = state["node_id"]
        builder._delete_subtree_no_history(nid)
        builder._restore_node_tree(state)

    builder.glob_history.add(
        _do_restore, _do_restore, (old_state,), (new_state,), {}, {}
    )

    builder.invalidate_cache()
    return True


def reparent_node(
    builder, node_id: str, new_parent_id: str, child_to_replace_id: Optional[str] = None
) -> bool:
    """
    Reparent a node to a new parent operation.

    If new_parent_id already has max children, child_to_replace_id specifies which child to delete.

    Args:
        node_id: Node to reparent
        new_parent_id: New parent operation node ID
        child_to_replace_id: Child of new_parent to delete (if parent is at capacity)

    Returns:
        True if successful
    """
    node = builder.get_node(node_id)
    new_parent = builder.get_node(new_parent_id)

    if not node or not new_parent:
        return False

    # Can't reparent to self or descendants
    all_descendants = builder.get_all_children_recursive(node_id)
    if new_parent_id == node_id or new_parent_id in all_descendants:
        return False

    # New parent must be operation
    if new_parent.node_type != "operation":
        return False

    # Check capacity
    required = builder._get_operand_count(new_parent.item_data.operation_type)
    current_children = len(new_parent.children)

    # If at capacity, must delete a child first
    if current_children >= required:
        if (
            child_to_replace_id is None
            or child_to_replace_id not in new_parent.children
        ):
            return False
        # Delete the child to make room
        builder.delete_node(child_to_replace_id)

    # Save state for undo
    old_parent_id = node.parent_id
    old_state = builder._save_node_tree_state(node_id)

    # Remove from old parent
    if old_parent_id:
        old_parent = builder.get_node(old_parent_id)
        if old_parent:
            old_parent.remove_child(node_id)
    else:
        # Was root
        if node_id in builder.root_children:
            builder.root_children.remove(node_id)

    # Add to new parent
    node.parent_id = new_parent_id
    new_parent.add_child(node_id)

    # Update new parent's operation args
    op = new_parent.item_data
    if hasattr(op, "args"):
        try:
            if isinstance(op.args, tuple):
                op.args = list(op.args) + [node_id]
            else:
                op.args.append(node_id)
        except Exception:
            op.args = getattr(op, "args", []) + [node_id]
    else:
        op.args = [node_id]

    # Register undo/redo
    def undo_reparent():
        # Restore old tree structure
        builder._delete_subtree_no_history(node_id)
        builder._restore_node_tree(old_state)

    builder.glob_history.add(
        undo_reparent,
        lambda: reparent_node(builder, node_id, new_parent_id, child_to_replace_id),
        (),
        (),
        {},
        {},
    )

    builder.invalidate_cache()
    return True
