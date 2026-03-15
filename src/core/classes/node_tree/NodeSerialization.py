from typing import Optional, Any, Dict
from src.core.SDFObjects import SDFOperation, SDFPrimitive
from src.core.classes.scene_tree.SceneNode import SceneNode


def save_node_tree_state(builder, node_id: str) -> Optional[Dict]:
    """
    Save a node and all its descendants to a dictionary for undo.

    Returns:
        Dictionary representing the entire subtree, or None
    """
    node = builder.get_node(node_id)
    if not node:
        return None

    state = {
        "node_id": node_id,
        "node_type": node.node_type,
        "item_data": node.item_data.to_dict(),
        "parent_id": node.parent_id,
        "children": [],
    }

    # Recursively save children
    for child_id in node.children:
        child_state = builder._save_node_tree_state(child_id)
        if child_state:
            state["children"].append(child_state)

    return state


def restore_node_tree(builder, node_tree_state: Dict):
    """
    Restore a saved node tree (used by undo).

    Args:
        node_tree_state: Dictionary from _save_node_tree_state
    """
    if not node_tree_state:
        return

    def restore_recursive(state, parent_id=None):
        """Recursively restore nodes."""
        node_id = state["node_id"]
        node_type = state["node_type"]
        item_data_dict = state["item_data"]

        # Reconstruct the data object
        if node_type == "primitive":
            item_data = SDFPrimitive(
                builder.selected_item_id,
                primitive_type=item_data_dict["primitive_type"],
                position=item_data_dict["position"],
                size_or_radius=item_data_dict["size_or_radius"],
                rotation=item_data_dict.get("rotation"),
                scale=item_data_dict.get("scale"),
                ui_name=item_data_dict.get("ui_name"),
                color=item_data_dict.get("color"),
                **item_data_dict.get("kwargs", {})
            )
        else:  # operation
            item_data = SDFOperation(
                item_data_dict["operation_type"],
                *item_data_dict["args"],
                ui_name=item_data_dict.get("ui_name")
            )
            # Restore smooth_k if present
            if item_data_dict.get("smooth_k") is not None:
                item_data.smooth_k = item_data_dict["smooth_k"]

        # Create node
        node = SceneNode(node_type, node_id, item_data, parent_id=parent_id)
        builder.scene_nodes[node_id] = node
        builder.id_to_node[node_id] = node

        # Recursively restore children
        for child_state in state.get("children", []):
            child_id = restore_recursive(child_state, parent_id=node_id)
            node.children.append(child_id)

        return node_id

    # Restore the tree
    root_id = restore_recursive(node_tree_state)

    # If this is a root node being restored, add it back to root
    if root_id and root_id not in builder.root_children:
        builder.root_children.append(root_id)


def reconstruct_node(builder, node_id: str, node_data: Dict):
    """
    Reconstruct a single node from serialized data.

    Args:
        node_id: ID of node
        node_data: Serialized node data
    """
    node_type = node_data["node_type"]
    item_data_dict = node_data["item_data"]
    parent_id = node_data.get("parent_id")
    children_ids = node_data.get("children", [])

    # Reconstruct the appropriate object type
    if node_type == "primitive":
        item_data = SDFPrimitive(
            builder.selected_item_id,
            primitive_type=item_data_dict["primitive_type"],
            position=item_data_dict["position"],
            size_or_radius=item_data_dict["size_or_radius"],
            rotation=item_data_dict.get("rotation"),
            scale=item_data_dict.get("scale"),
            ui_name=item_data_dict.get("ui_name"),
            color=item_data_dict.get("color"),
            **item_data_dict.get("kwargs", {})
        )
        item_data.properties = item_data_dict.get("properties")
    else:  # operation
        item_data = SDFOperation(
            item_data_dict["operation_type"],
            *item_data_dict["args"],
            ui_name=item_data_dict.get("ui_name")
        )
        # Restore smooth_k if present
        if item_data_dict.get("smooth_k") is not None:
            item_data.smooth_k = item_data_dict["smooth_k"]

    # Create the node
    node = SceneNode(node_type, node_id, item_data, parent_id=parent_id)
    node.children = children_ids

    # Store in maps
    builder.scene_nodes[node_id] = node
    builder.id_to_node[node_id] = node
