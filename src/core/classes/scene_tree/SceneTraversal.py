from typing import Optional, List, Tuple, Any
from .SceneNode import SceneNode


def get_children(builder, node_id: str) -> List[str]:
    """Get list of child IDs for a node."""
    node = builder.get_node(node_id)
    return node.children if node else []


def get_parent(builder, node_id: str) -> Optional[SceneNode]:
    """Get parent node of a node, or None if root."""
    node = builder.get_node(node_id)
    if node and node.parent_id:
        return builder.get_node(node.parent_id)
    return None


def get_root_nodes(builder) -> List[Tuple[str, SceneNode]]:
    """Get all root-level nodes as list of (node_id, node) tuples."""
    return [(nid, builder.scene_nodes[nid]) for nid in builder.root_children]


def get_all_nodes_flat(builder) -> List[Tuple[str, Any]]:
    """
    Get all nodes in flat list (for backwards compatibility).
    Traverses tree in depth-first order.

    Returns:
        List of (node_id, item_data) tuples
    """
    result = []

    def traverse(node_id):
        node = builder.scene_nodes.get(node_id)
        if node:
            result.append((node_id, node.item_data))
            for child_id in node.children:
                traverse(child_id)

    for root_id in builder.root_children:
        traverse(root_id)

    return result


def get_node_depth(builder, node_id: str) -> int:
    """Get depth of node (0 = root, 1 = child of root, etc.)"""
    depth = 0
    node = builder.get_node(node_id)
    while node and node.parent_id:
        depth += 1
        node = builder.get_node(node.parent_id)
    return depth


def get_all_children_recursive(builder, node_id: str) -> List[str]:
    """Get all descendant IDs (children, grandchildren, etc.)."""
    result = []
    node = builder.get_node(node_id)
    if not node:
        return result

    for child_id in node.children:
        result.append(child_id)
        result.extend(builder.get_all_children_recursive(child_id))

    return result
