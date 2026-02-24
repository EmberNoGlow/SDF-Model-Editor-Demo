
from typing import Optional, List, Tuple
from .SceneNode import SceneNode

def get_item_by_id(builder, node_id: str):
    """
    Return the underlying item (SDFPrimitive or SDFOperation) for node_id,
    or None if node isn't found.
    """
    node = builder.get_node(node_id)
    return node.item_data if node else None

def get_primitive_by_index(builder, idx: int):
    """
    Compatibility helper: return the primitive object at flat index idx
    from the compatibility .primitives property, or raise IndexError.
    """
    node_id, prim = builder.primitives[idx]
    return prim

def get_node_by_id(builder, node_id: str) -> Optional[SceneNode]:
    """Return the SceneNode for node_id (thin wrapper)."""
    return builder.get_node(node_id)

def get_node(builder, node_id: str) -> Optional[SceneNode]:
    """Get a node by ID, or None if not found."""
    return builder.scene_nodes.get(node_id)