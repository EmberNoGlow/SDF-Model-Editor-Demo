from typing import Dict, List, Any, Optional, Tuple


class SceneNode:
    """
    Represents a single node in the hierarchical scene tree.

    A node can be either:
    - An Operation (has children which are its operands/primitives)
    - A Primitive (leaf node or child of operation)

    Attributes:
        node_type: 'operation' or 'primitive'
        item_id: Unique identifier like 'd0', 'd1', etc.
        item_data: The actual SDFOperation or SDFPrimitive object
        parent_id: ID of parent node (None if root)
        children: List of child node IDs
    """

    def __init__(
        self, node_type: str, item_id: str, item_data, parent_id: Optional[str] = None
    ):
        self.node_type = node_type  # 'operation' or 'primitive'
        self.item_id = item_id
        self.item_data = item_data  # SDFOperation or SDFPrimitive
        self.parent_id = parent_id
        self.children = []  # List of child item_ids

    def add_child(self, child_id: str):
        """Add a child node ID."""
        if child_id not in self.children:
            self.children.append(child_id)

    def remove_child(self, child_id: str):
        """Remove a child node ID."""
        if child_id in self.children:
            self.children.remove(child_id)

    def to_dict(self) -> Dict:
        """Serialize node to dictionary for JSON storage."""
        return {
            "node_type": self.node_type,
            "item_id": self.item_id,
            "parent_id": self.parent_id,
            "children": self.children,
            "item_data": self.item_data.to_dict(),
        }
