"""
NEW HIERARCHICAL SCENE BUILDER

This module provides a complete refactoring of the scene architecture
from a flat primitives/operations model to a hierarchical tree model.

Key improvements:
- Operations are root-level nodes that own their operand primitives
- Clear parent-child relationships make dependencies obvious
- Auto-creation of primitives when operations are added
- Cascade deletion (deleting operation deletes its children)
- Intuitive tree-based UI similar to Blender/Godot
"""

import json
import numpy as np
import copy
import math
from typing import Dict, List, Any, Optional, Tuple

from src.classes.SDFObjects import SDFOperation, SDFPrimitive


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
    
    def __init__(self, node_type: str, item_id: str, item_data, parent_id: Optional[str] = None):
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
            'node_type': self.node_type,
            'item_id': self.item_id,
            'parent_id': self.parent_id,
            'children': self.children,
            'item_data': self.item_data.to_dict()
        }


class SDFSceneBuilder:
    """
    New hierarchical scene builder that organizes the scene as a tree
    instead of flat primitives and operations lists.
    
    Usage:
        builder = SDFSceneBuilderHierarchical(history, selected_id)
        
        # Add an operation (auto-creates children)
        union_id = builder.add_operation_with_auto_primitives(
            'union',
            auto_primitive_type='box',
            ui_name='Union 1'
        )
        
        # Add a standalone primitive at root
        sphere_id = builder.add_standalone_primitive(
            'sphere',
            position=[0, 0, 0],
            size_or_radius=0.5,
            ui_name='Sphere'
        )
        
        # Query the tree
        root_nodes = builder.get_root_nodes()
        children = builder.get_children(union_id)
    """
    
    def __init__(self, glob_history, selected_item_id):
        """
        Initialize the hierarchical scene builder.
        
        Args:
            glob_history: History/undo-redo manager
            selected_item_id: Reference to current selection (mutable reference)
        """
        self.scene_nodes = {}  # node_id -> SceneNode
        self.root_children = []  # IDs of root-level nodes
        self.next_id = 0
        self.glob_history = glob_history
        self.selected_item_id = selected_item_id
        
        # Reverse lookup for quick access
        self.id_to_node = {}  # item_id -> SceneNode
        
        # Cache for shader code generation
        self._shader_cache = None
        self._cache_valid = False

    
    def add_operation_with_auto_primitives(
        self,
        operation_type: str,
        ui_name: Optional[str] = None,
        auto_primitive_type: str = 'box',
        forced_op_id: Optional[str] = None
    ) -> str:
        """
        Add an operation at the ROOT LEVEL with auto-generated primitive children.
        
        This is the primary way to add operations in the new model.
        When you create a Union, it automatically creates two Box primitives as children.
        When you create an Invert, it creates one Box primitive as a child.
        
        Args:
            operation_type: Type of operation ('union', 'sub', 'inter', 'sunion', 
                          'ssub', 'sinter', 'mix', 'invert', 'round', 'onion', 'snoiseDisp')
            ui_name: Display name for the operation (default: operation type)
            auto_primitive_type: Type of primitives to auto-create ('box', 'sphere', etc.)
            forced_op_id: For undo/redo to recreate with same ID
        
        Returns:
            operation_op_id: ID of the created operation
        
        Example:
            >>> union_id = builder.add_operation_with_auto_primitives(
            ...     'union',
            ...     auto_primitive_type='sphere',
            ...     ui_name='My Union'
            ... )
            # Creates: Union (d0)
            #   ├── Sphere (d1)
            #   └── Sphere (d2)
        """
        # Determine operand count based on operation type
        operand_count = self._get_operand_count(operation_type)
        
        # Create unique operation ID
        op_id = forced_op_id or f"d{self.next_id}"
        self._ensure_op_id_unique(op_id)
        
        # Pre-allocate operand IDs
        operand_ids = [f"d{self.next_id + 1 + i}" for i in range(operand_count)]
        
        # Create operation with operand IDs as arguments
        operation = SDFOperation(
            operation_type,
            *operand_ids,
            ui_name=ui_name or operation_type
        )
        
        # Create operation node
        operation_node = SceneNode('operation', op_id, operation, parent_id=None)
        self.scene_nodes[op_id] = operation_node
        self.id_to_node[op_id] = operation_node
        
        # Add to root level
        self.root_children.append(op_id)
        
        # Auto-create primitive children
        for i, operand_id in enumerate(operand_ids):
            # Offset each primitive so they're visible
            position = [1.0 * i, 0.0, 0.0]
            
            # Create primitive
            primitive = SDFPrimitive(
                self.selected_item_id,
                auto_primitive_type,
                position,
                0.5,  # Default radius/size
                ui_name=f"{auto_primitive_type.title()} {i + 1}"
            )
            
            # Create primitive node as child of operation
            prim_node = SceneNode(
                'primitive',
                operand_id,
                primitive,
                parent_id=op_id
            )
            self.scene_nodes[operand_id] = prim_node
            self.id_to_node[operand_id] = prim_node
            
            # Add to operation's children
            operation_node.add_child(operand_id)
            
            self.next_id += 1
        
        self.next_id += 1  # Increment for next item
        
        # Register undo/redo
        redo_kwargs = {'forced_op_id': op_id}
        self.glob_history.add(
            self.delete_node,  # undo: delete
            self.add_operation_with_auto_primitives,  # redo: recreate
            (op_id,),  # undo args
            (operation_type, ui_name, auto_primitive_type),  # redo args
            {},
            redo_kwargs
        )
        
        self.invalidate_cache()
        return op_id
    

    def update_glob_history(self, new_value):
        """Update the global history reference."""
        self.glob_history = new_value
    
    def get_item_name(self, node_id: str) -> str:
        """Get the display name of a node (for compatibility)."""
        node = self.get_node(node_id)
        if node:
            return node.item_data.ui_name
        return node_id
    

    def modify_primitive_property(self, node_id: str, property_name: str, new_value):
        """Compatibility method for modifying primitive properties."""
        node = self.get_node(node_id)
        if not node or node.node_type != 'primitive':
            return False
        
        prim = node.item_data
        
        if property_name == 'position':
            prim.position = list(new_value)
        elif property_name == 'rotation':
            prim.rotation = list(new_value)
        elif property_name == 'scale':
            prim.scale = list(new_value)
        elif property_name == 'color':
            prim.color = list(new_value)
        
        self.invalidate_cache()
        return True





    # COMPATIBILITY METHODS (for old code that used primitives/operations lists)
    @property
    def primitives(self):
        """Compatibility property that returns primitives as flat list."""
        result = []
        for node_id, node in self.scene_nodes.items():
            if node.node_type == 'primitive':
                result.append((node_id, node.item_data))
        return result
    
    @property
    def operations(self):
        """Compatibility property that returns operations as flat list."""
        result = []
        for node_id, node in self.scene_nodes.items():
            if node.node_type == 'operation':
                result.append((node_id, node.item_data))
        return result
    
    def delete_item(self, node_id: str) -> bool:
        """Compatibility method for old delete_item calls."""
        return self.delete_node(node_id)


    def add_child_primitive(
        self,
        parent_op_id: str,
        primitive_type: str = 'box',
        position: List[float] = None,
        size_or_radius=0.5,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        ui_name: Optional[str] = None,
        color: Optional[List[float]] = None,
        forced_op_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Add a primitive as a child of an operation node.

        Returns the new primitive node id, or None on failure.
        """
        if position is None:
            position = [0.0, 0.0, 0.0]

        parent = self.get_node(parent_op_id)
        if not parent or parent.node_type != 'operation':
            return None

        # Check capacity
        required = self._get_operand_count(parent.item_data.operation_type)
        if len(parent.children) >= required:
            return None

        node_id = forced_op_id or f"d{self.next_id}"
        self._ensure_op_id_unique(node_id)

        primitive = SDFPrimitive(
            self.selected_item_id,
            primitive_type,
            position,
            size_or_radius,
            rotation,
            scale,
            ui_name or primitive_type,
            color,
            **(kwargs or {})
        )

        prim_node = SceneNode('primitive', node_id, primitive, parent_id=parent_op_id)
        self.scene_nodes[node_id] = prim_node
        self.id_to_node[node_id] = prim_node

        # Attach to parent node
        parent.add_child(node_id)

        # Update operation args (append the child id)
        op = parent.item_data
        if hasattr(op, 'args'):
            try:
                # prefer list
                if isinstance(op.args, tuple):
                    op.args = list(op.args) + [node_id]
                else:
                    op.args.append(node_id)
            except Exception:
                # Fallback: set args to single list
                op.args = getattr(op, 'args', []) + [node_id]
        else:
            op.args = [node_id]

        if not forced_op_id:
            self.next_id += 1

        # Register undo/redo: undo deletes node, redo re-creates as child with forced id
        redo_kwargs = copy.deepcopy(kwargs) if kwargs else {}
        redo_kwargs['forced_op_id'] = node_id
        self.glob_history.add(
            self.delete_node,
            self.add_child_primitive,
            (node_id,),
            (parent_op_id, primitive_type, copy.deepcopy(position), copy.deepcopy(size_or_radius),
            copy.deepcopy(rotation), copy.deepcopy(scale), ui_name, copy.deepcopy(color)),
            {},
            redo_kwargs
        )

        self.invalidate_cache()
        return node_id


    def add_child_operation(
        self,
        parent_op_id: str,
        operation_type: str,
        ui_name: Optional[str] = None,
        auto_primitive_type: str = 'box',
        forced_op_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Add an operation node as a child of an existing operation. The newly-added
        operation will be created with its auto-primitives (same behavior as
        add_operation_with_auto_primitives) and then attached as a child to the parent.

        Returns the new operation id or None on failure.
        """
        parent = self.get_node(parent_op_id)
        if not parent or parent.node_type != 'operation':
            return None

        # Check capacity of parent
        required = self._get_operand_count(parent.item_data.operation_type)
        if len(parent.children) >= required:
            return None

        # Create op id
        op_id = forced_op_id or f"d{self.next_id}"
        self._ensure_op_id_unique(op_id)

        # Determine operand count for the new child operation
        operand_count = self._get_operand_count(operation_type)
        operand_ids = [f"d{self.next_id + 1 + i}" for i in range(operand_count)]

        # Build the child operation (but parented to parent_op_id)
        operation = SDFOperation(
            operation_type,
            *operand_ids,
            ui_name=ui_name or operation_type
        )

        op_node = SceneNode('operation', op_id, operation, parent_id=parent_op_id)
        self.scene_nodes[op_id] = op_node
        self.id_to_node[op_id] = op_node

        # Attach to parent
        parent.add_child(op_id)

        # Ensure parent's args reference the op_id
        op_parent = parent.item_data
        if hasattr(op_parent, 'args'):
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
                self.selected_item_id,
                auto_primitive_type,
                position,
                0.5,
                ui_name=f"{auto_primitive_type.title()} {i + 1}"
            )
            prim_node = SceneNode('primitive', operand_id, primitive, parent_id=op_id)
            self.scene_nodes[operand_id] = prim_node
            self.id_to_node[operand_id] = prim_node
            op_node.add_child(operand_id)
            self.next_id += 1

        self.next_id += 1

        # Register undo/redo
        redo_kwargs = {'forced_op_id': op_id}
        self.glob_history.add(
            self.delete_node,
            self.add_child_operation,
            (op_id,),
            (parent_op_id, operation_type, ui_name, auto_primitive_type),
            {},
            redo_kwargs
        )

        self.invalidate_cache()
        return op_id


    def add_standalone_primitive(
        self,
        primitive_type: str,
        position: List[float],
        size_or_radius,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        ui_name: Optional[str] = None,
        color: Optional[List[float]] = None,
        forced_op_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Add a primitive at the ROOT LEVEL (not as a child of an operation).
        
        Use this for standalone primitives that aren't operands.
        
        Args:
            primitive_type: Type of primitive ('box', 'sphere', 'torus', etc.)
            position: [x, y, z] position
            size_or_radius: Size or radius parameter
            rotation: [rx, ry, rz] rotation in radians
            scale: [sx, sy, sz] scale factors
            ui_name: Display name
            color: [r, g, b] color
            forced_op_id: For undo/redo
            **kwargs: Additional primitive-specific parameters
        
        Returns:
            node_id: ID of the created primitive
        """
        op_id = forced_op_id or f"d{self.next_id}"
        self._ensure_op_id_unique(op_id)
        
        # Create primitive
        primitive = SDFPrimitive(
            self.selected_item_id,
            primitive_type,
            position,
            size_or_radius,
            rotation,
            scale,
            ui_name or primitive_type,
            color,
            **kwargs
        )
        
        # Create node
        prim_node = SceneNode('primitive', op_id, primitive, parent_id=None)
        self.scene_nodes[op_id] = prim_node
        self.id_to_node[op_id] = prim_node
        
        # Add to root
        self.root_children.append(op_id)
        
        if not forced_op_id:
            self.next_id += 1
        
        # Register undo/redo
        redo_kwargs = copy.deepcopy(kwargs) if kwargs else {}
        redo_kwargs['forced_op_id'] = op_id
        
        self.glob_history.add(
            self.delete_node,
            self.add_standalone_primitive,
            (op_id,),
            (primitive_type, copy.deepcopy(position), copy.deepcopy(size_or_radius),
             copy.deepcopy(rotation), copy.deepcopy(scale), ui_name, copy.deepcopy(color)),
            {},
            redo_kwargs
        )
        
        self.invalidate_cache()
        return op_id
    
    # =====================================================================
    # TREE NAVIGATION AND QUERIES
    # =====================================================================
    
    def get_item_by_id(self, node_id: str):
        """
        Return the underlying item (SDFPrimitive or SDFOperation) for node_id,
        or None if node isn't found.
        """
        node = self.get_node(node_id)
        return node.item_data if node else None

    def get_primitive_by_index(self, idx: int):
        """
        Compatibility helper: return the primitive object at flat index idx
        from the compatibility .primitives property, or raise IndexError.
        """
        node_id, prim = self.primitives[idx]
        return prim

    def get_node_by_id(self, node_id: str) -> Optional[SceneNode]:
        """Return the SceneNode for node_id (thin wrapper)."""
        return self.get_node(node_id)

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        """Get a node by ID, or None if not found."""
        return self.scene_nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[str]:
        """Get list of child IDs for a node."""
        node = self.get_node(node_id)
        return node.children if node else []
    
    def get_parent(self, node_id: str) -> Optional[SceneNode]:
        """Get parent node of a node, or None if root."""
        node = self.get_node(node_id)
        if node and node.parent_id:
            return self.get_node(node.parent_id)
        return None
    
    def get_root_nodes(self) -> List[Tuple[str, SceneNode]]:
        """Get all root-level nodes as list of (node_id, node) tuples."""
        return [(nid, self.scene_nodes[nid]) for nid in self.root_children]
    
    def get_all_nodes_flat(self) -> List[Tuple[str, Any]]:
        """
        Get all nodes in flat list (for backwards compatibility).
        Traverses tree in depth-first order.
        
        Returns:
            List of (node_id, item_data) tuples
        """
        result = []
        
        def traverse(node_id):
            node = self.scene_nodes.get(node_id)
            if node:
                result.append((node_id, node.item_data))
                for child_id in node.children:
                    traverse(child_id)
        
        for root_id in self.root_children:
            traverse(root_id)
        
        return result
    
    def get_node_depth(self, node_id: str) -> int:
        """Get depth of node (0 = root, 1 = child of root, etc.)"""
        depth = 0
        node = self.get_node(node_id)
        while node and node.parent_id:
            depth += 1
            node = self.get_node(node.parent_id)
        return depth
    
    def get_all_children_recursive(self, node_id: str) -> List[str]:
        """Get all descendant IDs (children, grandchildren, etc.)."""
        result = []
        node = self.get_node(node_id)
        if not node:
            return result
        
        for child_id in node.children:
            result.append(child_id)
            result.extend(self.get_all_children_recursive(child_id))
        
        return result
    
    # =====================================================================
    # NODE MODIFICATION
    # =====================================================================
    
    def rename_node(self, node_id: str, new_name: str) -> bool:
        """
        Rename a node (update ui_name).
        
        Args:
            node_id: ID of node to rename
            new_name: New display name
        
        Returns:
            True if successful
        """
        node = self.get_node(node_id)
        if not node:
            return False
        
        old_name = node.item_data.ui_name
        node.item_data.ui_name = new_name
        
        # Register undo/redo
        self.glob_history.add(
            lambda nid, old: self.rename_node(nid, old),
            lambda nid, new: self.rename_node(nid, new),
            (node_id, old_name),
            (node_id, new_name),
            {},
            {}
        )
        
        self.invalidate_cache()
        return True
    
    def move_root_node(self, node_id: str, new_index: int) -> bool:
        """
        Move a root-level node to a different position in root list.
        
        Args:
            node_id: ID of root node to move
            new_index: New position in root_children list
        
        Returns:
            True if successful
        """
        if node_id not in self.root_children:
            return False
        
        old_index = self.root_children.index(node_id)
        
        # Clamp new_index
        new_index = max(0, min(new_index, len(self.root_children) - 1))
        
        if new_index == old_index:
            return True
        
        # Move
        self.root_children.pop(old_index)
        self.root_children.insert(new_index, node_id)
        
        # Register undo/redo
        self.glob_history.add(
            self.move_root_node,
            self.move_root_node,
            (node_id, old_index),
            (node_id, new_index),
            {},
            {}
        )
        
        self.invalidate_cache()
        return True
    

    def _apply_primitive_state(self, node_id: str, state: dict) -> bool:
        """
        Internal helper to apply a saved primitive state to a node WITHOUT registering undo.
        Used as both the undo and redo callback so reapplying doesn't add nested history.
        """
        node = self.get_node(node_id)
        if not node or node.node_type != 'primitive':
            return False
        p = node.item_data
        p.primitive_type = state.get('primitive_type', p.primitive_type)
        p.position = list(state.get('position', p.position))
        p.size_or_radius = list(state.get('size_or_radius', p.size_or_radius))
        p.rotation = list(state.get('rotation', p.rotation))
        p.scale = list(state.get('scale', p.scale))
        p.color = list(state.get('color', p.color))
        p.kwargs = dict(state.get('kwargs', p.kwargs))
        return True

    def change_primitive_type(self, node_id: str, new_type: str, new_size_or_radius=None, **kwargs) -> bool:
        """
        Change a primitive's type in-place (keeps the node_id stable).
        Registers an undo/redo entry that restores the previous state and re-applies the new state.
        """
        node = self.get_node(node_id)
        if not node or node.node_type != 'primitive':
            return False

        prim = node.item_data
        # Save old state snapshot
        old_state = prim.to_dict()

        # Compute new state (apply changes to a copy)
        new_state = old_state.copy()
        new_state['primitive_type'] = new_type

        # Normalize size_or_radius into list/appropriate form
        if new_size_or_radius is not None:
            new_state['size_or_radius'] = new_size_or_radius
        else:
            # sensible defaults for common types
            default_map = {
                'box': [0.5, 0.5, 0.5],
                'round_box': [0.5, 0.5, 0.5],
                'sphere': [0.5],
                'torus': [0.5, 0.25],
                'cone': [0.5],
                'plane': [1.0],
                'hex_prism': [0.5, 0.5],
                'vertical_capsule': [1.0, 0.3],
                'capped_cylinder': [0.3, 1.0],
                'rounded_cylinder': [0.3, 0.3],
            }
            new_state['size_or_radius'] = default_map.get(new_type, old_state.get('size_or_radius', []))

        # Update kwargs if provided
        new_kwargs = dict(old_state.get('kwargs', {}))
        if kwargs:
            new_kwargs.update(kwargs)
        new_state['kwargs'] = new_kwargs

        # Immediately apply the new state (live)
        applied = self._apply_primitive_state(node_id, new_state)
        if not applied:
            return False

        # Register undo/redo using the internal apply helper (so undo/redo won't add additional history)
        self.glob_history.add(
            self._apply_primitive_state,   # undo: apply old state
            self._apply_primitive_state,   # redo: apply new state
            (node_id, old_state),
            (node_id, new_state),
            {},
            {}
        )

        self.invalidate_cache()
        return True



    def _alloc_id(self) -> str:
        """Allocate a new unique ID like d0, d1, ... and increment next_id."""
        op_id = f"d{self.next_id}"
        self.next_id += 1
        while op_id in self.scene_nodes:
            op_id = f"d{self.next_id}"
            self.next_id += 1
        return op_id

    def _ensure_op_id_unique(self, op_id: str) -> str:
        """
        Ensure a requested op_id is unique. If it's already present, return a fresh id.
        Do NOT delete existing nodes here.
        """
        if op_id in self.scene_nodes:
            return self._alloc_id()
        return op_id

    def _delete_subtree_no_history(self, node_id: str):
        """Delete a node and all descendants without recording history (internal helper)."""
        if node_id not in self.scene_nodes:
            return
        all_to_delete = [node_id] + self.get_all_children_recursive(node_id)
        for cid in all_to_delete:
            if cid in self.scene_nodes:
                # remove reference from parent if present
                parent = self.get_parent(cid)
                if parent and cid in parent.children:
                    parent.remove_child(cid)
                if cid in self.root_children:
                    try:
                        self.root_children.remove(cid)
                    except ValueError:
                        pass
                # delete maps
                if cid in self.scene_nodes:
                    del self.scene_nodes[cid]
                if cid in self.id_to_node:
                    del self.id_to_node[cid]

    def change_node_to_operation(self, node_id: str, operation_type: str, auto_primitive_type: str = 'box') -> bool:
        """
        Convert a primitive node into an operation node IN-PLACE (keeps node_id).
        The original primitive becomes the first operand (moved into a newly-created child).
        Additional operand primitives are created automatically.
        Registers undo/redo by saving the subtree before/after conversion.
        """
        node = self.get_node(node_id)
        if not node:
            return False

        # Save state for undo
        old_state = self._save_node_tree_state(node_id)

        # If node is already an operation, simply update its type & return
        if node.node_type == 'operation':
            node.item_data.operation_type = operation_type
            self.invalidate_cache()
            return True

        # node is primitive: preserve its SDFPrimitive as first child
        if node.node_type != 'primitive':
            return False

        old_prim = node.item_data

        # Create operand ids (keep count according to operation type)
        operand_count = self._get_operand_count(operation_type)
        operand_ids = [self._alloc_id() for _ in range(operand_count)]

        # Create new child nodes: move old primitive into operand_ids[0]
        child_nodes = []

        for i, oid in enumerate(operand_ids):
            if i == 0:
                # reuse the existing primitive object as child
                prim = old_prim
            else:
                # create default primitives for remaining operands
                prim = SDFPrimitive(self.selected_item_id, auto_primitive_type, [1.0 * i, 0.0, 0.0], 0.5, ui_name=f"{auto_primitive_type.title()} {i+1}")
            prim_node = SceneNode('primitive', oid, prim, parent_id=node_id)
            self.scene_nodes[oid] = prim_node
            self.id_to_node[oid] = prim_node
            child_nodes.append(oid)

        # Replace the current node to be an operation
        operation = SDFOperation(operation_type, *operand_ids, ui_name=operation_type)
        node.node_type = 'operation'
        node.item_data = operation
        node.children = child_nodes
        # ensure old primitive is no longer referenced as a root child
        if node_id in self.root_children:
            # node remains root; children are nested under it
            pass

        new_state = self._save_node_tree_state(node_id)

        # Register undo/redo: on undo restore old_state, on redo restore new_state
        def _do_restore(state):
            nid = state['node_id']
            # remove existing subtree
            self._delete_subtree_no_history(nid)
            self._restore_node_tree(state)

        self.glob_history.add(
            _do_restore,  # undo: restore old_state
            _do_restore,  # redo: restore new_state (we will pass new_state as redo args)
            (old_state,),
            (new_state,),
            {},
            {}
        )

        self.invalidate_cache()
        return True

    def change_node_to_primitive(self, node_id: str, primitive_type: str = 'box', position=None, size_or_radius=None, rotation=None, scale=None, color=None, **kwargs) -> bool:
        """
        Convert an operation node (and its children) into a single primitive node IN-PLACE.
        The operation's subtree is deleted; the node becomes a primitive with supplied parameters.
        Undo/redo is registered by saving the old subtree and the new subtree.
        """
        node = self.get_node(node_id)
        if not node:
            return False

        old_state = self._save_node_tree_state(node_id)

        # If already primitive, update its type/params
        if node.node_type == 'primitive':
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

            new_state = self._save_node_tree_state(node_id)

            def _do_restore(state):
                nid = state['node_id']
                self._delete_subtree_no_history(nid)
                self._restore_node_tree(state)

            self.glob_history.add(_do_restore, _do_restore, (old_state,), (new_state,), {}, {})
            self.invalidate_cache()
            return True

        # node is operation: delete its children and replace with a primitive
        # Remove children nodes
        child_ids = list(node.children)
        for cid in child_ids:
            self._delete_subtree_no_history(cid)

        # Replace with new primitive
        pos = position or [0.0, 0.0, 0.0]
        s_or_r = size_or_radius if size_or_radius is not None else (0.5 if primitive_type != 'box' else [0.5,0.5,0.5])
        prim = SDFPrimitive(self.selected_item_id, primitive_type, pos, s_or_r, rotation or [0.0,0.0,0.0], scale or [1.0,1.0,1.0], ui_name=primitive_type, color=color or [0.8,0.6,0.4], **(kwargs or {}))
        node.node_type = 'primitive'
        node.item_data = prim
        node.children = []

        new_state = self._save_node_tree_state(node_id)

        def _do_restore(state):
            nid = state['node_id']
            self._delete_subtree_no_history(nid)
            self._restore_node_tree(state)

        self.glob_history.add(_do_restore, _do_restore, (old_state,), (new_state,), {}, {})

        self.invalidate_cache()
        return True


    def update_selected_item_id(self, new_value):
        """Update the selected item reference."""
        self.selected_item_id = new_value
    
    # =====================================================================
    # DELETION (CASCADE DELETE CHILDREN)
    # =====================================================================
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node and all its descendants.
        
        Unlike the old system, deleting an operation automatically deletes
        all its primitive children. This enforces the tree constraint.
        
        Args:
            node_id: ID of node to delete
        
        Returns:
            True if successful
        """
        if node_id not in self.scene_nodes:
            return False
        
        node = self.scene_nodes[node_id]
        
        # Save entire subtree for undo
        deleted_state = self._save_node_tree_state(node_id)
        
        # Delete all descendants
        all_to_delete = [node_id] + self.get_all_children_recursive(node_id)
        for child_id in all_to_delete:
            self._delete_node_no_history(child_id)
        
        # Remove from parent's children
        if node.parent_id:
            parent = self.get_node(node.parent_id)
            if parent:
                parent.remove_child(node_id)
        
        # Remove from root if necessary
        if node_id in self.root_children:
            self.root_children.remove(node_id)
        
        # Register undo/redo
        self.glob_history.add(
            self._restore_node_tree,
            self.delete_node,
            (deleted_state,),
            (node_id,),
            {},
            {}
        )
        
        self.invalidate_cache()
        return True
    
    def _delete_node_no_history(self, node_id: str):
        """Internal delete without undo (used by cascade delete)."""
        if node_id in self.scene_nodes:
            del self.scene_nodes[node_id]
        if node_id in self.id_to_node:
            del self.id_to_node[node_id]
    
    # =====================================================================
    # SERIALIZATION (for undo/redo and file save/load)
    # =====================================================================
    
    def _save_node_tree_state(self, node_id: str) -> Optional[Dict]:
        """
        Save a node and all its descendants to a dictionary for undo.
        
        Returns:
            Dictionary representing the entire subtree, or None
        """
        node = self.get_node(node_id)
        if not node:
            return None
        
        state = {
            'node_id': node_id,
            'node_type': node.node_type,
            'item_data': node.item_data.to_dict(),
            'parent_id': node.parent_id,
            'children': []
        }
        
        # Recursively save children
        for child_id in node.children:
            child_state = self._save_node_tree_state(child_id)
            if child_state:
                state['children'].append(child_state)
        
        return state
    
    def _restore_node_tree(self, node_tree_state: Dict):
        """
        Restore a saved node tree (used by undo).
        
        Args:
            node_tree_state: Dictionary from _save_node_tree_state
        """
        if not node_tree_state:
            return
        
        def restore_recursive(state, parent_id=None):
            """Recursively restore nodes."""
            node_id = state['node_id']
            node_type = state['node_type']
            item_data_dict = state['item_data']
            
            # Reconstruct the data object
            if node_type == 'primitive':
                item_data = SDFPrimitive(
                    self.selected_item_id,
                    primitive_type=item_data_dict['primitive_type'],
                    position=item_data_dict['position'],
                    size_or_radius=item_data_dict['size_or_radius'],
                    rotation=item_data_dict.get('rotation'),
                    scale=item_data_dict.get('scale'),
                    ui_name=item_data_dict.get('ui_name'),
                    color=item_data_dict.get('color'),
                    **item_data_dict.get('kwargs', {})
                )
            else:  # operation
                item_data = SDFOperation(
                    item_data_dict['operation_type'],
                    *item_data_dict['args'],
                    ui_name=item_data_dict.get('ui_name')
                )
                # Restore smooth_k if present
                if item_data_dict.get('smooth_k') is not None:
                    item_data.smooth_k = item_data_dict['smooth_k']
            
            # Create node
            node = SceneNode(node_type, node_id, item_data, parent_id=parent_id)
            self.scene_nodes[node_id] = node
            self.id_to_node[node_id] = node
            
            # Recursively restore children
            for child_state in state.get('children', []):
                child_id = restore_recursive(child_state, parent_id=node_id)
                node.children.append(child_id)
            
            return node_id
        
        # Restore the tree
        root_id = restore_recursive(node_tree_state)
        
        # If this is a root node being restored, add it back to root
        if root_id and root_id not in self.root_children:
            self.root_children.append(root_id)
    
    def to_dict(self) -> Dict:
        """
        Serialize entire scene to dictionary for JSON save.
        
        Returns:
            Dictionary with 'next_id', 'root_children', 'nodes'
        """
        scene_dict = {
            'next_id': self.next_id,
            'root_children': self.root_children,
            'nodes': {}
        }
        
        # Serialize all nodes
        for node_id, node in self.scene_nodes.items():
            scene_dict['nodes'][node_id] = node.to_dict()
        
        return scene_dict
    
    def from_dict(self, scene_dict: Dict):
        """
        Load scene from dictionary (inverse of to_dict).
        
        Args:
            scene_dict: Dictionary from to_dict() or JSON
        """
        # Clear current scene
        self.scene_nodes.clear()
        self.id_to_node.clear()
        self.root_children.clear()
        
        # Restore basic properties
        self.next_id = scene_dict.get('next_id', 0)
        self.root_children = list(scene_dict.get('root_children', []))
        
        # Reconstruct all nodes
        nodes_dict = scene_dict.get('nodes', {})
        for node_id, node_data in nodes_dict.items():
            self._reconstruct_node(node_id, node_data)
        
        self.invalidate_cache()
    
    def _reconstruct_node(self, node_id: str, node_data: Dict):
        """
        Reconstruct a single node from serialized data.
        
        Args:
            node_id: ID of node
            node_data: Serialized node data
        """
        node_type = node_data['node_type']
        item_data_dict = node_data['item_data']
        parent_id = node_data.get('parent_id')
        children_ids = node_data.get('children', [])
        
        # Reconstruct the appropriate object type
        if node_type == 'primitive':
            item_data = SDFPrimitive(
                self.selected_item_id,
                primitive_type=item_data_dict['primitive_type'],
                position=item_data_dict['position'],
                size_or_radius=item_data_dict['size_or_radius'],
                rotation=item_data_dict.get('rotation'),
                scale=item_data_dict.get('scale'),
                ui_name=item_data_dict.get('ui_name'),
                color=item_data_dict.get('color'),
                **item_data_dict.get('kwargs', {})
            )
        else:  # operation
            item_data = SDFOperation(
                item_data_dict['operation_type'],
                *item_data_dict['args'],
                ui_name=item_data_dict.get('ui_name')
            )
            # Restore smooth_k if present
            if item_data_dict.get('smooth_k') is not None:
                item_data.smooth_k = item_data_dict['smooth_k']
        
        # Create the node
        node = SceneNode(node_type, node_id, item_data, parent_id=parent_id)
        node.children = children_ids
        
        # Store in maps
        self.scene_nodes[node_id] = node
        self.id_to_node[node_id] = node
    
    # =====================================================================
    # SHADER CODE GENERATION
    # =====================================================================
    
    def generate_raymarch_code(self) -> str:
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
            
            node = self.get_node(node_id)
            if not node:
                return None
            
            # Emit all children first (they are operands)
            child_ids = []
            for child_id in node.children:
                child_id_out = emit_node_code(child_id)
                if child_id_out:
                    child_ids.append(child_id_out)
            
            # Now emit this node
            if node.node_type == 'primitive':
                primitive = node.item_data
                primitive.update_selected_item_id(self.selected_item_id)
                
                # Emit transform code
                transform_code = primitive.generate_transform_code(node_id)
                if transform_code:
                    scene_lines.append(transform_code)
                
                # Emit SDF code
                sdf_code = primitive.generate_sdf_code(node_id)
                if sdf_code:
                    scene_lines.append(sdf_code)
                
                last_emitted_id = node_id
            
            elif node.node_type == 'operation':
                operation = node.item_data

                # Use child IDs as operands (they're already emitted)
                # Determine minimum operand count (1 or 2) and skip if not enough children
                required_ops = self._get_operand_count(operation.operation_type)
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
                if operation.operation_type in {'sunion', 'ssub', 'sinter', 'mix'}:
                    # These templates expect (d_a, d_b, k)
                    if len(op_args) < 3:
                        default_k = getattr(operation, 'smooth_k', 0.1)
                        op_args.append(default_k)
                elif operation.operation_type in {'round', 'onion', 'snoiseDisp'}:
                    # These templates expect (d_a, param)
                    if len(op_args) < 2:
                        default_param = getattr(operation, 'param', 0.1)
                        op_args.append(default_param)

                # Reconstruct operation with resolved arguments (so generate_code sees correct ids/params)
                op_copy = SDFOperation(
                    operation.operation_type,
                    *op_args,
                    ui_name=operation.ui_name
                )
                # Preserve smooth_k if present on the original operation object
                if hasattr(operation, 'smooth_k'):
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
        for root_id in self.root_children:
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
    
    # =====================================================================
    # CACHING AND PERFORMANCE
    # =====================================================================
    
    def invalidate_cache(self):
        """Invalidate shader code cache when tree changes."""
        self._cache_valid = False
        self._shader_cache = None
    
    def get_raymarch_code_cached(self) -> str:
        """
        Get shader code from cache if valid, otherwise regenerate.
        
        Returns:
            GLSL code string
        """
        if self._cache_valid and self._shader_cache:
            return self._shader_cache
        
        self._shader_cache = self.generate_raymarch_code()
        self._cache_valid = True
        return self._shader_cache
    
    # =====================================================================
    # HELPER FUNCTIONS
    # =====================================================================
    
    def _get_operand_count(self, operation_type: str) -> int:
        """Get number of operands required for an operation type."""
        single_operand_ops = {'invert', 'round', 'onion', 'snoiseDisp'}
        return 1 if operation_type in single_operand_ops else 2
    
    def _ensure_op_id_unique(self, op_id: str):
        """Remove any duplicate op_id before adding new one."""
        if op_id in self.scene_nodes:
            self._delete_node_no_history(op_id)


    # TODO: Refactor
    # =====================================================================
    # SAVE/LOAD FUNCTIONS 
    # =====================================================================

    def _recompute_next_id(self):
        """
        Ensure next_id is greater than any existing numeric part of node IDs.
        Node IDs are expected in the form 'd<number>'.
        """
        max_n = -1
        for nid in self.scene_nodes.keys():
            if isinstance(nid, str) and nid.startswith('d'):
                try:
                    n = int(nid[1:])
                    if n > max_n:
                        max_n = n
                except Exception:
                    continue
        # set to next free index
        self.next_id = max_n + 1 if max_n >= 0 else 0

    def to_json(self) -> str:
        """
        Return the scene serialized as a pretty JSON string.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def from_json(self, json_str: str) -> bool:
        """
        Load a scene from a JSON string. Returns True on success, False on failure.
        """
        try:
            scene_dict = json.loads(json_str)
            self.from_dict(scene_dict)
            # Recompute next_id to avoid collisions
            self._recompute_next_id()
            self.invalidate_cache()
            return True
        except Exception as e:
            print(f"SceneBuilder.from_json: failed to parse/load JSON: {e}")
            return False

    def save_to_file(self, filepath: str) -> tuple[bool, str]:
        import os
        """
        Save the current scene to a file in JSON format.

        Returns:
            (success: bool, message: str)
        """
        try:
            # Ensure parent folder exists
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)

            return True, f"Saved scene to {filepath}"
        except Exception as e:
            return False, f"Failed to save scene to {filepath}: {e}"

    def load_from_file(self, filepath: str) -> tuple[bool, str]:
        import os
        """
        Load scene from a JSON file produced by save_to_file().

        Returns:
            (success: bool, message: str)
        """
        if not os.path.exists(filepath):
            return False, f"File does not exist: {filepath}"
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                scene_dict = json.load(f)
            self.from_dict(scene_dict)
            # Recompute next_id to avoid collisions
            self._recompute_next_id()
            self.invalidate_cache()
            return True, f"Loaded scene from {filepath}"
        except Exception as e:
            return False, f"Failed to load scene from {filepath}: {e}"