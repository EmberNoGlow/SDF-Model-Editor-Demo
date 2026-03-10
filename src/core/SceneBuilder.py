"""
HIERARCHICAL SCENE BUILDER

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

from src.core.SDFObjects import SDFOperation, SDFPrimitive
from src.core.classes.scene_tree.SceneNode import SceneNode

from .classes.scene_tree.NodeFinder import *
from .classes.scene_tree.SceneTraversal import *
from .classes.save_load_helpers.SceneSerializer import *
from .classes.node_tree.NodeSerialization import *
from .classes.node_tree.NodeOperations import *
from .classes.node_tree.NodeMod import *

from .ShaderBuilder import *


class SDFSceneBuilder:
    """
    Hierarchical scene builder that organizes the scene as a tree
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

    def update_selected_item_id(self, new_value):
        """Update the selected item reference."""
        self.selected_item_id = new_value
    
    def update_glob_history(self, new_value):
        """Update the global history reference."""
        self.glob_history = new_value
    

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

    
    # =====================================================================
    # OPERATIONS WITH NODES
    # =====================================================================


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
        return add_operation_with_auto_primitives(self, operation_type, ui_name, auto_primitive_type, forced_op_id)

    def get_item_name(self, node_id: str) -> str:
        """Get the display name of a node (for compatibility)."""
        return get_item_name(self, node_id)

    def modify_primitive_property(self, node_id: str, property_name: str, new_value):
        """Compatibility method for modifying primitive properties."""
        return modify_primitive_property(self, node_id, property_name, new_value)

    def delete_item(self, node_id: str) -> bool:
        """Compatibility method for old delete_item calls."""
        return delete_item(self, node_id)

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
        return add_child_primitive(self, parent_op_id, primitive_type, position, size_or_radius, rotation, scale, ui_name, color, forced_op_id, **kwargs)

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
        return add_child_operation(self, parent_op_id, operation_type, ui_name, auto_primitive_type, forced_op_id)

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
        return add_standalone_primitive(self, primitive_type, position, size_or_radius, rotation, scale, ui_name, color, forced_op_id, **kwargs)


    
    # =====================================================================
    # TREE NAVIGATION AND QUERIES
    # =====================================================================

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        return get_node(self, node_id)

    def get_item_by_id(self, node_id: str):
        return get_item_by_id(self, node_id)

    def get_primitive_by_index(self, idx: int):
        return get_primitive_by_index(self, idx)

    def get_node_by_id(self, node_id: str) -> Optional[SceneNode]:
        return get_node_by_id(self, node_id)

    def get_children(self, node_id: str) -> List[str]:
        return get_children(self, node_id)

    def get_parent(self, node_id: str) -> Optional[SceneNode]:
        return get_parent(self, node_id)

    def get_root_nodes(self) -> List[Tuple[str, SceneNode]]:
        return get_root_nodes(self)

    def get_all_nodes_flat(self) -> List[Tuple[str, Any]]:
        return get_all_nodes_flat(self)

    def get_node_depth(self, node_id: str) -> int:
        return get_node_depth(self, node_id)


    def get_all_children_recursive(self, node_id: str) -> List[str]:
        return get_all_children_recursive(self, node_id)

    # =====================================================================
    # NODE MODIFICATION
    # =====================================================================
    
    def rename_node(builder, node_id: str, new_name: str) -> bool:
        """
        Rename a node (update ui_name).
        
        Args:
            node_id: ID of node to rename
            new_name: New display name
        
        Returns:
            True if successful
        """
        return rename_node(builder, node_id, new_name)


    def move_root_node(builder, node_id: str, new_index: int) -> bool:
        """
        Move a root-level node to a different position in root list.
        
        Args:
            node_id: ID of root node to move
            new_index: New position in root_children list
        
        Returns:
            True if successful
        """
        return move_root_node(builder, node_id, new_index)
    

    def move_root_node(builder, node_id: str, new_index: int) -> bool:
        """
        Move a root-level node to a different position in root list.
        
        Args:
            node_id: ID of root node to move
            new_index: New position in root_children list
        
        Returns:
            True if successful
        """
        return move_root_node(builder, node_id, new_index)


    def _apply_primitive_state(builder, node_id: str, state: dict) -> bool:
        """
        Internal helper to apply a saved primitive state to a node WITHOUT registering undo.
        Used as both the undo and redo callback so reapplying doesn't add nested history.
        """
        return apply_primitive_state(builder, node_id, state)


    def change_primitive_type(builder, node_id: str, new_type: str, new_size_or_radius=None, **kwargs) -> bool:
        """
        Change a primitive's type in-place (keeps the node_id stable).
        Registers an undo/redo entry that restores the previous state and re-applies the new state.
        """
        return change_primitive_type(builder, node_id, new_type, new_size_or_radius, **kwargs)


    def change_primitive_type(builder, node_id: str, new_type: str, new_size_or_radius=None, **kwargs) -> bool:
        """
        Change a primitive's type in-place (keeps the node_id stable).
        Registers an undo/redo entry that restores the previous state and re-applies the new state.
        """
        return change_primitive_type(builder, node_id, new_type, new_size_or_radius, **kwargs)


    def _alloc_id(builder) -> str:
        """Allocate a new unique ID like d0, d1, ... and increment next_id."""
        return alloc_id(builder)


    def _ensure_op_id_unique(builder, op_id: str) -> str:
        """
        Ensure a requested op_id is unique. If it's already present, return a fresh id.
        Do NOT delete existing nodes here.
        """
        return ensure_op_id_unique(builder, op_id)


    def _delete_subtree_no_history(builder, node_id: str):
        """Delete a node and all descendants without recording history (internal helper)."""
        return delete_subtree_no_history(builder, node_id)

    def change_node_to_operation(builder, node_id: str, operation_type: str, auto_primitive_type: str = 'box') -> bool:
        """
        Convert a primitive node into an operation node IN-PLACE (keeps node_id).
        The original primitive becomes the first operand (moved into a newly-created child).
        Additional operand primitives are created automatically.
        Registers undo/redo by saving the subtree before/after conversion.
        """
        return change_node_to_operation(builder, node_id, operation_type, auto_primitive_type)

    def change_node_to_primitive(builder, node_id: str, primitive_type: str = 'box', position=None, size_or_radius=None, rotation=None, scale=None, color=None, **kwargs) -> bool:
        """
        Convert an operation node (and its children) into a single primitive node IN-PLACE.
        The operation's subtree is deleted; the node becomes a primitive with supplied parameters.
        Undo/redo is registered by saving the old subtree and the new subtree.
        """
        return change_node_to_primitive(builder, node_id, primitive_type, position, size_or_radius, rotation, scale, color, **kwargs)

    def reparent_node(self, node_id: str, new_parent_id: str, child_to_replace_id: Optional[str] = None) -> bool:
        """
        Reparent a node to a new parent operation.
        
        Returns the new operation id or None on failure.
        """
        from .classes.node_tree.NodeMod import reparent_node
        return reparent_node(self, node_id, new_parent_id, child_to_replace_id)

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
    # SERIALIZATION (for undo/redo)
    # =====================================================================
    
    def _save_node_tree_state(self, node_id: str) -> Optional[Dict]:
        """
        Save a node and all its descendants to a dictionary for undo.

        Returns:
            Dictionary representing the entire subtree, or None
        """
        return save_node_tree_state(self, node_id)

    def _restore_node_tree(self, node_tree_state: Dict):
        """
        Restore a saved node tree (used by undo).

        Args:
            node_tree_state: Dictionary from _save_node_tree_state
        """
        restore_node_tree(self, node_tree_state)

    def _reconstruct_node(self, node_id: str, node_data: Dict):
        """
        Reconstruct a single node from serialized data.

        Args:
            node_id: ID of node
            node_data: Serialized node data
        """
        reconstruct_node(self, node_id, node_data)

    
    # =====================================================================
    # SHADER CODE GENERATION
    # =====================================================================
    
    def generate_raymarch_code(self):
        return generate_raymarch_code(self)
    
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


    # =====================================================================
    # SAVE/LOAD FUNCTIONS 
    # =====================================================================

    def _recompute_next_id(self):
        """Ensure next_id is greater than any existing numeric part of node IDs."""
        recompute_next_id(self)

    def to_json(self) -> str:
        """Return the scene serialized as a pretty JSON string."""
        return to_json(self)

    def from_json(self, json_str: str) -> bool:
        """Load a scene from a JSON string. Returns True on success, False on failure."""
        return from_json(self, json_str)

    def save_to_file(self, filepath: str) -> tuple[bool, str]:
        """Save the current scene to a file in JSON format."""
        return save_to_file(self, filepath)

    def load_from_file(self, filepath: str) -> tuple[bool, str]:
        """Load scene from a JSON file."""
        return load_from_file(self, filepath)