import json
import numpy as np
import copy
import math

from typing import Dict, List, Any
from src.classes import *

class SDFSceneBuilder:
    def __init__(self, glob_history, selected_item_id):
        self.primitives = []
        self.operations = []
        self.next_id = 0
        self.id_to_index = {}
        self.deleted_items_cache = {}  # Cache for restoring deleted items
        self.glob_history = glob_history
        self.selected_item_id = selected_item_id

    def update_glob_history(self, new_value):
        self.glob_history = new_value
    
    def update_selected_item_id(self, new_value):
        self.selected_item_id = new_value

    def _save_item_state(self, op_id):
        """Save the complete state of an item for undo/redo."""
        if op_id not in self.id_to_index:
            return None

        item_type, index = self.id_to_index[op_id]

        if item_type == 'primitive':
            primitive = self.primitives[index][1]
            return {
                'type': 'primitive',
                'op_id': op_id,
                'index': index,
                'data': primitive.to_dict()
            }
        else:
            operation = self.operations[index][1]
            return {
                'type': 'operation',
                'op_id': op_id,
                'index': index,
                'data': operation.to_dict()
            }

    def _get_all_dependent_items(self, op_id):
        """Get all operations that depend on this item (directly or indirectly)."""
        dependent = []

        def get_dependents(item_id):
            for op_id_check, operation in self.operations:
                # operation.args may include references to other op ids
                if item_id in operation.args and op_id_check not in dependent:
                    dependent.append(op_id_check)
                    get_dependents(op_id_check)  # Recursively get dependents of dependents

        get_dependents(op_id)
        return dependent

    def add_primitive(self, primitive_type, position, size_or_radius,
                      rotation=None, scale=None, ui_name=None, color=None,
                      forced_op_id=None, **kwargs):

        op_id = forced_op_id or f"d{self.next_id}"

        # Ensure uniqueness
        self._ensure_op_id_unique(op_id)

        primitive = SDFPrimitive(self.selected_item_id, primitive_type, position, size_or_radius, rotation, scale, ui_name, color, **kwargs)
        self.primitives.append((op_id, primitive))
        self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)

        # Always increment next_id if not forced
        if not forced_op_id:
            self.next_id += 1

        # Register undo/redo
        # redo should restore the same op_id; pass it via redo_kwargs
        redo_kwargs = copy.deepcopy(kwargs) if kwargs else {}
        redo_kwargs['forced_op_id'] = op_id

        self.glob_history.add(
            self.delete_item,
            self.add_primitive,
            (op_id,),
            (primitive_type, copy.deepcopy(position), copy.deepcopy(size_or_radius),
             copy.deepcopy(rotation), copy.deepcopy(scale), ui_name, copy.deepcopy(color)),
            {},
            redo_kwargs
        )

        return op_id

    def add_box(self, position, size, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("box", position, size, rotation, scale, ui_name, color)

    def add_roundbox(self, position, size, radius, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("round_box", position, size, rotation, scale, ui_name, color, radius=radius)

    def add_sphere(self, position, radius, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("sphere", position, radius, rotation, scale, ui_name, color)

    def add_torus(self, position, major_radius, minor_radius, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("torus", position, [major_radius, minor_radius], rotation, scale, ui_name, color)

    def add_cone(self, position, c_sin, c_cos, height, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("cone", position, [0.0], rotation, scale, ui_name, color, c_sin=c_sin, c_cos=c_cos, height=height)

    def add_plane(self, position, normal, h, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("plane", position, [0.0], rotation, scale, ui_name, color, normal=normal, h=h)

    def add_hex_prism(self, position, hex_radius, height, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("hex_prism", position, [hex_radius, height], rotation, scale, ui_name, color)

    def add_vertical_capsule(self, position, height, radius, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("vertical_capsule", position, [height, radius], rotation, scale, ui_name, color)

    def add_capped_cylinder(self, position, radius, height, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("capped_cylinder", position, [radius, height], rotation, scale, ui_name, color)

    def add_rounded_cylinder(self, position, radius_a, radius_b, height, rotation=None, scale=None, ui_name=None, color=None):
        return self.add_primitive("rounded_cylinder", position, [radius_a, radius_b], rotation, scale, ui_name, color, height=height)


    def add_group(self, member_ids: List[str], position=(0.0,0.0,0.0), rotation=None, scale=None, ui_name=None, color=None, forced_op_id=None):
        """
        Create a logical group primitive that contains member_ids (list of primitive op_ids).
        The grouped primitives are marked (kwargs['grouped'] = group_op_id) so they won't be emitted
        separately; the group primitive will emit their SDF as a union.
        """
        op_id = forced_op_id or f"d{self.next_id}"
        self._ensure_op_id_unique(op_id)

        # Create group primitive (size_or_radius left empty)
        primitive = SDFPrimitive(self.selected_item_id, "group", list(position), [0.0,0.0,0.0], rotation=rotation, scale=scale, ui_name=ui_name or "Group", color=color, members=list(member_ids))
        primitive.kwargs['members'] = list(member_ids)
        self.primitives.append((op_id, primitive))
        self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)

        # Mark members as grouped (so the generator will skip them)
        for mid in member_ids:
            if mid in self.id_to_index:
                itype, midx = self.id_to_index[mid]
                if itype == 'primitive':
                    self.primitives[midx][1].kwargs['grouped'] = op_id

        if not forced_op_id:
            self.next_id += 1

        # Undo/redo: create group -> delete_group on undo; re-add on redo
        self.glob_history.add(
            self._undo_delete_with_dependents, # easiest reuse of undo helper by saving state
            self._redo_operation_add,          # dummy redo: we'll just re-add (handled below)
            (self._save_item_state(op_id), []),
            (op_id,),
            {},
            {}
        )
        return op_id


    def add_member_to_group(self, group_op_id: str, member_op_id: str) -> bool:
        """
        Add an existing primitive (member_op_id) to a group (group_op_id).
        Marks the primitive's kwargs['grouped'] = group_op_id so the generator
        will fold it into the group's emission. Returns True on success.
        """
        if group_op_id not in self.id_to_index or member_op_id not in self.id_to_index:
            return False

        group_type, group_idx = self.id_to_index[group_op_id]
        if group_type != 'primitive':
            return False

        group_prim = self.primitives[group_idx][1]
        if group_prim.primitive_type != 'group':
            return False

        # Ensure member is a primitive and not already grouped
        mem_type, mem_idx = self.id_to_index[member_op_id]
        if mem_type != 'primitive':
            return False

        # Add member to group's list (avoid duplicates)
        members = group_prim.kwargs.get('members', [])
        if member_op_id in members:
            return False

        members.append(member_op_id)
        group_prim.kwargs['members'] = members

        # Mark primitive as grouped so it's skipped at top-level emission
        self.primitives[mem_idx][1].kwargs['grouped'] = group_op_id
        return True

    def remove_member_from_group(self, group_op_id: str, member_op_id: str) -> bool:
        """
        Remove a primitive from a group and unmark it so it will be emitted
        as a top-level primitive again.
        """
        if group_op_id not in self.id_to_index or member_op_id not in self.id_to_index:
            return False

        group_type, group_idx = self.id_to_index[group_op_id]
        if group_type != 'primitive':
            return False

        group_prim = self.primitives[group_idx][1]
        if group_prim.primitive_type != 'group':
            return False

        members = group_prim.kwargs.get('members', [])
        if member_op_id not in members:
            return False

        members.remove(member_op_id)
        group_prim.kwargs['members'] = members

        # Unmark the primitive so it will be emitted at top-level again (if present)
        if member_op_id in self.id_to_index:
            mem_type, mem_idx = self.id_to_index[member_op_id]
            if mem_type == 'primitive':
                self.primitives[mem_idx][1].kwargs.pop('grouped', None)
        return True


    def apply_group_transform(self, group_op_id, new_pos, new_rot, new_scale):
        """
        Apply a transform (position, rotation, scale) to members of group `group_op_id`.
        This mutates member primitives' position/rotation/scale so the shader sees the change.
        Records an undo/redo entry for the whole group transform operation.
        """
        if group_op_id not in self.id_to_index:
            return False

        item_type, idx = self.id_to_index[group_op_id]
        if item_type != 'primitive':
            return False

        group = self.primitives[idx][1]
        if group.primitive_type != 'group':
            return False

        old_group_pos = copy.deepcopy(group.position)
        old_group_rot = copy.deepcopy(group.rotation)
        old_group_scale = copy.deepcopy(group.scale)

        members = list(group.kwargs.get('members', []))
        # Save old member states
        old_states = []
        new_states = []

        for mid in members:
            if mid not in self.id_to_index:
                continue
            mtype, midx = self.id_to_index[mid]
            if mtype != 'primitive':
                continue
            mem = self.primitives[midx][1]
            old_states.append((mid, copy.deepcopy(mem.position), copy.deepcopy(mem.rotation), copy.deepcopy(mem.scale)))

        # Apply transform to members (position rotation scale change around group's origin)
        # Compute deltas
        old_pos = np.array(old_group_pos, dtype=float)
        new_pos_arr = np.array(new_pos, dtype=float)
        delta_pos = new_pos_arr - old_pos

        old_rot = np.array(old_group_rot, dtype=float)
        new_rot_arr = np.array(new_rot, dtype=float)
        delta_rot = new_rot_arr - old_rot  # radians

        old_s = np.array(old_group_scale, dtype=float)
        new_s = np.array(new_scale, dtype=float)
        # avoid division by zero
        scale_ratio = np.where(old_s == 0.0, 1.0, new_s / old_s)

        # Build rotation matrices for delta_rot (apply in X, Y, Z order)
        def rotation_matrix_from_euler(rx, ry, rz):
            cx, sx = math.cos(rx), math.sin(rx)
            cy, sy = math.cos(ry), math.sin(ry)
            cz, sz = math.cos(rz), math.sin(rz)
            Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
            return Rz @ Rx @ Ry

        R = rotation_matrix_from_euler(delta_rot[0], delta_rot[1], delta_rot[2])

        for mid in members:
            if mid not in self.id_to_index:
                continue
            mtype, midx = self.id_to_index[mid]
            if mtype != 'primitive':
                continue
            mem = self.primitives[midx][1]

            # relative vector from group's origin to member
            rel = np.array(mem.position, dtype=float) - old_pos
            # scale about group origin
            rel = rel * scale_ratio
            # rotate about group origin
            rel = R.dot(rel)
            # translate by delta_pos
            new_world = new_pos_arr + rel

            # Update member attributes
            old_pos_mem = copy.deepcopy(mem.position)
            old_rot_mem = copy.deepcopy(mem.rotation)
            old_scale_mem = copy.deepcopy(mem.scale)

            mem.position = [float(x) for x in new_world]
            # add delta rotation to member rotation
            mem.rotation = [float(old_rot_mem[i] + delta_rot[i]) for i in range(3)]
            # multiply member scale by ratio
            mem.scale = [float(old_scale_mem[i] * scale_ratio[i]) for i in range(3)]

            new_states.append((mid, copy.deepcopy(mem.position), copy.deepcopy(mem.rotation), copy.deepcopy(mem.scale)))

        # Update the group primitive's stored values (so inspector reflects the new state)
        group.position = [float(x) for x in new_pos_arr]
        group.rotation = [float(x) for x in new_rot_arr]
        group.scale = [float(x) for x in new_s]

        # Register history entry for undo/redo (restore all member states and group)
        self.glob_history.add(
            self._undo_group_transform,
            self._redo_group_transform,
            (group_op_id, old_group_pos, old_group_rot, old_group_scale, old_states),
            (group_op_id, copy.deepcopy(group.position), copy.deepcopy(group.rotation), copy.deepcopy(group.scale), new_states),
            {},
            {}
        )

        return True

    def _undo_group_transform(self, group_op_id, old_group_pos, old_group_rot, old_group_scale, old_states):
        # Restore group primitive and member primitives to old states
        if group_op_id in self.id_to_index:
            itype, idx = self.id_to_index[group_op_id]
            if itype == 'primitive':
                group = self.primitives[idx][1]
                group.position = old_group_pos
                group.rotation = old_group_rot
                group.scale = old_group_scale

        for (mid, pos, rot, scale) in old_states:
            if mid in self.id_to_index:
                itype, midx = self.id_to_index[mid]
                if itype == 'primitive':
                    mem = self.primitives[midx][1]
                    mem.position = pos
                    mem.rotation = rot
                    mem.scale = scale

    def _redo_group_transform(self, group_op_id, new_group_pos, new_group_rot, new_group_scale, new_states):
        # Reapply group transform (same format as apply_group_transform result)
        if group_op_id in self.id_to_index:
            itype, idx = self.id_to_index[group_op_id]
            if itype == 'primitive':
                group = self.primitives[idx][1]
                group.position = new_group_pos
                group.rotation = new_group_rot
                group.scale = new_group_scale

        for (mid, pos, rot, scale) in new_states:
            if mid in self.id_to_index:
                itype, midx = self.id_to_index[mid]
                if itype == 'primitive':
                    mem = self.primitives[midx][1]
                    mem.position = pos
                    mem.rotation = rot
                    mem.scale = scale

    def add_operation(self, operation_type, *args, ui_name=None, forced_op_id=None):
        """
        Add an operation. Accepts forced_op_id so undo/redo can recreate the same id.
        """
        op_id = forced_op_id or f"d{self.next_id}"

        # Ensure uniqueness before adding
        self._ensure_op_id_unique(op_id)

        operation = SDFOperation(operation_type, *args, ui_name=ui_name)
        self.operations.append((op_id, operation))
        self.id_to_index[op_id] = ('operation', len(self.operations) - 1)

        if not forced_op_id:
            self.next_id += 1

        # Register undo/redo for operations
        redo_kwargs = {'forced_op_id': op_id}
        self.glob_history.add(
            self._undo_operation_delete,
            self._redo_operation_add,
            (op_id, operation_type, copy.deepcopy(args), copy.deepcopy(ui_name)),
            (copy.deepcopy(operation_type), copy.deepcopy(args), copy.deepcopy(ui_name)),
            {},
            redo_kwargs
        )

        return op_id

    def sunion(self, d_a, d_b, k=0.05, ui_name=None):
        return self.add_operation("sunion", d_a, d_b, k, ui_name=ui_name)

    def ssub(self, d_a, d_b, k=0.05, ui_name=None):
        return self.add_operation("ssub", d_a, d_b, k, ui_name=ui_name)

    def sinter(self, d_a, d_b, k=0.05, ui_name=None):
        return self.add_operation("sinter", d_a, d_b, k, ui_name=ui_name)

    def mix(self, d_a, d_b, k=0.5, ui_name=None):
        return self.add_operation("mix", d_a, d_b, k, ui_name=ui_name)

    def invert(self, d_a, ui_name=None):
        return self.add_operation("invert", d_a, ui_name=ui_name)

    def sub(self, d_a, d_b, ui_name=None):
        return self.add_operation("sub", d_a, d_b, ui_name=ui_name)

    def union(self, d_a, d_b, ui_name=None):
        return self.add_operation("union", d_a, d_b, ui_name=ui_name)

    def inter(self, d_a, d_b, ui_name=None):
        return self.add_operation("inter", d_a, d_b, ui_name=ui_name)

    def xor(self, d_a, d_b, ui_name=None):
        return self.add_operation("xor", d_a, d_b, ui_name=ui_name)

    def round(self, d_a, radius, ui_name=None):
        return self.add_operation("round", d_a, radius, ui_name=ui_name)

    def onion(self, d_a, thickness, ui_name=None):
        return self.add_operation("onion", d_a, thickness, ui_name=ui_name)
    
    def snoiseDisp(self, d_a, thickness, ui_name=None):
        return self.add_operation("snoiseDisp", d_a, thickness, ui_name=ui_name)
    

    def _ensure_op_id_unique(self, op_id):
        """Remove any duplicate op_id from primitives or operations before adding new one."""
        # Remove from primitives
        self.primitives = [(pid, prim) for pid, prim in self.primitives if pid != op_id]

        # Remove from operations
        self.operations = [(oid, op) for oid, op in self.operations if oid != op_id]

        # Remove from mapping
        if op_id in self.id_to_index:
            del self.id_to_index[op_id]

        # Update all indices after removal
        for i, (pid, _) in enumerate(self.primitives):
            self.id_to_index[pid] = ('primitive', i)
        for i, (oid, _) in enumerate(self.operations):
            self.id_to_index[oid] = ('operation', i)

    def add_pointer(self, position=(0.0, 0.0, 0.0), func='pointer_identity', ui_name=None, color=None, forced_op_id=None, **kwargs):
        """
        Add a pointer primitive. `func` is the name of a GLSL function in the sdf library
        that takes (vec3 p, vec3 pos) and returns vec3 p (transformed).
        """
        # Store the chosen function name in kwargs so it will be serialized
        kwargs = dict(kwargs) if kwargs else {}
        kwargs['func'] = func
        op_id = self.add_primitive("pointer", position, [0.0, 0.0, 0.0], rotation=None, scale=None, ui_name=ui_name or "Pointer", color=color, forced_op_id=forced_op_id, **kwargs)
        return op_id



    def _undo_operation_delete(self, op_id, operation_type, args, ui_name):
        """Helper to restore a deleted operation (used by history)."""
        # Make sure this op_id will be unique (remove any current duplicates)
        self._ensure_op_id_unique(op_id)
        self.operations.append((op_id, SDFOperation(operation_type, *args, ui_name=ui_name)))
        self.id_to_index[op_id] = ('operation', len(self.operations) - 1)

    def _redo_operation_add(self, operation_type, args, ui_name, forced_op_id=None):
        """Helper to add an operation for redo (preserve op id if provided)."""
        return self.add_operation(operation_type, *args, ui_name=ui_name, forced_op_id=forced_op_id)

    def delete_item(self, op_id):
        """Delete a primitive, group or operation by its ID, with full undo support."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]

        # Save state of the item being deleted
        deleted_item_state = self._save_item_state(op_id)

        # Get all dependent items BEFORE deletion
        dependent_ops = self._get_all_dependent_items(op_id)
        dependent_states = [self._save_item_state(dep_id) for dep_id in dependent_ops]
        dependent_states = [s for s in dependent_states if s is not None]

        if item_type == 'primitive':
            primitive = self.primitives[index][1]
            # If deleting a group, unmark its members so they will be emitted again
            if primitive.primitive_type == 'group':
                members = primitive.kwargs.get('members', [])
                for mid in members:
                    if mid in self.id_to_index:
                        mtype, midx = self.id_to_index[mid]
                        if mtype == 'primitive':
                            try:
                                self.primitives[midx][1].kwargs.pop('grouped', None)
                            except Exception:
                                pass

            # Remove the primitive
            del self.primitives[index]
            # Update indices for all primitives after this one
            for i in range(index, len(self.primitives)):
                prim_op_id = self.primitives[i][0]
                self.id_to_index[prim_op_id] = ('primitive', i)
        else:
            # Remove the operation
            del self.operations[index]
            # Update indices for all operations after this one
            for i in range(index, len(self.operations)):
                op_op_id = self.operations[i][0]
                self.id_to_index[op_op_id] = ('operation', i)

        # Remove from mapping
        if op_id in self.id_to_index:
            del self.id_to_index[op_id]

        # Remove any operations that depend on this deleted item
        for dep_id in dependent_ops:
            if dep_id in self.id_to_index:
                dep_item_type, dep_index = self.id_to_index[dep_id]
                if dep_item_type == 'operation':
                    # Find exact tuple index in operations list for this dep_id
                    for i, (oid, _) in enumerate(self.operations):
                        if oid == dep_id:
                            del self.operations[i]
                            # Update indices
                            for j in range(i, len(self.operations)):
                                op_op_id = self.operations[j][0]
                                self.id_to_index[op_op_id] = ('operation', j)
                            break
                if dep_id in self.id_to_index:
                    del self.id_to_index[dep_id]

        # Register undo/redo for the deletion (restores dependents and item)
        self.glob_history.add(
            self._undo_delete_with_dependents,
            self._redo_delete_with_dependents,
            (deleted_item_state, dependent_states),
            (op_id,),
            {},
            {}
        )

        return True



    def _insert_primitive_at(self, index, op_id, primitive):
        # clamp index
        if index < 0:
            index = 0
        if index > len(self.primitives):
            index = len(self.primitives)
        self.primitives.insert(index, (op_id, primitive))
        # update id mapping for primitives
        for i, (pid, _) in enumerate(self.primitives):
            self.id_to_index[pid] = ('primitive', i)

    def _insert_operation_at(self, index, op_id, operation):
        # clamp index
        if index < 0:
            index = 0
        if index > len(self.operations):
            index = len(self.operations)
        self.operations.insert(index, (op_id, operation))
        # update id mapping for operations
        for i, (oid, _) in enumerate(self.operations):
            self.id_to_index[oid] = ('operation', i)

    def _undo_delete_with_dependents(self, deleted_item_state, dependent_states):
        """Restore a deleted item and all its dependent operations at their original indices."""
        if deleted_item_state is None:
            return

        # Restore the main item at its original index (if present)
        item = deleted_item_state
        op_id = item['op_id']
        original_index = item.get('index', None)

        # Ensure uniqueness before restoring
        self._ensure_op_id_unique(op_id)

        if item['type'] == 'primitive':
            prim_dict = item['data']
            primitive = SDFPrimitive(
                self.selected_item_id,
                primitive_type=prim_dict["primitive_type"],
                position=prim_dict["position"],
                size_or_radius=prim_dict["size_or_radius"],
                rotation=prim_dict.get("rotation", [0.0, 0.0, 0.0]),
                scale=prim_dict.get("scale", [1.0, 1.0, 1.0]),
                ui_name=prim_dict.get("ui_name"),
                color=prim_dict.get("color", [0.8, 0.6, 0.4]),
                **prim_dict.get("kwargs", {})
            )
            # insert at saved index if available
            if original_index is None:
                self.primitives.append((op_id, primitive))
                self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)
            else:
                self._insert_primitive_at(original_index, op_id, primitive)
        else:
            op_dict = item['data']
            operation = SDFOperation(
                op_dict["operation_type"],
                *op_dict["args"],
                ui_name=op_dict.get("ui_name")
            )
            if op_dict.get("smooth_k") is not None:
                operation.smooth_k = op_dict["smooth_k"]
            if original_index is None:
                self.operations.append((op_id, operation))
                self.id_to_index[op_id] = ('operation', len(self.operations) - 1)
            else:
                self._insert_operation_at(original_index, op_id, operation)

        # Restore dependent operations at their saved indices.
        # Skip invalid/null dependent states; sort by index ascending so insertion doesn't invalidate later indices.
        valid_dep_states = [s for s in (dependent_states or []) if s]
        # Filter for operation-type states only (dependents are operations)
        valid_dep_states = [s for s in valid_dep_states if s.get('type') == 'operation']
        # sort by their original index (missing index -> large number -> appended at end)
        def dep_index_key(s):
            try:
                return s.get('index', 10**9)
            except Exception:
                return 10**9
        valid_dep_states.sort(key=dep_index_key)

        for dep_state in valid_dep_states:
            dep_id = dep_state['op_id']
            # ensure uniqueness
            self._ensure_op_id_unique(dep_id)
            op_dict = dep_state['data']
            operation = SDFOperation(
                op_dict["operation_type"],
                *op_dict["args"],
                ui_name=op_dict.get("ui_name")
            )
            if op_dict.get("smooth_k") is not None:
                operation.smooth_k = op_dict["smooth_k"]
            dep_index = dep_state.get('index', None)
            if dep_index is None:
                # append at end
                self.operations.append((dep_id, operation))
                self.id_to_index[dep_id] = ('operation', len(self.operations) - 1)
            else:
                self._insert_operation_at(dep_index, dep_id, operation)

        # Recompute next_id to avoid future duplicates
        all_ids = [int(op_id[1:]) for op_id, _ in (self.primitives + self.operations) if op_id.startswith('d')]
        if all_ids:
            self.next_id = max(all_ids) + 1

    def _redo_delete_with_dependents(self, op_id):
        """Redo deletion of an item and all its dependents."""
        self.delete_item(op_id)

    # ---- Property change helpers ----
    def _set_primitive_property(self, op_id, property_name, value):
        """Set primitive property without recording history (used by undo/redo)."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]
        if item_type != 'primitive':
            return False

        primitive = self.primitives[index][1]

        if property_name == 'position':
            primitive.position = list(value)
        elif property_name == 'size_or_radius':
            primitive.size_or_radius = list(value) if isinstance(value, (list, tuple)) else [value]
        elif property_name == 'rotation':
            primitive.rotation = list(value)
        elif property_name == 'scale':
            primitive.scale = list(value)
        elif property_name == 'color':
            primitive.color = list(value)
        elif property_name.startswith('kwargs.'):
            kwarg_name = property_name[7:]
            primitive.kwargs[kwarg_name] = value
        return True

    def modify_primitive_property(self, op_id, property_name, old_value, new_value):
        """Track modifications to primitive properties for undo/redo."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]
        if item_type != 'primitive':
            return False

        # Register the modification in history
        self.glob_history.add(
            self._undo_property_change,
            self._redo_property_change,
            (op_id, property_name, copy.deepcopy(old_value)),
            (op_id, property_name, copy.deepcopy(new_value)),
            {},
            {}
        )

        # Apply the new value without creating another history entry
        return self._set_primitive_property(op_id, property_name, new_value)

    def _undo_property_change(self, op_id, property_name, old_value):
        """Restore old property value (without creating history)."""
        self._set_primitive_property(op_id, property_name, old_value)

    def _redo_property_change(self, op_id, property_name, new_value):
        """Reapply property change (without creating history)."""
        self._set_primitive_property(op_id, property_name, new_value)

    def _set_operation_parameter(self, op_id, param_name, value):
        """Set operation parameter without recording history (used by undo/redo)."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]
        if item_type != 'operation':
            return False

        operation = self.operations[index][1]

        if param_name == 'smooth_k':
            operation.smooth_k = value
            if len(operation.args) >= 3:
                operation.args[2] = value
        elif param_name == 'float_param':
            operation.float_param = value
            if len(operation.args) >= 2:
                operation.args[1] = value
        elif param_name.startswith('args['):
            # Handle args like "args[0]", "args[1]", etc.
            arg_index = int(param_name.split('[')[1].split(']')[0])
            if arg_index < len(operation.args):
                operation.args[arg_index] = value
        return True

    def modify_operation_parameter(self, op_id, param_name, old_value, new_value):
        """Track modifications to operation parameters for undo/redo."""
        if op_id not in self.id_to_index:
            return False

        item_type, index = self.id_to_index[op_id]
        if item_type != 'operation':
            return False

        self.glob_history.add(
            self._undo_op_param_change,
            self._redo_op_param_change,
            (op_id, param_name, copy.deepcopy(old_value)),
            (op_id, param_name, copy.deepcopy(new_value)),
            {},
            {}
        )

        # Apply the new value without creating another history entry
        return self._set_operation_parameter(op_id, param_name, new_value)

    def _undo_op_param_change(self, op_id, param_name, old_value):
        """Restore old operation parameter value (without creating history)."""
        self._set_operation_parameter(op_id, param_name, old_value)

    def _redo_op_param_change(self, op_id, param_name, new_value):
        """Reapply operation parameter change (without creating history)."""
        self._set_operation_parameter(op_id, param_name, new_value)



    def _move_item_no_history(self, op_id, new_index):
        """Move an existing item to new_index within its list without creating a history entry.

        For primitives the behavior is unchanged.

        For operations we allow the move but then sanitize operation arguments:
        - After the move we ensure every operation's operand references only items
        that come earlier in the combined order (primitives then operations).
        - If an operand would refer to an item that comes later (i.e. becomes invalid),
        we replace it with the nearest higher-level item (the nearest item that
        appears before the operation in the combined ordering). If none exists we
        leave the argument unchanged (usually only happens in degenerate scenes).
        """
        if op_id not in self.id_to_index:
            return False

        item_type, old_index = self.id_to_index[op_id]

        if item_type == 'primitive':
            item = self.primitives.pop(old_index)
            # clamp
            new_index = max(0, min(new_index, len(self.primitives)))
            self.primitives.insert(new_index, item)
            # update indices
            for i, (pid, _) in enumerate(self.primitives):
                self.id_to_index[pid] = ('primitive', i)
            return True

        # --- Operation move ---
        # Allow insertion positions from 0..len(self.operations)
        desired_new_index = max(0, min(new_index, len(self.operations)))

        # Remove the item from the list
        item = self.operations.pop(old_index)

        # Adjust insertion index because the list is now shorter if removing an earlier element
        insert_index = desired_new_index
        if insert_index > old_index:
            insert_index -= 1

        # Clamp final insertion index
        insert_index = max(0, min(insert_index, len(self.operations)))
        # Insert
        self.operations.insert(insert_index, item)

        # Update id mapping for operations (and keep primitives mapping as-is)
        for i, (oid, _) in enumerate(self.operations):
            self.id_to_index[oid] = ('operation', i)

        # --- Sanitize operands so every operation only references items declared earlier ---
        # Build combined order: primitives first, then operations (their current order)
        combined = []
        for pid, _ in self.primitives:
            combined.append(pid)
        for oid, _ in self.operations:
            combined.append(oid)

        combined_index = {opid: idx for idx, opid in enumerate(combined)}

        # Helper: find nearest valid prior item id (< limit_idx), returns None if none
        def find_nearest_prior(limit_idx):
            for k in range(limit_idx - 1, -1, -1):
                return combined[k]
            return None

        # Iterate through operations and fix arguments that point to items that come
        # at or after the operation itself (invalid).
        for op_idx, (cur_op_id, cur_op) in enumerate(self.operations):
            # compute combined index of this operation
            if cur_op_id not in combined_index:
                continue
            cur_combined_idx = combined_index[cur_op_id]

            new_args = []
            changed = False
            for arg in cur_op.args:
                # Only adjust string references that exist in combined_index
                if isinstance(arg, str) and arg in combined_index:
                    arg_combined_idx = combined_index[arg]
                    if arg_combined_idx >= cur_combined_idx:
                        # invalid reference: pick the nearest prior item
                        replacement = find_nearest_prior(cur_combined_idx)
                        if replacement is not None and replacement != arg:
                            new_args.append(replacement)
                            changed = True
                            continue
                        # if no valid prior found, fall through and keep original arg (degenerate case)
                new_args.append(arg)

            if changed:
                # apply sanitized args (no history recorded here)
                cur_op.args = new_args

        # After possibly changing args, done. id_to_index already updated.
        return True

    def move_item(self, op_id, new_index):
        """
        Public API to move an item within its section (primitives or operations).
        Records undo/redo so the action can be reverted.
        """
        if op_id not in self.id_to_index:
            return False

        item_type, old_index = self.id_to_index[op_id]

        # Record undo/redo entries that call the same low-level move (no-history).
        self.glob_history.add(
            self._move_item_no_history,  # undo: move back
            self._move_item_no_history,  # redo: move to new_index again
            (op_id, old_index),
            (op_id, new_index),
            {},
            {}
        )

        # Apply the move
        return self._move_item_no_history(op_id, new_index)






    # --- Remaining methods unchanged (but included for completeness) ---
    def get_all_items(self):
        """Get all items in order: primitives then operations."""
        return self.primitives + self.operations

    def get_valid_operands(self, current_op_id):
        """Get all valid operands for an operation (excluding itself and operations that reference it)."""
        all_items = self.get_all_items()
        valid_items = []

        # Find the index of current operation
        current_index = -1
        for idx, (item_id, _) in enumerate(all_items):
            if item_id == current_op_id:
                current_index = idx
                break

        # Only allow items that come before the current operation
        for idx, item in enumerate(all_items):
            if idx < current_index:
                valid_items.append(item)

        return valid_items

    def get_item_name(self, op_id):
        """Get the display name of an item."""
        if op_id not in self.id_to_index:
            return op_id

        item_type, index = self.id_to_index[op_id]
        if item_type == 'primitive':
            return self.primitives[index][1].ui_name
        else:
            return self.operations[index][1].ui_name

    def generate_raymarch_code(self):
        """
        Generate the GLSL code for the entire scene.
        Groups are expanded in-place (they union their member primitives and
        include any operations that only reference members of the group).
        """
        scene_lines = []

        # replacements: original_op_id (e.g. "d3") -> replacement_op_id (usually group id "d10")
        # This is used to remap operation operands that referenced grouped primitives/ops.
        replacements = {}

        # Keep track of operations that have been folded into groups (so we skip them later)
        folded_ops = set()

        # Helper to append raw lines safely
        def add_lines(s):
            if s:
                scene_lines.append(s)

        # iterate primitives
        for op_id, primitive in self.primitives:
            # Skip primitives that were folded into groups (they are emitted by their group)
            if primitive.kwargs.get('grouped', None) is not None:
                continue
            
            primitive.update_selected_item_id(self.selected_item_id)
            if primitive.primitive_type == 'group':
                # Expand group: for each member, emit transform + sdf code under synthetic ids,
                # then compute the group's final distance as the min() and take color of closest.
                members = primitive.kwargs.get('members', [])
                if not members:
                    add_lines(f"float {op_id} = 1000.0;\n    vec3 col{op_id} = vec3(0.0);")
                    continue

                # Build mapping from original member id -> synthetic id
                member_synth_ids = []
                synth_map = {}  # original_id -> synthetic_id
                member_index = 0

                for mid in members:
                    if mid not in self.id_to_index:
                        continue
                    mtype, midx = self.id_to_index[mid]
                    if mtype != 'primitive':
                        continue
                    mem = self.primitives[midx][1]
                    synth_id = f"{op_id}_m{member_index}"
                    member_index += 1
                    member_synth_ids.append(synth_id)
                    synth_map[mid] = synth_id

                    # Emit transform & sdf for the synthetic id
                    temp_prim = SDFPrimitive(
                        self.selected_item_id,
                        primitive_type=mem.primitive_type,
                        position=copy.deepcopy(mem.position),
                        size_or_radius=copy.deepcopy(mem.size_or_radius),
                        rotation=copy.deepcopy(mem.rotation),
                        scale=copy.deepcopy(mem.scale),
                        ui_name=mem.ui_name,
                        color=copy.deepcopy(mem.color),
                        **copy.deepcopy(mem.kwargs)
                    )
                    add_lines(temp_prim.generate_transform_code(synth_id))
                    sdf_code = temp_prim.generate_sdf_code(synth_id)
                    if sdf_code:
                        add_lines(sdf_code)

                    # Map original member id -> group's op_id so external ops referring to the member
                    # after grouping will know to use the group's result.
                    replacements[mid] = op_id

                # Collect operations that reference only members (all string args point inside members)
                ops_to_fold = []
                for orig_oid, op in self.operations:
                    # collect referenced item ids from op.args
                    referenced = [a for a in op.args if isinstance(a, str)]
                    if not referenced:
                        continue
                    # if every referenced id is a group member, we can fold this op
                    if all((r in members) for r in referenced):
                        ops_to_fold.append((orig_oid, op))

                # Emit folded operations in the group's scope using synthetic ids
                folded_synth_ids = []
                op_fold_index = 0
                for orig_oid, op in ops_to_fold:
                    # build remapped args: if arg refers to a member or a previously folded op, map it to the synthetic id
                    remapped_args = []
                    for a in op.args:
                        if isinstance(a, str):
                            if a in synth_map:
                                remapped_args.append(synth_map[a])
                            else:
                                # if the arg is another folded operation we've already handled, map it
                                if a in synth_map:
                                    remapped_args.append(synth_map[a])
                                else:
                                    remapped_args.append(a)
                        else:
                            remapped_args.append(a)

                    synth_op_id = f"{op_id}_op{op_fold_index}"
                    op_fold_index += 1
                    # Create a transient copy of operation with remapped args
                    op_copy = SDFOperation(op.operation_type, *remapped_args, ui_name=op.ui_name)
                    op_copy.smooth_k = getattr(op, "smooth_k", None)
                    op_copy.float_param = getattr(op, "float_param", None)

                    # Emit operation code using synthetic id
                    code = op_copy.generate_code(synth_op_id)
                    add_lines(code)

                    # Register mapping: original operation id -> group id, so outer ops referencing orig_oid are remapped to group
                    replacements[orig_oid] = op_id
                    # Also make this operation's synthetic id available for subsequent folded ops
                    synth_map[orig_oid] = synth_op_id
                    folded_synth_ids.append(synth_op_id)
                    folded_ops.add(orig_oid)

                # Build final list of candidate outputs for the group's "result"
                final_outputs = []
                final_outputs.extend(member_synth_ids)
                final_outputs.extend(folded_synth_ids)

                # Create min-of final outputs and pick color of closest
                if final_outputs:
                    first = final_outputs[0]
                    block = f"float d_{op_id} = {first};\n    vec3 c_{op_id} = col{first};"
                    for fname in final_outputs[1:]:
                        block += f"\n    if ({fname} < d_{op_id}) {{ d_{op_id} = {fname}; c_{op_id} = col{fname}; }}"
                    block += f"\n    float {op_id} = d_{op_id};\n    vec3 col{op_id} = c_{op_id};"
                    add_lines(block)
                else:
                    add_lines(f"float {op_id} = 1000.0;\n    vec3 col{op_id} = vec3(0.0);")

            else:
                # normal primitive
                add_lines(primitive.generate_transform_code(op_id))
                add_lines(primitive.generate_sdf_code(op_id))

        # operations - remap arguments that point to grouped primitives or folded ops
        for op_id, operation in self.operations:
            # Skip operations that were folded into a group
            if op_id in folded_ops:
                continue

            # Build remapped args for this operation (do not mutate original operation)
            remapped_args = []
            for a in operation.args:
                if isinstance(a, str) and a in replacements:
                    remapped_args.append(replacements[a])
                else:
                    remapped_args.append(a)

            # Create transient operation copy with remapped args
            op_copy = SDFOperation(operation.operation_type, *remapped_args, ui_name=operation.ui_name)
            op_copy.smooth_k = getattr(operation, "smooth_k", None)
            op_copy.float_param = getattr(operation, "float_param", None)

            code = op_copy.generate_code(op_id)
            scene_lines.append(code)

        if scene_lines:
            scene_code = "\n    ".join(scene_lines)
            # Determine last id used to construct the return
            last_id = None
            last_col_id = None
            if self.operations:
                # find last operation not folded (if last op folded then fallback)
                for op_id, _ in reversed(self.operations):
                    if op_id in folded_ops:
                        continue
                    last_id = op_id
                    break
            if last_id is None:
                # find last non-grouped primitive
                for op_id, prim in reversed(self.primitives):
                    if prim.kwargs.get('grouped', None) is None or prim.primitive_type == 'group':
                        last_id = op_id
                        break

            if last_id is None:
                return "return vec4(0.0, 0.0, 0.0, 1000.0);"

            last_col_id = f"col{last_id}"
            scene_code += f"\n    return vec4({last_col_id}, {last_id});"
            return scene_code
        else:
            return "return vec4(0.0, 0.0, 0.0, 1000.0);"


    def to_dict(self):
        """Convert the entire scene to a dictionary for JSON serialization."""
        scene_dict = {
            "primitives": [],
            "operations": [],
            "sprites": []
        }

        # Serialize primitives
        for op_id, primitive in self.primitives:
            prim_dict = primitive.to_dict()
            prim_dict["op_id"] = op_id
            scene_dict["primitives"].append(prim_dict)

        # Serialize operations
        for op_id, operation in self.operations:
            op_dict = operation.to_dict()
            op_dict["op_id"] = op_id
            scene_dict["operations"].append(op_dict)

        # Serialize sprites if the global sprites_array exists
        sprs = globals().get("sprites_array", None)
        if sprs:
            for spr in sprs:
                # Use the Sprite.to_dict so we have a single canonical representation
                scene_dict["sprites"].append(spr.to_dict())

        return scene_dict

    def from_dict(self, scene_dict):
        """Load a scene from a dictionary (inverse of to_dict)."""
        # Clear current scene
        self.primitives.clear()
        self.operations.clear()
        self.id_to_index.clear()
        self.next_id = 0

        # Rebuild sprites_array first so sprite_index references in primitives are valid
        global sprites_array
        sprites_array = []
        for s in scene_dict.get("sprites", []):
            # Preserve sampler name if present; otherwise create a stable default name
            sampler_name = s.get("SprTexture", f"sprTex{len(sprites_array)}")
            # Read optional texture path (may be None)
            texture_path = s.get("texture_path", None)

            spr = Sprite(
                planePoint=tuple(s.get("planePoint", (0.0, 0.0, 0.0))),
                planeNormal=tuple(s.get("planeNormal", (0.0, 0.0, 1.0))),
                planeWidth=float(s.get("planeWidth", 1.0)),
                planeHeight=float(s.get("planeHeight", 1.0)),
                SprTexture=sampler_name,
                uvSize=tuple(s.get("uvSize", (1.0, 1.0))),
                Alpha=float(s.get("Alpha", 1.0)),
                LOD=float(s.get("LOD", 0.0))
            )
            # restore optional texture path and tex size (GL texture won't be loaded here)
            spr.texture_path = texture_path
            tex_size = s.get("tex_size", None)
            if tex_size:
                try:
                    spr.tex_size = (int(tex_size[0]), int(tex_size[1]))
                except Exception:
                    spr.tex_size = (0, 0)

            sprites_array.append(spr)

        # Load primitives
        for prim_dict in scene_dict.get("primitives", []):
            op_id = prim_dict["op_id"]

            # Normalize kwargs and make sprite_index an int if present
            kwargs = dict(prim_dict.get("kwargs", {}))
            if "sprite_index" in kwargs:
                try:
                    # Some JSON writers may have stored this as a string; cast to int
                    kwargs["sprite_index"] = int(kwargs["sprite_index"])
                except Exception:
                    # If invalid, fallback to 0 (or None if you prefer)
                    kwargs["sprite_index"] = 0

            primitive = SDFPrimitive(
                self.selected_item_id,
                primitive_type=prim_dict["primitive_type"],
                position=prim_dict["position"],
                size_or_radius=prim_dict["size_or_radius"],
                rotation=prim_dict.get("rotation", [0.0, 0.0, 0.0]),
                scale=prim_dict.get("scale", [1.0, 1.0, 1.0]),
                ui_name=prim_dict.get("ui_name"),
                color=prim_dict.get("color", [0.8, 0.6, 0.4]),
                **kwargs
            )

            # If this primitive is a sprite, ensure the sprite_index is valid (defensive)
            if primitive.primitive_type == "sprite":
                sprite_idx = primitive.kwargs.get("sprite_index", None)
                if sprite_idx is None:
                    # attempt to guess by sampler name if available (compatibility)
                    spr_name = prim_dict.get("kwargs", {}).get("SprTexture", None)
                    if spr_name:
                        # find first matching sprite sampler
                        found_idx = None
                        for i, s in enumerate(sprites_array):
                            if s.SprTexture == spr_name:
                                found_idx = i
                                break
                        primitive.kwargs["sprite_index"] = found_idx if found_idx is not None else 0
                    else:
                        primitive.kwargs["sprite_index"] = 0
                else:
                    # clamp to valid range
                    if not isinstance(sprite_idx, int) or sprite_idx < 0 or sprite_idx >= len(sprites_array):
                        # out-of-range or invalid, clamp and warn
                        primitive.kwargs["sprite_index"] = max(0, min(len(sprites_array) - 1, int(sprite_idx) if isinstance(sprite_idx, int) else 0))
                        print(f"Warning: sprite_index for primitive {op_id} was invalid or out-of-range; clamped to {primitive.kwargs['sprite_index']}")

            self.primitives.append((op_id, primitive))
            self.id_to_index[op_id] = ('primitive', len(self.primitives) - 1)

            # Update next_id
            try:
                prim_num = int(op_id[1:])  # Extract number from "d0", "d1", etc.
                self.next_id = max(self.next_id, prim_num + 1)
            except Exception:
                pass

        # Load operations (unchanged)
        for op_dict in scene_dict.get("operations", []):
            op_id = op_dict["op_id"]

            operation = SDFOperation(
                op_dict["operation_type"],     # Pass as positional first
                *op_dict["args"],              # Then unpack the rest of the positional args
                ui_name=op_dict.get("ui_name") # Finally, keyword arguments
            )

            # Restore smooth_k if it was set
            if op_dict.get("smooth_k") is not None:
                operation.smooth_k = op_dict["smooth_k"]

            self.operations.append((op_id, operation))
            self.id_to_index[op_id] = ('operation', len(self.operations) - 1)

            # Update next_id
            try:
                op_num = int(op_id[1:])
                self.next_id = max(self.next_id, op_num + 1)
            except Exception:
                pass
        
    
    def save_to_json(self, filepath):
        # Save the scene to a JSON file.
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True, f"Scene saved to {filepath}"
        except Exception as e:
            return False, f"Error saving scene: {str(e)}"

    def load_from_json(self, filepath):
        # Load a scene from a JSON file.
        try:
            with open(filepath, 'r') as f:
                scene_dict = json.load(f)
            self.from_dict(scene_dict)
            return True, f"Scene loaded from {filepath}"
        except FileNotFoundError:
            return False, f"File not found: {filepath}"
        except json.JSONDecodeError:
            return False, f"Invalid JSON file: {filepath}"
        except Exception as e:
            return False, f"Error loading scene: {str(e)}"
