import json
import os
from typing import Dict

def recompute_next_id(builder):
    max_n = -1
    for nid in builder.scene_nodes.keys():
        if isinstance(nid, str) and nid.startswith('d'):
            try:
                n = int(nid[1:])
                if n > max_n:
                    max_n = n
            except Exception:
                continue
    # set to next free index
    builder.next_id = max_n + 1 if max_n >= 0 else 0


def to_dict(builder) -> Dict:
    """
    Serialize entire scene to dictionary for JSON save.
    
    Returns:
        Dictionary with 'next_id', 'root_children', 'nodes'
    """
    scene_dict = {
        'next_id': builder.next_id,
        'root_children': builder.root_children,
        'nodes': {}
    }
    
    # Serialize all nodes
    for node_id, node in builder.scene_nodes.items():
        scene_dict['nodes'][node_id] = node.to_dict()
    
    return scene_dict

def from_dict(builder, scene_dict: Dict):
    """
    Load scene from dictionary (inverse of to_dict).
    
    Args:
        scene_dict: Dictionary from to_dict() or JSON
    """
    # Clear current scene
    builder.scene_nodes.clear()
    builder.id_to_node.clear()
    builder.root_children.clear()
    
    # Restore basic properties
    builder.next_id = scene_dict.get('next_id', 0)
    builder.root_children = list(scene_dict.get('root_children', []))
    
    # Reconstruct all nodes
    nodes_dict = scene_dict.get('nodes', {})
    for node_id, node_data in nodes_dict.items():
        builder._reconstruct_node(node_id, node_data)
    
    builder.invalidate_cache()


def to_json(builder) -> str:
    return json.dumps(to_dict(builder), indent=2, sort_keys=True)

def from_json(builder, json_str: str) -> bool:
    try:
        scene_dict = json.loads(json_str)
        from_dict(builder, scene_dict)
        # Recompute next_id to avoid collisions
        recompute_next_id(builder)
        builder.invalidate_cache()
        return True
    except Exception as e:
        print(f"SceneBuilder.from_json: failed to parse/load JSON: {e}")
        return False

def save_to_file(builder, filepath: str) -> tuple[bool, str]:
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
            json.dump(to_dict(builder), f, indent=2, sort_keys=True)

        return True, f"Saved scene to {filepath}"
    except Exception as e:
        return False, f"Failed to save scene to {filepath}: {e}"

def load_from_file(builder, filepath: str) -> tuple[bool, str]:
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
        from_dict(builder, scene_dict)
        # Recompute next_id to avoid collisions
        recompute_next_id(builder)
        builder.invalidate_cache()
        return True, f"Loaded scene from {filepath}"
    except Exception as e:
        return False, f"Failed to load scene from {filepath}: {e}"