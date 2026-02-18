
import os
import json
from typing import Dict, List, Any

def save_user_config(file_path: str, config_data: Dict[str, Any]) -> None:
    # Save configuration data to a JSON file.
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(config_data, file, indent=2)  # Pretty-print with 2-space indent
    except IOError as e:
        raise IOError(f"Error writing config file {file_path}: {e}")


def load_user_config(file_path: str) -> Dict[str, Any]:
    # Load user configuration from a JSON file into a Python dictionary.
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return {}  # Return empty dict for new/empty files

        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {file_path}: {e}")
    except IOError as e:
        raise IOError(f"Error reading config file {file_path}: {e}")
