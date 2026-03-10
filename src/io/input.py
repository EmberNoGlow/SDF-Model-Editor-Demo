import imgui
import glfw

ShortCuts = {
    "Rename" : (glfw.KEY_F2),
    "Add" : (glfw.KEY_A, "CTRL"), 
    "Delete" : (glfw.KEY_DELETE),
    "Compile" : (glfw.KEY_B, "CTRL"),
    "Undo" : (glfw.KEY_Z, "CTRL"),
    "Redo" : (glfw.KEY_Z, "CTRL", "SHIFT"),
    "Redo2" : (glfw.KEY_Y, "CTRL"),
    "Move" : (glfw.KEY_G),
    "Rotate" : (glfw.KEY_R),
    "X" : (glfw.KEY_X),
    "Y" : (glfw.KEY_Y),
    "Z" : (glfw.KEY_Z),
    "Open" : (glfw.KEY_O, "CTRL"),
    "Save" : (glfw.KEY_S, "CTRL"),
    "Duplicate": (glfw.KEY_D, "CTRL"),
}


io = None

def get_io():
    # It won't work if you don't have the imgui itself, lol
    global io
    io = imgui.get_io()
    return io


def input_handle(action : str) -> bool:
    if io is None: # You forgot to get io before calling
        get_io()

    # Helper function to get the live state of a modifier ID
    def get_live_modifier_state(modifier_id):
        if modifier_id == "CTRL":
            return io.key_ctrl
        if modifier_id == "SHIFT":
            return io.key_shift
        return False
    
    keys_required = ShortCuts.get(action)
    
    if keys_required is None:
        return False

    # Ensure keys_required is always iterable (a tuple)
    if not isinstance(keys_required, tuple):
        keys_required = (keys_required,)

    # --- STEP 1: Check if ALL conditions are met (Is the combination currently held?) ---
    all_keys_down_this_frame = True
    main_key_code = None # Store the main key code for debouncing later

    for key_check in keys_required:
        
        if isinstance(key_check, int):
            # Standard key code: Must be currently down
            if not io.keys_down[key_check]:
                all_keys_down_this_frame = False
                break
            # Store this as the potential main key to check for initial press
            main_key_code = key_check
                
        elif isinstance(key_check, str):
            # Modifier: Must be currently down
            if not get_live_modifier_state(key_check):
                all_keys_down_this_frame = False
                break

    if not all_keys_down_this_frame:
        return False # Combo is not active right now

    # --- STEP 2: Debounce (Did the key press START this frame?) ---
    
    # Case A: Single Key (like F2)
    if len(keys_required) == 1 and main_key_code is not None:
        if io.keys_down[main_key_code]:
            return True
    
    # Case B: Combination Key (like Ctrl+A)
    elif len(keys_required) > 1 and main_key_code is not None:
        if io.keys_down[main_key_code]:
            return True
            
    # If we reach here, the combination is held, but the trigger key wasn't *newly* pressed this frame.
    return False