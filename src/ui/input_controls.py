import imgui
import time

from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import (
    glBindTexture, glTexParameteri, GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST
)

# --- Font ---
def rebuild_imgui_fonts(renderer : GlfwRenderer, base_font_path="path/to/your/font.ttf", base_font_size=16.0):
    # base_font_size is in logical points; multiply by framebuffer scale for pixel-perfect atlas
    io = imgui.get_io()
    fb_scale_x, fb_scale_y = io.display_fb_scale

    # clear existing fonts and add scaled font
    io.fonts.clear()
    pixel_size = base_font_size * max(fb_scale_x, fb_scale_y)
    io.fonts.add_font_from_file_ttf(base_font_path, pixel_size)

    # rebuild texture and let the renderer upload it
    renderer.refresh_font_texture()

    # force nearest filtering if you want crisp text at integer scales
    tex_id = io.fonts.texture_id
    if tex_id:
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)




def HSpinner(value, value_step, name, width=16, height=16, under=True, repeat_delay=0.25, repeat_rate=0.1):
    # Combined spinner with input field above buttons with auto-repeat on hold
    imgui.begin_group()

    # Input field at top
    imgui.push_item_width(width)
    input_changed, value = imgui.input_float(f"##input_{name}", value, 0, 0, "%.3f")
    imgui.pop_item_width()
    if under==False: imgui.same_line()

    btn_changed = False
    width_mul = 2.0 if under else 5.0

    # Create unique IDs for buttons
    minus_btn_id = f"##btn_{name}_minus"
    plus_btn_id = f"##btn_{name}_plus"
    
    # Track button state and timing
    if not hasattr(HSpinner, "button_states"):
        HSpinner.button_states = {}
    
    # Minus button (left)
    if imgui.button(f"-{minus_btn_id}", (width/width_mul)-1, height):
        value -= value_step
        btn_changed = True
    
    # Check if minus button is held down
    if imgui.is_item_active() and imgui.is_mouse_down(0):  # Left mouse button
        current_time = time.time()
        btn_key = minus_btn_id
        
        # Initialize button state if not exists
        if btn_key not in HSpinner.button_states:
            HSpinner.button_states[btn_key] = {
                'first_press_time': current_time,
                'last_repeat_time': current_time,
                'has_repeated': False
            }
        
        state = HSpinner.button_states[btn_key]
        
        # Check if enough time has passed for first repeat
        if not state['has_repeated'] and (current_time - state['first_press_time']) >= repeat_delay:
            value -= value_step
            btn_changed = True
            state['has_repeated'] = True
            state['last_repeat_time'] = current_time
        # Check if enough time has passed for subsequent repeats
        elif state['has_repeated'] and (current_time - state['last_repeat_time']) >= repeat_rate:
            value -= value_step
            btn_changed = True
            state['last_repeat_time'] = current_time
    else:
        # Reset button state when not pressed
        minus_btn_key = minus_btn_id
        if minus_btn_key in HSpinner.button_states:
            del HSpinner.button_states[minus_btn_key]

    imgui.same_line(0, 2)

    # Plus button (right)
    if imgui.button(f"+{plus_btn_id}", (width/width_mul)-1, height):
        value += value_step
        btn_changed = True
    
    # Check if plus button is held down
    if imgui.is_item_active() and imgui.is_mouse_down(0):  # Left mouse button
        current_time = time.time()
        btn_key = plus_btn_id
        
        # Initialize button state if not exists
        if btn_key not in HSpinner.button_states:
            HSpinner.button_states[btn_key] = {
                'first_press_time': current_time,
                'last_repeat_time': current_time,
                'has_repeated': False
            }
        
        state = HSpinner.button_states[btn_key]
        
        # Check if enough time has passed for first repeat
        if not state['has_repeated'] and (current_time - state['first_press_time']) >= repeat_delay:
            value += value_step
            btn_changed = True
            state['has_repeated'] = True
            state['last_repeat_time'] = current_time
        # Check if enough time has passed for subsequent repeats
        elif state['has_repeated'] and (current_time - state['last_repeat_time']) >= repeat_rate:
            value += value_step
            btn_changed = True
            state['last_repeat_time'] = current_time
    else:
        # Reset button state when not pressed
        plus_btn_key = plus_btn_id
        if plus_btn_key in HSpinner.button_states:
            del HSpinner.button_states[plus_btn_key]

    imgui.end_group()
    return input_changed or btn_changed, value

def input_vec3(name, vector, value_step=0.1, item_width=60):
    # Handles a 3D vector input with separate HSpinners for each component
    imgui.begin_group()
    changed = False
    for i, axis in enumerate(['x', 'y', 'z']):
        c, vector[i] = HSpinner(vector[i], value_step, f"{name}_{axis}", item_width)
        changed = changed or c
        if i < 2:
            imgui.same_line()
    imgui.end_group()

    imgui.same_line()
    imgui.text(name)
    return changed, vector


def input_vec2(name, vector, value_step=0.1, item_width=60):
    # Handles a 3D vector input with separate HSpinners for each component
    imgui.begin_group()
    changed = False
    for i, axis in enumerate(['x', 'y']):
        c, vector[i] = HSpinner(vector[i], value_step, f"{name}_{axis}", item_width)
        changed = changed or c
        if i < 1:
            imgui.same_line()
    imgui.end_group()

    imgui.same_line()
    imgui.text(name)
    return changed, vector


def input_float(name, value, value_step=0.1, item_width=60):
    imgui.begin_group()
    changed, value = HSpinner(value, value_step, f"{name}_f", item_width, 20, False)
    imgui.same_line()
    imgui.end_group()

    imgui.same_line()
    imgui.text(name)
    return changed, value