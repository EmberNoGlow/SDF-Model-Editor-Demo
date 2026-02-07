import imgui

# --- Defined Palette ---
bg_dark = (0.12, 0.11, 0.09, 1.0)        # Very dark background
panel_dark = (0.18, 0.16, 0.13, 1.0)     # Slightly lighter panel/frame
accent = (0.608, 0.067, 0.118, 1.0)      # Primary Dark Red/Crimson (Buttons, Active Header)
hover = (0.902, 0.125, 0.125, 1.0)       # Bright Red/Scarlet (Hover/Active State)
text_light = (0.92, 0.90, 0.80, 1.0)     # Off-white text

muted_accent = (0.4, 0.04, 0.08, 1.0) 
child_bg = (0.20, 0.18, 0.15, 1.0)
dim_background = (0.0, 0.0, 0.0, 0.7) 
border_color = (0.25, 0.23, 0.20, 1.0)




def setup_theme():
    style = imgui.get_style()

    # --- Defined Palette ---
    global bg_dark, panel_dark, accent, hover, text_light, muted_accent, child_bg, dim_background, border_color

    # ---------------------------------------------
    # Set key colors based on the defined palette
    # ---------------------------------------------
    
    # Core Backgrounds
    style.colors[imgui.COLOR_WINDOW_BACKGROUND] = bg_dark
    style.colors[imgui.COLOR_POPUP_BACKGROUND] = panel_dark
    style.colors[imgui.COLOR_CHILD_BACKGROUND] = child_bg
    style.colors[imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND] = dim_background
    style.colors[imgui.COLOR_NAV_WINDOWING_DIM_BACKGROUND] = dim_background
    
    # Text & Borders
    style.colors[imgui.COLOR_TEXT] = text_light
    style.colors[imgui.COLOR_TEXT_DISABLED] = (0.5, 0.5, 0.5, 1.0)
    style.colors[imgui.COLOR_BORDER] = border_color
    style.colors[imgui.COLOR_BORDER_SHADOW] = (0.0, 0.0, 0.0, 0.0)
    
    # Frames, Inputs, and Sliders (Use panel_dark or a slightly lighter version)
    style.colors[imgui.COLOR_FRAME_BACKGROUND] = panel_dark
    style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.25, 0.22, 0.19, 1.0)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = child_bg 

    style.colors[imgui.COLOR_SLIDER_GRAB] = accent
    style.colors[imgui.COLOR_SLIDER_GRAB_ACTIVE] = hover
    
    # Buttons
    style.colors[imgui.COLOR_BUTTON] = accent
    style.colors[imgui.COLOR_BUTTON_HOVERED] = hover
    style.colors[imgui.COLOR_BUTTON_ACTIVE] = hover
    
    # Headers & Tabs (Using accent for focus)
    style.colors[imgui.COLOR_HEADER] = muted_accent
    style.colors[imgui.COLOR_HEADER_HOVERED] = accent
    style.colors[imgui.COLOR_HEADER_ACTIVE] = hover
    
    style.colors[imgui.COLOR_TAB] = muted_accent
    style.colors[imgui.COLOR_TAB_HOVERED] = accent
    style.colors[imgui.COLOR_TAB_ACTIVE] = hover
    style.colors[imgui.COLOR_TAB_UNFOCUSED] = (0.15, 0.14, 0.12, 1.0)
    style.colors[imgui.COLOR_TAB_UNFOCUSED_ACTIVE] = (0.25, 0.1, 0.1, 1.0)
    
    # Menu Bar & Title Bar (Use primary background colors)
    style.colors[imgui.COLOR_MENUBAR_BACKGROUND] = bg_dark
    style.colors[imgui.COLOR_TITLE_BACKGROUND] = panel_dark
    style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = accent
    style.colors[imgui.COLOR_TITLE_BACKGROUND_COLLAPSED] = panel_dark
    
    # Scrollbar
    style.colors[imgui.COLOR_SCROLLBAR_BACKGROUND] = (0.15, 0.14, 0.12, 0.5)
    style.colors[imgui.COLOR_SCROLLBAR_GRAB] = (0.35, 0.33, 0.30, 1.0)
    style.colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = accent
    style.colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = hover
    
    # Navigation/Highlight
    style.colors[imgui.COLOR_NAV_HIGHLIGHT] = hover
    style.colors[imgui.COLOR_CHECK_MARK] = text_light
    
    # Tables
    style.colors[imgui.COLOR_TABLE_HEADER_BACKGROUND] = panel_dark
    style.colors[imgui.COLOR_TABLE_BORDER_STRONG] = border_color
    style.colors[imgui.COLOR_TABLE_BORDER_LIGHT] = (0.25, 0.23, 0.20, 0.5)
    style.colors[imgui.COLOR_TABLE_ROW_BACKGROUND] = child_bg
    style.colors[imgui.COLOR_TABLE_ROW_BACKGROUND_ALT] = (0.19, 0.17, 0.14, 1.0)
    
    # Selection
    style.colors[imgui.COLOR_DRAG_DROP_TARGET] = (0.3, 0.4, 0.2, 0.5)
    style.colors[imgui.COLOR_TEXT_SELECTED_BACKGROUND] = accent
    
    # Geometry
    style.window_padding = (10, 10)
    style.frame_padding = (5, 3)
    style.item_spacing = (6, 4)

    style.frame_rounding = 2.0
    style.window_rounding = 5.0
    style.grab_rounding = 3.0
