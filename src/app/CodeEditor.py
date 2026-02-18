import tkinter as tk
from tkinter import font as tkfont
import re


class GLSLEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self._processing_event = False
        self.title("GLSL Editor")
        # Set initial geometry and minimum size
        self.geometry("1000x600") 
        self.minsize(650, 400)

        # --- Theme ---
        self.colors = {
            "bg": "#282828",
            "fg": "#d4d4d4",
            "cursor": "#aeafad",
            "select_bg": "#264f78",
            "line_bg": "#0A0A0A",
            "line_fg": "#858585",
            "keyword": "#ffa3fc",
            "type": "#fb9e9e",
            "builtin": "#ffbda3",
            "string": "#ce9178",
            "comment": "#6a9955",
            "number": "#b5cea8",
            # New colors for the sidebar
            "sidebar_bg": "#1e1e1e",
            "icon_color": "#cccccc",
        }
        
        self.sidebar_width = 300 # Initial width of the sidebar
        self.sidebar_open = False # State of the sideba
        self.rec = False

        self._build_ui()
        self._setup_syntax()
        self._bind_events()

        # Insert a small example so highlighting is visible immediately
        demo_code = """\
/* Input Data:
    p (vec3) - RayMarching point position
    sceneRes (vec4) - scene data, where:
        First three values - RGB color;
        Fourth value - distance;
Output: 
    sceneRes (vec4) - merge your scene code with the main code as shown in the example. */
    
// Example of a custom scene: Box and Sphere with Smooth Union
// 1. Define colors
vec3 box_col = vec3(0.6,0.8,0.4);
vec3 sph_col = vec3(1.0, 0.7, 0.5);

// 2. Calculate the SDF (using sdf library)
float box = sdBox(p - vec3(0.0), vec3(0.5));
float sph = sdSphere(p - vec3(0.0, 0.5, 0.0), 0.5);

// 3. Combine the box and sphere SDF using the smooth union operator
float box_and_sph = SmoothUnion(box, sph, 0.1); // Adjust the smoothing factor (k = 0.1)

// 4. Combine the box and sphere Color using "mixColorSmooth"
vec3 box_and_sph_col = mixColorSmooth(box_col, sph_col, box, sph, 0.1);

// 5. Combine the resulting SDF with the scene distance using Union (Note: NOT smooth)
sceneRes.w = Union(sceneRes.w, box_and_sph);

// 6. Combine the color with the sceneRes color using mixColorSmooth
// with k=0.001 since we are NOT using a smooth Union
sceneRes.rgb = mixColorSmooth(sceneRes.rgb, box_and_sph_col, sceneRes.w, box_and_sph, 0.001);

// Tip: You can also use the ternary operator for NON-smooth color blending:
// sceneRes.rgb = (sceneRes.w < box_and_sph) ? sceneRes.rgb : box_and_sph_col;

"""
        self.text.insert("1.0", demo_code)
        self.highlight_syntax()
        self.update_line_numbers()
        
        # Initialize the read-only panel
        self._setup_readonly_panel()
        self.hide_readonly_panel()


    # -------------------- UI SETUP --------------------

    def _build_ui(self):
        # Main container set to grid layout
        self.main_container = tk.Frame(self, bg=self.colors["bg"])
        self.main_container.pack(fill="both", expand=True)
        
        # Grid configuration for the main content area (Editor + Sidebar)
        self.main_container.grid_rowconfigure(1, weight=1) # Editor row expands vertically
        self.main_container.grid_columnconfigure(1, weight=1) # Editor column expands horizontally

        # 1. Top Bar (to hold the icon and title)
        top_bar = tk.Frame(self.main_container, bg=self.colors["line_bg"])
        top_bar.grid(row=0, column=0, columnspan=4, sticky="ew")
        
        # Icon Button (Upper Left Corner)
        self.preview_button = tk.Button(
            top_bar,
            text="☷ preview code",
            bg=self.colors["line_bg"],
            fg=self.colors["icon_color"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["icon_color"],
            command=self.toggle_readonly_panel,
            font=("Segoe UI", 12, "bold"),
            borderwidth=0,
            width=12
        )
        self.preview_button.pack(side="right", padx=15, pady=2)

        # Title Label
        title_label = tk.Label(
            top_bar,
            text="GLSL Editor",
            bg=self.colors["line_bg"],
            fg=self.colors["fg"],
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        )
        title_label.pack(side="left", padx=10)


        # Recompile Button
        self.recompile_button = tk.Button(
            top_bar,
            text="◼ recompile",
            bg=self.colors["line_bg"],
            fg=self.colors["icon_color"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["icon_color"],
            command=self.recompile_signal,
            font=("Segoe UI", 12, "bold"),
            borderwidth=0,
            width=12
        )
        self.recompile_button.pack(side="left", padx=5, pady=2)


        # --- Code Editor Components (Grid Layout) ---
        
        # Line numbers panel (Column 0)
        self.line_numbers = tk.Text(
            self.main_container,
            width=5,
            padx=5,
            takefocus=0,
            border=0,
            background=self.colors["line_bg"],
            foreground=self.colors["line_fg"],
            state="disabled",
            wrap="none",
        )
        self.line_numbers.grid(row=1, column=0, sticky="ns")

        # Scrollbars setup 
        self.v_scroll = tk.Scrollbar(self.main_container, orient="vertical")
        self.v_scroll.grid(row=1, column=2, sticky="ns")

        self.h_scroll = tk.Scrollbar(self.main_container, orient="horizontal")
        self.h_scroll.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Text widget (code editor) (Column 1)
        font_family = "Consolas"
        self.text_font = tkfont.Font(family=font_family, size=11)

        self.text = tk.Text(
            self.main_container,
            bg=self.colors["bg"],
            fg=self.colors["fg"],
            insertbackground=self.colors["cursor"],
            selectbackground=self.colors["select_bg"],
            undo=True,
            maxundo=-1,
            wrap="none",
            padx=8,
            pady=8,
            border=0,
            font=self.text_font,
            yscrollcommand=self._on_textscroll,
            xscrollcommand=self.h_scroll.set,
        )
        self.text.grid(row=1, column=1, sticky="nsew")
        
        self.v_scroll.config(command=self._on_vscroll)
        self.h_scroll.config(command=self.text.xview)
        
        # 3. Read-Only Panel Container (Column 3, dynamically managed)
        self.readonly_frame = tk.Frame(self.main_container, bg=self.colors["sidebar_bg"], bd=1, relief="flat")
        self.readonly_frame.grid_rowconfigure(0, weight=1) # Inner frame content expands vertically
        self.readonly_frame.grid_columnconfigure(1, weight=1) # Inner text widget expands horizontally

        # Resize Handle (Placed inside the readonly_frame, but positioned by its own grid location)
        self.resize_handle = tk.Frame(self.readonly_frame, width=5, bg=self.colors["line_bg"], cursor="sb_h_double_arrow")
        self.resize_handle.grid(row=0, column=0, sticky="ns")
        self.resize_handle.bind("<B1-Motion>", self._resize_panel)
        self.resize_handle.bind("<ButtonPress-1>", self._start_resize)

    def _setup_readonly_panel(self):
        """Sets up the readonly text widget inside the sidebar frame."""
        
        # Read-Only Text Widget (Column 1 of the readonly_frame)
        self.readonly_text = tk.Text(
            self.readonly_frame,
            bg=self.colors["sidebar_bg"],
            fg=self.colors["fg"],
            undo=False,
            wrap="word",
            padx=8,
            pady=8,
            border=0,
            font=self.text_font,
            state="disabled" # Crucial: makes it non-editable
        )
        self.readonly_text.grid(row=0, column=1, sticky="nsew")
        
        # Apply syntax highlighting configuration to the readonly widget
        for tag in ("keyword", "type", "builtin", "number", "string", "comment"):
            self.readonly_text.tag_configure(tag, foreground=self.colors[tag])

    # -------------------- SIDEBAR TOGGLE & RESIZE LOGIC --------------------
    
    def recompile_signal(self):
        self.rec = True


    def toggle_readonly_panel(self):
        """Shows or hides the right panel."""
        if self.sidebar_open:
            self.hide_readonly_panel()
        else:
            self.show_readonly_panel()
            
    def show_readonly_panel(self):
        """Displays the panel on the right and sets up grid weights."""
        self.sidebar_open = True
        
        # 1. Place the sidebar frame in the grid (column 3)
        self.readonly_frame.grid(row=1, column=3, rowspan=2, sticky="ns")
        
        # 2. Configure the column width for the sidebar
        self.readonly_frame.config(width=self.sidebar_width)
        self.readonly_frame.grid_propagate(False) 
        
        # 3. Configure the grid column structure of the main container
        self.main_container.grid_columnconfigure(3, minsize=self.sidebar_width, weight=0)
        
        # 4. Update content and redraw
        self.update_readonly_content()
        self.preview_button.config(text="⇶ close preview", fg="#A9A9A9") # Change icon to close indicator

    def hide_readonly_panel(self):
        """Hides the panel and allows the editor to take full space."""
        self.sidebar_open = False
        
        # 1. Remove the sidebar frame from the grid
        self.readonly_frame.grid_forget()
        
        # 2. Reset the column configuration to collapse column 3
        self.main_container.grid_columnconfigure(3, weight=0, minsize=0)

        # 3. Update icon
        self.preview_button.config(text="☷ preview code", fg=self.colors["icon_color"])
        
    def update_readonly_content(self):
        """Copies content and reapplies highlighting to the read-only widget."""
        if not self.sidebar_open:
            return

        code = self.text.get("1.0", "end-1c")
        code = self.format_glsl_code(code)
        
        self.readonly_text.config(state="normal")
        self.readonly_text.delete("1.0", "end")
        self.readonly_text.insert("1.0", code)
        
        # Re-apply highlighting
        for tag in ("keyword", "type", "builtin", "number", "string", "comment"):
            pattern = self.syntax_patterns[tag]
            for match in pattern.finditer(code):
                start, end = match.span()
                start_index = f"1.0+{start}c"
                end_index = f"1.0+{end}c"
                self.readonly_text.tag_add(tag, start_index, end_index)

        self.readonly_text.config(state="disabled")
        
    # --- Resizing Logic ---
    
    def _start_resize(self, event):
        """Begins the resize operation."""
        self._resize_start_x = event.x_root
        self._initial_width = self.sidebar_width
        
    def _resize_panel(self, event):
        """Handles the mouse motion while dragging the handle."""
        delta_x = event.x_root - self._resize_start_x
        
        new_width = self._initial_width - delta_x
        
        MIN_WIDTH = 300
        MAX_WIDTH = self.winfo_width() // 2
        
        if MIN_WIDTH < new_width < MAX_WIDTH:
            self.sidebar_width = new_width
            
            # Update the frame width configuration directly
            self.readonly_frame.config(width=self.sidebar_width)
            
            # Update content to reflect new wrapping
            self.update_readonly_content()

    # -------------------- EVENT BINDINGS & UPDATES --------------------
    
    def _bind_events(self):
        # Content changes
        self.text.bind("<KeyRelease>", self._on_key_release)
        self.text.bind("<ButtonRelease-1>", self._on_cursor_move)
        self.text.bind("<MouseWheel>", self._on_mouse_wheel)
        self.text.bind("<Button-4>", self._on_mouse_wheel)
        self.text.bind("<Button-5>", self._on_mouse_wheel)

        # Auto-pair characters
        for ch in ("(", "[", "{", '"', "'"):
            self.text.bind(ch, self._handle_auto_pair)

        # Indentation helpers
        self.text.bind("<Tab>", self._handle_tab)
        self.text.bind("<Return>", self._handle_return)

        # Clipboard bindings
        self.text.bind("<Control-a>", self._select_all)
        self.text.bind("<Control-A>", self._select_all)
        self.text.bind("<Control-c>", self._copy)
        self.text.bind("<Control-C>", self._copy)
        self.text.bind("<Control-v>", self._paste)
        self.text.bind("<Control-V>", self._paste)
        self.text.bind("<Control-x>", self._cut)
        self.text.bind("<Control-X>", self._cut)

        # Layout monitoring
        self.text.bind("<Configure>", self._on_editor_resize)


    def _on_key_release(self, event=None):
        if self._processing_event:
            return
        self._processing_event = True
        
        self.update_line_numbers()
        self.highlight_syntax()
        if self.sidebar_open:
            self.update_readonly_content()
        
        self._processing_event = False

    def _on_cursor_move(self, event=None):
        if self._processing_event:
            return
        self._processing_event = True
        
        self.update_line_numbers()
        
        self._processing_event = False
        
    def _on_editor_resize(self, event):
        # This helps ensure grid layouts manage space correctly when the window is resized
        self.update_line_numbers()

    # -------------------- SCROLLING --------------------

    def _on_textscroll(self, *args):
        self.line_numbers.yview_moveto(args[0])
        self.v_scroll.set(*args)

    def _on_vscroll(self, *args):
        self.text.yview(*args)
        self.line_numbers.yview(*args)

    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self.text.yview_scroll(-1, "units")
            self.line_numbers.yview_scroll(-1, "units")
        else:
            self.text.yview_scroll(1, "units")
            self.line_numbers.yview_scroll(1, "units")
        return "break"

    # -------------------- LINE NUMBERS --------------------

    def update_line_numbers(self, event=None):
        self.line_numbers.config(state="normal")
        self.line_numbers.delete("1.0", "end")

        content = self.text.get("1.0", "end-1c")
        line_count = content.count("\n") + 1
        lines = "\n".join(str(i) for i in range(1, line_count + 1))

        self.line_numbers.insert("1.0", lines)
        self.line_numbers.config(state="disabled")

    # -------------------- SYNTAX HIGHLIGHTING --------------------

    def _setup_syntax(self):
        # GLSL keyword/type/builtin sets
        glsl_keywords = [
            "attribute", "const", "uniform", "varying", "layout",
            "centroid", "flat", "smooth", "noperspective",
            "break", "continue", "do", "for", "while",
            "switch", "case", "default", "if", "else",
            "in", "out", "inout", "invariant",
            "discard", "return",
            "lowp", "mediump", "highp", "precision",
            "struct"
        ]

        glsl_types = [
            "void", "bool", "int", "uint", "float", "double",
            "vec2", "vec3", "vec4",
            "bvec2", "bvec3", "bvec4",
            "ivec2", "ivec3", "ivec4",
            "uvec2", "uvec3", "uvec4",
            "mat2", "mat3", "mat4",
            "sampler1D", "sampler2D", "sampler3D",
            "samplerCube", "sampler2DShadow",
            "samplerCubeShadow",
            "sampler2DArray", "sampler2DArrayShadow",
            "isampler2D", "usampler2D"
        ]

        glsl_builtins = [
            "radians", "degrees", "sin", "cos", "tan",
            "asin", "acos", "atan", "pow", "exp", "log",
            "exp2", "log2", "sqrt", "inversesqrt",
            "abs", "sign", "floor", "ceil", "fract", "mod",
            "min", "max", "clamp", "mix", "step", "smoothstep",
            "length", "distance", "dot", "cross", "normalize",
            "faceforward", "reflect", "refract",
            "matrixCompMult",
            "lessThan", "lessThanEqual",
            "greaterThan", "greaterThanEqual",
            "equal", "notEqual", "any", "all", "not",
            "texture", "texture2D", "textureCube", 
            "rgb", "rgba", "r", "g", "b", "a", 
            "xyz", "xyzw", "xy", "xz", "x", "y", "z", "w"
        ]

        # Compile regex patterns
        self.syntax_patterns = {
            "keyword": re.compile(
                r"\b(?:"
                + "|".join(re.escape(k) for k in glsl_keywords)
                + r")\b"
            ),
            "type": re.compile(
                r"\b(?:"
                + "|".join(re.escape(t) for t in glsl_types)
                + r")\b"
            ),
            "builtin": re.compile(
                r"\b(?:"
                + "|".join(re.escape(b) for b in glsl_builtins)
                + r")\b"
            ),
            "number": re.compile(r"\b\d+(\.\d+)?\b"),
            "string": re.compile(
                r'"([^"\\]|\\.)*"|'   # double-quoted
                r"'([^'\\]|\\.)*'"    # single-quoted
            ),
            "comment": re.compile(r"//.*|/\*[\s\S]*?\*/"),
        }

        # Tag configuration (colors)
        self.text.tag_configure("keyword", foreground=self.colors["keyword"])
        self.text.tag_configure("type", foreground=self.colors["type"])
        self.text.tag_configure("builtin", foreground=self.colors["builtin"])
        self.text.tag_configure("number", foreground=self.colors["number"])
        self.text.tag_configure("string", foreground=self.colors["string"])
        self.text.tag_configure("comment", foreground=self.colors["comment"])

    def highlight_syntax(self, event=None):
        text = self.text
        for tag in ("keyword", "type", "builtin", "number", "string", "comment"):
            text.tag_remove(tag, "1.0", "end")

        code = text.get("1.0", "end-1c")
        order = ["keyword", "type", "builtin", "number", "string", "comment"]

        for tag in order:
            pattern = self.syntax_patterns[tag]
            for match in pattern.finditer(code):
                start, end = match.span()
                start_index = f"1.0+{start}c"
                end_index = f"1.0+{end}c"
                text.tag_add(tag, start_index, end_index)
    

    def format_glsl_code(self, code: str) -> str:
        """
        Formats a string of GLSL code by:
        1. Removing comments (//, /*, */) while preserving original empty lines.
        2. Prepending and appending specific boilerplate code.
        3. Increasing indentation by 4 spaces.
        """
        import re

        # --- 1. Remove comments while preserving empty lines ---
        lines = []
        in_multiline_comment = False

        for line in code.splitlines():
            stripped = line.strip()

            # Handle multi-line comments (/* ... */)
            if in_multiline_comment:
                if "*/" in line:
                    in_multiline_comment = False
                    # Keep the line if it has non-comment content after */
                    remaining = line.split("*/", 1)[1].strip()
                    if remaining:
                        lines.append(remaining)
                    else:
                        lines.append("")  # Preserve empty line
                continue  # Skip lines inside multi-line comments

            # Check for start of multi-line comment
            if "/*" in line:
                in_multiline_comment = True
                # Keep content before /*
                before_comment = line.split("/*", 1)[0].strip()
                if before_comment:
                    lines.append(before_comment)
                else:
                    lines.append("")  # Preserve empty line
                continue

            # Remove single-line comments (//)
            if "//" in line:
                before_comment = line.split("//", 1)[0].strip()
                if before_comment:
                    lines.append(before_comment)
                else:
                    lines.append("")  # Preserve empty line
            else:
                lines.append(line)  # Keep non-comment lines as-is

        # Rejoin lines (preserves original empty lines)
        code = "\n".join(lines)

        # --- 4. Increase indentation by 4 spaces ---
        INDENT_SPACE = "    "  # 4 spaces

        # Normalize tabs to spaces
        code_normalized = code.replace('\t', INDENT_SPACE)

        # Apply indentation (only to non-empty lines)
        indented_lines = []
        for line in code_normalized.splitlines():
            if line.strip():  # Only indent non-empty lines
                indented_lines.append(INDENT_SPACE + line)
            else:
                indented_lines.append("")  # Preserve empty lines

        indented_code = "\n".join(indented_lines)

        # --- 2. Add prefix ---
        prefix = (
            "vec4 map(vec3 p) {\n"
            "    vec4 sceneRes = getSceneDist(p);\n"
            "    // --- Custom Scene Code --- \n"
        )

        # --- 3. Add suffix ---
        suffix = (
            "\n    // --- End Custom Scene Code ---"
            "\n    return sceneRes;\n"
            "}"
        )

        # Combine all parts
        formatted_code = prefix + indented_code + suffix

        return formatted_code



    # -------------------- EDITING HANDLERS (Auto-pair, Tab, Enter) --------------------

    def _handle_auto_pair(self, event):
        text = event.widget
        char = event.char
        pairs = {"(": ")", "[": "]", "{": "}", '"': '"', "'": "'"}
        closing = pairs.get(char)
        if not closing:
            return

        try:
            sel_start = text.index("sel.first")
            sel_end = text.index("sel.last")
            selected = text.get(sel_start, sel_end)
            text.delete(sel_start, sel_end)
            text.insert(sel_start, char + selected + closing)
            text.mark_set("insert", f"{sel_start}+{len(char + selected + closing)}c")
        except tk.TclError:
            text.insert("insert", char + closing)
            text.mark_set("insert", "insert-1c")

        return "break"

    def _handle_tab(self, event):
        event.widget.insert("insert", "    ")
        return "break"

    def _handle_return(self, event):
        text = event.widget
        line_start = text.index("insert linestart")
        line_text = text.get(line_start, "insert")
        indent = re.match(r"\s*", line_text).group(0)
        text.insert("insert", "\n" + indent)
        return "break"

    # -------------------- CLIPBOARD OPERATIONS --------------------
    
    def get_plain_text(self):
        return self.text.get("1.0", "end-1c")

    def _select_all(self, event=None):
        self.text.tag_add("sel", "1.0", "end")
        return "break"

    def _copy(self, event=None):
        try:
            selected_text = self.text.get("sel.first", "sel.last")
            self.clipboard_clear()
            self.clipboard_append(selected_text)
        except tk.TclError:
            pass
        return "break"

    def _cut(self, event=None):
        try:
            selected_text = self.text.get("sel.first", "sel.last")
            self.clipboard_clear()
            self.clipboard_append(selected_text)
            self.text.delete("sel.first", "sel.last")
        except tk.TclError:
            pass
        return "break"

    def _paste(self, event=None):
        try:
            clipboard_text = self.clipboard_get()
            self.text.insert("insert", clipboard_text)
        except tk.TclError:
            pass
        return "break"


if __name__ == '__main__':
    GLSLEditor().mainloop()