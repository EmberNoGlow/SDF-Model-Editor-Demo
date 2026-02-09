import tkinter as tk
from tkinter import font as tkfont
import re


class GLSLEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self._processing_event = False
        self.title("GLSL Editor")
        self.geometry("900x600")
        self.minsize(600, 400)

        # --- Theme (VS Code–like) ---
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
        }

        self._build_ui()
        self._setup_syntax()
        self._bind_events()

        # Insert a small example so highlighting is visible immediately
        demo_code = """// GLSL fragment shader example
#version 330 core

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D texture1;

void main()
{
    vec4 texColor = texture(texture1, TexCoords);
    FragColor = vec4(texColor.rgb, 1.0);
}
"""
        self.text.insert("1.0", demo_code)
        self.highlight_syntax()
        self.update_line_numbers()

    # -------------------- UI SETUP --------------------

    def _build_ui(self):
        # Main container
        main_frame = tk.Frame(self, bg=self.colors["bg"])
        main_frame.pack(fill="both", expand=True)

        # Top "title bar" label
        title_label = tk.Label(
            main_frame,
            text="GLSL Editor",
            bg=self.colors["line_bg"],
            fg=self.colors["fg"],
            anchor="w",
            padx=10,
            font=("Segoe UI", 10, "bold"),
        )
        title_label.pack(fill="x", side="top")

        # Editor area (line numbers + text + scrollbars)
        editor_frame = tk.Frame(main_frame, bg=self.colors["bg"])
        editor_frame.pack(fill="both", expand=True)

        # Line numbers panel
        self.line_numbers = tk.Text(
            editor_frame,
            width=5,
            padx=5,
            takefocus=0,
            border=0,
            background=self.colors["line_bg"],
            foreground=self.colors["line_fg"],
            state="disabled",
            wrap="none",
        )
        self.line_numbers.pack(side="left", fill="y")

        # Scrollbars
        self.v_scroll = tk.Scrollbar(editor_frame, orient="vertical")
        self.v_scroll.pack(side="right", fill="y")

        self.h_scroll = tk.Scrollbar(editor_frame, orient="horizontal")
        self.h_scroll.pack(side="bottom", fill="x")

        # Text widget (code editor)
        font_family = "Consolas"  # Fallback: "Courier New" or any monospace
        self.text_font = tkfont.Font(family=font_family, size=11)

        self.text = tk.Text(
            editor_frame,
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

        self.text.pack(side="left", fill="both", expand=True)

        self.v_scroll.config(command=self._on_vscroll)
        self.h_scroll.config(command=self.text.xview)

    # -------------------- SYNTAX HIGHLIGHTING SETUP --------------------

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
            # Trig & exponentials
            "radians", "degrees", "sin", "cos", "tan",
            "asin", "acos", "atan", "pow", "exp", "log",
            "exp2", "log2", "sqrt", "inversesqrt",
            # Common functions
            "abs", "sign", "floor", "ceil", "fract", "mod",
            "min", "max", "clamp", "mix", "step", "smoothstep",
            # Geometry
            "length", "distance", "dot", "cross", "normalize",
            "faceforward", "reflect", "refract",
            # Matrices
            "matrixCompMult",
            # Relational
            "lessThan", "lessThanEqual",
            "greaterThan", "greaterThanEqual",
            "equal", "notEqual", "any", "all", "not",
            # Texture functions
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
            # Single-line // ... and multi-line /* ... */
            "comment": re.compile(r"//.*|/\*[\s\S]*?\*/"),
        }

        # Tag configuration (colors)
        self.text.tag_configure("keyword", foreground=self.colors["keyword"])
        self.text.tag_configure("type", foreground=self.colors["type"])
        self.text.tag_configure("builtin", foreground=self.colors["builtin"])
        self.text.tag_configure("number", foreground=self.colors["number"])
        self.text.tag_configure("string", foreground=self.colors["string"])
        self.text.tag_configure("comment", foreground=self.colors["comment"])

    # -------------------- EVENT BINDINGS --------------------

    def _bind_events(self):
        # Content changes
        self.text.bind("<KeyRelease>", self._on_key_release)
        self.text.bind("<ButtonRelease-1>", self._on_cursor_move)
        self.text.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows/macOS
        self.text.bind("<Button-4>", self._on_mouse_wheel)    # Linux scroll up
        self.text.bind("<Button-5>", self._on_mouse_wheel)    # Linux scroll down

        # Auto-pair characters
        for ch in ("(", "[", "{", '"', "'"):
            self.text.bind(ch, self._handle_auto_pair)

        # Indentation helpers
        self.text.bind("<Tab>", self._handle_tab)
        self.text.bind("<Return>", self._handle_return)

    # -------------------- SCROLLING --------------------

    def _on_textscroll(self, *args):
        # Sync line numbers with text vertical scroll
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

    # -------------------- EVENT HANDLERS --------------------

    def _on_key_release(self, event=None):
        if self._processing_event:
            return
        self._processing_event = True
        
        self.update_line_numbers()
        self.highlight_syntax()
        
        self._processing_event = False

    def _on_cursor_move(self, event=None):
        if self._processing_event:
            return
        self._processing_event = True
        
        self.update_line_numbers()
        
        self._processing_event = False

    def _handle_auto_pair(self, event):
        """
        Auto-insert closing pair for (), [], {}, "", ''.
        If text is selected, wrap the selection.
        """
        text = event.widget
        char = event.char
        pairs = {"(": ")", "[": "]", "{": "}", '"': '"', "'": "'"}
        closing = pairs.get(char)
        if not closing:
            return

        try:
            # If there's a selection, wrap it
            sel_start = text.index("sel.first")
            sel_end = text.index("sel.last")
            selected = text.get(sel_start, sel_end)
            text.delete(sel_start, sel_end)
            text.insert(sel_start, char + selected + closing)
            text.mark_set("insert", f"{sel_start}+{len(char + selected + closing)}c")
        except tk.TclError:
            # No selection: insert pair and move cursor between
            text.insert("insert", char + closing)
            text.mark_set("insert", "insert-1c")

        return "break"

    def _handle_tab(self, event):
        # Insert 4 spaces instead of losing focus
        event.widget.insert("insert", "    ")
        return "break"

    def _handle_return(self, event):
        """
        Keep current indentation on new line.
        """
        text = event.widget
        line_start = text.index("insert linestart")
        line_text = text.get(line_start, "insert")
        indent = re.match(r"\s*", line_text).group(0)
        text.insert("insert", "\n" + indent)
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

    def highlight_syntax(self, event=None):
        """
        Basic GLSL syntax highlighting.
        For simplicity, re-highlights the entire buffer on each key release.
        """
        text = self.text

        # Remove all previous tags
        for tag in ("keyword", "type", "builtin", "number", "string", "comment"):
            text.tag_remove(tag, "1.0", "end")

        code = text.get("1.0", "end-1c")

        # Apply in an order so comments/strings override keyword colors
        order = ["keyword", "type", "builtin", "number", "string", "comment"]

        for tag in order:
            pattern = self.syntax_patterns[tag]
            for match in pattern.finditer(code):
                start, end = match.span()
                start_index = f"1.0+{start}c"
                end_index = f"1.0+{end}c"
                text.tag_add(tag, start_index, end_index)


if __name__ == "__main__":
    app = GLSLEditor()
    app.mainloop()