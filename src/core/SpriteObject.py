from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

class Sprite:
    """
    A structure to hold the parameters defining a sprite,
    including its projection plane and texture information.
    """
    def __init__(self,
        planePoint, planeNormal,
        planeWidth: float, planeHeight: float,
        SprTexture=None, uvSize=(1.0, 1.0),
        Alpha: float = 1.0, LOD: float = 0.0
    ):
        # Store the data as instance attributes
        self.planePoint = list(planePoint)
        self.planeNormal = list(planeNormal)
        self.planeWidth = float(planeWidth)
        self.planeHeight = float(planeHeight)

        # Sampler name (used in shader code). If none provided, use a stable fallback;
        # the scene loader usually passes an explicit name, but this keeps things robust.
        self.SprTexture = SprTexture if SprTexture else f"sprTex_{id(self)}"
        self.uvSize = list(uvSize)
        self.Alpha = float(Alpha)
        self.LOD = float(LOD)

        # GL texture handle (created when loading image from disk). None => not loaded
        self.texture_id = None
        self.tex_size = (0, 0)

        # Optional: store path so we can restore textures on load if desired
        self.texture_path = None

    def to_dict(self):
        """
        Return a serializable dict representation. We avoid saving GL handles.
        We store the sampler name (SprTexture) so the shader uniform name remains stable
        across save/load, and also optional texture_path if available so the user can
        reload textures on scene load.
        """
        return {
            "planePoint": list(self.planePoint),
            "planeNormal": list(self.planeNormal),
            "planeWidth": float(self.planeWidth),
            "planeHeight": float(self.planeHeight),
            "SprTexture": self.SprTexture,
            "uvSize": list(self.uvSize),
            "Alpha": float(self.Alpha),
            "LOD": float(self.LOD),
            "texture_path": self.texture_path if self.texture_path else None,
            "tex_size": [int(self.tex_size[0]), int(self.tex_size[1])]
        }

    def generate_spr_code(self):
        # NOTE: This injects literal values into the shader. The sampler is passed
        # as the identifier self.SprTexture (must match the uniform declared).
        code = (
            f"col = Sprite("
            f"ro,rd,"
            f"vec3({self.planePoint[0]},{self.planePoint[1]},{self.planePoint[2]}),"
            f"vec3({self.planeNormal[0]},{self.planeNormal[1]},{self.planeNormal[2]}),"
            f"{self.planeWidth:.6f},"
            f"-{self.planeHeight:.6f},"
            f"col, d,"
            f"{self.SprTexture},"  # sampler uniform name (no quotes)
            f"vec2({self.uvSize[0]:.6f},{self.uvSize[1]:.6f}),"
            f"{self.Alpha:.6f},"
            f"{self.LOD:.6f}"
            f");\n"
        )

        return code

    def generate_uniforms_code(self):
        # Return a sampler declaration using the sampler name
        return f"uniform sampler2D {self.SprTexture};\n"

    def load_texture_from_file(self, filepath):
        """Load an image from disk and upload to GL as an RGBA texture. Returns True on success."""
        try:
            img = Image.open(filepath).convert("RGBA")
            w, h = img.size
            img_data = img.tobytes("raw", "RGBA", 0, -1)

            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            # Upload
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)

            # If an old texture existed, delete it
            if self.texture_id:
                try:
                    glDeleteTextures(1, [self.texture_id])
                except Exception:
                    pass

            self.texture_id = tex
            self.tex_size = (w, h)
            # Persist the file path so scene saves can include it
            self.texture_path = filepath
            return True
        except Exception as e:
            print(f"Failed to load sprite texture '{filepath}': {e}")
            return False

    def free_texture(self):
        if self.texture_id:
            try:
                glDeleteTextures(1, [self.texture_id])
            except Exception:
                pass
            self.texture_id = None
            self.tex_size = (0, 0)
            # keep texture_path (we may want to attempt reload next time)


def bind_sprite_textures(uniforms, sprites_array):
    """
    Bind loaded sprite textures to texture units and upload the sampler uniform indices.
    Assumes texture unit 0 may be used for accumulation/render targets, so start at unit 1.
    """
    base_unit = 1
    for i, spr in enumerate(sprites_array):
        loc = uniforms.get(spr.SprTexture, -1) if uniforms else -1
        unit = base_unit + i
        if spr.texture_id is not None and loc is not None and loc != -1:
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, spr.texture_id)
            # Tell shader which texture unit to sample from
            glUniform1i(loc, unit)
        else:
            # If texture not loaded, bind 0 to keep behavior stable
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, 0)
            if loc != -1:
                glUniform1i(loc, unit)
    # restore active texture to 0
    glActiveTexture(GL_TEXTURE0)