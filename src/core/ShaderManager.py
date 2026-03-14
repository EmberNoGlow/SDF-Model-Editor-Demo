from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import hashlib

from ..rendering.uniforms_helper import get_uniform_locations
from ..utils.postproc_code import generate_postproc_code
from ..core.classes.save_load_helpers.ShaderLoader import load_shader_code
from ..app.data.consts import cn

class ShaderManager:
    def __init__(self, vertex_shader_src: str, sdf_library_src: str, state):
        """
        vertex_shader_src: full vertex shader source string
        sdf_library_src: SDF library GLSL string
        state: your global 'st' object (holds shader_names, shader_choice, sprites_array, etc.)
        """
        self.vertex_shader_src = vertex_shader_src
        self.sdf_library_src = sdf_library_src
        self.st = state  # st


    # --- Internal helpers ---


    def _build_fragment_shader(self, scene_builder) -> str:
        """Build final fragment shader source from templates + scene + postproc."""
        scene_code = scene_builder.generate_raymarch_code()
        postproc_code, additional_uniforms = generate_postproc_code(self.st.sprites_array)

        selected_fragment_shader = load_shader_code(self.st.shader_names[self.st.shader_choice])

        fragment_shader = selected_fragment_shader
        fragment_shader = fragment_shader.replace("{SDF_LIBRARY}", self.sdf_library_src)
        fragment_shader = fragment_shader.replace("{SCENE_CODE}", scene_code)
        fragment_shader = fragment_shader.replace("{FOV_ANGLE_VAL}", str(cn['FOV_ANGLE']))
        fragment_shader = fragment_shader.replace("{POSTPROC}", postproc_code)
        fragment_shader = fragment_shader.replace("{ADDITIONAL_UNIFORMS}", additional_uniforms)
        fragment_shader = fragment_shader.replace("{ADDITIONAL_SCENE_CODE}", self.st.additional_scene_code)

        return fragment_shader

    def _get_shader_hash(self, fragment_shader_src: str) -> str:
        """Generate a hash of the current shader code for caching."""
        shader_code = f"{self.vertex_shader_src}\n{fragment_shader_src}\n{self.st.shader_names[self.st.shader_choice]}"
        return hashlib.md5(shader_code.encode("utf-8")).hexdigest()

    def _compile_program(self, vertex_src: str, fragment_src: str):
        """Compile and link a GL program, return (program, uniform_locations_dict)."""
        program = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        uniforms = get_uniform_locations(program)
        return program, uniforms

    # ---- Public API ----

    def get_or_compile(self, scene_builder):
        """
        Returns (shader_program, uniform_locations_dict).
        Uses st.shader_cache for caching and st.shader_compile_error for error reporting.
        """
        # Build fragment shader from current scene + settings
        fragment_shader_src = self._build_fragment_shader(scene_builder)

        # Hash for cache
        shader_hash = self._get_shader_hash(fragment_shader_src)

        # Cache hit
        if shader_hash in self.st.shader_cache:
            shader_program, uniforms = self.st.shader_cache[shader_hash]
            self.st.shader_compile_error = None
            return shader_program, uniforms

        # Cache miss → compile
        try:
            shader_program, uniforms = self._compile_program(
                self.vertex_shader_src,
                fragment_shader_src
            )
            # Store in cache
            self.st.shader_cache[shader_hash] = (shader_program, uniforms)
            self.st.shader_compile_error = None
            return shader_program, uniforms

        except Exception as e:
            self.st.shader_compile_error = str(e)
            print(f"Shader compilation error: {e}")
            return None, None

    def invalidate_cache(self):
        """Optional: delete all GL programs in cache and clear it."""
        for prog, _uniforms in self.st.shader_cache.values():
            try:
                glDeleteProgram(prog)
            except Exception:
                pass
        self.st.shader_cache.clear()
