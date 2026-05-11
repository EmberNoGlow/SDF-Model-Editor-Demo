"""Main rendering pipeline."""
import time
from OpenGL.GL import *
from src.app.data.states import st
from src.app.data.consts import cn
from .fbo import (
    setup_framebuffer,
    clear_accumulation_fbos,
)
from ..core.SpriteObject import bind_sprite_textures
from ..core.classes.uniform_managers.set_specific_uniforms_helper import set_move_pos_uniform, set_move_rot_uniform


def render_scene_main(window, scene_builder, camera, vao, vbo, display_vao, display_vbo, display_shader):
    """Main rendering pipeline orchestration."""
    from .window import handle_window_resize, apply_resolution_scale
    from .render_pass import rendering_pass
    
    width, height, menu_bar_height, panel_width, rendering_width, rendering_height = (
        handle_window_resize(window)
    )

    apply_resolution_scale(rendering_width, rendering_height)

    use_accumulation = rendering_pass(
        st,
        st.shader,
        display_shader,
        vao,
        display_vao,
        st.uniform_locs,
        rendering_width,
        rendering_height,
        width,
        height,
        panel_width,
        menu_bar_height,
        clear_accumulation_fbos,
        bind_sprite_textures,
        set_move_pos_uniform,
        set_move_rot_uniform,
    )

    return (
        use_accumulation,
        width,
        height,
        menu_bar_height,
        panel_width,
        rendering_width,
        rendering_height,
    )


def render_framebuffer_scaled(
    width,
    height,
    menu_bar_height,
    panel_width,
    rendering_width,
    rendering_height,
    display_shader,
):
    """Render to framebuffer at scaled resolution."""
    if st.shader is None or st.shader_choice == 1:
        return None, None, None, None, None

    if display_shader is None or st.resolution_scale == 1.0:
        return None, None, None, None, None

    (
        framebuffer_output,
        st.scaled_rendering_width,
        st.scaled_rendering_height,
        fbo,
        render_texture,
        fbo_width,
        fbo_height,
    ) = setup_framebuffer(
        st.scaled_rendering_width,
        st.scaled_rendering_height,
        st.fbo,
        st.render_texture,
        st.fbo_width,
        st.fbo_height,
    )

    if not framebuffer_output:
        return None, None, None, None, None

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, st.scaled_rendering_width, st.scaled_rendering_height)
    glClear(GL_COLOR_BUFFER_BIT)

    _render_to_framebuffer(
        display_shader,
        render_texture,
        width,
        height,
        rendering_width,
        rendering_height,
        panel_width,
        menu_bar_height,
    )

    return fbo, render_texture, st.fbo_width, st.fbo_height, framebuffer_output


def _render_to_framebuffer(
    display_shader,
    render_texture,
    width,
    height,
    rendering_width,
    rendering_height,
    panel_width,
    menu_bar_height,
):
    """Internal: render scaled scene to framebuffer texture."""
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, width, height)

    glUseProgram(display_shader)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, render_texture)
    glUniform1i(glGetUniformLocation(display_shader, "renderTexture"), 0)

    glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
    glBindVertexArray(st.display_vao)
    glDrawArrays(GL_QUADS, 0, 4)
    glBindVertexArray(0)

    glViewport(0, 0, width, height)


def render_direct_screen(
    width, height, menu_bar_height, panel_width, rendering_width, rendering_height
):
    """Direct rendering to screen when scale is 1.0."""
    if st.shader is None:
        return

    glUseProgram(st.shader)

    if st.uniform_locs is not None:
        current_time_uniform = time.time() - st.start_time
        glUniform1f(st.uniform_locs["time"], current_time_uniform)
        glUniform2f(st.uniform_locs["resolution"], rendering_width, rendering_height)
        glUniform2f(
            st.uniform_locs["viewportOffset"],
            float(panel_width),
            float(menu_bar_height),
        )
        glUniform1f(st.uniform_locs["camYaw"], st.cam_yaw)
        glUniform1f(st.uniform_locs["camPitch"], st.cam_pitch)
        glUniform1f(st.uniform_locs["radius"], st.cam_radius)
        glUniform3f(
            st.uniform_locs["CamOrbit"],
            st.cam_orbit[0],
            st.cam_orbit[1],
            st.cam_orbit[2],
        )
        set_move_pos_uniform(st.shader, st.uniform_locs, st.drag_position)
        set_move_rot_uniform(st.shader, st.uniform_locs, st.drag_rot_position)

    if rendering_width > 0 and rendering_height > 0:
        glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
        glBindVertexArray(st.vao)
        bind_sprite_textures(st.uniform_locs, st.sprites_array)
        glDrawArrays(GL_QUADS, 0, 4)

    glViewport(0, 0, width, height)