import time
from OpenGL.GL import *

def rendering_pass( #🤯
    st,
    shader,
    display_shader,
    vao,
    display_vao,
    uniform_locs,
    scaled_rendering_width,
    scaled_rendering_height,
    rendering_width,
    rendering_height,
    width,
    height,
    panel_width,
    menu_bar_height,
    frame_count,
    max_frames,
    start_time,
    drag_position,
    drag_rot_position,
    accumulation_fbos,
    accumulation_textures,
    accumulation_width,
    accumulation_height,
    current_accum_index,
    setup_accumulation_buffer,
    bind_sprite_textures,
    set_move_pos_uniform,
    set_move_rot_uniform
):
    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)

    # --- Setup accumulation buffer if using cycles shader ---
    use_accumulation = 0
    accbuffer_output = False

    accbuffer_output, \
    scaled_rendering_width, scaled_rendering_height, \
    accumulation_fbos, accumulation_textures, \
    accumulation_width, accumulation_height = setup_accumulation_buffer(
        scaled_rendering_width, scaled_rendering_height,
        accumulation_fbos, accumulation_textures,
        accumulation_width, accumulation_height
    )

    if st.shader_choice == 1:  # cycles.glsl
        if accbuffer_output:
            use_accumulation = 1

    # --- RENDER TO ACCUMULATION BUFFER ---
    if shader is not None and st.shader_choice == 1 and use_accumulation == 1:
        write_buffer = current_accum_index
        read_buffer = 1 - current_accum_index

        glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[write_buffer])
        glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)

        if frame_count == 0:
            glClear(GL_COLOR_BUFFER_BIT)

        if frame_count < max_frames:
            glUseProgram(shader)

            if uniform_locs is not None:
                current_time_uniform = time.time() - start_time
                glUniform1f(uniform_locs['time'], current_time_uniform)
                glUniform2f(uniform_locs['resolution'], scaled_rendering_width, scaled_rendering_height)
                glUniform2f(uniform_locs['viewportOffset'], 0.0, 0.0)
                glUniform1f(uniform_locs['camYaw'], st.cam_yaw)
                glUniform1f(uniform_locs['camPitch'], st.cam_pitch)
                glUniform1f(uniform_locs['radius'], st.cam_radius)
                glUniform3f(uniform_locs['CamOrbit'], *st.cam_orbit)
                glUniform1i(uniform_locs['frameIndex'], frame_count)
                glUniform1i(uniform_locs['maxFrames'], max_frames)

                set_move_pos_uniform(shader, uniform_locs, drag_position)
                set_move_rot_uniform(shader, uniform_locs, drag_rot_position)

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, accumulation_textures[read_buffer])
                glUniform1i(uniform_locs['accumulationTexture'], 0)
                glUniform1i(uniform_locs['useAccumulation'], 1)

                glUniform3f(uniform_locs['col_sky_top'], *st.sky_top_color)
                glUniform3f(uniform_locs['col_sky_bottom'], *st.sky_bottom_color)
                glUniform3f(uniform_locs['LightDir'], *st.LightDir)

            bind_sprite_textures(uniform_locs, st.sprites_array)
            glBindVertexArray(vao)
            glDrawArrays(GL_QUADS, 0, 4)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, width, height)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, accumulation_textures[write_buffer])

        glUseProgram(display_shader)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, accumulation_textures[write_buffer])
        glUniform1i(glGetUniformLocation(display_shader, "renderTexture"), 0)

        glUniform1i(
            glGetUniformLocation(display_shader, "isAccumulation"),
            1 if frame_count >= max_frames else 0
        )

        glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
        bind_sprite_textures(uniform_locs, st.sprites_array)
        glBindVertexArray(display_vao)
        glDrawArrays(GL_QUADS, 0, 4)
        glBindVertexArray(0)

        glViewport(0, 0, width, height)

        return (
            scaled_rendering_width,
            scaled_rendering_height,
            accumulation_fbos,
            accumulation_textures,
            accumulation_width,
            accumulation_height,
            1 - current_accum_index,
            use_accumulation
        )

    # --- RENDER DIRECTLY ---
    elif shader is not None:
        glUseProgram(shader)

        if uniform_locs is not None:
            current_time_uniform = time.time() - start_time
            glUniform1f(uniform_locs['time'], current_time_uniform)
            glUniform2f(uniform_locs['resolution'], rendering_width, rendering_height)
            glUniform2f(uniform_locs['viewportOffset'], float(panel_width), float(menu_bar_height))
            glUniform1f(uniform_locs['camYaw'], st.cam_yaw)
            glUniform1f(uniform_locs['camPitch'], st.cam_pitch)
            glUniform1f(uniform_locs['radius'], st.cam_radius)
            glUniform3f(uniform_locs['CamOrbit'], *st.cam_orbit)
            glUniform1i(uniform_locs['frameIndex'], 0)
            glUniform1i(uniform_locs['useAccumulation'], 0)

            set_move_pos_uniform(shader, uniform_locs, drag_position)
            set_move_rot_uniform(shader, uniform_locs, drag_rot_position)

            glUniform3f(uniform_locs['col_sky_top'], *st.sky_top_color)
            glUniform3f(uniform_locs['col_sky_bottom'], *st.sky_bottom_color)
            glUniform1i(uniform_locs['grid_enabled'], st.GridEnabled)
            glUniform3f(uniform_locs['LightDir'], *st.LightDir)

        if rendering_width > 0 and rendering_height > 0:
            glViewport(panel_width, menu_bar_height, rendering_width, rendering_height)
            glBindVertexArray(vao)
            bind_sprite_textures(uniform_locs, st.sprites_array)
            glDrawArrays(GL_QUADS, 0, 4)

        glViewport(0, 0, width, height)

    return (
        scaled_rendering_width,
        scaled_rendering_height,
        accumulation_fbos,
        accumulation_textures,
        accumulation_width,
        accumulation_height,
        current_accum_index,
        use_accumulation
    )
