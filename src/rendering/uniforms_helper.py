from OpenGL.GL import glGetUniformLocation


def get_uniform_locations(shader_program, sprites_array=None):
    # Get all uniform locations for the shader program.
    uniforms = {
        "time": glGetUniformLocation(shader_program, "time"),
        "resolution": glGetUniformLocation(shader_program, "resolution"),
        "viewportOffset": glGetUniformLocation(shader_program, "viewportOffset"),
        "camYaw": glGetUniformLocation(shader_program, "camYaw"),
        "camPitch": glGetUniformLocation(shader_program, "camPitch"),
        "radius": glGetUniformLocation(shader_program, "radius"),
        "CamOrbit": glGetUniformLocation(shader_program, "CamOrbit"),
        "frameIndex": glGetUniformLocation(shader_program, "frameIndex"),
        "accumulationTexture": glGetUniformLocation(
            shader_program, "accumulationTexture"
        ),
        "useAccumulation": glGetUniformLocation(shader_program, "useAccumulation"),
        "col_sky_top": glGetUniformLocation(shader_program, "SkyColorTop"),
        "col_sky_bottom": glGetUniformLocation(shader_program, "SkyColorBottom"),
        "grid_enabled": glGetUniformLocation(shader_program, "GridEnabled"),
        "move_pos": glGetUniformLocation(shader_program, "MovePos"),
        "move_rot": glGetUniformLocation(shader_program, "MoveRot"),
        "maxFrames": glGetUniformLocation(shader_program, "MaxFrames"),
        "LightDir": glGetUniformLocation(shader_program, "LightDir"),
    }

    # Register sprite sampler uniforms (dynamic)
    # sprites_array is in outer scope; it's the list of Sprite objects used for postprocessing
    try:
        for spr in sprites_array:
            # Use sampler name string as key, store location (may be -1 if unused)
            uniforms[spr.SprTexture] = glGetUniformLocation(
                shader_program, spr.SprTexture
            )
    except Exception:
        # If sprites_array is not defined yet, skip (defensive)
        pass

    return uniforms
