from OpenGL.GL import glGetUniformLocation, glUniform3f


def set_move_pos_uniform(shader_program, uniform_locs, pos):
    """
    Safely set the MovePos uniform. If the cached uniform location is missing (-1 or None),
    query it dynamically and cache it. Only call glUniform if the location exists.
    """
    if uniform_locs is None or shader_program is None:
        return
    move_key = "move_pos"
    loc = uniform_locs.get(move_key, None)
    if loc is None or loc == -1:
        # Query the active program for the location (this is safe and will return -1 if not declared)
        loc = glGetUniformLocation(shader_program, "MovePos")
        uniform_locs[move_key] = loc
    if loc != -1:
        glUniform3f(loc, float(pos[0]), float(pos[1]), float(pos[2]))


def set_move_rot_uniform(shader_program, uniform_locs, rot):
    """
    Safely set the MoveRot uniform. If the cached uniform location is missing (-1 or None),
    query it dynamically and cache it. Only call glUniform if the location exists.
    """
    if uniform_locs is None or shader_program is None:
        return
    move_key = "move_rot"
    loc = uniform_locs.get(move_key, None)
    if loc is None or loc == -1:
        # Query the active program for the location (this is safe and will return -1 if not declared)
        loc = glGetUniformLocation(shader_program, "MoveRot")
        uniform_locs[move_key] = loc
    if loc != -1:
        glUniform3f(loc, float(rot[0]), float(rot[1]), float(rot[2]))
