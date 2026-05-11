"""OpenGL resource cleanup helpers."""

from OpenGL.GL import (
    glDeleteBuffers,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteTextures,
    glDeleteVertexArrays,
)


def _safe_delete(delete_fn, *args):
    """Best-effort delete to avoid shutdown crashes."""
    try:
        delete_fn(*args)
    except Exception:
        pass


def cleanup_resources(fbo, render_texture, display_shader, display_vao, display_vbo, vao, vbo):
    """Release OpenGL resources created by the app."""
    if fbo:
        _safe_delete(glDeleteFramebuffers, 1, [fbo])
    if render_texture:
        _safe_delete(glDeleteTextures, 1, [render_texture])
    if display_shader:
        _safe_delete(glDeleteProgram, display_shader)
    if display_vao:
        _safe_delete(glDeleteVertexArrays, 1, [display_vao])
    if display_vbo:
        _safe_delete(glDeleteBuffers, 1, [display_vbo])
    if vao:
        _safe_delete(glDeleteVertexArrays, 1, [vao])
    if vbo:
        _safe_delete(glDeleteBuffers, 1, [vbo])
