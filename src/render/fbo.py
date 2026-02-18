from OpenGL.GL import *

def clear_accumulation_fbos(accumulation_fbos,scaled_rendering_width,scaled_rendering_height):
    # Reset accumulation buffers so no stale data is read later
    if accumulation_fbos[0] is not None and accumulation_fbos[1] is not None:
        # store current viewport to restore later if you need; here we assume you will set proper viewport when drawing
        glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[0])
        glViewport(0, 0, scaled_rendering_width, scaled_rendering_height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, accumulation_fbos[1])
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

def setup_framebuffer(width, height, fbo, render_texture, fbo_width, fbo_height):
    """Create or update framebuffer for rendering at scaled resolution."""
    
    # Only recreate if size changed
    if fbo is None or fbo_width != width or fbo_height != height:
        # Delete old framebuffer if it exists
        if fbo is not None:
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [render_texture])
        
        # Create framebuffer
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        # Create texture to render to
        render_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, render_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Attach texture to framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_texture, 0)
        
        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("Error: Framebuffer is not complete!")
            return False, width, height, fbo, render_texture, fbo_width, fbo_height
        
        fbo_width = width
        fbo_height = height
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return True, width, height, fbo, render_texture, fbo_width, fbo_height
    return True, width, height, fbo, render_texture, fbo_width, fbo_height



def setup_accumulation_buffer(width, height, accumulation_fbos, accumulation_textures, accumulation_width, accumulation_height):
    """Create or update accumulation buffers (double-buffered) for temporal filtering."""

    # If already set up for this size and both buffers exist, nothing to do.
    if (accumulation_width == width and accumulation_height == height and
            accumulation_fbos[0] is not None and accumulation_fbos[1] is not None and
            accumulation_textures[0] is not None and accumulation_textures[1] is not None):
        return True, width, height, accumulation_fbos, accumulation_textures, accumulation_width, accumulation_height

    # Delete old buffers/textures if they exist
    for i in range(2):
        if accumulation_fbos[i] is not None:
            try:
                glDeleteFramebuffers(1, [accumulation_fbos[i]])
            except Exception:
                pass
            accumulation_fbos[i] = None
        if accumulation_textures[i] is not None:
            try:
                glDeleteTextures(1, [accumulation_textures[i]])
            except Exception:
                pass
            accumulation_textures[i] = None

    # Create two FBO/texture pairs
    for i in range(2):
        fbo_i = glGenFramebuffers(1)
        tex_i = glGenTextures(1)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_i)
        glBindTexture(GL_TEXTURE_2D, tex_i)

        # Allocate floating point RGBA texture for accumulation
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Attach texture to the framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_i, 0)

        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print(f"Error: Accumulation framebuffer {i} is not complete!")
            # Clean up what we created so far
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            for j in range(2):
                if accumulation_fbos[j] is not None:
                    try:
                        glDeleteFramebuffers(1, [accumulation_fbos[j]])
                    except Exception:
                        pass
                    accumulation_fbos[j] = None
                if accumulation_textures[j] is not None:
                    try:
                        glDeleteTextures(1, [accumulation_textures[j]])
                    except Exception:
                        pass
                    accumulation_textures[j] = None
            return False, width, height, accumulation_fbos, accumulation_textures, accumulation_width, accumulation_height

        # Store handles
        accumulation_fbos[i] = fbo_i
        accumulation_textures[i] = tex_i

    # Update size bookkeeping and unbind framebuffer
    accumulation_width = width
    accumulation_height = height
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return True, width, height, accumulation_fbos, accumulation_textures, accumulation_width, accumulation_height
