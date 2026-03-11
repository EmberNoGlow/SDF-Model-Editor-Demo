from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

def init_vao_vbo():
    vertices = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    vertices = (GLfloat * len(vertices))(*vertices)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    
    try:
        # Quad with texture coordinates for displaying the rendered texture
        quad_vertices = [
            # positions   # tex coords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ]
        quad_vertices = (GLfloat * len(quad_vertices))(*quad_vertices)
        
        display_vao = glGenVertexArrays(1)
        glBindVertexArray(display_vao)
        display_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, display_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(quad_vertices) * 4, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)  # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))  # tex coord
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
    except Exception as e:
        print(f"Warning: Could not create display shader: {e}")
        print("Falling back to direct rendering (resolution scale may not work correctly)")
    
    # Simple shader for displaying texture
    display_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
    """
    
    display_fragment_shader = """
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D renderTexture;
    uniform int isAccumulation;

    void main() {
        vec4 tex = texture(renderTexture, TexCoord);

        if (isAccumulation == 1) {
            FragColor = vec4(tex.rgb, 1.0);
        } else {
            FragColor = vec4(tex.rgb, 1.0);
        }
    }
    """
    
    display_shader = None
    display_vao = None
    display_vbo = None
    
    try:
        display_shader = compileProgram(
            compileShader(display_vertex_shader, GL_VERTEX_SHADER),
            compileShader(display_fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Quad with texture coordinates for displaying the rendered texture
        quad_vertices = [
            # positions   # tex coords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ]
        quad_vertices = (GLfloat * len(quad_vertices))(*quad_vertices)
        
        display_vao = glGenVertexArrays(1)
        glBindVertexArray(display_vao)
        display_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, display_vbo)
        glBufferData(GL_ARRAY_BUFFER, len(quad_vertices) * 4, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, None)  # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))  # tex coord
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
    except Exception as e:
        print(f"Warning: Could not create display shader: {e}")
        print("Falling back to direct rendering (resolution scale may not work correctly)")
        display_shader = None
    
    return vao, vbo, display_vao, display_vbo, display_shader
