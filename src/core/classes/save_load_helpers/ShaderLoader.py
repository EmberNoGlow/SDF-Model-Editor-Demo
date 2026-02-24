def load_shader_code(file_path):
    # Load shader code from a file and return it as a string.
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Shader file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading shader file {file_path}: {e}")


# Load shader files with error handling
def load_shaders():
    # Return: vertex_shader, fragment_shader_template, sdf_library
    try:
        # Vertex shader source code
        vertex_shader = load_shader_code("shaders/vertex_shader.glsl")
        
        # SDF Library
        sdf_library = load_shader_code("shaders/sdf_library.glsl")
        
        # Fragment shader template
        fragment_shader_template = load_shader_code("shaders/fragment/template.glsl")
        
        return vertex_shader, fragment_shader_template, sdf_library
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading shader files: {e}")
        print("Please ensure all shader files are present in the project directory.")
        exit(1)
        return