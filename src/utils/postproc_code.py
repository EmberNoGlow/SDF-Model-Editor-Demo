def generate_postproc_code(Sprites):
    uni_code = ""
    code = ""
    for spr in Sprites:
        code += spr.generate_spr_code()
        uni_code += spr.generate_uniforms_code()
    return code, uni_code
