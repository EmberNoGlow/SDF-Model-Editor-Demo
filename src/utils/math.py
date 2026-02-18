import math

def orbital_to_cartesian(_yaw, _pitch, _radius):
    yaw_rad = _yaw
    pitch_rad = _pitch

    x = _radius * math.cos(pitch_rad) * math.cos(yaw_rad)
    y = _radius * math.sin(pitch_rad)                    
    z = _radius * math.cos(pitch_rad) * math.sin(yaw_rad)

    return (x, y, z)