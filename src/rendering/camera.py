import math

class Camera:
    def __init__(self):
        self.cam_pitch = 0.0
        self.cam_yaw = 0.0

        self.cam_pan_x = 0.0
        self.cam_pan_y = 0.0


        self.setup()


    def setup(self):
        self.forward_x = math.cos(self.cam_pitch) * math.sin(self.cam_yaw)
        self.forward_y = math.sin(self.cam_pitch)
        self.forward_z = math.cos(self.cam_pitch) * math.cos(self.cam_yaw)


        self.right_x = math.cos(self.cam_yaw)
        self.right_y = 0
        self.right_z = -math.sin(self.cam_yaw)


        self.up_x = self.forward_y * self.right_z - self.forward_z * self.right_y
        self.up_y = self.forward_z * self.right_x - self.forward_x * self.right_z
        self.up_z = self.forward_x * self.right_y - self.forward_y * self.right_x


        self.orbit_center_offset_x = self.cam_pan_x * self.right_x + self.cam_pan_y * self.up_x
        self.orbit_center_offset_y = self.cam_pan_x * self.right_y + self.cam_pan_y * self.up_y
        self.orbit_center_offset_z = self.cam_pan_x * self.right_z + self.cam_pan_y * self.up_z

        self.cam_orbit = (
            self.orbit_center_offset_z, # Yoow! (Correctly)
            self.orbit_center_offset_y,
            self.orbit_center_offset_x
        )

    def update(self, target_yaw, target_pitch, target_pan_y, target_pan_x, factor):
        # Update angles
        self.cam_yaw += (target_yaw - self.cam_yaw) * factor
        self.cam_pitch += (target_pitch - self.cam_pitch) * factor
        
        # Update pan
        self.cam_pan_y += (target_pan_y - self.cam_pan_y) * factor
        self.cam_pan_x -= (target_pan_x + self.cam_pan_x) * factor
        
        # Re-calc
        self.setup()

        return self.cam_yaw, self.cam_pitch, self.cam_pan_y, self.cam_pan_x