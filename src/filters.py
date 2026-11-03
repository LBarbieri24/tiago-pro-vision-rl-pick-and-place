import numpy as np
import math


class ExponentialMovingAverage:
    def __init__(self, init_pos, alpha=0.20):
        self.x = np.array([init_pos, 0.0], dtype=np.float32)
        self.alpha = alpha

    def predict(self, dt, q_scale=1.0):
        # EMA has no motion model or state propagation, prediction is a no-op (holds the last state)
        pass

    def update(self, z_pos, r_pos=None):  # r_pos ignored but accepted for API compatibility
        # standard EMA formula
        self.x[0] = self.alpha * z_pos + (1.0 - self.alpha) * self.x[0]
        # Velocity is not tracked, set to 0
        self.x[1] = 0.0
        return True


class ExponentialMovingAverageYaw:
    def __init__(self, init_yaw, alpha=0.20):
        # Represent yaw as a unit vector [cos(yaw), sin(yaw)] to handle wrap-around gracefully
        self.x = np.array([math.cos(init_yaw), math.sin(init_yaw)], dtype=np.float32)
        self.alpha = alpha

    def predict(self, dt, q_scale=1.0):
        pass

    def update(self, z_yaw, r_yaw=None):
        z = np.array([math.cos(z_yaw), math.sin(z_yaw)], dtype=np.float32)
        self.x = self.alpha * z + (1.0 - self.alpha) * self.x
        
        # Normalize the unit vector to keep it valid
        norm = np.linalg.norm(self.x)
        if norm > 1e-5:
            self.x = self.x / norm
        return True

    def get_yaw(self):
        return float(math.atan2(self.x[1], self.x[0]))


class ExponentialMovingAverage1D:
    def __init__(self, init_val, alpha=0.20):
        self.x = float(init_val)
        self.alpha = alpha

    def predict(self, dt, q_scale=1.0):
        pass

    def update(self, z_val, r_val=None):
        self.x = self.alpha * z_val + (1.0 - self.alpha) * self.x
        return True

