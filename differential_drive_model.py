"""
Differential drive robot model for path tracking simulation.

State: [x, y, yaw, v, omega]
Input: [a, alpha]  (linear acceleration, angular acceleration)
"""

import math
import numpy as np

# ---------- parameters ----------
dt = 0.05          # time step [s]
wheel_radius = 0.1 # [m]
axle_width = 0.3   # [m]

v_max = 0.6                    # max linear velocity [m/s]
omega_max = np.deg2rad(90.0)   # max angular velocity [rad/s]

accel_max = 2.0                    # max linear acceleration [m/s^2]
omega_accel_max = np.deg2rad(90.0) # max angular acceleration [rad/s^2]


class State:
    """Differential drive robot state."""
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.omega = omega

    def copy(self):
        return State(self.x, self.y, self.yaw, self.v, self.omega)


def update(state, a, alpha):
    """
    Advance the state by one time-step using the differential-drive kinematics.

    Parameters
    ----------
    state : State
    a     : float – linear acceleration  [m/s^2]
    alpha : float – angular acceleration [rad/s^2]

    Returns
    -------
    state : State (mutated in-place and returned)
    """
    a     = np.clip(a,     -accel_max,     accel_max)
    alpha = np.clip(alpha, -omega_accel_max, omega_accel_max)

    state.v     = np.clip(state.v     + a     * dt, -v_max,     v_max)
    state.omega = np.clip(state.omega + alpha * dt, -omega_max, omega_max)

    state.x   += state.v * math.cos(state.yaw) * dt
    state.y   += state.v * math.sin(state.yaw) * dt
    state.yaw  = pi_2_pi(state.yaw + state.omega * dt)

    return state


def pi_2_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
