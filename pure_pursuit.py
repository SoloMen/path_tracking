"""
Improved Pure Pursuit path tracking for differential drive.

Key improvements over original:
  1. Reset target_ind to nearest point EVERY step (critical bug fix)
  2. Separate lookahead index from target index
  3. Cross-track error feedback for lateral correction
  4. Curvature-adaptive speed reduction on sharp turns
  5. Larger, smoother lookahead distance
"""

import math
import numpy as np
from differential_drive_model import State, update, pi_2_pi, dt
from common import (
    generate_test_path, calc_target_index, calc_path_heading,
    animate_and_save, speed_control, run_simulation, omega_max,
)

# ── tuning ──────────────────────────────────────────────────────────
Lf_min   = 0.8       # minimum look-ahead [m]
k_lf     = 3.0       # look-ahead = k_lf * |v|  → at 0.5 m/s → 1.5 m
K_omega  = 3.0       # angular-velocity P gain
K_cte    = 1.0       # cross-track error feedback gain
Lf_head  = 1.0       # heading-reference look-ahead (for cross-track)

TARGET_SPEED = 0.5
MAX_TIME     = 300.0
GOAL_TOL     = 0.5


# ── helpers ─────────────────────────────────────────────────────────
def _advance_index(cx, cy, target_ind, Lf):
    """Walk forward Lf metres along path from target_ind. Returns index."""
    dist = 0.0
    while dist < Lf and (target_ind + 1) < len(cx):
        seg = math.hypot(cx[target_ind + 1] - cx[target_ind],
                         cy[target_ind + 1] - cy[target_ind])
        if dist + seg > Lf:
            break
        dist += seg
        target_ind += 1
    return min(target_ind, len(cx) - 2)


def _cross_track_error(state, cx, cy, cyaw, target_ind):
    """Signed lateral distance from the path at the closest point."""
    ti = min(target_ind, len(cx) - 2)
    # path direction
    dx_p = cx[ti + 1] - cx[ti]
    dy_p = cy[ti + 1] - cy[ti]
    path_len = math.hypot(dx_p, dy_p)
    if path_len < 1e-8:
        pu_x, pu_y = math.cos(cyaw[ti]), math.sin(cyaw[ti])
    else:
        pu_x, pu_y = dx_p / path_len, dy_p / path_len
    # vector from path point to robot
    rx = state.x - cx[ti]
    ry = state.y - cy[ti]
    # cross product gives signed lateral distance
    return pu_x * ry - pu_y * rx


# ── control function ────────────────────────────────────────────────
def control(state, cx, cy, cyaw, target_ind, target_speed):
    # 1) Reset target_ind to nearest point on path (CRITICAL FIX)
    new_ind, _ = calc_target_index(state, cx, cy)
    if new_ind >= target_ind:
        target_ind = new_ind

    # 2) Compute cross-track error at nearest point
    e = _cross_track_error(state, cx, cy, cyaw, target_ind)

    # 3) Find lookahead point
    Lf = max(Lf_min, k_lf * abs(state.v))
    la_ind = _advance_index(cx, cy, target_ind, Lf)
    la_ind = min(la_ind, len(cx) - 1)

    # 4) Pure pursuit heading error from lookahead point
    tx, ty = cx[la_ind], cy[la_ind]
    alpha = pi_2_pi(math.atan2(ty - state.y, tx - state.x) - state.yaw)

    # 5) Target omega = pure pursuit curvature + cross-track correction
    if abs(state.v) > 1e-3:
        pp_omega = state.v * 2.0 * math.sin(alpha) / Lf
    else:
        pp_omega = 2.0 * alpha / dt
    target_omega = pp_omega - K_cte * e
    target_omega = np.clip(target_omega, -omega_max, omega_max)

    # 6) Curvature-adaptive speed: slow down on sharp turns
    curvature = abs(2.0 * math.sin(alpha) / max(Lf, 0.1))
    speed_factor = 1.0 / (1.0 + 3.0 * curvature)
    adjusted_speed = target_speed * speed_factor
    adjusted_speed = max(adjusted_speed, 0.15)  # never go too slow

    # 7) Outputs
    a     = speed_control(adjusted_speed, state.v)
    alpha_cmd = K_omega * (target_omega - state.omega)
    return a, alpha_cmd, target_ind


# ── main ────────────────────────────────────────────────────────────
def main():
    cx, cy = generate_test_path()
    print("Pure Pursuit (improved) – simulation start")
    traj = run_simulation(control, cx, cy, TARGET_SPEED, MAX_TIME, GOAL_TOL)
    print(f"  Goal reached: {traj['find_goal']}")
    animate_and_save(cx, cy, traj, "Pure_Pursuit")


if __name__ == "__main__":
    main()
