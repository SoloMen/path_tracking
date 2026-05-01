"""
Shared utilities for path tracking simulation:
  - car model drawing
  - animated GIF generation
  - path generation
  - target-index lookup
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, PillowWriter

from differential_drive_model import State as DDState, update as dd_update, pi_2_pi, dt, v_max, omega_max

# ── car visualisation dimensions (enlarged for visibility) ──────────
CAR_L = 1.2     # body length  [m]
CAR_W = 0.6     # body width   [m]
WHEEL_L = 0.35  # wheel length [m]
WHEEL_W = 0.12  # wheel width  [m]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── path & index helpers ────────────────────────────────────────────
def generate_test_path():
    """Sinusoidal test path (same as original)."""
    cx = np.arange(0, 50, 0.1)
    cy = np.sin(cx / 5.0) * cx / 2.0
    return cx.tolist(), cy.tolist()


def calc_target_index(state, cx, cy):
    """Return (index, min_distance) of the closest point on the path."""
    dx = np.asarray(cx) - state.x
    dy = np.asarray(cy) - state.y
    d  = np.hypot(dx, dy)
    ind = int(np.argmin(d))
    return ind, float(d[ind])


def calc_path_heading(cx, cy):
    """Heading angle at every path point."""
    cx, cy = np.asarray(cx), np.asarray(cy)
    dx = np.diff(cx, append=cx[-1])
    dy = np.diff(cy, append=cy[-1])
    cyaw = np.arctan2(dy, dx).tolist()
    return cyaw


# ── car drawing ─────────────────────────────────────────────────────
def draw_car(ax, x, y, yaw):
    """
    Draw a differential-drive robot at (x, y, yaw).
    Returns list of patches added to *ax*.
    """
    t = Affine2D().rotate(yaw).translate(x, y) + ax.transData
    pp = []

    # body
    body = patches.FancyBboxPatch(
        (-CAR_L / 2, -CAR_W / 2), CAR_L, CAR_W,
        boxstyle="round,pad=0.04",
        facecolor="#4A90D9", edgecolor="#2C5F8A",
        linewidth=1.5, alpha=0.9, zorder=10,
    )
    body.set_transform(t)
    ax.add_patch(body)
    pp.append(body)

    # left wheel
    lw = patches.Rectangle(
        (-WHEEL_L / 2, CAR_W / 2 - 0.01), WHEEL_L, WHEEL_W,
        facecolor="#333333", edgecolor="black", linewidth=0.5, zorder=11,
    )
    lw.set_transform(t)
    ax.add_patch(lw)
    pp.append(lw)

    # right wheel
    rw = patches.Rectangle(
        (-WHEEL_L / 2, -CAR_W / 2 - WHEEL_W + 0.01), WHEEL_L, WHEEL_W,
        facecolor="#333333", edgecolor="black", linewidth=0.5, zorder=11,
    )
    rw.set_transform(t)
    ax.add_patch(rw)
    pp.append(rw)

    # front arrow
    front = plt.Polygon(
        [(CAR_L / 2, -CAR_W / 4),
         (CAR_L / 2 + 0.3, 0),
         (CAR_L / 2, CAR_W / 4)],
        closed=True,
        facecolor="#E74C3C", edgecolor="#C0392B", linewidth=1, zorder=12,
    )
    front.set_transform(t)
    ax.add_patch(front)
    pp.append(front)

    return pp


# ── animation & GIF ─────────────────────────────────────────────────
def animate_and_save(cx, cy, traj, algorithm_name,
                     filename=None, frame_skip=6, fps=15):
    """
    Build an animated GIF + optional interactive window showing the car
    tracking the reference path.

    Interactive mode (TkAgg):
        - Popup window shows the animation in real-time as it plays
        - GIF saves in the background after animation completes
        - Window closes automatically when done
    Headless mode (Agg / NO_DISPLAY=1):
        - GIF only, no window
    """
    if filename is None:
        filename = os.path.join(OUTPUT_DIR, f"{algorithm_name}.gif")

    tx  = traj["x"]
    ty  = traj["y"]
    tw  = traj["yaw"]
    tv  = traj["v"]
    tow = traj.get("omega", traj.get("delta", []))
    tt  = traj["t"]
    ti  = traj["target_ind"]

    path_x = np.array(cx)
    path_y = np.array(cy)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(f"{algorithm_name} Path Tracking", fontsize=14, fontweight="bold")

    # ── top: tracking ──────────────────────────────────────────────────
    ax1.plot(path_x, path_y, "-r", label="Reference", linewidth=2, alpha=0.5)
    traj_line, = ax1.plot([], [], "-b", label="Trajectory", linewidth=1.5)
    tgt_mk,    = ax1.plot([], [], "xg", markersize=10, markeredgewidth=2,
                          label="Target")
    ax1.plot(cx[0], cy[0], "go", markersize=8, label="Start", zorder=5)
    ax1.plot(cx[-1], cy[-1], "r*", markersize=12, label="Goal", zorder=5)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")

    # ── bottom: velocity ────────────────────────────────────────────────
    v_line,     = ax2.plot([], [], "-r", label="v [m/s]", linewidth=1)
    has_omega = "omega" in traj
    has_delta = "delta" in traj
    omega_label = "\u03c9 [rad/s]" if has_omega else "\u03b4 [rad]"
    omega_line, = ax2.plot([], [], "-b", label=omega_label, linewidth=1)
    ax2.set_xlim(0, tt[-1] if tt else 1)
    vabs = max(
        max(abs(v) for v in tv) if tv else 0.5,
        max(abs(w) for w in tow) if tow else 0.5,
        0.8,
    )
    ax2.set_ylim(-vabs * 1.2, vabs * 1.2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlabel("Time [s]")

    car_patches = []
    n_total  = len(tx)
    n_frames = max(1, n_total // frame_skip)

    # Progress annotation
    progress_txt = ax1.text(
        0.02, 0.98, "", transform=ax1.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def _init():
        traj_line.set_data([], [])
        tgt_mk.set_data([], [])
        v_line.set_data([], [])
        omega_line.set_data([], [])
        progress_txt.set_text("")
        return []

    def _update(frame):
        nonlocal car_patches
        i = min(frame * frame_skip, n_total - 1)
        progress = min(1.0, i / max(1, n_total - 1))

        traj_line.set_data(tx[:i + 1], ty[:i + 1])

        tind = min(ti[i], len(cx) - 1) if i < len(ti) else len(cx) - 1
        tgt_mk.set_data([cx[tind]], [cy[tind]])

        v_line.set_data(tt[:i + 1], tv[:i + 1])
        omega_line.set_data(tt[:i + 1], tow[:i + 1])

        t_elapsed = tt[i] if i < len(tt) else 0
        progress_txt.set_text(f"t={t_elapsed:.1f}s  progress={progress*100:.0f}%")

        for p in car_patches:
            p.remove()
        car_patches = draw_car(ax1, tx[i], ty[i], tw[i])

        # adaptive view
        x_robot = tx[i]; y_robot = ty[i]
        x_frac = x_robot / max(path_x.max(), 1)
        if x_frac < 0.3:
            win_x, win_y = 6, 5
        elif x_frac < 0.7:
            win_x, win_y = 12, 10
        else:
            win_x, win_y = 20, 15
        ax1.set_xlim(x_robot - win_x, x_robot + win_x)
        ax1.set_ylim(y_robot - win_y, y_robot + win_y)
        return []

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         init_func=_init, interval=int(1000 / fps),
                         blit=False, repeat=False)

    print(f"[{algorithm_name}] Saving {filename} …")
    anim.save(filename, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"[{algorithm_name}] Saved  {filename}")


# ── simple speed controller ─────────────────────────────────────────
def speed_control(target_speed, current_speed, Kp=3.0):
    """Proportional speed controller with saturation."""
    return np.clip(Kp * (target_speed - current_speed), -5.0, 5.0)


# ── common simulation loop (differential drive) ──────────────────
def run_simulation(control_func, cx, cy, target_speed,
                   max_time=200.0, goal_tol=0.5):
    """
    Closed-loop simulation for differential-drive robots.

    control_func signature:
        (state, cx, cy, cyaw, target_ind, target_speed)
        -> (a, alpha, target_ind)
        where alpha is angular acceleration [rad/s^2]

    Returns dict {x, y, yaw, v, omega, t, target_ind, find_goal}
    """
    cyaw = calc_path_heading(cx, cy)
    state = DDState(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0)

    rec = dict(x=[0.0], y=[0.0], yaw=[0.0], v=[0.0], omega=[0.0],
               t=[0.0], target_ind=[0])
    target_ind, _ = calc_target_index(state, cx, cy)
    find_goal = False
    t = 0.0

    while t <= max_time:
        a, alpha, target_ind = control_func(state, cx, cy, cyaw,
                                             target_ind, target_speed)
        state = dd_update(state, a, alpha)
        t += dt

        dx = state.x - cx[-1]
        dy = state.y - cy[-1]
        if math.hypot(dx, dy) <= goal_tol:
            find_goal = True
            break

        rec["x"].append(state.x)
        rec["y"].append(state.y)
        rec["yaw"].append(state.yaw)
        rec["v"].append(state.v)
        rec["omega"].append(state.omega)
        rec["t"].append(t)
        rec["target_ind"].append(target_ind)
    else:
        print("  ⚠ Time out!")

    rec["find_goal"] = find_goal
    return rec