#!/usr/bin/env python3
"""Animated demo of jam_sim.py -> jam_demo.gif  (needs numpy, matplotlib, pillow)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jam_sim as S

DT = 0.05
STRIDE = 3
base_traj, base_js, _ = S.fly(False)
traj, js, _ = S.fly(True)
n = len(traj)
frames = list(range(0, n, STRIDE)) + [n - 1] * 20   # hold last frame

xs = np.linspace(-0.5, 4.5, 200)
ys = np.linspace(0.0, 3.0, 120)
Z = np.array([[S.jam_level(np.array([x, y])) for x in xs] for y in ys])

fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [2, 1]})
a0.contourf(xs, ys, Z, levels=20, cmap="Reds", alpha=0.75)
a0.plot(*base_traj.T, color="gray", ls="--", lw=1, label="no avoidance")
a0.plot(*S.START, "go", ms=9)
a0.plot(*S.GOAL, "g*", ms=15)
a0.plot(*S.JAMMER, "rx", ms=12, mew=3)
a0.set_aspect("equal"); a0.set_xlim(-0.5, 4.5); a0.set_ylim(0, 3)
a0.set_title("Drone route (jammer hidden from drone)")
a0.legend(loc="upper right", fontsize=8)
trail, = a0.plot([], [], "b-", lw=2)
drone, = a0.plot([], [], "bo", ms=11, mec="k")
status = a0.text(0.02, 0.94, "", transform=a0.transAxes, fontsize=12,
                 weight="bold", va="top")

t = np.arange(n) * DT
a1.set_xlim(0, t[-1]); a1.set_ylim(0, 1.05)
a1.axhline(S.Avoider(S.GOAL).detect, color="orange", ls=":", label="detect threshold")
a1.plot(t, base_js[:n] if len(base_js) >= n else np.r_[base_js, np.full(n - len(base_js), np.nan)],
        color="gray", ls="--", lw=1, label="no avoidance")
line, = a1.plot([], [], "b-", lw=2, label="with avoidance")
a1.set_xlabel("time (s)"); a1.set_ylabel("jam level"); a1.set_title("Jam exposure")
a1.legend(fontsize=8, loc="upper left")
plt.tight_layout()


def update(i):
    trail.set_data(traj[:i + 1, 0], traj[:i + 1, 1])
    drone.set_data([traj[i, 0]], [traj[i, 1]])
    line.set_data(t[:i], js[:i])
    j = js[min(i, len(js) - 1)]
    if np.linalg.norm(traj[i] - S.GOAL) < 0.1:
        status.set_text("GOAL REACHED"); status.set_color("green")
    elif j > 0.06:
        status.set_text("JAM DETECTED - REROUTING"); status.set_color("red")
    else:
        status.set_text("CRUISING"); status.set_color("navy")
    return trail, drone, line, status


anim = FuncAnimation(fig, update, frames=frames, blit=False)
anim.save("jam_demo.gif", writer=PillowWriter(fps=15))
print("saved", len(frames), "frames")