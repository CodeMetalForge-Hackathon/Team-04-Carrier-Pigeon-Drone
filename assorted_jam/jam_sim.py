#!/usr/bin/env python3
"""2D jammer-avoidance sim.

The drone flies start -> goal. A hidden jammer degrades a noisy scalar
"jam level" the drone can measure (it never sees the jammer position).
The drone estimates the local gradient of jam level from a sliding window of
(position, measurement) samples (plus a small dither so the fit is
well-conditioned), then blends goal attraction with repulsion up the
negative gradient and a tangential bias to avoid getting stuck.

Run:  python3 jam_sim.py            -> prints metrics, writes jam_sim.png
Deps: numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- world / jammer model ----------------
START = np.array([0.0, 1.5])
GOAL = np.array([4.0, 1.5])
JAMMER = np.array([2.0, 1.6])      # hidden from the controller
R_JAM = 0.6                        # "radius" scale of the jam field
NOISE = 0.01                       # measurement noise (std)


def jam_level(p, rng=None):
    """0 = clean, ->1 = deep inside jammer. Smooth, monotonic in distance."""
    d2 = np.sum((p - JAMMER) ** 2)
    j = 1.0 / (1.0 + d2 / R_JAM ** 2)
    if rng is not None:
        j += rng.normal(0, NOISE)
    return j


# ---------------- controller ----------------
class Avoider:
    def __init__(self, goal, v_max=0.5, window=25, detect=0.06,
                 k_rep=2.0, k_tan=0.8, dither=0.04, enabled=True):
        self.goal, self.v_max = goal, v_max
        self.window, self.detect = window, detect
        self.k_rep, self.k_tan, self.dither = k_rep, k_tan, dither
        self.enabled = enabled
        self.hist = []             # (x, y, j)
        self.t = 0.0
        self.tan_sign = None       # commit to one side once

    def est_gradient(self):
        if len(self.hist) < self.window:
            return None
        H = np.array(self.hist[-self.window:])
        X = np.c_[H[:, 0] - H[:, 0].mean(), H[:, 1] - H[:, 1].mean()]
        y = H[:, 2] - H[:, 2].mean()
        # ridge-regularised least squares for slope in x,y
        A = X.T @ X + 1e-4 * np.eye(2)
        return np.linalg.solve(A, X.T @ y)

    def step(self, pos, j, dt):
        self.t += dt
        self.hist.append((pos[0], pos[1], j))
        to_goal = self.goal - pos
        dist = np.linalg.norm(to_goal)
        v = to_goal / (dist + 1e-9)                      # attraction

        g = self.est_gradient() if self.enabled else None
        if g is not None and j > self.detect:
            gn = np.linalg.norm(g)
            if gn > 1e-6:
                away = -g / gn                            # toward cleaner signal
                # tangential direction (perpendicular to away); commit to a side
                tan = np.array([-away[1], away[0]])
                if self.tan_sign is None:
                    self.tan_sign = 1.0 if np.dot(tan, v) >= 0 else -1.0
                strength = min(1.0, (j - self.detect) / 0.3)
                v = v + strength * (self.k_rep * away
                                    + self.k_tan * self.tan_sign * tan)
        elif j <= self.detect * 0.5:
            self.tan_sign = None                          # clear: reset

        n = np.linalg.norm(v)
        v = v / (n + 1e-9) * self.v_max
        # slow near goal, small perpendicular dither for gradient observability
        v = v * min(1.0, dist / 0.3)
        dth = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-9)
        v = v + self.dither * np.sin(2 * np.pi * 1.5 * self.t) * dth * 4
        return v


# ---------------- simulation ----------------
def fly(enabled, seed=0, dt=0.05, t_max=40.0):
    rng = np.random.default_rng(seed)
    pos = START.copy()
    ctl = Avoider(GOAL, enabled=enabled)
    traj, js = [pos.copy()], []
    for k in range(int(t_max / dt)):
        j = jam_level(pos, rng)
        js.append(jam_level(pos))                        # true level for plots
        v = ctl.step(pos, j, dt)
        pos = pos + v * dt
        traj.append(pos.copy())
        if np.linalg.norm(GOAL - pos) < 0.08:
            break
    return np.array(traj), np.array(js), (k + 1) * dt


def metrics(name, traj, js, t):
    dmin = np.min(np.linalg.norm(traj - JAMMER, axis=1))
    reached = np.linalg.norm(traj[-1] - GOAL) < 0.1
    print(f"{name:10s} reached={reached}  time={t:5.1f}s  "
          f"min_dist_to_jammer={dmin:.2f}m  peak_jam={js.max():.2f}  "
          f"mean_jam={js.mean():.2f}")


def main():
    base = fly(False)
    avoid = fly(True)
    metrics("baseline", *base[:2], base[2])
    metrics("avoid", *avoid[:2], avoid[2])

    xs = np.linspace(-0.5, 4.5, 250)
    ys = np.linspace(0.0, 3.0, 150)
    Z = np.array([[jam_level(np.array([x, y])) for x in xs] for y in ys])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6),
                           gridspec_kw={"width_ratios": [2, 1]})
    im = ax[0].contourf(xs, ys, Z, levels=20, cmap="Reds", alpha=0.8)
    plt.colorbar(im, ax=ax[0], label="jam level")
    ax[0].plot(*base[0].T, "k--", label="no avoidance")
    ax[0].plot(*avoid[0].T, "b-", lw=2, label="with avoidance")
    ax[0].plot(*START, "go", ms=9, label="start")
    ax[0].plot(*GOAL, "g*", ms=14, label="goal")
    ax[0].plot(*JAMMER, "rx", ms=12, mew=3, label="jammer (hidden)")
    ax[0].set_aspect("equal")
    ax[0].set_title("Trajectories")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(np.arange(len(base[1])) * 0.05, base[1], "k--", label="no avoidance")
    ax[1].plot(np.arange(len(avoid[1])) * 0.05, avoid[1], "b-", label="with avoidance")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("true jam level")
    ax[1].set_title("Exposure")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("jam_sim.png", dpi=130)


if __name__ == "__main__":
    main()
