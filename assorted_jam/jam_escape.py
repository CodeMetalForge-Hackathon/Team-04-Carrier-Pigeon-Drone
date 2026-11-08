#!/usr/bin/env python3
"""Escape-and-land demo (units: km).

Protocol being demonstrated:
  1. Launch from INSIDE a jammed area; navigate out of it (no destination).
  2. Once the jam level has stayed below a threshold, land.
  3. Sit on the ground in a low-power listening state until a friendly
     handshake arrives (simulated as a random delay).

The drone never sees the jammers. It senses one noisy combined jam level,
estimates the local gradient from a sliding window of (position, reading)
samples, and heads downhill. All numbers (speed, battery rates, handshake
delay) are stylised, not real hardware figures.

Run:  python3 jam_escape.py   -> stats, jam_escape_grid.png, jam_escape.gif
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

X0, X1, Y0, Y1 = -4.0, 21.0, -4.0, 18.0   # plotting window; flight is unbounded
V = 0.8                 # km per sim-second
DT, DT_WAIT = 0.05, 0.5
NOISE = 0.01
LAND_J = 0.05           # smoothed jam level must stay below this ...
LAND_HOLD = 2.0         # ... for this many seconds before landing
LAND_TIME = 2.0         # descent duration
FLY_BURN, HOVER_BURN, IDLE_BURN = 1.0, 1.0, 0.05   # battery % per second (illustrative)
T_FLY_MAX = 60.0


# ---------------- world ----------------
def jam_level(p, jammers, rng=None):
    keep = 1.0
    for c, r in jammers:
        d = np.linalg.norm(p - c)
        keep *= 1.0 - 1.0 / (1.0 + (d / r) ** 4)
    j = 1.0 - keep
    return j + (rng.normal(0, NOISE) if rng is not None else 0.0)


def make_layout(seed):
    rng = np.random.default_rng(5000 + seed)
    while True:
        n = int(rng.integers(3, 6))
        jammers = [(np.array([rng.uniform(3.0, 14.0), rng.uniform(3.0, 11.0)]),
                    rng.uniform(1.2, 2.2)) for _ in range(n)]
        c, r = jammers[int(rng.integers(n))]
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.35, 0.8) * r
        start = c + rad * np.array([np.cos(ang), np.sin(ang)])
        if jam_level(start, jammers) < 0.4:       # make sure we really launch jammed
            continue
        return dict(jammers=jammers, start=start, seed=seed,
                    handshake_delay=rng.uniform(10, 30))


# ---------------- controller ----------------
class EscapeController:
    """Heads downhill on the sensed jam level; no destination."""
    def __init__(self, window=30):
        self.window, self.hist = window, []
        self.heading = None
        self.t = 0.0
        self.low_since = None

    def gradient(self):
        if len(self.hist) < self.window:
            return None
        H = np.array(self.hist[-self.window:])
        X = np.c_[H[:, 0] - H[:, 0].mean(), H[:, 1] - H[:, 1].mean()]
        y = H[:, 2] - H[:, 2].mean()
        return np.linalg.solve(X.T @ X + 1e-4 * np.eye(2), X.T @ y)

    def smoothed_j(self):
        return np.mean([h[2] for h in self.hist[-20:]])

    def step(self, pos, j, dt):
        self.t += dt
        self.hist.append((pos[0], pos[1], j))
        g = self.gradient()
        if g is not None:
            gn = np.linalg.norm(g)
            if gn > 1e-3:
                away, conf = -g / gn, min(1.0, gn / 0.05)
                if self.heading is None:
                    self.heading = away
                else:
                    a = 0.08 * conf
                    h = (1 - a) * self.heading + a * away
                    self.heading = h / np.linalg.norm(h)
        if self.heading is None:                     # takeoff probe: small circle
            w = 2 * np.pi * 0.5
            return 0.5 * np.array([np.cos(w * self.t), np.sin(w * self.t)])
        perp = np.array([-self.heading[1], self.heading[0]])
        return V * self.heading + 0.3 * V * np.sin(2 * np.pi * 1.5 * self.t) * perp

    def clear_enough(self):
        """True once smoothed jam level has stayed low for LAND_HOLD seconds."""
        if len(self.hist) < 20:
            return False
        if self.smoothed_j() < LAND_J:
            if self.low_since is None:
                self.low_since = self.t
            return self.t - self.low_since >= LAND_HOLD
        self.low_since = None
        return False


# ---------------- simulation ----------------
def simulate(L, seed=0):
    rng = np.random.default_rng(seed)
    pos, ctl = L["start"].copy(), EscapeController()
    R = dict(t=[], x=[], y=[], j=[], batt=[], phase=[])
    batt, t = 100.0, 0.0

    def log(phase, j_true):
        R["t"].append(t); R["x"].append(pos[0]); R["y"].append(pos[1])
        R["j"].append(j_true); R["batt"].append(batt); R["phase"].append(phase)

    landed = False
    while t < T_FLY_MAX:
        log("fly", jam_level(pos, L["jammers"]))
        v = ctl.step(pos, jam_level(pos, L["jammers"], rng), DT)
        pos = pos + v * DT
        batt -= FLY_BURN * DT; t += DT
        if t > 3.0 and ctl.clear_enough():
            landed = True
            break
    if landed:
        for _ in range(int(LAND_TIME / DT)):          # descent
            log("land", jam_level(pos, L["jammers"])); batt -= FLY_BURN * DT; t += DT
        t_land = t
        while t < t_land + L["handshake_delay"]:      # low-power listening
            log("wait", jam_level(pos, L["jammers"])); batt -= IDLE_BURN * DT_WAIT; t += DT_WAIT
        log("done", jam_level(pos, L["jammers"]))
    out = {k: np.array(v) for k, v in R.items()}
    out.update(landed=landed, land_j=out["j"][-1],
               t_land=(t_land if landed else None),
               escape_km=float(np.linalg.norm(pos - L["start"])))
    return out


# ---------------- plotting ----------------
_grid_cache = {}


def jam_grid(L):
    if L["seed"] not in _grid_cache:
        xs, ys = np.linspace(X0, X1, 150), np.linspace(Y0, Y1, 130)
        Z = np.array([[jam_level(np.array([x, y]), L["jammers"]) for x in xs] for y in ys])
        _grid_cache[L["seed"]] = (xs, ys, Z)
    return _grid_cache[L["seed"]]


def draw_world(ax, L):
    xs, ys, Z = jam_grid(L)
    ax.imshow(Z, origin="lower", extent=[X0, X1, Y0, Y1], cmap="Reds",
              vmin=0, vmax=1, alpha=0.8, aspect="equal")
    ax.contour(xs, ys, Z, levels=[LAND_J], colors="orange", linestyles="--", linewidths=1.2)
    for c, _ in L["jammers"]:
        ax.plot(*c, "rx", ms=8, mew=2)
    ax.plot(*L["start"], "go", ms=8)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)


def title(L, r):
    if r["landed"]:
        return (f"seed {L['seed']}: landed after {r['t_land']:.0f}s, "
                f"{r['escape_km']:.1f} km out, jam {r['land_j']:.2f}")
    return f"seed {L['seed']}: DID NOT ESCAPE in {T_FLY_MAX:.0f}s"


def main():
    N = 100
    layouts = [make_layout(s) for s in range(N)]
    res = [simulate(L) for L in layouts]
    good = [bool(r["landed"] and r["land_j"] < 0.06) for r in res]
    ok = [r for r, g in zip(res, good) if g]
    print(f"{N} random layouts: escaped and landed clear in {len(ok)}/{N}")
    print(f"  mean time to land {np.mean([r['t_land'] for r in ok]):.1f}s, "
          f"mean distance flown out {np.mean([r['escape_km'] for r in ok]):.1f} km")
    lb = [r["batt"][np.where(r["phase"] == "land")[0][-1]] for r in ok]
    print(f"  mean battery when landed {np.mean(lb):.0f}%")
    print(f"  failed seeds: {[L['seed'] for L, g in zip(layouts, good) if not g]}")

    # ---- grid ----
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, L, r in zip(axs.ravel(), layouts[:6], res[:6]):
        draw_world(ax, L)
        fl = r["phase"] != "wait"
        ax.plot(r["x"][fl], r["y"][fl], "b-", lw=2)
        ax.plot(r["x"][-1], r["y"][-1], "s", color="navy", ms=9)
        ax.set_title(title(L, r), fontsize=9)
    plt.tight_layout(); plt.savefig("jam_escape_grid.png", dpi=110); plt.close(fig)

    # ---- GIF: two successful layouts ----
    pick = [(L, r) for L, r, g in zip(layouts, res, good) if g][:2]
    frames = []
    for L, r in pick:
        idx = []
        for i, ph in enumerate(r["phase"]):
            if (ph in ("fly", "land") and i % 5 == 0) or ph in ("wait", "done"):
                idx.append(i)
        idx += [len(r["t"]) - 1] * 15
        frames += [(L, r, i) for i in idx]
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1])
    a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
    plt.tight_layout()

    def update(k):
        L, r, i = frames[k]
        a0.clear(); a1.clear(); a2.clear()
        draw_world(a0, L)
        fl = r["phase"][:i + 1] != "wait"
        a0.plot(r["x"][:i + 1][fl], r["y"][:i + 1][fl], "b-", lw=2)
        ph = r["phase"][i]
        a0.plot(r["x"][i], r["y"][i], marker="o" if ph == "fly" else "s",
                color="blue" if ph == "fly" else "navy", ms=11 if ph == "fly" else 10, mec="k", zorder=5)
        txt, col = {"fly": ("INSIDE JAM - ESCAPING" if r["j"][i] > LAND_J else "CLEAR - CONFIRMING", "red"),
                    "land": ("LANDING", "darkorange"),
                    "wait": ("LANDED - LISTENING FOR HANDSHAKE (low power)", "navy"),
                    "done": ("HANDSHAKE RECEIVED", "green")}[ph]
        if ph == "fly" and r["j"][i] <= LAND_J:
            col = "darkorange"
        a0.text(0.02, 0.97, txt, transform=a0.transAxes, fontsize=12, weight="bold",
                color=col, va="top")
        a0.set_title(f"Launch inside a jam zone, seed {L['seed']}  (orange dashes = landing threshold)", fontsize=10)
        tt = r["t"][:i + 1]
        a1.plot(tt, r["j"][:i + 1], "r-", lw=2); a1.axhline(LAND_J, color="orange", ls=":")
        a1.set_xlim(0, r["t"][-1]); a1.set_ylim(0, 1.05); a1.set_ylabel("jam level")
        a2.plot(tt, r["batt"][:i + 1], "g-", lw=2)
        a2.plot([0, r["t"][-1]], [100, max(0, 100 - HOVER_BURN * r["t"][-1])], color="gray", ls="--",
                label="if it kept hovering")
        a2.set_xlim(0, r["t"][-1]); a2.set_ylim(0, 105)
        a2.set_ylabel("battery %"); a2.set_xlabel("time (s)"); a2.legend(fontsize=8, loc="lower left")

    FuncAnimation(fig, update, frames=len(frames)).save("jam_escape.gif",
                                                         writer=PillowWriter(fps=15))


if __name__ == "__main__":
    main()