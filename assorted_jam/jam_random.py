#!/usr/bin/env python3
"""Randomised multi-jammer demo (units: km). Static jammers only, no waypoints.

Each seed draws a new layout: two hostile jammers on the LEFT side of the map,
two allied fleets on the RIGHT, and a drone start on the left edge. The drone
never sees the jammers; it only senses one noisy combined jam level and uses the
gradient-probing controller from jam_sim.py to reach the nearest allied fleet.
Layouts where the straight-line route would not cross a jam zone are rejected,
so every scenario genuinely forces a detour.

Run:  python3 jam_random.py   -> jam_random_grid.png, jam_random.gif, stats printed
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from jam_sim import Avoider

NOISE, V_MAX, DT, EDGE_W, DETECT = 0.01, 0.8, 0.05, 0.45, 0.10
X_MAX, Y_MAX = 17.0, 14.0


class RobustAvoider(Avoider):
    """Avoider + (1) low-pass filtered 'away' direction so noise can't flip it when
    the drone is nearly stationary, (2) stall escape: if progress toward the goal
    stalls inside a jam shoulder, flip which way it skirts the zone, holding each
    choice longer every time (so it can travel far enough to find a gap)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.away_f, self.hold, self.t_ref, self.d_ref = None, 6.0, 0.0, None

    def step(self, pos, j, dt):
        self.t += dt
        self.hist.append((pos[0], pos[1], j))
        to_goal = self.goal - pos
        dist = np.linalg.norm(to_goal)
        v = to_goal / (dist + 1e-9)
        g = self.est_gradient() if self.enabled else None
        if g is not None and j > self.detect:
            gn = np.linalg.norm(g)
            if gn > 1e-3:
                away = -g / gn
                conf = min(1.0, gn / 0.1)
                if self.away_f is None:
                    self.away_f = away
                else:
                    a = 0.1 * conf
                    self.away_f = (1 - a) * self.away_f + a * away
                    self.away_f /= np.linalg.norm(self.away_f)
            if self.away_f is not None:
                af = self.away_f
                tan = np.array([-af[1], af[0]])
                if self.tan_sign is None:
                    self.tan_sign = 1.0 if np.dot(tan, v) >= 0 else -1.0
                    self.t_ref, self.d_ref = self.t, dist
                elif self.t - self.t_ref > self.hold:
                    if self.d_ref - dist < 0.3:            # stalled -> try other side
                        self.tan_sign *= -1
                        self.hold *= 1.8
                    self.t_ref, self.d_ref = self.t, dist
                strength = min(1.0, (j - self.detect) / 0.3)
                v = v + strength * (self.k_rep * af + self.k_tan * self.tan_sign * tan)
        elif j <= self.detect * 0.5:
            self.away_f, self.tan_sign, self.hold, self.d_ref = None, None, 6.0, None
        n = np.linalg.norm(v)
        v = v / (n + 1e-9) * self.v_max * min(1.0, dist / 0.3)
        dth = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-9)
        return v + self.dither * np.sin(2 * np.pi * 1.5 * self.t) * dth * 4


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def jam_level(p, jammers, rng=None):
    j = sum(1.0 / (1.0 + np.exp((np.linalg.norm(p - c) - r) / EDGE_W)) for c, r in jammers)
    j = min(j, 1.0)
    return j + (rng.normal(0, NOISE) if rng is not None else 0.0)


def passable(L, thr=0.15, h=0.2):
    """Is there any route from start to goal that stays below jam level thr?"""
    from collections import deque
    nx, ny = int(X_MAX / h) + 1, int(Y_MAX / h) + 1
    free = np.array([[jam_level(np.array([i * h, k * h]), L["jammers"]) < thr
                      for k in range(ny)] for i in range(nx)])
    s0 = (int(L["start"][0] / h), int(L["start"][1] / h))
    g0 = (int(L["goal"][0] / h), int(L["goal"][1] / h))
    seen, q = {s0}, deque([s0])
    while q:
        x, y = q.popleft()
        if abs(x - g0[0]) <= 1 and abs(y - g0[1]) <= 1:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if 0 <= n[0] < nx and 0 <= n[1] < ny and n not in seen and free[n]:
                seen.add(n); q.append(n)
    return False


def make_layout(seed):
    rng = np.random.default_rng(1000 + seed)
    while True:
        jammers = [(np.array([rng.uniform(3.0, 8.0), rng.uniform(2.5, 11.5)]),
                    rng.uniform(2.0, 4.0)) for _ in range(2)]
        allies = [(np.array([rng.uniform(13.0, 15.5), rng.uniform(2.5, 11.5)]),
                   rng.uniform(2.5, 3.5)) for _ in range(2)]
        start = np.array([0.8, rng.uniform(2.0, 12.0)])
        # jammers shouldn't swallow each other's centre, start, or the goals
        if np.linalg.norm(jammers[0][0] - jammers[1][0]) < 2.5:
            continue
        goal = min(allies, key=lambda a: np.linalg.norm(a[0] - start))[0]
        if jam_level(start, jammers) > 0.03 or jam_level(goal, jammers) > 0.03:
            continue
        # require the straight route to actually cross >= 1 jam zone
        if not any(seg_dist(c, start, goal) < r - 0.2 for c, r in jammers):
            continue
        L = dict(jammers=jammers, allies=allies, start=start, goal=goal, seed=seed)
        if not passable(L):      # reject layouts with no jam-free route at all
            continue
        return L


def fly(L, enabled, t_max=45.0, seed=0):
    rng = np.random.default_rng(seed)
    pos = L["start"].copy()
    ctl = RobustAvoider(L["goal"], v_max=V_MAX, detect=DETECT,
                  dither=0.04 * V_MAX / 0.5, enabled=enabled)
    P, J, t = [pos.copy()], [], 0.0
    while t < t_max:
        J.append(jam_level(pos, L["jammers"]))
        v = ctl.step(pos, jam_level(pos, L["jammers"], rng), DT)
        pos = pos + v * DT
        P.append(pos.copy())
        t += DT
        if np.linalg.norm(L["goal"] - pos) < 0.3:
            break
    P = np.array(P)
    return dict(pos=P[:len(J)], j=np.array(J), t=len(J) * DT,
                done=bool(np.linalg.norm(L["goal"] - P[-1]) < 0.3))


def draw_world(ax, L):
    ax.set_xlim(0, X_MAX); ax.set_ylim(0, Y_MAX); ax.set_aspect("equal")
    ax.set_facecolor("#eef4fb")
    for i, (c, r) in enumerate(L["allies"]):
        ax.add_patch(Circle(c, r, fc="#3b7dd8", alpha=0.12, ec="#3b7dd8", ls="--"))
        ax.plot(*c, "o", color="#3b7dd8", ms=7)
    for i, (c, r) in enumerate(L["jammers"]):
        ax.add_patch(Circle(c, r, fc="#d83b3b", alpha=0.18, ec="#d83b3b", ls="--"))
        ax.plot(*c, "o", color="#d83b3b", ms=7)
    ax.plot(*L["start"], "go", ms=8)
    ax.plot(*L["goal"], "g*", ms=14)


def summary(L, a, b):
    return (f"seed {L['seed']}: avoid peak jam {a['j'].max():.2f} "
            f"(baseline {b['j'].max():.2f}), "
            f"{'reached' if a['done'] else 'FAILED'} in {a['t']:.1f}s")


def main():
    # --- stats over many random layouts ---
    N = 60
    ok, peak_a, peak_b, times, fails = 0, [], [], [], []
    layouts = [make_layout(s) for s in range(N)]
    for L in layouts:
        a, b = fly(L, True), fly(L, False)
        ok += a["done"]
        peak_a.append(a["j"].max()); peak_b.append(b["j"].max()); times.append(a["t"])
        if not a["done"]:
            fails.append(L["seed"])
    print(f"{N} random layouts: reached goal in {ok}/{N}")
    print(f"mean peak jam  baseline {np.mean(peak_b):.2f}  vs  avoiding {np.mean(peak_a):.2f}")
    print(f"mean mission time (avoiding) {np.mean(times):.1f}s;  failed seeds: {fails}")

    # --- grid of 6 example layouts ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, L in zip(axs.ravel(), layouts[:6]):
        a, b = fly(L, True), fly(L, False)
        draw_world(ax, L)
        ax.plot(*b["pos"].T, color="gray", ls="--", lw=1.2, label="no avoidance")
        ax.plot(*a["pos"].T, "b-", lw=2, label="with avoidance")
        ax.set_title(summary(L, a, b), fontsize=9)
    axs[0, 0].legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig("jam_random_grid.png", dpi=110)
    plt.close(fig)

    # --- GIF: three layouts back to back ---
    runs = []
    good = [L for L in layouts if fly(L, True)["done"]][:3]
    for L in good:
        a, b = fly(L, True), fly(L, False)
        n = len(a["j"])
        runs += [(L, a, b, i) for i in list(range(0, n, 4)) + [n - 1] * 8]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 5),
                                 gridspec_kw={"width_ratios": [2.1, 1]})
    plt.tight_layout()

    def update(k):
        L, a, b, i = runs[k]
        a0.clear(); a1.clear()
        draw_world(a0, L)
        a0.plot(*b["pos"].T, color="gray", ls="--", lw=1, label="no avoidance")
        a0.plot(*a["pos"][:i + 1].T, "b-", lw=2)
        a0.plot(*a["pos"][i], "bo", ms=11, mec="k", zorder=5)
        a0.set_title(f"Random layout, seed {L['seed']} (jammers hidden from drone)")
        a0.legend(loc="lower left", fontsize=8)
        j = a["j"][i]
        if i >= len(a["j"]) - 1 and a["done"]:
            txt, col = "REACHED ALLIED FLEET", "green"
        elif j > DETECT:
            txt, col = "JAM DETECTED - REROUTING", "red"
        else:
            txt, col = "CLEAR - heading to allies", "navy"
        a0.text(0.02, 0.97, txt, transform=a0.transAxes, fontsize=12, weight="bold",
                color=col, va="top")
        tt = np.arange(len(a["j"])) * DT
        a1.set_xlim(0, 25); a1.set_ylim(0, 1.05)
        a1.axhline(DETECT, color="orange", ls=":", label="detect threshold")
        a1.plot(tt[:i], a["j"][:i], "r-", lw=2, label="jam level (avoiding)")
        a1.plot(np.arange(len(b["j"])) * DT, b["j"], color="gray", ls="--", label="no avoidance")
        a1.set_xlabel("time (s)"); a1.set_title("Jam exposure"); a1.legend(fontsize=8)

    FuncAnimation(fig, update, frames=len(runs)).save(
        "jam_random.gif", writer=PillowWriter(fps=15))


if __name__ == "__main__":
    main()