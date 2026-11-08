#!/usr/bin/env python3
"""Multi-jammer demo based on the user's battlefield sketch (units: km).

Layout was read off the sketch at ~21.5 px/km (the sketch's 7.5 km drone-to-
frigate label and 2.5 km drone-to-fleet label both fit that scale), so all
positions/radii are approximate and the scenario is notional.

Mission: drone starts at its sketched position, reaches the safe Buffer Zone,
then continues to the northern Allied Fleet. Two hostile jammers (Hostile
Frigate, Coastal EW) are hidden from the drone; it only senses one noisy
combined jam level and uses the gradient-probing controller from jam_sim.py.

Run:  python3 jam_battle.py   -> jam_battle_static.gif, jam_battle_moving.gif
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from jam_sim import Avoider

NOISE = 0.01
V_MAX = 0.8          # km per sim-second (stylised; not a real airspeed)
DT = 0.05
EDGE_W = 0.45        # km, softness of the jam-zone edge

DRONE_START = np.array([10.7, 4.3])
WAYPOINTS = [("Buffer Zone", np.array([10.6, 10.2])),
             ("Allied Fleet N", np.array([13.0, 9.3]))]
ALLIES = [("Allied Fleet N", np.array([14.4, 9.3]), 3.7),
          ("Allied Fleet S", np.array([13.2, 4.3]), 4.0)]


class Jammer:
    def __init__(self, name, pos, radius, vel=(0.0, 0.0)):
        self.name, self.p0, self.r = name, np.array(pos, float), radius
        self.vel = np.array(vel, float)

    def pos(self, t):
        return self.p0 + self.vel * t


def make_jammers(moving):
    return [Jammer("Coastal EW", (2.0, 9.3), 5.5),
            Jammer("Hostile Frigate", (3.3, 4.3), 7.5,
                   vel=(0.08, 0.03) if moving else (0, 0))]


def jam_level(p, jammers, t, rng=None):
    j = 0.0
    for J in jammers:
        d = np.linalg.norm(p - J.pos(t))
        j += 1.0 / (1.0 + np.exp((d - J.r) / EDGE_W))
    j = min(j, 1.0)
    if rng is not None:
        j += rng.normal(0, NOISE)
    return j


def link_quality(p, j_true):
    """Toy comms model: best allied link (range falloff) x (1 - jam)."""
    best = 0.0
    for _, c, r in ALLIES:
        best = max(best, max(0.0, 1 - np.linalg.norm(p - c) / r))
    return best * (1 - j_true)


def fly(enabled, moving, seed=1, t_max=40.0):
    rng = np.random.default_rng(seed)
    jammers = make_jammers(moving)
    pos = DRONE_START.copy()
    ctl = Avoider(WAYPOINTS[0][1], v_max=V_MAX, detect=0.10, dither=0.04 * V_MAX / 0.5,
                  k_rep=2.0, k_tan=0.8, enabled=enabled)
    wp, t = 0, 0.0
    rec = dict(pos=[pos.copy()], j=[], q=[], jp=[], wp=[], t=[])
    while t < t_max:
        j_meas = jam_level(pos, jammers, t, rng)
        j_true = jam_level(pos, jammers, t)
        rec["j"].append(j_true); rec["q"].append(link_quality(pos, j_true))
        rec["jp"].append([J.pos(t).copy() for J in jammers])
        rec["wp"].append(wp); rec["t"].append(t)
        v = ctl.step(pos, j_meas, DT)
        pos = pos + v * DT
        rec["pos"].append(pos.copy())
        t += DT
        if np.linalg.norm(WAYPOINTS[wp][1] - pos) < 0.25:
            if wp == len(WAYPOINTS) - 1:
                break
            wp += 1
            ctl.goal = WAYPOINTS[wp][1]
    for k in ("pos", "j", "q", "wp", "t"):
        rec[k] = np.array(rec[k])
    rec["pos"] = rec["pos"][:len(rec["j"])]
    rec["jp"] = np.array(rec["jp"])
    rec["done"] = wp == len(WAYPOINTS) - 1 and t < t_max
    return rec, jammers


def report(name, rec, jammers):
    mind = [np.min(np.linalg.norm(rec["pos"] - rec["jp"][:, i], axis=1)) - J.r
            for i, J in enumerate(jammers)]
    print(f"{name:22s} finished={rec['done']}  time={rec['t'][-1]:5.1f}s  "
          f"peak_jam={rec['j'].max():.2f}  mean_link={rec['q'].mean():.2f}  "
          + "  ".join(f"{J.name}: {m:+.2f} km vs edge" for J, m in zip(jammers, mind)))


def animate(rec, base, jammers, outfile, title):
    n = len(rec["j"])
    frames = list(range(0, n, 4)) + [n - 1] * 25
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 5),
                                 gridspec_kw={"width_ratios": [2.1, 1]})
    a0.set_xlim(0, 17); a0.set_ylim(0, 14); a0.set_aspect("equal")
    a0.set_facecolor("#eef4fb"); a0.set_title(title)
    a0.set_xlabel("km"); a0.set_ylabel("km")
    for nm, c, r in ALLIES:
        a0.add_patch(Circle(c, r, fc="#3b7dd8", alpha=0.12, ec="#3b7dd8", ls="--"))
        a0.plot(*c, "o", color="#3b7dd8", ms=8)
        a0.text(c[0], c[1] + 0.4, nm, color="#2457a6", ha="center", fontsize=8)
    jc, jd = [], []
    for J in jammers:
        jc.append(a0.add_patch(Circle(J.p0, J.r, fc="#d83b3b", alpha=0.15,
                                      ec="#d83b3b", ls="--")))
        d, = a0.plot(*J.p0, "o", color="#d83b3b", ms=8); jd.append(d)
    jt = [a0.text(*J.p0, J.name, color="#a02020", ha="center", fontsize=8,
                  va="top") for J in jammers]
    for nm, w in WAYPOINTS[:1]:
        a0.plot(*w, "gs", ms=9); a0.text(w[0], w[1] + 0.35, nm, color="green",
                                         ha="center", fontsize=8)
    a0.plot(*base["pos"].T, color="gray", ls="--", lw=1, label="no avoidance")
    a0.legend(loc="lower left", fontsize=8)
    trail, = a0.plot([], [], "b-", lw=2)
    drone, = a0.plot([], [], "bo", ms=11, mec="k", zorder=5)
    status = a0.text(0.02, 0.97, "", transform=a0.transAxes, fontsize=12,
                     weight="bold", va="top")
    t = rec["t"]
    a1.set_xlim(0, t[-1]); a1.set_ylim(0, 1.05)
    a1.axhline(0.10, color="orange", ls=":", label="detect threshold")
    jl, = a1.plot([], [], "r-", lw=2, label="jam level (drone sensed)")
    ql, = a1.plot([], [], "g-", lw=2, label="link quality to fleet")
    a1.set_xlabel("time (s)"); a1.legend(fontsize=8, loc="upper right")
    a1.set_title("Exposure and link")
    plt.tight_layout()

    def update(i):
        for k, J in enumerate(jammers):
            c = rec["jp"][i, k]
            jc[k].center = tuple(c); jd[k].set_data([c[0]], [c[1]]); jt[k].set_position(c - [0, 0.4])
        trail.set_data(rec["pos"][:i + 1, 0], rec["pos"][:i + 1, 1])
        drone.set_data([rec["pos"][i, 0]], [rec["pos"][i, 1]])
        jl.set_data(t[:i], rec["j"][:i]); ql.set_data(t[:i], rec["q"][:i])
        if i >= n - 1 and rec["done"]:
            status.set_text("DELIVERED TO ALLIED FLEET"); status.set_color("green")
        elif rec["j"][i] > 0.10:
            status.set_text("JAM DETECTED - REROUTING"); status.set_color("red")
        else:
            tgt = WAYPOINTS[rec["wp"][i]][0]
            status.set_text(f"CLEAR - heading to {tgt}"); status.set_color("navy")
        return trail, drone

    FuncAnimation(fig, update, frames=frames).save(outfile, writer=PillowWriter(fps=15))
    plt.close(fig)


def main():
    for moving, tag, title in [(False, "static", "Static jammers"),
                               (True, "moving", "Hostile Frigate closing in (jammer moves)")]:
        rec, jam = fly(True, moving)
        base, _ = fly(False, moving)
        report(f"{tag} / avoid", rec, jam)
        report(f"{tag} / baseline", base, jam)
        animate(rec, base, jam, f"jam_battle_{tag}.gif", title)


if __name__ == "__main__":
    main()