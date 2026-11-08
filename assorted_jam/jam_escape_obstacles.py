#!/usr/bin/env python3
"""Escape-and-land demo WITH obstacle avoidance from simple range sensors.

Adds to jam_escape.py:
  * random circular obstacles (stylised buildings/trees),
  * a simulated "multi-ranger": 4 (or 8) fixed-direction distance sensors with
    limited range and noise (no camera, no map),
  * avoidance = push away from close readings + slide sideways along the obstacle,
    blended on top of the existing downhill-on-jam-level escape heading.

Compares: no avoidance vs 4 sensors vs 8 sensors over many random layouts.
Run:  python3 jam_escape_obstacles.py -> stats, jam_obstacles_grid.png, jam_obstacles.gif
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import jam_escape as J

V, DT, DT_WAIT = J.V, J.DT, J.DT_WAIT
RANGE_MAX = 2.0       # km (stylised sensor range)
RANGE_NOISE = 0.03
R_SAFE = 1.0          # start pushing away inside this distance
K_PUSH, K_SLIDE = 1.2, 0.8
BODY_R = 0.10         # collision if centre gets within obstacle radius + this

DIRS4 = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)], float)
_d = np.sqrt(0.5)
DIRS8 = np.vstack([DIRS4, [(_d, _d), (-_d, _d), (-_d, -_d), (_d, -_d)]])


def make_obstacles(L, n=14):
    rng = np.random.default_rng(9000 + L["seed"])
    obs = []
    while len(obs) < n:
        c = np.array([rng.uniform(-3, 19), rng.uniform(-3, 15)])
        r = rng.uniform(0.4, 0.9)
        if np.linalg.norm(c - L["start"]) < r + 1.5:      # keep the launch point clear
            continue
        obs.append((c, r))
    return obs


def ray_range(pos, d, obstacles, rng):
    best = RANGE_MAX
    for c, r in obstacles:
        pc = c - pos
        b = np.dot(pc, d)
        disc = b * b - (np.dot(pc, pc) - r * r)
        if disc >= 0:
            t = b - np.sqrt(disc)
            if 0 < t < best:
                best = t
    return float(np.clip(best + rng.normal(0, RANGE_NOISE), 0.02, RANGE_MAX))


def avoid(v_cmd, dirs, ranges, slide_sign):
    push = np.zeros(2)
    for d, r in zip(dirs, ranges):
        if r < R_SAFE:
            push -= d * (1.0 / r - 1.0 / R_SAFE)
    out = v_cmd + V * K_PUSH * push
    pn = np.linalg.norm(push)
    if pn > 0.05:
        tan = np.array([-push[1], push[0]]) / pn
        dots = np.dot(tan, v_cmd)
        if abs(dots) > 0.1 * V:
            slide_sign = 1.0 if dots > 0 else -1.0
        out = out + V * K_SLIDE * min(1.0, pn) * slide_sign * tan
    sp = np.linalg.norm(out)
    if sp > 1.2 * V:
        out *= 1.2 * V / sp
    return out, slide_sign


def simulate(L, obstacles, dirs, seed=0):
    rng = np.random.default_rng(seed)
    pos, ctl = L["start"].copy(), J.EscapeController()
    R = dict(t=[], x=[], y=[], j=[], batt=[], phase=[], rng=[])
    batt, t, slide, crashed = 100.0, 0.0, 1.0, False

    def log(phase, ranges):
        R["t"].append(t); R["x"].append(pos[0]); R["y"].append(pos[1])
        R["j"].append(J.jam_level(pos, L["jammers"])); R["batt"].append(batt)
        R["phase"].append(phase); R["rng"].append(ranges)

    landed = False
    nd = len(DIRS4)
    while t < J.T_FLY_MAX:
        ranges = ([ray_range(pos, d, obstacles, rng) for d in dirs]
                  if dirs is not None else [RANGE_MAX] * nd)
        log("fly", ranges)
        v = ctl.step(pos, J.jam_level(pos, L["jammers"], rng), DT)
        if dirs is not None:
            v, slide = avoid(v, dirs, ranges, slide)
        pos = pos + v * DT
        batt -= J.FLY_BURN * DT; t += DT
        if any(np.linalg.norm(pos - c) < r + BODY_R for c, r in obstacles):
            crashed = True
            break
        if t > 3.0 and ctl.clear_enough():
            landed = True
            break
    if landed:
        for _ in range(int(J.LAND_TIME / DT)):
            log("land", ranges); batt -= J.FLY_BURN * DT; t += DT
        t_land = t
        while t < t_land + L["handshake_delay"]:
            log("wait", ranges); batt -= J.IDLE_BURN * DT_WAIT; t += DT_WAIT
        log("done", ranges)
    out = {k: np.array(v) for k, v in R.items()}
    out.update(landed=landed, crashed=crashed, land_j=out["j"][-1],
               t_land=(t_land if landed else None))
    return out


def draw(ax, L, obstacles):
    J.draw_world(ax, L)
    for c, r in obstacles:
        ax.add_patch(Circle(c, r, fc="#555555", ec="k", alpha=0.9, zorder=3))


def main():
    N = 100
    layouts = [J.make_layout(s) for s in range(N)]
    obs = [make_obstacles(L) for L in layouts]
    cfgs = [("no sensors", None), ("4 sensors", DIRS4), ("8 sensors", DIRS8)]
    results = {}
    for name, dirs in cfgs:
        res = [simulate(L, o, dirs) for L, o in zip(layouts, obs)]
        results[name] = res
        ok = sum(bool(r["landed"] and r["land_j"] < 0.06) for r in res)
        crash = sum(r["crashed"] for r in res)
        print(f"{name:10s}: escaped+landed clear {ok:3d}/{N},  crashed {crash:3d}/{N}")

    r4 = results["4 sensors"]
    # grid of the first 6 layouts (4 sensors) with no-sensor path for contrast
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, L, o, a, b in zip(axs.ravel(), layouts[:6], obs[:6], r4[:6], results["no sensors"][:6]):
        draw(ax, L, o)
        fl = b["phase"] != "wait"
        ax.plot(b["x"][fl], b["y"][fl], color="gray", ls="--", lw=1.2)
        if b["crashed"]:
            ax.plot(b["x"][-1], b["y"][-1], "kx", ms=11, mew=3, zorder=6)
        fl = a["phase"] != "wait"
        ax.plot(a["x"][fl], a["y"][fl], "b-", lw=2, zorder=5)
        end = "CRASHED" if a["crashed"] else ("landed" if a["landed"] else "no escape")
        ax.plot(a["x"][-1], a["y"][-1], "s" if a["landed"] else "kx", color="navy" if a["landed"] else "k",
                ms=9, zorder=6)
        ax.set_title(f"seed {L['seed']}: 4 sensors -> {end}; no sensors -> "
                     f"{'CRASHED' if b['crashed'] else 'ok'}", fontsize=9)
    plt.tight_layout(); plt.savefig("jam_obstacles_grid.png", dpi=110); plt.close(fig)

    # GIF: two layouts where no-sensor crashes but 4 sensors succeed
    pick = [(L, o, a, b) for L, o, a, b in zip(layouts, obs, r4, results["no sensors"])
            if b["crashed"] and a["landed"] and a["land_j"] < 0.06][:2]
    frames = []
    for L, o, a, b in pick:
        idx = [i for i, ph in enumerate(a["phase"])
               if (ph in ("fly", "land") and i % 5 == 0) or (ph in ("wait", "done") and i % 2 == 0)]
        idx += [len(a["t"]) - 1] * 15
        frames += [(L, o, a, b, i) for i in idx]
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1])
    a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
    plt.tight_layout()

    def update(k):
        L, o, a, b, i = frames[k]
        a0.clear(); a1.clear(); a2.clear()
        draw(a0, L, o)
        fl = b["phase"] != "wait"
        a0.plot(b["x"][fl], b["y"][fl], color="gray", ls="--", lw=1, label="no sensors (crashes)")
        a0.plot(b["x"][-1], b["y"][-1], "kx", ms=11, mew=3, zorder=6)
        sel = a["phase"][:i + 1] != "wait"
        a0.plot(a["x"][:i + 1][sel], a["y"][:i + 1][sel], "b-", lw=2, zorder=5, label="with 4 sensors")
        p = np.array([a["x"][i], a["y"][i]])
        ph = a["phase"][i]
        if ph == "fly":
            for d, r in zip(DIRS4, a["rng"][i]):
                col = "red" if r < R_SAFE else "orange"
                a0.plot([p[0], p[0] + d[0] * r], [p[1], p[1] + d[1] * r], color=col, lw=1.5, zorder=4)
        a0.plot(*p, marker="o" if ph == "fly" else "s", color="blue" if ph == "fly" else "navy",
                ms=11 if ph == "fly" else 10, mec="k", zorder=7)
        near = ph == "fly" and min(a["rng"][i]) < R_SAFE
        txt, col = {"fly": ("OBSTACLE NEAR - STEERING AROUND" if near else
                            ("INSIDE JAM - ESCAPING" if a["j"][i] > J.LAND_J else "CLEAR - CONFIRMING"),
                            "darkorange" if near else "red"),
                    "land": ("LANDING", "darkorange"),
                    "wait": ("LANDED - LISTENING FOR HANDSHAKE", "navy"),
                    "done": ("HANDSHAKE RECEIVED", "green")}[ph]
        a0.text(0.02, 0.97, txt, transform=a0.transAxes, fontsize=12, weight="bold", color=col, va="top")
        a0.set_title(f"Seed {L['seed']}: rays show the 4 range sensors (red = obstacle close)", fontsize=10)
        a0.legend(loc="lower right", fontsize=8)
        tt = a["t"][:i + 1]
        a1.plot(tt, a["j"][:i + 1], "r-", lw=2); a1.axhline(J.LAND_J, color="orange", ls=":")
        a1.set_xlim(0, a["t"][-1]); a1.set_ylim(0, 1.05); a1.set_ylabel("jam level")
        a2.plot(tt, np.min(a["rng"][:i + 1], axis=1), "k-", lw=1.5)
        a2.axhline(R_SAFE, color="red", ls=":")
        a2.set_xlim(0, a["t"][-1]); a2.set_ylim(0, RANGE_MAX + 0.1)
        a2.set_ylabel("closest obstacle (km)"); a2.set_xlabel("time (s)")

    FuncAnimation(fig, update, frames=len(frames)).save("jam_obstacles.gif",
                                                         writer=PillowWriter(fps=15))


if __name__ == "__main__":
    main()
