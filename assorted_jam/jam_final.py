#!/usr/bin/env python3
"""FINAL DEMO: lightweight drone escapes a jammed area, uses obstacle attenuation shadows, lands,
and leaves a recorded path a rescuer can follow back to the soldier. Units: km (stylised).
Standalone: numpy, matplotlib, Pillow.  Crazyflie-class is only a size reference; sensors are assumed custom.

Assumed hardware: flow-style velocity sensing + gyro; THREE range lasers (front, +/-30 deg);
a jam-level sensor. No camera, no map, no position fixes.

Behaviour (all built in)
  * Launch INSIDE a jam zone; head downhill on the sensed jam level; yaw to face travel.
  * Avoid obstacles with the 3 beams. While still jammed and slipping past an obstacle, lean a little
    TOWARD recently-seen obstacles (short hit memory) to slide into their attenuation shadow.
  * Slow down when the jam level first drops; land after 2 s of low jam AND >= LAND_CLEAR of forward clearance.
  * Memory: log a dead-reckoned position every 0.5 s (flow velocity + gyro yaw, with realistic errors).
  * Rescue: the rescuer knows where the drone LANDED; reversing the recorded displacement gives an estimate
    of the soldier's launch point. The radius of the green circle is calibrated by Monte Carlo (90th pct).

Two tests
  1. RANDOM: random obstacle maps (circles / walls, 2-4 obstacles) with 3-5 random jammers.
  2. SHADOW COURSE: one hard-coded course with a big "building" between the jammer and the escape path.

Run:  python3 jam_final.py -> stats, jam_final_random_grid.png, jam_final_course.png, jam_final.gif
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------------------------------------------------- parameters
X0, X1, Y0, Y1 = -3.0, 19.0, -3.0, 15.0     # plotting window; flight is unbounded
V, DT, DT_WAIT, NOISE = 0.8, 0.05, 0.5, 0.01
LAND_J, LAND_HOLD, LAND_TIME = 0.05, 2.0, 2.0
FLY_BURN, IDLE_BURN, T_FLY_MAX = 1.0, 0.05, 60.0     # battery %/s (illustrative)
RANGE_MAX, RANGE_NOISE, R_SAFE = 2.0, 0.03, 1.0
K_PUSH, K_SLIDE, BODY_R = 1.2, 0.8, 0.10
SENS_ANGLE = np.radians(30.0)                 # side lasers: +/-30 deg from the nose
YAW_RATE = 3.0                                 # rad/s, how fast the drone can turn to face travel
LOG_PERIOD, BYTES_PER_PT = 0.5, 6              # memory: (t,x,y) as 3 x int16 ~ 6 bytes
FLOW_SCALE_SD, FLOW_NOISE = 0.03, 0.03         # flow velocity: 3% scale error (per flight), 0.03 km/s noise
GYRO_BIAS_SD = np.radians(0.3)                 # yaw drift rate sd (rad/s) - gyro integration error
HIT_MEM, BIAS_RANGE = 6.0, 2.5                  # s of obstacle-hit memory; only lean toward things this close (km)
CONFIRM_SPEED = 0.3                            # fraction of speed while confirming a low-jam spot
K_SHADOW, R_HUG_MIN = 0.9, 0.5                  # shadow bias strength; never hug closer than this (km)
LAND_CLEAR = 1.0                               # km: min obstacle distance (all 4 sensors) to land
WALL_T = 0.25                                  # wall thickness (km)
ALPHA_CIRCLE, WALL_ATT = 1.2, 0.30             # attenuation strengths
DIRS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)], float)   # the 4 sensors


# ------------------------------------------------------------- obstacles
class Obstacles:
    def __init__(self, kind, C=None, R=None, A=None, B=None):
        self.kind = kind
        self.C = np.zeros((0, 2)) if C is None else C
        self.R = np.zeros(0) if R is None else R
        self.A = np.zeros((0, 2)) if A is None else A
        self.B = np.zeros((0, 2)) if B is None else B


def seg_dist(P, A, B):
    """distance from point(s) P (N,2) to each segment (W,2)-(W,2) -> (N,W)"""
    e = B - A
    t = np.clip(((P[:, None, :] - A[None]) * e[None]).sum(-1) / (e * e).sum(-1)[None], 0, 1)
    return np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * e[None]), axis=-1)


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def attenuation(J0, P, obs):
    """Fraction of each jammer's strength that reaches each point. -> (N, J)"""
    N, nJ = len(P), len(J0)
    f = np.ones((N, nJ))
    if obs is None:
        return f
    d = P[:, None, :] - J0[None, :, :]                      # (N,J,2) jammer -> point
    if len(obs.R):
        dd = d[:, :, None, :]
        oc = J0[None, :, None, :] - obs.C[None, None, :, :]
        a = (dd * dd).sum(-1)
        b = 2 * (dd * oc).sum(-1)
        cc = (oc * oc).sum(-1) - (obs.R ** 2)[None, None, :]
        sq = np.sqrt(np.maximum(b * b - 4 * a * cc, 0))
        t1 = np.clip((-b - sq) / (2 * a + 1e-12), 0, 1)
        t2 = np.clip((-b + sq) / (2 * a + 1e-12), 0, 1)
        chord = (t2 - t1) * np.sqrt(a)
        f *= np.exp(-ALPHA_CIRCLE * chord.sum(-1))
    if len(obs.A):
        e = (obs.B - obs.A)[None, None, :, :]
        dd = d[:, :, None, :]
        w = obs.A[None, None, :, :] - J0[None, :, None, :]
        den = cross(dd, e)
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        t, s = cross(w, e) / den, cross(w, dd) / den
        hit = ok & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
        f *= WALL_ATT ** hit.sum(-1)
    return f


def jam_field(P, jam, obs, rng=None):
    """Combined jam level at points P (N,2) (or a single point)."""
    P = np.atleast_2d(P)
    d = np.linalg.norm(P[:, None, :] - jam["C"][None], axis=-1)
    eff = (1.0 / (1.0 + (d / jam["R"][None]) ** 4)) * attenuation(jam["C"], P, obs)
    j = 1.0 - np.prod(1.0 - eff, axis=1)
    if rng is not None:
        j = j + rng.normal(0, NOISE, size=j.shape)
    return j


# ------------------------------------------------------------- layouts
def make_layout(seed, kind):
    ra = np.random.default_rng(5000 + seed)        # same jammers/start for both map types
    while True:
        n = int(ra.integers(3, 6))
        C = np.c_[ra.uniform(3, 14, n), ra.uniform(3, 11, n)]
        R = ra.uniform(1.2, 2.2, n)
        k = int(ra.integers(n))
        ang, rad = ra.uniform(0, 2 * np.pi), ra.uniform(0.35, 0.8) * R[k]
        start = C[k] + rad * np.array([np.cos(ang), np.sin(ang)])
        delay = ra.uniform(10, 30)
        jam = dict(C=C, R=R)
        if jam_field(start, jam, None)[0] >= 0.4:
            break
    ro = np.random.default_rng(7000 + seed + (0 if kind == "circles" else 50000))
    while True:
        if kind == "circles":
            m = int(ro.integers(2, 5))
            Cc = np.c_[ro.uniform(1, 16, m), ro.uniform(0, 13, m)]
            Rr = ro.uniform(0.4, 0.9, m)
            if np.any(np.linalg.norm(Cc - start, axis=1) < Rr + 1.5):
                continue
            obs = Obstacles("circles", C=Cc, R=Rr)
        else:
            m = int(ro.integers(2, 5))
            c = np.c_[ro.uniform(1, 16, m), ro.uniform(0, 13, m)]
            half = ro.uniform(1.5, 3.5, m)[:, None]
            th = ro.uniform(0, np.pi, m)
            u = np.c_[np.cos(th), np.sin(th)]
            A, B = c - half * u, c + half * u
            if np.any(seg_dist(start[None], A, B) < 1.5):
                continue
            obs = Obstacles("walls", A=A, B=B)
        if jam_field(start, jam, obs)[0] >= 0.4:
            return dict(jam=jam, obs=obs, start=start, seed=seed, kind=kind, delay=delay)


# ------------------------------------------------------------- sensors / avoidance
def ray_range(pos, d, obs, rng):
    best = RANGE_MAX
    if len(obs.R):
        pc = obs.C - pos
        b = pc @ d
        disc = b * b - ((pc * pc).sum(-1) - obs.R ** 2)
        with np.errstate(invalid="ignore"):
            t = b - np.sqrt(disc)
        t = t[(disc >= 0) & (t > 0)]
        if len(t):
            best = min(best, t.min())
    if len(obs.A):
        e = obs.B - obs.A
        den = cross(d[None], e)
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        w = obs.A - pos
        t, s = cross(w, e) / den, cross(w, np.broadcast_to(d, w.shape)) / den
        hit = ok & (t > 0) & (s >= 0) & (s <= 1)
        if hit.any():
            best = min(best, t[hit].min() - WALL_T / 2)
    return float(np.clip(best + rng.normal(0, RANGE_NOISE), 0.02, RANGE_MAX))


def collided(pos, obs):
    if len(obs.R) and np.any(np.linalg.norm(obs.C - pos, axis=1) < obs.R + BODY_R):
        return True
    return bool(len(obs.A) and np.any(seg_dist(pos[None], obs.A, obs.B) < WALL_T / 2 + BODY_R))


def avoid(v_cmd, ranges, slide):
    push = np.zeros(2)
    for d, r in zip(DIRS, ranges):
        if r < R_SAFE:
            push -= d * (1.0 / r - 1.0 / R_SAFE)
    out = v_cmd + V * K_PUSH * push
    pn = np.linalg.norm(push)
    if pn > 0.05:
        tan = np.array([-push[1], push[0]]) / pn
        dots = np.dot(tan, v_cmd)
        if abs(dots) > 0.1 * V:
            slide = 1.0 if dots > 0 else -1.0
        out = out + V * K_SLIDE * min(1.0, pn) * slide * tan
    sp = np.linalg.norm(out)
    return (out * 1.2 * V / sp if sp > 1.2 * V else out), slide



def beam_dirs(yaw):
    return np.array([(np.cos(yaw + a), np.sin(yaw + a)) for a in (0.0, SENS_ANGLE, -SENS_ANGLE)])


def avoid(v_cmd, dirs, ranges, slide, in_jam=False, shadow_bias=False, bias_vec=None):
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
            slide = 1.0 if dots > 0 else -1.0
        out = out + V * K_SLIDE * min(1.0, pn) * slide * tan
    biased = False
    if shadow_bias and in_jam and bias_vec is not None:   # still jammed, just slipped past an obstacle:
        out = out + V * K_SHADOW * bias_vec                # lean a little TOWARD it (its lee side is quiet)
        biased = True
    sp = np.linalg.norm(out)
    return (out * 1.2 * V / sp if sp > 1.2 * V else out), slide, biased


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------- drone controller
class EscapeController:
    """Downhill on the sensed jam level (sliding-window plane fit), smoothed heading."""
    def __init__(self, window=30):
        self.window, self.hist, self.heading, self.t, self.low_since = window, [], None, 0.0, None

    def gradient(self):
        if len(self.hist) < self.window:
            return None
        H = np.array(self.hist[-self.window:])
        X = np.c_[H[:, 0] - H[:, 0].mean(), H[:, 1] - H[:, 1].mean()]
        return np.linalg.solve(X.T @ X + 1e-4 * np.eye(2), X.T @ (H[:, 2] - H[:, 2].mean()))

    def step(self, pos, j, dt):
        self.t += dt
        self.hist.append((pos[0], pos[1], j))
        g = self.gradient()
        if g is not None and np.linalg.norm(g) > 1e-3:
            gn = np.linalg.norm(g)
            away, conf = -g / gn, min(1.0, gn / 0.05)
            if self.heading is None:
                self.heading = away
            else:
                h = (1 - 0.08 * conf) * self.heading + 0.08 * conf * away
                self.heading = h / np.linalg.norm(h)
        if self.heading is None:                                   # takeoff probe circle
            w = 2 * np.pi * 0.5
            return 0.5 * np.array([np.cos(w * self.t), np.sin(w * self.t)])
        perp = np.array([-self.heading[1], self.heading[0]])
        return V * self.heading + 0.3 * V * np.sin(2 * np.pi * 1.5 * self.t) * perp

    def clear_enough(self, space_ok=True):
        """Jam low for LAND_HOLD s AND (space_ok: far enough from every obstacle)."""
        if len(self.hist) < 20:
            return False
        if space_ok and np.mean([h[2] for h in self.hist[-20:]]) < LAND_J:
            self.low_since = self.t if self.low_since is None else self.low_since
            return self.t - self.low_since >= LAND_HOLD
        self.low_since = None
        return False


# ------------------------------------------------------------- simulation
def simulate(L, seed=0):
    rng = np.random.default_rng([seed, L["seed"] + 1])      # fresh sensor errors per run
    obs, jam = L["obs"], L["jam"]
    pos, yaw = L["start"].copy(), 0.0
    est = L["start"].copy()                                   # dead-reckoned position (start frame)
    flow_scale = 1.0 + rng.normal(0, FLOW_SCALE_SD)
    gyro_bias = rng.normal(0, GYRO_BIAS_SD)
    yaw_est_err = 0.0
    ctl, slide, t, batt, tick, path_len, expo = EscapeController(), 1.0, 0.0, 100.0, 0, 0.0, 0.0
    R = dict(t=[], x=[], y=[], ex=[], ey=[], j=[], batt=[], phase=[], rng=[], yaw=[], bias=[])
    mem = []
    hits = []
    flag = [False]

    def log(phase, j, ranges):
        for k, v in zip(R, (t, pos[0], pos[1], est[0], est[1], j, batt, phase, ranges, yaw, flag[0])):
            R[k].append(v)

    landed = crashed = False
    ranges = [RANGE_MAX] * 3
    while t < T_FLY_MAX:
        j_true = float(jam_field(pos, jam, obs)[0])
        dirs = beam_dirs(yaw)
        ranges = [ray_range(pos, d, obs, rng) for d in dirs]
        log("fly", j_true, ranges)
        for d_, r_ in zip(dirs, ranges):                     # remember where beams recently hit something
            if r_ < RANGE_MAX - 0.1:
                hits.append((t, est[0] + d_[0] * r_, est[1] + d_[1] * r_))
        hits[:] = [h for h in hits if t - h[0] <= HIT_MEM]
        bias_vec = None
        if hits and min(ranges) > R_HUG_MIN:
            H = np.array(hits)[:, 1:]
            dd = np.linalg.norm(H - est, axis=1)
            k_ = int(np.argmin(dd))
            if R_HUG_MIN < dd[k_] < BIAS_RANGE:
                bias_vec = (H[k_] - est) / dd[k_]
        expo += j_true * DT
        if tick % int(LOG_PERIOD / DT) == 0:
            mem.append((t, est[0], est[1]))
        j_meas = j_true + rng.normal(0, NOISE)
        v = ctl.step(est, j_meas, DT)
        v, slide, flag[0] = avoid(v, dirs, ranges, slide, j_meas > LAND_J, True, bias_vec)        # shadow bias is always on
        if ctl.low_since is not None:                      # jam just dropped: slow down while confirming
            v = v * CONFIRM_SPEED
        if np.linalg.norm(v) > 0.05:
            yaw += np.clip(wrap(np.arctan2(v[1], v[0]) - yaw), -YAW_RATE * DT, YAW_RATE * DT)
        step = v * DT
        pos = pos + step
        path_len += np.linalg.norm(step)
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        vb = np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]]) * flow_scale + rng.normal(0, FLOW_NOISE, 2)
        yaw_est_err += gyro_bias * DT
        ye = yaw + yaw_est_err
        est = est + np.array([np.cos(ye) * vb[0] - np.sin(ye) * vb[1],
                              np.sin(ye) * vb[0] + np.cos(ye) * vb[1]]) * DT
        batt -= FLY_BURN * DT; t += DT; tick += 1
        if collided(pos, obs):
            crashed = True
            break
        if t > 3.0 and ctl.clear_enough(min(ranges) >= LAND_CLEAR):
            landed = True
            break
    flag[0] = False
    if landed:
        j_land = float(jam_field(pos, jam, obs)[0])
        for _ in range(int(LAND_TIME / DT)):
            log("land", j_land, ranges); batt -= FLY_BURN * DT; t += DT; expo += j_land * DT
        mem.append((t, est[0], est[1]))
        t_land = t
        while t < t_land + L["delay"]:
            log("wait", j_land, ranges); batt -= IDLE_BURN * DT_WAIT; t += DT_WAIT
        log("done", j_land, ranges)
    out = {k: np.array(v) for k, v in R.items()}
    mem = np.array(mem)
    j_open = float(jam_field(pos, jam, None)[0])
    rec_len = float(np.linalg.norm(np.diff(mem[:, 1:], axis=0), axis=1).sum()) if len(mem) > 1 else 0.0
    d_est = mem[-1, 1:] - mem[0, 1:]
    out.update(landed=landed, crashed=crashed, land_j=out["j"][-1], j_open=j_open,
               clear_land=(float(min(ranges)) if landed else None),
               sheltered=bool(landed and j_open > 1.5 * LAND_J),
               mem=mem, path_len=path_len, rec_len=rec_len, exposure=expo,
               land_pos=pos.copy(), start=L["start"].copy(),
               start_err=float(np.linalg.norm(d_est - (pos - L["start"]))),   # error of rescuer's start estimate
               drift_series=np.linalg.norm(np.c_[out["ex"] - out["x"], out["ey"] - out["y"]], axis=1),
               exposure_series=np.cumsum(out["j"] * np.where(np.isin(out["phase"], ["fly", "land"]), DT, 0.0)))
    return out


def make_course(jitter=(0.0, 0.0), seed=-1):
    """The single hard-coded course: one big jammer, a 'building' dead ahead of the escape path,
    and two decoy obstacles further out."""
    jam = dict(C=np.array([[3.0, 6.0]]), R=np.array([4.0]))
    obs = Obstacles("circles", C=np.array([[7.8, 6.6], [11.0, 10.2], [11.2, 3.0]]),
                    R=np.array([1.5, 0.7, 0.8]))
    return dict(jam=jam, obs=obs, start=np.array([4.09, 6.14]) + np.array(jitter), seed=seed,
                kind="course", delay=20.0)
# ------------------------------------------------------------- drawing
_cache = {}


def draw(ax, L):
    key = (L["seed"], L["kind"])
    xs, ys = np.linspace(X0, X1, 150), np.linspace(Y0, Y1, 130)
    if key not in _cache:
        XX, YY = np.meshgrid(xs, ys)
        _cache[key] = jam_field(np.c_[XX.ravel(), YY.ravel()], L["jam"], L["obs"]).reshape(XX.shape)
    Z = _cache[key]
    ax.imshow(Z, origin="lower", extent=[X0, X1, Y0, Y1], cmap="Reds", vmin=0, vmax=1,
              alpha=0.8, aspect="equal")
    ax.contour(xs, ys, Z, levels=[LAND_J], colors="orange", linestyles="--", linewidths=1.2)
    for c in L["jam"]["C"]:
        ax.plot(*c, "rx", ms=8, mew=2)
    o = L["obs"]
    for c, r in zip(o.C, o.R):
        ax.add_patch(Circle(c, r, fc="#555555", ec="k", zorder=3))
    for a, b in zip(o.A, o.B):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#333333", lw=5, solid_capstyle="butt", zorder=3)
    ax.plot(*L["start"], marker="*", color="gold", mec="k", ms=15, zorder=4)   # soldier / launch point
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)



def course_batch(n, offset=0):
    return [simulate(make_course(jitter=np.random.default_rng(i + offset).uniform(-0.15, 0.15, 2),
                                 seed=i + offset), seed=0) for i in range(n)]


def random_batch(kind, n, offset=0):
    Ls = [make_layout(s + offset, kind) for s in range(n)]
    return Ls, [simulate(L) for L in Ls]


def t_land(r):
    return float(r["t"][np.where(r["phase"] == "land")[0][0]])


def recon(r, k):
    """Rescuer's view: reverse the recorded displacement from the known landing spot."""
    shift = r["land_pos"] - r["mem"][-1, 1:]
    return r["mem"][:, 1:] + shift, k * r["rec_len"]


def draw_rescue(ax, r, k, label=True):
    path, rad = recon(r, k)
    ax.plot(path[:, 0], path[:, 1], "--", color="green", lw=1.8, zorder=6)
    ax.add_patch(Circle(path[0], rad, fc="green", alpha=0.18, ec="green", lw=1.5, zorder=2))
    if label:
        ax.text(path[0][0], path[0][1] + rad + 0.3, "estimated soldier location", color="darkgreen",
                ha="center", fontsize=8, zorder=8)


def summarize(name, rs):
    ok = [r for r in rs if r["landed"] and r["land_j"] < 0.06]
    ld = [r for r in rs if r["landed"]]
    print(f"  {name}: landed clear {len(ok)}/{len(rs)}, crashed {sum(r['crashed'] for r in rs)}, "
          f"no escape {sum((not r['landed']) and (not r['crashed']) for r in rs)}, "
          f"landed in an obstacle shadow {sum(r['sheltered'] for r in ld)}/{len(ld)}, "
          f"time to land {np.mean([t_land(r) for r in ld]):.1f}s, "
          f"landing {np.mean([np.linalg.norm(r['land_pos'] - r['start']) for r in ld]):.1f} km from launch")


def verdict(r):
    return "CRASHED" if r["crashed"] else ("landed" if r["landed"] else "no escape")


def main():
    N = 60
    print("RANDOM obstacle maps (calibration set)")
    Lc, rc = random_batch("circles", N)
    Lw, rw = random_batch("walls", N)
    summarize("circles", rc); summarize("walls  ", rw)
    print("\nSHADOW COURSE (hard-coded; 60 runs with small random launch offsets)")
    cr = course_batch(N)
    summarize("course ", cr)

    cal = [r for r in rc + rw + cr if r["landed"]]            # calibrate on BOTH tests together
    ratios = [r["start_err"] / r["rec_len"] for r in cal]
    k90 = float(np.percentile(ratios, 90))
    print(f"\nRESCUE calibration (random maps + shadow course pooled): start-location error averages "
          f"{100 * np.mean(ratios):.1f}% of the recorded path; 90th percentile {100 * k90:.1f}% "
          f"-> green circle radius = {100 * k90:.1f}% of recorded path length")

    print("\nHOLD-OUT check of the green circle on fresh runs (not used for calibration)")
    _, fc = random_batch("circles", 30, offset=N)
    _, fw = random_batch("walls", 30, offset=N)
    fcourse = course_batch(30, offset=N)
    for name, group in (("random maps", [r for r in fc + fw if r["landed"]]),
                        ("shadow course", [r for r in fcourse if r["landed"]])):
        cov = np.mean([r["start_err"] <= k90 * r["rec_len"] for r in group])
        print(f"  {name}: circle contained the true start {100 * cov:.0f}% of the time "
              f"(mean radius {np.mean([k90 * r['rec_len'] for r in group]):.2f} km)")

    # ---------- random grid: top circles, bottom walls
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for row, (Ls, rs) in enumerate(((Lc, rc), (Lw, rw))):
        for col in range(3):
            ax, L, r = axs[row, col], Ls[col], rs[col]
            draw(ax, L)
            m = r["phase"] != "wait"
            ax.plot(r["x"][m], r["y"][m], "b-", lw=2, zorder=5)
            ax.plot(*r["land_pos"], "s" if r["landed"] else "kx", color="navy" if r["landed"] else "k",
                    ms=9, zorder=7)
            if r["landed"]:
                draw_rescue(ax, r, k90, label=False)
            ax.set_title(f"{L['kind']}, seed {L['seed']}: {verdict(r)}"
                         f"{', in shadow' if r['sheltered'] else ''}; start estimate off by {r['start_err']:.2f} km",
                         fontsize=9)
    fig.text(0.5, 0.005, "blue = drone's actual path   gold star = soldier (launch)   navy square = landing spot   "
             "green = rescuer's backtrack and estimated-location radius", ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.savefig("jam_final_random_grid.png", dpi=110)
    plt.close(fig)

    # ---------- course figure
    L0 = make_course(seed=0)
    r0 = simulate(L0, 0)
    fig, ax = plt.subplots(figsize=(10, 7))
    draw(ax, L0)
    m = r0["phase"] != "wait"
    ax.plot(r0["x"][m], r0["y"][m], "b-", lw=2.5, zorder=5)
    ax.plot(*r0["land_pos"], "s", color="navy", ms=11, zorder=7)
    draw_rescue(ax, r0, k90)
    ax.set_title(f"Shadow course: landed {np.linalg.norm(r0['land_pos'] - r0['start']):.1f} km from launch"
                 f"{' (in shadow)' if r0['sheltered'] else ''}; green circle = rescuer's estimate of the soldier")
    plt.tight_layout(); plt.savefig("jam_final_course.png", dpi=110); plt.close(fig)

    # ---------- GIF: course, then a random circles map, then a random walls map
    def best(Ls, rs):
        c = [(L, r) for L, r in list(zip(Ls, rs))[:25] if r["landed"] and r["land_j"] < 0.06]
        return max(c, key=lambda x: int(x[1]["bias"].sum()))
    runs = [("Shadow course (hard-coded)", L0, r0), ("Random map: circles",) + best(Lc, rc),
            ("Random map: walls",) + best(Lw, rw)]
    frames = []
    for title, L, r in runs:
        idx = [i for i, ph in enumerate(r["phase"])
               if (ph in ("fly", "land") and i % 6 == 0) or (ph in ("wait", "done") and i % 4 == 0)]
        idx += [len(r["t"]) - 1] * 20
        frames += [(title, L, r, i) for i in idx]
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1])
    a0, a1, a2 = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    plt.tight_layout()

    def update(k):
        title, L, r, i = frames[k]
        a0.clear(); a1.clear(); a2.clear()
        draw(a0, L)
        sel = r["phase"][:i + 1] != "wait"
        a0.plot(r["x"][:i + 1][sel], r["y"][:i + 1][sel], "b-", lw=2, zorder=5)
        p, ph, yw = np.array([r["x"][i], r["y"][i]]), r["phase"][i], r["yaw"][i]
        if ph == "fly":
            for d, r_ in zip(beam_dirs(yw), r["rng"][i]):
                a0.plot([p[0], p[0] + d[0] * r_], [p[1], p[1] + d[1] * r_],
                        color="red" if r_ < R_SAFE else "orange", lw=1.5, zorder=4)
        a0.plot(*p, marker="o" if ph == "fly" else "s", color="blue" if ph == "fly" else "navy",
                ms=11 if ph == "fly" else 10, mec="k", zorder=7)
        if ph in ("wait", "done"):
            draw_rescue(a0, r, k90)
        if ph == "fly":
            if r["bias"][i]:
                txt, col = "SLIPPING AROUND - LEANING INTO SHADOW", "darkgreen"
            elif r["j"][i] > LAND_J:
                txt, col = "INSIDE JAM - ESCAPING", "red"
            else:
                txt, col = "JAM LOW - CONFIRMING", "darkorange"
        else:
            txt, col = {"land": ("LANDING IN OBSTACLE SHADOW" if r["sheltered"] else "LANDING", "darkorange"),
                        "wait": ("LANDED - RESCUER BACKTRACKS FROM LANDING SPOT", "darkgreen"),
                        "done": ("HANDSHAKE RECEIVED", "green")}[ph]
        a0.text(0.02, 0.97, txt, transform=a0.transAxes, fontsize=11, weight="bold", color=col, va="top")
        a0.text(0.02, 0.03, "gold star = soldier (launch)   green = rescuer's backtrack and location radius",
                transform=a0.transAxes, fontsize=8, color="darkgreen")
        a0.set_title(title, fontsize=10)
        tt = r["t"][:i + 1]
        a1.plot(tt, r["j"][:i + 1], "r-", lw=2); a1.axhline(LAND_J, color="orange", ls=":")
        a1.set_xlim(0, r["t"][-1]); a1.set_ylim(0, 1.05); a1.set_ylabel("jam level")
        a2.plot(tt, np.min(r["rng"][:i + 1], axis=1), "k-", lw=1.5)
        a2.axhline(R_SAFE, color="red", ls=":"); a2.axhline(LAND_CLEAR, color="green", ls=":")
        a2.set_xlim(0, r["t"][-1]); a2.set_ylim(0, RANGE_MAX + 0.1)
        a2.set_ylabel("closest obstacle (km)"); a2.set_xlabel("time (s)")

    FuncAnimation(fig, update, frames=len(frames)).save("jam_final.gif",
                                                         writer=PillowWriter(fps=15))


if __name__ == "__main__":
    main()
