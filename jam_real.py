#!/usr/bin/env python3
"""REAL-UNITS DEMO: a light drone escapes a jammed area, lands, and leaves a recorded path a rescuer can follow
back to the soldier.  SI units (metres, seconds, dB).  Standalone: numpy, matplotlib, Pillow.

Physical assumptions (all editable in the PARAMETERS block)
  * Jammers: effective denial radius ~1 km (0.8-1.2 km).  Received jam power falls as 1/d^2 (free space) and powers add.
    "0 dB" = the level where the link is denied (d = R for one jammer).  The drone lands when it is 10 dB BELOW that.
  * Jam sensing: received level in dB, with 2 dB of slowly-varying fading (50 m decorrelation) + 0.5 dB white noise.
  * Drone: 5 m/s cruise, three TFmini-class lidars (12 m, front and +/-30 deg), obstacles 20-400 m across.
  * Navigation memory: tactical-grade IMU + visual/optical velocity sensing + magnetometer start heading.
  * Escape controller: fly an L-shaped probe, then fit a single-source 1/d^2 model to the whole signal history
    (using dead-reckoned positions) and fly directly away from the fitted jam centre.

Run:  python3 jam_real.py   ->  stats, jam_real_random_grid.png, jam_real_course.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from multiprocessing import Pool

# ------------------------------------------------------------- PARAMETERS
V, DT, LOG_EVERY = 5.0, 0.1, 5                   # cruise speed m/s, physics step s, log every 5 steps (0.5 s)
T_FLY_MAX = 1800.0                               # s: battery endurance assumed (30 min)
LAND_TIME = 3.0                                  # s to descend
# lidar (TFmini-class)
RANGE_MAX, RANGE_NOISE, R_SAFE = 12.0, 0.10, 6.0
K_PUSH, K_SLIDE, BODY_R = 1.2, 0.8, 0.5
SENS_ANGLES = np.radians([-60.0, -30.0, 0.0, 30.0, 60.0])   # 5 lidar beams relative to heading
SENS_WEIGHTS = np.cos(SENS_ANGLES)               # react less to off-axis hits: 1.0 ahead, 0.87 at 30 deg, 0.5 at 60 deg
YAW_RATE = 3.0                                   # rad/s
# jam physics
D_FLOOR = 30.0                                   # m: near-field floor for the 1/d^2 law
MARGIN_DB = 10.0                                 # land when this far BELOW the denial level
FADE_SD_DB, FADE_DIST, WHITE_SD_DB = 2.0, 50.0, 0.5
# escape controller
SAMPLE_DT, FIT_EVERY, HEAD_TAU = 2.0, 8.0, 10.0
A_PRIOR_DB, A_PRIOR_SD, FIT_NOISE_DB = 60.0, 4.0, 3.0   # prior: a 'typical' jammer is denial-strong out to ~1 km (20log10(1000))
T_PROBE, T_LEG = 60.0, 30.0                      # L-shaped probe: 30 s one way, 30 s at 90 deg
LAND_AVG_N, LAND_HOLD, CONFIRM_SPEED, LAND_CLEAR = 4, 20.0, 0.3, 8.0
# obstacles + attenuation (placeholders; the user does not need these defended)
ALPHA_DB_PER_M, WALL_DB, WALL_T = 0.1, 15.0, 4.0
HIT_MEM, BIAS_RANGE, K_SHADOW, R_HUG_MIN = 6.0, 15.0, 0.9, 3.0
# memory / dead reckoning
LOG_PERIOD, BYTES_PER_PT = 1.0, 6
INIT_HEADING_SD = np.radians(2.0)                # magnetometer start-heading error (1 sigma)
GYRO_BIAS_SD = np.radians(0.003)                 # residual gyro bias, rad/s (~11 deg/h), 1 sigma
FLOW_SCALE_SD, FLOW_NOISE = 0.01, 0.10           # velocity scale error 1%, noise 0.1 m/s
CLEAR_DB = -6.0                                  # "landed clear" if TRUE level at landing is below this
S = 1e-3                                         # plot in km


# ------------------------------------------------------------- obstacles
class Obstacles:
    def __init__(self, kind, C=None, R=None, A=None, B=None):
        self.kind = kind
        self.C = np.zeros((0, 2)) if C is None else C
        self.R = np.zeros(0) if R is None else R
        self.A = np.zeros((0, 2)) if A is None else A
        self.B = np.zeros((0, 2)) if B is None else B


def seg_dist(P, A, B):
    e = B - A
    t = np.clip(((P[:, None, :] - A[None]) * e[None]).sum(-1) / (e * e).sum(-1)[None], 0, 1)
    return np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * e[None]), axis=-1)


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def attenuation(J0, P, obs):
    """Linear power fraction of each jammer reaching each point (N,J). Circles: dB per metre of chord; walls: dB per crossing."""
    N, nJ = len(P), len(J0)
    f = np.ones((N, nJ))
    if obs is None:
        return f
    d = P[:, None, :] - J0[None, :, :]
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
        f *= 10.0 ** (-0.1 * ALPHA_DB_PER_M * chord.sum(-1))
    if len(obs.A):
        e = (obs.B - obs.A)[None, None, :, :]
        dd = d[:, :, None, :]
        w = obs.A[None, None, :, :] - J0[None, :, None, :]
        den = cross(dd, e)
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        t, s = cross(w, e) / den, cross(w, dd) / den
        hit = ok & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
        f *= 10.0 ** (-0.1 * WALL_DB * hit.sum(-1))
    return f


def jam_lin(P, jam, obs):
    """Jam power relative to the denial level (1.0 = 0 dB = denied). Inverse-square, powers add."""
    P = np.atleast_2d(P)
    d = np.maximum(np.linalg.norm(P[:, None, :] - jam["C"][None], axis=-1), D_FLOOR)
    return ((jam["R"][None] / d) ** 2 * attenuation(jam["C"], P, obs)).sum(1)


def jam_db(P, jam, obs):
    return 10.0 * np.log10(jam_lin(P, jam, obs) + 1e-12)


# ------------------------------------------------------------- layouts
def make_layout(seed, kind):
    ra = np.random.default_rng(5000 + seed)
    while True:
        n = int(ra.integers(2, 5))
        C = np.c_[ra.uniform(0, 3000, n), ra.uniform(0, 2500, n)]
        R = ra.uniform(800, 1200, n)
        k = int(ra.integers(n))
        ang, rad = ra.uniform(0, 2 * np.pi), ra.uniform(0.3, 0.8) * R[k]
        start = C[k] + rad * np.array([np.cos(ang), np.sin(ang)])
        jam = dict(C=C, R=R)
        if jam_db(start, jam, None)[0] >= 1.5:
            break
    delay = ra.uniform(60, 180)
    ro = np.random.default_rng(7000 + seed + (0 if kind == "circles" else 50000))
    while True:
        if kind == "circles":
            m = 12                                         # hills / lakes / building blocks, 200-1000 m across
            Cc = start + np.c_[ro.uniform(-3500, 3500, m), ro.uniform(-3500, 3500, m)]
            Rr = ro.uniform(100, 500, m)
            if np.any(np.linalg.norm(Cc - start, axis=1) < Rr + 60):
                continue
            obs = Obstacles("circles", C=Cc, R=Rr)
        else:
            m = 8                                          # ridgelines / embankments, 0.4-1.6 km long
            c = start + np.c_[ro.uniform(-3500, 3500, m), ro.uniform(-3500, 3500, m)]
            half = ro.uniform(200, 800, m)[:, None]
            th = ro.uniform(0, np.pi, m)
            u = np.c_[np.cos(th), np.sin(th)]
            A, B = c - half * u, c + half * u
            if np.any(seg_dist(start[None], A, B) < 60):
                continue
            obs = Obstacles("walls", A=A, B=B)
        if jam_db(start, jam, obs)[0] >= 1.5:
            return dict(jam=jam, obs=obs, start=start, seed=seed, kind=kind, delay=delay)


def make_course(jitter=(0.0, 0.0), seed=-1):
    """Hand-built shadow course: one jammer, a big 'hill/building block' dead ahead of the escape path, two decoys."""
    jam = dict(C=np.array([[0.0, 0.0]]), R=np.array([1000.0]))
    obs = Obstacles("circles", C=np.array([[1100.0, 60.0], [1700.0, 800.0], [1650.0, -700.0]]),
                    R=np.array([250.0, 120.0, 150.0]))          # a big hill dead ahead, two decoy hills
    return dict(jam=jam, obs=obs, start=np.array([300.0, 40.0]) + np.asarray(jitter), seed=seed,
                kind="course", delay=120.0, theta0=np.radians(-30.0))   # launched already heading toward the hill


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
    return float(np.clip(best + rng.normal(0, RANGE_NOISE), 0.05, RANGE_MAX))


def collided(pos, obs):
    if len(obs.R) and np.any(np.linalg.norm(obs.C - pos, axis=1) < obs.R + BODY_R):
        return True
    return bool(len(obs.A) and np.any(seg_dist(pos[None], obs.A, obs.B) < WALL_T / 2 + BODY_R))

# 5 beams fanned across the front (SENS_ANGLES)
def beam_dirs(yaw):
    return np.array([(np.cos(yaw + a), np.sin(yaw + a)) for a in SENS_ANGLES])


def avoid(v_cmd, dirs, ranges, slide, in_jam=False, bias_vec=None):
    push = np.zeros(2)
    for d, r, w in zip(dirs, ranges, SENS_WEIGHTS):
        if r < R_SAFE:
            push -= w * d * (1.0 / r - 1.0 / R_SAFE) * R_SAFE    # dimensionless: 1 at r = R_SAFE/2 dead ahead; scaled by beam weight
    out = v_cmd + V * K_PUSH * push
    pn = np.linalg.norm(push)
    if pn > 0.05:
        tan = np.array([-push[1], push[0]]) / pn
        dots = np.dot(tan, v_cmd)
        if abs(dots) > 0.1 * V:
            slide = 1.0 if dots > 0 else -1.0
        out = out + V * K_SLIDE * min(1.0, pn) * slide * tan
    biased = False
    if in_jam and bias_vec is not None:       # still jammed and just slipped past an obstacle: lean toward its quiet lee side
        out = out + V * K_SHADOW * bias_vec
        biased = True
    sp = np.linalg.norm(out)
    return (out * 1.2 * V / sp if sp > 1.2 * V else out), slide, biased


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------- escape controller
def fit_source(Sarr):
    """Fit L = A - 20 log10(|p - c|) to the whole signal history; return the source position c (grid search + refine)."""
    P, Lm = Sarr[:, :2], Sarr[:, 2]
    if len(P) > 200:
        sel = np.linspace(0, len(P) - 1, 200).astype(int)
        P, Lm = P[sel], Lm[sel]
    cen = P[0]

    def cost(cx, cy):
        d = np.sqrt((P[None, :, 0] - cx[:, None]) ** 2 + (P[None, :, 1] - cy[:, None]) ** 2)
        r = Lm[None, :] + 20.0 * np.log10(np.maximum(d, D_FLOOR))          # = A for the right c
        w, wa = 1.0 / FIT_NOISE_DB ** 2, 1.0 / A_PRIOR_SD ** 2
        A = (w * r.sum(1) + wa * A_PRIOR_DB) / (w * r.shape[1] + wa)       # best A, pulled toward the prior
        return (w * ((r - A[:, None]) ** 2).sum(1) + wa * (A - A_PRIOR_DB) ** 2)

    g = np.arange(-3000.0, 3001.0, 200.0)
    GX, GY = np.meshgrid(cen[0] + g, cen[1] + g)
    gx, gy = GX.ravel(), GY.ravel()
    i = int(np.argmin(cost(gx, gy)))
    f = np.arange(-400.0, 401.0, 25.0)
    FX, FY = np.meshgrid(gx[i] + f, gy[i] + f)
    fx, fy = FX.ravel(), FY.ravel()
    j = int(np.argmin(cost(fx, fy)))
    return np.array([fx[j], fy[j]])


class EscapeController:
    def __init__(self, theta0):
        self.t, self.S, self.next_s, self.next_fit = 0.0, [], 0.0, T_PROBE
        self.theta0, self.c_hat, self.low_since = theta0, None, None
        self.heading = np.array([np.cos(theta0), np.sin(theta0)])

    def avg(self, n):
        return float(np.mean([s[2] for s in self.S[-n:]])) if len(self.S) >= n else np.inf

    def step(self, est, L, dt):
        self.t += dt
        if self.t >= self.next_s:
            self.S.append((est[0], est[1], L)); self.next_s += SAMPLE_DT
        if self.t < T_PROBE:
            th = self.theta0 + (0.0 if self.t < T_LEG else np.pi / 2)
            self.heading = np.array([np.cos(th), np.sin(th)])
        else:
            if self.c_hat is None or self.t >= self.next_fit:
                self.c_hat = fit_source(np.array(self.S)); self.next_fit = self.t + FIT_EVERY
            dv = est - self.c_hat
            n = np.linalg.norm(dv)
            away = dv / n if n > 1.0 else self.heading
            a = 1.0 - np.exp(-dt / HEAD_TAU)
            h = (1 - a) * self.heading + a * away
            self.heading = h / np.linalg.norm(h)
        return V * self.heading

    def clear_enough(self, space_ok=True):
        if self.t > T_PROBE and space_ok and self.avg(LAND_AVG_N) < -MARGIN_DB:
            self.low_since = self.t if self.low_since is None else self.low_since
            return self.t - self.low_since >= LAND_HOLD
        self.low_since = None
        return False


# ------------------------------------------------------------- simulation
def deadreckon(dp, yaw_t, t, scale, init_err, bias):
    """Re-create the dead-reckoned displacement from true displacements dp (n,2) at times t (n,)."""
    e = init_err + bias * t
    c, s = np.cos(e), np.sin(e)
    return scale * np.c_[c * dp[:, 0] - s * dp[:, 1], s * dp[:, 0] + c * dp[:, 1]]


def simulate(L, seed=0):
    rng = np.random.default_rng([seed, L["seed"] + 1])
    obs, jam = L["obs"], L["jam"]
    pos, yaw = L["start"].copy(), 0.0
    est = L["start"].copy()
    flow_scale = 1.0 + rng.normal(0, FLOW_SCALE_SD)
    gyro_bias = rng.normal(0, GYRO_BIAS_SD)
    yaw_err = rng.normal(0, INIT_HEADING_SD)
    fade = rng.normal(0, FADE_SD_DB)
    ctl, slide, t, tick, path_len, t_denied, n_enc, in_enc = EscapeController(L.get('theta0', rng.uniform(0, 2 * np.pi)) + (rng.normal(0, 0.1) if 'theta0' in L else 0.0)), 1.0, 0.0, 0, 0.0, 0.0, 0, False
    K = dict(t=[], x=[], y=[], ex=[], ey=[], j=[], phase=[], rng=[], yaw=[], bias=[])
    mem, hits, flag = [], [], [False]

    def log(phase, j, ranges):
        for k, v in zip(K, (t, pos[0], pos[1], est[0], est[1], j, phase, ranges, yaw, flag[0])):
            K[k].append(v)

    landed = crashed = False
    ranges = [RANGE_MAX] * len(SENS_ANGLES)
    while t < T_FLY_MAX:
        j_true = float(jam_db(pos, jam, obs)[0])
        dirs = beam_dirs(yaw)
        ranges = [ray_range(pos, d, obs, rng) for d in dirs]
        if tick % LOG_EVERY == 0:
            log("fly", j_true, ranges)
        mr = min(ranges)
        if mr < R_SAFE and not in_enc:
            n_enc += 1
        in_enc = mr < R_SAFE
        for d_, r_ in zip(dirs, ranges):
            if r_ < RANGE_MAX - 0.3:
                hits.append((t, est[0] + d_[0] * r_, est[1] + d_[1] * r_))
        hits[:] = [h for h in hits if t - h[0] <= HIT_MEM]
        bias_vec = None
        if hits and mr > R_HUG_MIN:
            H = np.array(hits)[:, 1:]
            dd = np.linalg.norm(H - est, axis=1)
            k_ = int(np.argmin(dd))
            if R_HUG_MIN < dd[k_] < BIAS_RANGE:
                bias_vec = (H[k_] - est) / dd[k_]
        if j_true > 0:
            t_denied += DT
        if tick % int(LOG_PERIOD / DT) == 0:
            mem.append((t, est[0], est[1]))
        c_ = path_len                                     # fading is correlated over distance travelled
        L_meas = j_true + fade + rng.normal(0, WHITE_SD_DB)
        v = ctl.step(est, L_meas, DT)
        in_jam = ctl.avg(LAND_AVG_N) > -MARGIN_DB
        v, slide, flag[0] = avoid(v, dirs, ranges, slide, in_jam, bias_vec)
        if ctl.low_since is not None:
            v = v * CONFIRM_SPEED
        if np.linalg.norm(v) > 0.05:
            yaw += np.clip(wrap(np.arctan2(v[1], v[0]) - yaw), -YAW_RATE * DT, YAW_RATE * DT)
        step = v * DT
        pos = pos + step
        ds = np.linalg.norm(step)
        path_len += ds
        rho = np.exp(-ds / FADE_DIST)
        fade = rho * fade + np.sqrt(1 - rho * rho) * FADE_SD_DB * rng.normal()
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        vb = np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]]) * flow_scale + rng.normal(0, FLOW_NOISE, 2)
        ye = yaw + yaw_err + gyro_bias * t
        est = est + np.array([np.cos(ye) * vb[0] - np.sin(ye) * vb[1],
                              np.sin(ye) * vb[0] + np.cos(ye) * vb[1]]) * DT
        t += DT; tick += 1
        if collided(pos, obs):
            crashed = True
            break
        if ctl.clear_enough(min(ranges) >= LAND_CLEAR):
            landed = True
            break
    t_fly = t
    flag[0] = False
    if landed:
        j_land = float(jam_db(pos, jam, obs)[0])
        for _ in range(int(LAND_TIME / 0.5)):
            log("land", j_land, ranges); t += 0.5
        mem.append((t, est[0], est[1]))
        t_land = t
        while t < t_land + L["delay"]:
            log("wait", j_land, ranges); t += 2.0
        log("done", j_land, ranges)
    else:
        log("fly", float(jam_db(pos, jam, obs)[0]), ranges)
    out = {k: np.array(v) for k, v in K.items()}
    mem = np.array(mem)
    j_open = float(jam_db(pos, jam, None)[0])
    rec_len = float(np.linalg.norm(np.diff(mem[:, 1:], axis=0), axis=1).sum()) if len(mem) > 1 else 0.0
    d_est = mem[-1, 1:] - mem[0, 1:]
    out.update(landed=landed, crashed=crashed, land_db=float(out["j"][-1]), j_open_db=j_open,
               sheltered=bool(landed and j_open > -MARGIN_DB + 3.0),
               mem=mem, path_len=path_len, rec_len=rec_len, t_fly=t_fly, t_denied=t_denied, n_enc=n_enc,
               land_pos=pos.copy(), start=L["start"].copy(),
               start_err=float(np.linalg.norm(d_est - (pos - L["start"]))))
    return out


def _sim_task(a):
    L, seed = a
    return simulate(L, seed)


def run_many(Ls, pool):
    return pool.map(_sim_task, [(L, 0) for L in Ls], chunksize=1)


def course_layouts(n, offset=0):
    return [make_course(jitter=np.random.default_rng(i + offset).uniform(-40, 40, 2), seed=i + offset) for i in range(n)]


# ------------------------------------------------------------- rescue
def recon(r, k):
    shift = r["land_pos"] - r["mem"][-1, 1:]
    return r["mem"][:, 1:] + shift, k * r["rec_len"]


def draw_rescue(ax, r, k, label=True):
    path, rad = recon(r, k)
    ax.plot(path[:, 0] * S, path[:, 1] * S, "--", color="green", lw=1.8, zorder=6)
    ax.add_patch(Circle(path[0] * S, rad * S, fc="green", alpha=0.25, ec="green", lw=1.5, zorder=2))
    if label:
        ax.text(path[0][0] * S, path[0][1] * S + rad * S + 0.15, "estimated soldier location", color="darkgreen",
                ha="center", fontsize=8, zorder=8)


def sens_ratio(runs, init_deg=None, bias_deg_s=None, scale_sd=None, draws=10, seed=1):
    """90th-percentile start-error / path-length if the navigation errors were different (re-dead-reckon the true tracks)."""
    g = np.random.default_rng(seed)
    init = np.radians(INIT_HEADING_SD * 0 + (np.degrees(INIT_HEADING_SD) if init_deg is None else init_deg))
    bias = np.radians(np.degrees(GYRO_BIAS_SD) if bias_deg_s is None else bias_deg_s)
    ssd = FLOW_SCALE_SD if scale_sd is None else scale_sd
    ratios = []
    for r in runs:
        m = r["phase"] == "fly"
        m = m & (r["j"] == r["j"])
        pts = np.c_[r["x"][m], r["y"][m]]
        tt = r["t"][m]
        if len(pts) < 5:
            continue
        dp = np.diff(pts, axis=0)
        L = np.linalg.norm(dp, axis=1).sum()
        true_d = dp.sum(0)
        for _ in range(draws):
            e = deadreckon(dp, None, tt[:-1], 1.0 + g.normal(0, ssd), g.normal(0, init), g.normal(0, bias))
            ratios.append(np.linalg.norm(e.sum(0) - true_d) / L)
    return float(np.percentile(ratios, 90)), float(np.mean(ratios))


# ------------------------------------------------------------- drawing
def view_box(L, r, pad=700.0, minw=3500.0):
    pts = [L["jam"]["C"], L["start"][None], np.c_[r["x"], r["y"]], r["land_pos"][None]]
    P = np.vstack(pts)
    x0, y0 = P.min(0) - pad
    x1, y1 = P.max(0) + pad
    w, h = max(x1 - x0, minw), max(y1 - y0, minw)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(w, h)
    return (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)


_zc = {}


def draw(ax, L, box, n=130):
    x0, x1, y0, y1 = box
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    key = (L["kind"], L["seed"], n, tuple(np.round(box, 0)))
    if key not in _zc:
        XX, YY = np.meshgrid(xs, ys)
        _zc[key] = jam_db(np.c_[XX.ravel(), YY.ravel()], L["jam"], L["obs"]).reshape(XX.shape)
    Z = _zc[key]
    ax.imshow(Z, origin="lower", extent=[x0 * S, x1 * S, y0 * S, y1 * S], cmap="Reds", vmin=-MARGIN_DB, vmax=15,
              alpha=0.8, aspect="equal")
    ax.contour(xs * S, ys * S, Z, levels=[0.0], colors="k", linestyles=":", linewidths=1.0)
    ax.contour(xs * S, ys * S, Z, levels=[-MARGIN_DB], colors="orange", linestyles="--", linewidths=1.2)
    for c in L["jam"]["C"]:
        ax.plot(c[0] * S, c[1] * S, "rx", ms=8, mew=2)
    o = L["obs"]
    for c, r in zip(o.C, o.R):
        ax.add_patch(Circle(c * S, r * S, fc="#444444", ec="k", lw=1.0, zorder=3))
    for a, b in zip(o.A, o.B):
        ax.plot([a[0] * S, b[0] * S], [a[1] * S, b[1] * S], color="#222222", lw=2.5, solid_capstyle="butt", zorder=3)
    ax.plot(L["start"][0] * S, L["start"][1] * S, marker="*", color="gold", mec="k", ms=15, zorder=4)
    ax.set_xlim(x0 * S, x1 * S); ax.set_ylim(y0 * S, y1 * S)


def summarize(name, rs):
    ld = [r for r in rs if r["landed"]]
    ok = [r for r in ld if r["land_db"] <= CLEAR_DB]
    tl = [r["t_fly"] / 60 for r in ld]
    marg = [-r["land_db"] for r in ld]
    print(f"  {name}: landed clear {len(ok)}/{len(rs)}, crashed {sum(r['crashed'] for r in rs)}, "
          f"no escape in {T_FLY_MAX / 60:.0f} min {sum((not r['landed']) and (not r['crashed']) for r in rs)}; "
          f"obstacle encounters in {100 * np.mean([r['n_enc'] > 0 for r in rs]):.0f}% of runs; "
          f"shadow landings {sum(r['sheltered'] for r in ld)}/{len(ld)}")
    if ld:
        print(f"      flight time median {np.median(tl):.1f} min (90th pct {np.percentile(tl, 90):.1f}, max {max(tl):.1f}); "
              f"path {np.median([r['path_len'] for r in ld]) / 1000:.1f} km; landing {np.median([np.linalg.norm(r['land_pos'] - r['start']) for r in ld]) / 1000:.1f} km from launch; "
              f"margin at landing median {np.median(marg):.1f} dB (10th pct {np.percentile(marg, 10):.1f}); "
              f"time inside the denial zone median {np.median([r['t_denied'] for r in ld]) / 60:.1f} min")


def verdict(r):
    return "CRASHED" if r["crashed"] else ("landed" if r["landed"] else "no escape")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    pool = Pool(2)
    Lc = [make_layout(s, "circles") for s in range(N)]
    Lw = [make_layout(s, "walls") for s in range(N)]
    Lk = course_layouts(N)
    print(f"Simulating {N} circle fields...", flush=True); rc = run_many(Lc, pool)
    print(f"Simulating {N} wall fields...", flush=True); rw = run_many(Lw, pool)
    print(f"Simulating {N} shadow-course runs...", flush=True); rk = run_many(Lk, pool)
    print(f"Real units: jam radius ~1 km (1/d^2, power sum), V={V} m/s, lidar {RANGE_MAX:.0f} m, land at {MARGIN_DB:.0f} dB margin, "
          f"endurance cap {T_FLY_MAX / 60:.0f} min")
    print("RANDOM terrain (12 hills 0.2-1 km across / 8 ridgelines 0.4-1.6 km long, 2-4 jammers)")
    summarize("circles", rc); summarize("walls  ", rw)
    print("SHADOW COURSE (hard-coded; launch offsets +/-40 m)")
    summarize("course ", rk)

    cal = [r for r in rc + rw + rk if r["landed"] and r["rec_len"] > 0]
    ratios = [r["start_err"] / r["rec_len"] for r in cal]
    k90 = float(np.percentile(ratios, 90))
    print(f"\nRESCUE calibration (pooled): start-location error averages {100 * np.mean(ratios):.1f}% of the recorded path; "
          f"90th percentile {100 * k90:.1f}%  (typical path {np.median([r['rec_len'] for r in cal]) / 1000:.1f} km "
          f"-> circle radius about {k90 * np.median([r['rec_len'] for r in cal]):.0f} m)")

    # hold-out
    Lh = [make_layout(s + N, "circles") for s in range(N // 2)] + [make_layout(s + N, "walls") for s in range(N // 2)]
    print("Running hold-out simulations...", flush=True)
    fh, fk = run_many(Lh, pool), run_many(course_layouts(N // 2, offset=N), pool)
    print("HOLD-OUT check on fresh runs:")
    for name, grp in (("random fields", [r for r in fh if r["landed"]]), ("shadow course", [r for r in fk if r["landed"]])):
        cov = np.mean([r["start_err"] <= k90 * r["rec_len"] for r in grp])
        print(f"  {name}: circle contained the true start {100 * cov:.0f}% of the time "
              f"(mean radius {np.mean([k90 * r['rec_len'] for r in grp]):.0f} m)")

    print("\nSENSITIVITY: 90th-percentile start error as % of path, if the navigation errors were different")
    base = [r for r in rc + rw if r["landed"]]
    print(f"  as simulated (init heading {np.degrees(INIT_HEADING_SD):.0f} deg, gyro bias {np.degrees(GYRO_BIAS_SD):.3f} deg/s, "
          f"scale {100 * FLOW_SCALE_SD:.0f}%): {100 * sens_ratio(base)[0]:.1f}%")
    for lab, kw in (("start heading 0.5 deg", dict(init_deg=0.5)), ("start heading 1 deg", dict(init_deg=1.0)),
                    ("start heading 5 deg", dict(init_deg=5.0)), ("start heading 10 deg (magnetometer fooled)", dict(init_deg=10.0)),
                    ("gyro bias 0.0003 deg/s (~1 deg/h)", dict(bias_deg_s=0.0003)),
                    ("gyro bias 0.03 deg/s (consumer MEMS)", dict(bias_deg_s=0.03)),
                    ("velocity scale error 0.3%", dict(scale_sd=0.003)), ("velocity scale error 3%", dict(scale_sd=0.03))):
        print(f"  {lab}: {100 * sens_ratio(base, **kw)[0]:.1f}%")
    pool.close()

    # ---------- random grid: top circles, bottom walls
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for row, (Ls, rs) in enumerate(((Lc, rc), (Lw, rw))):
        for col in range(3):
            ax, L, r = axs[row, col], Ls[col], rs[col]
            box = view_box(L, r)
            draw(ax, L, box)
            m = r["phase"] != "wait"
            ax.plot(r["x"][m] * S, r["y"][m] * S, "b-", lw=1.8, zorder=5)
            ax.plot(*(r["land_pos"] * S), "s" if r["landed"] else "x", color="navy" if r["landed"] else "k", ms=8, zorder=7)
            if r["landed"]:
                draw_rescue(ax, r, k90, label=False)
            ax.set_title(f"{L['kind']}, seed {L['seed']}: {verdict(r)}"
                         f"{', in shadow' if r['sheltered'] else ''}; {r['t_fly'] / 60:.0f} min, "
                         f"{-r['land_db']:.0f} dB margin; start est. off {r['start_err']:.0f} m", fontsize=8)
            ax.set_xlabel("km")
    fig.text(0.5, 0.005, "blue = actual path   gold star = soldier (launch)   navy square = landing   black dotted = denial boundary (0 dB)   "
             "orange dashed = 10 dB margin   green = rescuer's backtrack + location radius", ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.savefig("jam_real_random_grid.png", dpi=100)
    plt.close(fig)

    # ---------- course figure
    ik = next((i for i, r in enumerate(rk) if r["landed"] and r["sheltered"] and r["land_db"] <= CLEAR_DB), 0)
    L0, r0 = Lk[ik], rk[ik]                      # first typical success on the course (the batch above reports every failure)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7))
    for a_, box in ((ax, view_box(L0, r0)), (bx, (300, 2100, -900, 900))):
        draw(a_, L0, box)
        m = r0["phase"] != "wait"
        a_.plot(r0["x"][m] * S, r0["y"][m] * S, "b-", lw=2.2, zorder=5)
        a_.plot(*(r0["land_pos"] * S), "s", color="navy", ms=9, zorder=7)
        draw_rescue(a_, r0, k90, label=(a_ is ax))
        a_.set_xlabel("km")
    ax.set_title(f"Shadow course: {verdict(r0)} {np.linalg.norm(r0['land_pos'] - r0['start']) / 1000:.1f} km from launch, "
                 f"{r0['t_fly'] / 60:.1f} min{' (in shadow)' if r0['sheltered'] else ''}")
    bx.set_title("zoom on the obstacle")
    plt.tight_layout(); plt.savefig("jam_real_course.png", dpi=100); plt.close(fig)

    print("Done: saved jam_real_random_grid.png and jam_real_course.png", flush=True)



if __name__ == "__main__":
    main()
