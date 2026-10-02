#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocol_stability_test.py -- is the prime topological imprint a kinetic
artefact, or a robust structural property?

Two independent defenses against the "glassy kinetic trap" objection, in one
driver. Neither claims equilibrium sampling; together they establish that the
measured effect does not depend on how the chain was cooled, and that the state
in which it is measured does not drift.

  STAGE B -- PROTOCOL INDEPENDENCE
    N = 4x10^4, arms True and S1, at 1x / 2x / 4x / 8x the production annealing
    length (8k / 16k / 32k / 64k steps), n = 40 per arm per protocol.
    A kinetic artefact moves with the cooling rate; a structural property does
    not. The test is whether the ordering gap
          Delta rho = rho_True - rho_S1
    is CONSTANT across a factor of 8 in annealing length. Reported as a weighted
    regression of Delta rho on log2(protocol) with the slope tested against zero,
    plus a chi^2 test of the constant-gap hypothesis.
    The 1x point is imported from s5_scaling_results.csv (already computed);
    only 2x, 4x and 8x are simulated.

  STAGE C -- LOW-TEMPERATURE METASTABILITY
    Fully annealed N = 4x10^4 conformations are held at fixed kT = 0.003 for
    N_HOLD integration steps and rho is logged every 1000 steps. A flat trace
    shows the measurement is not being taken mid-relaxation.
      * dt is HALVED to 0.004 and the per-bead DISPLACEMENT CAP IS REMOVED, so
        the propagator is a clean Euler-Maruyama step with no state-dependent
        truncation. (At dt=0.004 and kT=0.003 the cap never binds; the script
        logs the observed maximum displacement so this can be verified, not
        assumed.)
      * The pairwise force is UNCHANGED from production (WCA capped at 30). This
        is deliberate: the question is whether the state is stable under the
        dynamics that produced it. Annealed conformations retain ~40-80 pairs
        inside r_cap = 0.988, so the cap is active and switching it off would
        guarantee a relaxation artefact rather than test metastability.
      * Energies are reported using the conservative potential whose gradient IS
        that capped force (linear extension below r_cap), so E is well defined
        while the trajectory is bit-identical to production dynamics.
      * Both True and S1 are held, n = 5 each. The S1 control matters: drift in
        both arms would indicate incomplete thermalisation generally, whereas
        drift in True alone would undermine the specific claim. Set
        PDTP_C_ARMS=True to run only the prime arm.

ENGINEERING
  Per-trajectory fsync CSV checkpointing; exact auto-resume; resilient process
  pool with sequential fallback; per-task exception isolation. Stage C
  trajectories are long (hours each), so they additionally checkpoint their
  coordinates and partial trace to .npz every PDTP_C_CKPT steps and resume
  mid-trajectory.

USAGE
  python3 protocol_stability_test.py                 # both stages, auto-resume
  PDTP_STAGE=B python3 protocol_stability_test.py    # one stage only
  PDTP_WORKERS=15 python3 protocol_stability_test.py
  PDTP_C_HOLD=250000 python3 protocol_stability_test.py   # shorter Stage C
  PDTP_C_ARMS=True python3 protocol_stability_test.py     # prime arm only
  PDTP_ANALYZE_ONLY=1 python3 protocol_stability_test.py  # re-run statistics
  SMOKE=1 python3 protocol_stability_test.py         # seconds-long sanity run

COST (single core, calibrated at ~0.1125 s/step for N=4x10^4)
  Stage B: 2 arms x 40 runs x (16k+32k+64k) steps        ~ 280 core-hours
  Stage C: 10 trajectories x 10^6 steps                  ~ 310 core-hours
  Total ~ 590 core-hours; ~40 h wall clock on 15 cores. Stage C dominates and
  can be quartered with PDTP_C_HOLD=250000, which still holds the chain for
  1000 reduced time units versus the 64 spanned by the entire anneal.
Dependencies: numpy, scipy.
"""

import os
_WREQ = os.environ.get("PDTP_WORKERS")
if _WREQ is None or int(_WREQ) != 1:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import sys, time, csv, traceback
from datetime import datetime
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
import concurrent.futures as cf

# ============================== configuration ================================
SMOKE      = os.environ.get("SMOKE", "0") == "1"
BASE_SEED  = 20261101
STAGE_SEL  = os.environ.get("PDTP_STAGE", "BC").upper()
MAX_RUN    = int(os.environ.get("PDTP_MAXRUN", 10**9))
ANALYZE    = os.environ.get("PDTP_ANALYZE_ONLY", "0") == "1"
try:
    N_WORKERS = int(os.environ["PDTP_WORKERS"])
except (KeyError, ValueError):
    N_WORKERS = max(1, (os.cpu_count() or 1) - 1)

N_SYS      = 800 if SMOKE else 40_000
BASE_STEPS = 200 if SMOKE else 8_000                       # 1x production anneal
MULTS      = [1, 2, 4] if SMOKE else [1, 2, 4, 8]
N_ENS      = 3 if SMOKE else 40
C_ARMS     = os.environ.get("PDTP_C_ARMS", "True,S1").split(",")
C_N        = 2 if SMOKE else 5
C_HOLD     = int(os.environ.get("PDTP_C_HOLD", 4_000 if SMOKE else 1_000_000))
C_LOG      = int(os.environ.get("PDTP_C_LOG", 200 if SMOKE else 1_000))
C_CKPT     = int(os.environ.get("PDTP_C_CKPT", 2_000 if SMOKE else 100_000))
C_DT       = float(os.environ.get("PDTP_C_DT", 0.004))     # halved; no disp cap

CSV_B      = os.environ.get("PDTP_CSV_B", "protocol_independence_results.csv")
CSV_C      = os.environ.get("PDTP_CSV_C", "metastability_traces.csv")
PRIOR_CSV  = "s5_scaling_results.csv"                      # source of the 1x point
CKPT_DIR   = os.environ.get("PDTP_CKPT_DIR", "stage_c_ckpt")

HDR_B = ["N", "Model", "Steps", "Run_ID", "spearman_rho", "dR"]
HDR_C = ["N", "Model", "Run_ID", "step", "spearman_rho", "dR", "energy", "max_disp"]

# force field / dynamics -- identical to production
B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003
R_CAP = 0.9880711080486  # where |F_WCA| = FCAP; below this the force is constant


# ============================== arithmetic core ==============================
def sieve_primes(n):
    ip = np.ones(n + 1, dtype=bool); ip[:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if ip[i]: ip[i * i::i] = False
    return ip

def gap_from_mask(is_anchor, N):
    anchors = np.where(is_anchor)[0]; g = np.zeros(N)
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b - a > 1: g[a + 1:b] = b - a
    g[0:anchors[0]] = anchors[0]; g[anchors[-1] + 1:N] = N - anchors[-1]; return g

def stiffness_from_gap(g, is_anchor):
    k = np.full(len(g), K_HINGE); comp = ~is_anchor
    k[comp] = np.minimum(K_HINGE * np.exp(ALPHA * g[comp]), K_MAX); return k

def mask_shuffle(primes, N, rng):
    gaps = np.diff(primes).copy(); rng.shuffle(gaps)
    newpos = primes[0] + np.concatenate(([0], np.cumsum(gaps)))
    mask = np.zeros(N, dtype=bool); mask[newpos] = True; return mask

def build_arm(model, N, rng):
    sb = sieve_primes(N - 1); primes = np.where(sb)[0]
    is_anchor = sb.copy() if model == "True" else mask_shuffle(primes, N, rng)
    return stiffness_from_gap(gap_from_mask(is_anchor, N), is_anchor)


# ================================ physics ====================================
def _scatter_add(F, idx, vals):
    n = F.shape[0]
    for c in range(3): F[:, c] += np.bincount(idx, weights=vals[:, c], minlength=n)

def _pairs(r):
    tree = cKDTree(r); p = tree.query_pairs(RC_LJ * SIGMA, output_type="ndarray")
    if len(p) == 0: return p[:0], p[:0]
    i, j = p[:, 0], p[:, 1]; keep = np.abs(i - j) > 1
    return i[keep], j[keep]

def compute_forces(r, kbend):
    n = r.shape[0]; F = np.zeros_like(r)
    d = r[1:] - r[:-1]; L = np.linalg.norm(d, axis=1) + 1e-12
    fb = (K_BOND * (L - B0) / L)[:, None] * d; F[0:n - 1] += fb; F[1:n] -= fb
    a, b = r[1:-1] - r[:-2], r[2:] - r[1:-1]
    na = np.linalg.norm(a, axis=1) + 1e-12; nb = np.linalg.norm(b, axis=1) + 1e-12
    ah, bh = a / na[:, None], b / nb[:, None]
    cos = np.clip(np.sum(ah * bh, axis=1), -1.0, 1.0); kk = kbend[1:-1][:, None]
    dda = (bh - cos[:, None] * ah) / na[:, None]; ddb = (ah - cos[:, None] * bh) / nb[:, None]
    F[0:n - 2] += -kk * dda; F[2:n] += kk * ddb; F[1:n - 1] += kk * (dda - ddb)
    i, j = _pairs(r)
    if len(i) > 0:
        for s in range(0, len(i), 4_000_000):
            ii, jj = i[s:s + 4_000_000], j[s:s + 4_000_000]
            rij = r[ii] - r[jj]; dist = np.linalg.norm(rij, axis=1) + 1e-12
            sr6 = (SIGMA / dist) ** 6
            fmag = np.clip(24 * EPS_LJ * (2 * sr6 ** 2 - sr6) / dist, 0.0, FCAP_LJ)
            fvec = (fmag / dist)[:, None] * rij
            _scatter_add(F, ii, fvec); _scatter_add(F, jj, -fvec)
    if K_CONF > 0.0: F += -K_CONF * (r - r.mean(axis=0))
    return F

def total_energy(r, kbend):
    """Configurational energy under the conservative potential whose gradient is
    exactly the production (capped) force. Used as a Stage-C drift diagnostic."""
    d = r[1:] - r[:-1]; L = np.linalg.norm(d, axis=1)
    e = 0.5 * K_BOND * np.sum((L - B0) ** 2)
    a, b = r[1:-1] - r[:-2], r[2:] - r[1:-1]
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    e += np.sum(kbend[1:-1] * (1.0 - np.clip(cos, -1, 1)))
    e += 0.5 * K_CONF * np.sum((r - r.mean(axis=0)) ** 2)
    i, j = _pairs(r)
    if len(i) > 0:
        dist = np.linalg.norm(r[i] - r[j], axis=1)
        u_wca = lambda x: 4 * EPS_LJ * ((SIGMA / x) ** 12 - (SIGMA / x) ** 6) + EPS_LJ
        out = np.where(dist >= R_CAP, u_wca(np.clip(dist, R_CAP, None)),
                       u_wca(R_CAP) + FCAP_LJ * (R_CAP - dist))
        e += float(np.sum(out))
    return float(e)

def init_chain(n, rng, persistence=0.55):
    dirs = np.zeros((n, 3)); dv = rng.standard_normal(3); dv /= np.linalg.norm(dv); dirs[0] = dv
    for i in range(1, n):
        dv = persistence * dv + (1 - persistence) * rng.standard_normal(3)
        dv /= np.linalg.norm(dv) + 1e-12; dirs[i] = dv
    r = np.cumsum(dirs, axis=0) * B0; return r - r.mean(axis=0)

def anneal(r, kbend, n_steps, rng):
    """Production protocol: geometric cooling, dt=0.008, displacement cap ON."""
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1); kT = KT_HI * (KT_LO / KT_HI) ** frac
        F = compute_forces(r, kbend)
        disp = MOBILITY * F * DT + np.sqrt(2.0 * MOBILITY * kT * DT) * rng.standard_normal(r.shape)
        dm = np.linalg.norm(disp, axis=1); big = dm > MAX_DISP
        if np.any(big): disp[big] *= (MAX_DISP / dm[big])[:, None]
        r += disp
    return r

def hold_isothermal(r, kbend, n_steps, rng, dt=C_DT, kT=KT_LO):
    """Stage C: fixed kT, halved dt, NO displacement cap. Yields the observed
    maximum per-bead displacement so the removal can be verified empirically."""
    md = 0.0
    for _ in range(n_steps):
        F = compute_forces(r, kbend)
        disp = MOBILITY * F * dt + np.sqrt(2.0 * MOBILITY * kT * dt) * rng.standard_normal(r.shape)
        md = max(md, float(np.max(np.linalg.norm(disp, axis=1))))
        r += disp
    return r, md

def observables(r, kbend):
    rad = np.linalg.norm(r - r.mean(axis=0), axis=1)
    rho, _ = stats.spearmanr(kbend, rad)
    hi = kbend > np.percentile(kbend, 80); lo = kbend < np.percentile(kbend, 20)
    return float(rho), float(rad[hi].mean() - rad[lo].mean())


# ================================ CSV I/O ====================================
def _ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _fmt(sec):
    if sec is None or sec != sec or sec < 0: return "?"
    h, rr = divmod(int(sec), 3600); m, s = divmod(rr, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

def append_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row); f.flush()
        try: os.fsync(f.fileno())
        except OSError: pass

def load_rows(path):
    out = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rdr = csv.reader(f); next(rdr, None)
            for r in rdr:
                if r: out.append(r)
    return out

def _find(name):
    for c in (name, os.path.join("data", name), os.path.join("..", "data", name)):
        if os.path.exists(c): return c
    return None

def import_1x():
    """Bring the already-computed 1x anneal (True/S1 at N, BASE_STEPS) across."""
    src = _find(PRIOR_CSV)
    if src is None:
        print(f"[{_ts()}]  {PRIOR_CSV} not found -- the 1x protocol point will be simulated.")
        return
    have = {(r[1], int(r[2]), int(r[3])) for r in load_rows(CSV_B)}
    added = 0
    with open(src, newline="") as f:
        rdr = csv.reader(f); next(rdr, None)
        for r in rdr:
            if len(r) < 6: continue
            if int(r[0]) == N_SYS and r[1] in ("True", "S1") and r[2] == "NA" and int(r[3]) < N_ENS:
                key = (r[1], BASE_STEPS, int(r[3]))
                if key not in have:
                    append_row(CSV_B, HDR_B, [N_SYS, r[1], BASE_STEPS, r[3], r[4], r[5]])
                    added += 1
    if added: print(f"[{_ts()}]  imported {added} rows for the 1x protocol from {src}")


# ============================== job execution ================================
def seed_B(model, steps, run):
    return int(BASE_SEED + {"True": 0, "S1": 1}[model] * 1_000_000 + steps * 7 + run)

def seed_C(model, run):
    return int(BASE_SEED + 500_000 + {"True": 0, "S1": 1}[model] * 10_000 + run)

def run_job_B(job):
    _, model, steps, run = job
    t0 = time.time()
    rng = np.random.default_rng(seed_B(model, steps, run))
    kbend = build_arm(model, N_SYS, rng)
    r = init_chain(N_SYS, rng)
    r = anneal(r, kbend, steps, rng)
    rho, dR = observables(r, kbend)
    return ("B", model, steps, run, rho, dR, time.time() - t0)

def run_job_C(job):
    """Anneal once (production protocol), then hold isothermally, logging rho.
    Checkpoints coordinates + partial trace so a multi-hour run can resume."""
    _, model, run = job
    t0 = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    ck = os.path.join(CKPT_DIR, f"c_N{N_SYS}_{model}_r{run}.npz")
    rng = np.random.default_rng(seed_C(model, run))
    kbend = build_arm(model, N_SYS, rng)

    if os.path.exists(ck):
        z = np.load(ck)
        r = z["r"].astype(np.float64); done = int(z["step"]); md = float(z["max_disp"])
        rng = np.random.default_rng(seed_C(model, run) + 10_000 + done)
    else:
        r = init_chain(N_SYS, rng)
        r = anneal(r, kbend, BASE_STEPS, rng)          # the production state
        done, md = 0, 0.0
        rho, dR = observables(r, kbend)
        append_row(CSV_C, HDR_C, [N_SYS, model, run, 0, f"{rho:.6f}", f"{dR:.6f}",
                                  f"{total_energy(r, kbend):.4f}", "0.0"])
        np.savez_compressed(ck, r=r.astype(np.float32), step=0, max_disp=0.0)

    while done < C_HOLD:
        chunk = min(C_LOG, C_HOLD - done)
        r, m = hold_isothermal(r, kbend, chunk, rng)
        md = max(md, m); done += chunk
        rho, dR = observables(r, kbend)
        append_row(CSV_C, HDR_C, [N_SYS, model, run, done, f"{rho:.6f}", f"{dR:.6f}",
                                  f"{total_energy(r, kbend):.4f}", f"{md:.4f}"])
        if done % C_CKPT == 0 or done >= C_HOLD:
            np.savez_compressed(ck, r=r.astype(np.float32), step=done, max_disp=md)
    return ("C", model, 0, run, float("nan"), float("nan"), time.time() - t0)


def build_queue():
    jobs = []
    if "B" in STAGE_SEL:
        have = {(r[1], int(r[2]), int(r[3])) for r in load_rows(CSV_B)}
        for mult in MULTS:
            steps = BASE_STEPS * mult
            for model in ("True", "S1"):
                for run in range(N_ENS):
                    if (model, steps, run) not in have:
                        jobs.append(("B", model, steps, run))
    if "C" in STAGE_SEL:
        done_c = {}
        for r in load_rows(CSV_C):
            done_c[(r[1], int(r[2]))] = max(done_c.get((r[1], int(r[2])), 0), int(r[3]))
        for model in C_ARMS:
            for run in range(C_N):
                if done_c.get((model, run), -1) < C_HOLD:
                    jobs.append(("C", model, run))
    return jobs


# ================================ analysis ===================================
def welch(a, b):
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    d = (a.mean() - b.mean()) / (np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2) + 1e-12)
    return a.mean() - b.mean(), p, d

def analyze_B():
    rows = load_rows(CSV_B)
    if not rows: print("\n[Stage B] no data yet."); return
    d = {}
    for r in rows:
        d.setdefault((r[1], int(r[2])), []).append(float(r[4]))
    d = {k: np.array(v) for k, v in d.items()}
    steps = sorted({k[1] for k in d})
    print("\n" + "=" * 78)
    print("STAGE B -- PROTOCOL INDEPENDENCE   (is the gap constant vs cooling rate?)")
    print("=" * 78)
    print(f"  {'anneal':>8} {'mult':>5} {'rho_True':>18} {'rho_S1':>18} {'gap':>9} {'p':>9} {'d':>6}")
    G, E, X = [], [], []
    for st in steps:
        T = d.get(("True", st)); S = d.get(("S1", st))
        if T is None or S is None or min(len(T), len(S)) < 2: continue
        g, p, dd = welch(T, S)
        se = np.sqrt(T.std(ddof=1)**2 / len(T) + S.std(ddof=1)**2 / len(S))
        G.append(g); E.append(se); X.append(np.log2(st / BASE_STEPS))
        print(f"  {st:>8} {st//BASE_STEPS:>4}x "
              f"{T.mean():>+9.4f}+/-{T.std(ddof=1)/np.sqrt(len(T)):.4f} "
              f"{S.mean():>+9.4f}+/-{S.std(ddof=1)/np.sqrt(len(S)):.4f} "
              f"{g:>+9.4f} {p:>9.1e} {dd:>+6.2f}")
    if len(G) < 2:
        print("  (need >=2 protocols for the constancy test)"); return
    G, E, X = np.array(G), np.array(E), np.array(X)
    w = 1 / E**2
    gbar = np.sum(w * G) / np.sum(w); chi2 = np.sum(((G - gbar) / E)**2); dof = len(G) - 1
    Sx, Sxx = np.sum(w * X), np.sum(w * X * X)
    det = np.sum(w) * Sxx - Sx**2
    slope = (np.sum(w) * np.sum(w * X * G) - Sx * np.sum(w * G)) / det
    slope_se = np.sqrt(np.sum(w) / det)
    print(f"\n  weighted mean gap        : {gbar:+.4f}")
    print(f"  constant-gap hypothesis  : chi2 = {chi2:.2f} / {dof} dof   "
          f"p = {1 - stats.chi2.cdf(chi2, dof):.3f}   (large p = constant)")
    print(f"  slope vs log2(protocol)  : {slope:+.5f} +/- {slope_se:.5f}   "
          f"({abs(slope)/slope_se:.1f} sigma from zero)")
    verdict = ("CONSTANT -- the gap does not track the cooling rate; the effect is "
               "not a kinetic artefact." if (1 - stats.chi2.cdf(chi2, dof)) > 0.05
               and abs(slope) / slope_se < 2 else
               "DRIFTING -- the gap depends on annealing length; report this honestly "
               "and investigate before claiming protocol independence.")
    print(f"  => {verdict}")

def analyze_C():
    rows = load_rows(CSV_C)
    if not rows: print("\n[Stage C] no data yet."); return
    tr = {}
    for r in rows:
        tr.setdefault((r[1], int(r[2])), []).append((int(r[3]), float(r[4]), float(r[6])))
    print("\n" + "=" * 78)
    print("STAGE C -- LOW-TEMPERATURE METASTABILITY   (does rho drift while held?)")
    print("=" * 78)
    REF_GAP = 0.0457     # the N=4x10^4 True-S1 gap: the scale that matters
    print(f"  reference: the effect being defended is a gap of {REF_GAP:.4f} in rho")
    print(f"  {'arm':>6} {'run':>4} {'steps':>9} {'rho drift':>11} {'vs gap':>8} "
          f"{'slope sigma':>12} {'dE/E':>9} {'max disp':>9}")
    agg = {}
    for (model, run), pts in sorted(tr.items()):
        pts.sort()
        t = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
        if len(t) < 10: continue
        sl, ic, _, _, se = stats.linregress(t, y)
        drift = sl * (t[-1] - t[0])
        de = (e[-1] - e[0]) / abs(e[0] + 1e-12)
        md = max(float(r[7]) for r in rows if r[1] == model and int(r[2]) == run)
        agg.setdefault(model, []).append((drift, sl / (se + 1e-30)))
        print(f"  {model:>6} {run:>4} {int(t[-1]):>9} {drift:>+11.5f} "
              f"{100*abs(drift)/REF_GAP:>7.1f}% {sl/(se+1e-30):>+12.1f} "
              f"{100*de:>+8.2f}% {md:>9.4f}")
    print()
    for model, v in agg.items():
        dr = np.array([x[0] for x in v])
        print(f"  {model:>6}: mean drift over the hold = {dr.mean():+.5f} "
              f"({100*abs(dr.mean())/REF_GAP:.1f}% of the gap)")
    print(f"\n  Displacement cap: production value was {MAX_DISP}; Stage C removed it and")
    print(f"  ran at dt={C_DT}. If 'max disp' above stays well below {MAX_DISP}, the cap")
    print(f"  was never binding and its removal changed nothing -- verified, not assumed.")
    print("  A drift small compared with the gap means the measurement is taken in a")
    print("  stable basin, not mid-relaxation.")


# ================================= driver ====================================
def _sequential(jobs, prog):
    for job in jobs:
        try:
            res = run_job_B(job) if job[0] == "B" else run_job_C(job)
            if res[0] == "B":
                append_row(CSV_B, HDR_B, [N_SYS, res[1], res[2], res[3],
                                          f"{res[4]:.6f}", f"{res[5]:.6f}"])
            prog.tick(res)
        except KeyboardInterrupt: raise
        except Exception:
            print(f"[{_ts()}]  ERROR on {job} (will retry next run):\n{traceback.format_exc()}",
                  flush=True)

def _pool(jobs, prog, workers):
    try:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_job_B if j[0] == "B" else run_job_C, j): j for j in jobs}
            for fut in cf.as_completed(futs):
                job = futs[fut]
                try:
                    res = fut.result()
                    if res[0] == "B":
                        append_row(CSV_B, HDR_B, [N_SYS, res[1], res[2], res[3],
                                                  f"{res[4]:.6f}", f"{res[5]:.6f}"])
                    prog.tick(res)
                except Exception:
                    print(f"[{_ts()}]  ERROR on {job} (will retry next run):\n"
                          f"{traceback.format_exc()}", flush=True)
    except (cf.process.BrokenProcessPool, OSError) as e:
        print(f"[{_ts()}]  pool broke ({e}); finishing SEQUENTIALLY.", flush=True)
        _sequential(build_queue(), prog)

class Progress:
    def __init__(self, total, workers):
        self.n = 0; self.total = total; self.workers = max(1, workers); self.t = []
    def tick(self, res):
        self.n += 1; self.t.append(res[6])
        eta = (sum(self.t[-10:]) / len(self.t[-10:])) * (self.total - self.n) / self.workers
        tag = (f"B {res[1]:<5} {res[2]:>6} steps" if res[0] == "B"
               else f"C {res[1]:<5} hold {C_HOLD}")
        extra = f"rho={res[4]:+.4f}" if res[0] == "B" else "trace complete"
        print(f"[{_ts()}]  {tag}  run {res[3]:02d}  {extra}  ({_fmt(res[6])})   "
              f"[{self.n}/{self.total} | ETA {_fmt(eta)}]", flush=True)

def main():
    print("=" * 78)
    print(f"protocol_stability_test  |  start {_ts()}   stages={STAGE_SEL}")
    print(f"  Stage B: N={N_SYS}, arms True/S1, protocols "
          f"{[BASE_STEPS*m for m in MULTS]} steps, n={N_ENS}")
    print(f"  Stage C: arms {C_ARMS}, n={C_N}, hold {C_HOLD} steps at kT={KT_LO}, "
          f"dt={C_DT}, no displacement cap")
    print(f"  workers={N_WORKERS}   CSVs: {CSV_B} / {CSV_C}")
    print("=" * 78, flush=True)
    if ANALYZE:
        analyze_B(); analyze_C(); return
    if "B" in STAGE_SEL: import_1x()
    jobs = build_queue()
    if MAX_RUN < len(jobs):
        jobs = jobs[:MAX_RUN]; print(f"[{_ts()}]  capped to {len(jobs)} jobs this session.")
    print(f"[{_ts()}]  {len(jobs)} jobs to run.", flush=True)
    if not jobs:
        print(f"[{_ts()}]  nothing to do."); analyze_B(); analyze_C(); return
    prog = Progress(len(jobs), N_WORKERS); t0 = time.time()
    try:
        (_sequential if N_WORKERS <= 1 else lambda j, p: _pool(j, p, N_WORKERS))(jobs, prog)
    except KeyboardInterrupt:
        print(f"\n[{_ts()}]  interrupted; progress saved, rerun to resume.", flush=True)
        sys.exit(130)
    print(f"\n[{_ts()}]  session done in {_fmt(time.time()-t0)}.")
    analyze_B(); analyze_C()


if __name__ == "__main__":
    main()
