#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s5_converged_scaling.py -- does the scale-free law xi ~ n_gap survive relaxation?

The Letter measured xi(N) at the PRODUCTION protocol, which the Stage B/C tests
showed sits ~0.95 relaxation times into a collapse requiring ~60. This driver
re-measures xi at CONVERGED depth at four (optionally five) sizes and asks
whether the proportional law survives.

It also runs the control that the Letter never ran, and which is the single
thing most likely to kill the headline. Read STAGE 2 before deciding the budget.

--------------------------------------------------------------------------------
STAGE 0 -- tau_relax CALIBRATION.  MEASURE IT; DO NOT EXTRAPOLATE.
--------------------------------------------------------------------------------
The premise "tau_relax grows with N, so matched steps means mismatched depth" is
plausible but is probably FALSE for this force field, and the difference decides
the budget. Three facts:

  * The dominant relaxation mode is the confinement spring, not chain dynamics.
    F_conf = -k_conf (r - r_cm) with mu*k_conf = 0.02 gives an analytic mode time
    1/(mu k_conf) = 50 tau_LJ. The Stage C measurement is 67 tau_LJ. Ratio 1.35.
  * That mode is N-INDEPENDENT. A Rouse internal mode would give tau ~ N^2
    (a factor 4 between 40k and 80k); a Zimm-like one ~N^1.76. They disagree with
    each other by more than the thing we are trying to control.
  * The True and S1 arms relaxed with time constants agreeing to 0.1%
    (67.32 vs 67.26 tau_LJ). A sequence-dependent internal rearrangement would
    not do that; a shared confinement mode would.

So the physics says N-independent and the assumed scalings say N^1.76-N^2. Both
cannot be right, guessing low ruins the experiment and guessing high wastes 4x
the compute. Stage 0 measures tau_relax directly at every size for ~180 core-h,
about 8% of the main run, and sets the per-size protocol depth from the
measurement. It also fits the SECOND, slow mode seen in Stage C
(tau_slow ~ 6e5 steps, amplitude ~0.003) and verifies that it moves both arms
equally -- it is the gap, not rho, that must be converged.

STAGE 0 also records R_g and the interior packing fraction at the end of every
calibration trajectory. Stage B never did this, so we do not actually know that
the 8x anneal reaches melt density; we only know its rho agrees with the state
that does. Any size whose converged protocol does not reach phi > 0.45 is not
converged, whatever rho says.

--------------------------------------------------------------------------------
STAGE 1 -- THE S5 LADDERS AT CONVERGED DEPTH
--------------------------------------------------------------------------------
Per size: arms True, S1, and S5 at six block lengths bracketing the expected
crossing, n = 40 each. Retention f(L) = (rho_S1 - rho_S5)/(rho_S1 - rho_true);
xi is the half-retention crossing, bootstrapped over trajectories.

--------------------------------------------------------------------------------
STAGE 2 -- THE S6' SCALE CONTROL.  THIS IS NOT OPTIONAL.
--------------------------------------------------------------------------------
Here is the confound. xi ~ 0.33 n_gap says the correlation length is always a
fixed fraction of whatever window is sampled. That is the signature of scale-free
order. It is ALSO the signature of an observable that responds to the smooth
prime-number-theorem envelope, because <g_i> ~ ln i spans the entire sequence BY
CONSTRUCTION: block-scrambling a monotone trend destroys a fraction of it that
depends only on L/n_gap, which would give xi proportional to n_gap trivially and
with no arithmetic content whatever.

The Letter's S6' surrogate -- exact gap multiset, exact ln i gradient, all fine
correlation erased -- was used to show that the gradient alone recovers only
32-44% of the ORDERING EFFECT. It was never used to measure a CORRELATION LENGTH.
So the scale-free law currently has no control at all.

Stage 2 runs the S5 ladder on S6' at two sizes and extracts xi_S6'(N).

    If xi_S6' is flat or short      -> the scale-free law is arithmetic. Landmark.
    If xi_S6' ~ 0.33 n_gap as well  -> the law is the prime number theorem in
                                       disguise and the headline must be withdrawn.

Run Stage 2 BEFORE committing to the full five-size Stage 1. It costs ~12% of it
and it is the result that decides whether the rest is worth buying.

--------------------------------------------------------------------------------
USAGE
  PDTP_COST=1 python3 s5_converged_scaling.py     # budget table, runs nothing
  PDTP_STAGE=0 python3 s5_converged_scaling.py    # calibration only  (DO FIRST)
  PDTP_STAGE=2 python3 s5_converged_scaling.py    # S6' control       (DO SECOND)
  PDTP_STAGE=1 python3 s5_converged_scaling.py    # the main ladders
  PDTP_ANALYZE_ONLY=1 python3 s5_converged_scaling.py
  SMOKE=1 python3 s5_converged_scaling.py         # seconds-long sanity run
  PDTP_SIZES=10000,20000,40000,80000,160000 ...   # add the 5th size (see below)

Dependencies: numpy, scipy.
"""

import os
_WREQ = os.environ.get("PDTP_WORKERS")
if _WREQ is None or int(_WREQ) != 1:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import sys, time, csv, json, traceback
from datetime import datetime
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
from scipy.optimize import curve_fit, brentq
import concurrent.futures as cf

# ============================== configuration ================================
SMOKE     = os.environ.get("SMOKE", "0") == "1"
BASE_SEED = 20261201
STAGE_SEL = os.environ.get("PDTP_STAGE", "012")
ANALYZE   = os.environ.get("PDTP_ANALYZE_ONLY", "0") == "1"
COSTONLY  = os.environ.get("PDTP_COST", "0") == "1"
try:
    N_WORKERS = int(os.environ["PDTP_WORKERS"])
except (KeyError, ValueError):
    N_WORKERS = max(1, (os.cpu_count() or 1) - 1)

SIZES = [int(x) for x in os.environ.get(
    "PDTP_SIZES", "800,1600" if SMOKE else "10000,20000,40000,80000").split(",")]
N_ENS = 3 if SMOKE else int(os.environ.get("PDTP_N", 40))

# Stage 0 calibration
CAL_N     = 2 if SMOKE else 3          # trajectories per arm per size
CAL_HOLD  = 4_000 if SMOKE else int(os.environ.get("PDTP_CAL_HOLD", 250_000))
MAX_MULT  = int(os.environ.get("PDTP_MAX_MULT", 32))   # hard cap on protocol depth
CAL_LOG   = 200 if SMOKE else 2_000

# Stage 1/2 converged protocol. BASE_MULT is overwritten per size by Stage 0.
PROD_STEPS   = 200 if SMOKE else 8_000     # the 1x production anneal
DEFAULT_MULT = 2 if SMOKE else 4           # Stage B showed 4x == 8x at N=4e4
POST_HOLD    = 200 if SMOKE else 0         # optional isothermal top-up, steps

# S5 block ladder, as multiples of the prior xi = XI_FRAC * n_gap
XI_FRAC   = 0.326
L_LADDER  = [0.15, 0.28, 0.50, 0.90, 1.60, 2.80]
S6_SIZES  = [800] if SMOKE else [int(x) for x in
             os.environ.get("PDTP_S6_SIZES", "20000,80000").split(",")]

CSV_0 = os.environ.get("PDTP_CSV0", "tau_calibration.csv")
CSV_1 = os.environ.get("PDTP_CSV1", "s5_converged_results.csv")
CSV_2 = os.environ.get("PDTP_CSV2", "s6prime_scale_control.csv")
CKPT  = os.environ.get("PDTP_CKPT_DIR", "converged_ckpt")

HDR_0 = ["N", "Model", "Run_ID", "step", "spearman_rho", "dR", "Rg", "phi", "energy"]
HDR_1 = ["N", "Model", "L_block", "Run_ID", "steps", "spearman_rho", "dR", "Rg", "phi"]

# force field / dynamics -- IDENTICAL to production. Do not touch.
B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003
HOLD_DT = 0.004
R_CAP = brentq(lambda r: 24 * EPS_LJ * (2 * (SIGMA / r) ** 12 - (SIGMA / r) ** 6) / r
               - FCAP_LJ, 0.5, RC_LJ)

# cost model: seconds per integration step, single core, calibrated at N=4e4
SEC_PER_STEP_40K = float(os.environ.get("PDTP_SPS", 0.1125))


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

def _mask_from_gaps(p0, gaps, N):
    pos = p0 + np.concatenate(([0], np.cumsum(gaps)))
    mask = np.zeros(N, dtype=bool); mask[pos] = True; return mask

def gaps_shuffle(gaps, rng):
    """S1: destroy all order, keep the multiset exactly."""
    return rng.permutation(gaps)

def gaps_local_scramble(gaps, L, rng):
    """S5: keep blocks of length L in their true global positions, randomise
    the gaps inside each block. L=1 -> true chain; L>=n_gap -> S1."""
    out = gaps.copy(); ng = len(out)
    for s in range(0, ng, L):
        blk = out[s:s + L]
        out[s:s + L] = rng.permutation(blk)
    return out

def gaps_s6prime(gaps, rng):
    """S6': exact true gap multiset, rearranged to follow ONLY the smooth
    logarithmic envelope. Rank-map each true gap onto an independent
    PNT-trended key kappa_k ~ Exp(ln p_k^smooth), p_k^smooth ~ k(ln k + ln ln k).
    Reproduces <g_i> ~ ln i; erases every fine correlation; multiset exact."""
    ng = len(gaps)
    k = np.arange(1, ng + 1, dtype=float)
    kk = np.maximum(k, 3.0)
    p_smooth = kk * (np.log(kk) + np.log(np.log(kk)))
    scale = np.maximum(np.log(np.maximum(p_smooth, np.e)), 1e-6)
    keys = rng.exponential(scale=scale)
    slots_by_key = np.argsort(keys, kind="stable")   # slot indices, increasing key
    out = np.empty_like(gaps)
    out[slots_by_key] = np.sort(gaps)                # rth smallest gap -> rth smallest key
    return out

def build_arm(model, N, rng, L=None):
    """model in {True, S1, S5, S6p, S6p_S5}. L required for the scramble arms."""
    sb = sieve_primes(N - 1); primes = np.where(sb)[0]
    if model == "True":
        is_anchor = sb.copy()
    else:
        g = np.diff(primes)
        if   model == "S1":      g = gaps_shuffle(g, rng)
        elif model == "S5":      g = gaps_local_scramble(g, int(L), rng)
        elif model == "S6p":     g = gaps_s6prime(g, rng)
        elif model == "S6p_S5":  g = gaps_local_scramble(gaps_s6prime(g, rng), int(L), rng)
        else: raise ValueError(model)
        is_anchor = _mask_from_gaps(primes[0], g, N)
    return stiffness_from_gap(gap_from_mask(is_anchor, N), is_anchor)

def n_gap(N):
    return int(sieve_primes(N - 1).sum()) - 1


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
    d = r[1:] - r[:-1]; L = np.linalg.norm(d, axis=1)
    e = 0.5 * K_BOND * np.sum((L - B0) ** 2)
    a, b = r[1:-1] - r[:-2], r[2:] - r[1:-1]
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    e += np.sum(kbend[1:-1] * (1.0 - np.clip(cos, -1, 1)))
    e += 0.5 * K_CONF * np.sum((r - r.mean(axis=0)) ** 2)
    i, j = _pairs(r)
    if len(i) > 0:
        dist = np.linalg.norm(r[i] - r[j], axis=1)
        u = lambda x: 4 * EPS_LJ * ((SIGMA / x) ** 12 - (SIGMA / x) ** 6) + EPS_LJ
        out = np.where(dist >= R_CAP, u(np.clip(dist, R_CAP, None)),
                       u(R_CAP) + FCAP_LJ * (R_CAP - dist))
        e += float(np.sum(out))
    return float(e)

def init_chain(n, rng, persistence=0.55):
    dirs = np.zeros((n, 3)); dv = rng.standard_normal(3); dv /= np.linalg.norm(dv); dirs[0] = dv
    for i in range(1, n):
        dv = persistence * dv + (1 - persistence) * rng.standard_normal(3)
        dv /= np.linalg.norm(dv) + 1e-12; dirs[i] = dv
    r = np.cumsum(dirs, axis=0) * B0; return r - r.mean(axis=0)

def anneal(r, kbend, n_steps, rng):
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1); kT = KT_HI * (KT_LO / KT_HI) ** frac
        F = compute_forces(r, kbend)
        disp = MOBILITY * F * DT + np.sqrt(2.0 * MOBILITY * kT * DT) * rng.standard_normal(r.shape)
        dm = np.linalg.norm(disp, axis=1); big = dm > MAX_DISP
        if np.any(big): disp[big] *= (MAX_DISP / dm[big])[:, None]
        r += disp
    return r

def hold_isothermal(r, kbend, n_steps, rng, dt=HOLD_DT, kT=KT_LO):
    md = 0.0
    for _ in range(n_steps):
        F = compute_forces(r, kbend)
        disp = MOBILITY * F * dt + np.sqrt(2.0 * MOBILITY * kT * dt) * rng.standard_normal(r.shape)
        md = max(md, float(np.max(np.linalg.norm(disp, axis=1))))
        r += disp
    return r, md

def observables(r, kbend):
    """rho, dR, Rg, and the equivalent-sphere packing fraction. The last two are
    the convergence diagnostics Stage B lacked: a state that is not at melt
    density is not converged, whatever rho says."""
    rad = np.linalg.norm(r - r.mean(axis=0), axis=1)
    rho, _ = stats.spearmanr(kbend, rad)
    hi = kbend > np.percentile(kbend, 80); lo = kbend < np.percentile(kbend, 20)
    Rg = float(np.sqrt(np.mean(rad ** 2)))
    Reff = Rg * np.sqrt(5.0 / 3.0)
    phi = len(r) * (np.pi / 6.0) / (4.0 / 3.0 * np.pi * Reff ** 3)
    return float(rho), float(rad[hi].mean() - rad[lo].mean()), Rg, float(phi)


# ================================ CSV I/O ====================================
def _ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_row(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row); f.flush(); os.fsync(f.fileno())

def load_rows(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        rd = csv.reader(f); next(rd, None)
        return [r for r in rd if r]

def done_keys(path, ncols):
    return {tuple(r[:ncols]) for r in load_rows(path)}


# ============================== job execution ================================
def seed_of(*parts):
    h = BASE_SEED
    for p in parts: h = (h * 1_000_003 + hash(str(p))) % (2 ** 63 - 1)
    return h

def run_cal(job):
    """Stage 0: anneal at production depth, then hold, logging rho/Rg/phi."""
    _, N, model, run = job
    rng = np.random.default_rng(seed_of("cal", N, model, run))
    kb = build_arm(model, N, rng)
    r = anneal(init_chain(N, rng), kb, PROD_STEPS, rng)
    rho, dR, Rg, phi = observables(r, kb)
    append_row(CSV_0, HDR_0, [N, model, run, 0, f"{rho:.6f}", f"{dR:.6f}",
                              f"{Rg:.4f}", f"{phi:.4f}", f"{total_energy(r,kb):.4f}"])
    done = 0
    while done < CAL_HOLD:
        chunk = min(CAL_LOG, CAL_HOLD - done)
        r, _ = hold_isothermal(r, kb, chunk, rng); done += chunk
        rho, dR, Rg, phi = observables(r, kb)
        append_row(CSV_0, HDR_0, [N, model, run, done, f"{rho:.6f}", f"{dR:.6f}",
                                  f"{Rg:.4f}", f"{phi:.4f}", f"{total_energy(r,kb):.4f}"])
    return job

def run_ladder(job):
    """Stage 1/2: one trajectory at converged depth."""
    tag, N, model, L, run, steps = job
    rng = np.random.default_rng(seed_of(tag, N, model, L, run))
    kb = build_arm(model, N, rng, L=L)
    r = anneal(init_chain(N, rng), kb, steps, rng)
    if POST_HOLD > 0:
        r, _ = hold_isothermal(r, kb, POST_HOLD, rng)
    rho, dR, Rg, phi = observables(r, kb)
    path = CSV_1 if tag == "S1L" else CSV_2
    append_row(path, HDR_1, [N, model, "NA" if L is None else int(L), run, steps,
                             f"{rho:.6f}", f"{dR:.6f}", f"{Rg:.4f}", f"{phi:.4f}"])
    return job


# =============================== queue building ==============================
def converged_depth(N):
    """Protocol depth for size N, in annealing steps. Read from Stage 0 if the
    calibration has been run; otherwise the Stage-B default (4x production),
    which is established at N=4e4 only."""
    if os.path.exists("converged_protocol.json"):
        d = json.load(open("converged_protocol.json"))
        if str(N) in d: return int(d[str(N)])
    return DEFAULT_MULT * PROD_STEPS

def ladder_for(N):
    ng = n_gap(N)
    xs = sorted({max(2, min(ng - 1, int(round(f * XI_FRAC * ng)))) for f in L_LADDER})
    return xs

def build_queue():
    jobs = []
    if "0" in STAGE_SEL:
        have = done_keys(CSV_0, 3)
        for N in SIZES:
            for m in ("True", "S1"):
                for run in range(CAL_N):
                    if (str(N), m, str(run)) not in have:
                        jobs.append(("CAL", N, m, run))
    if "1" in STAGE_SEL:
        have = done_keys(CSV_1, 4)
        for N in SIZES:
            st = converged_depth(N)
            arms = [("True", None), ("S1", None)] + [("S5", L) for L in ladder_for(N)]
            for m, L in arms:
                for run in range(N_ENS):
                    key = (str(N), m, "NA" if L is None else str(int(L)), str(run))
                    if key not in have:
                        jobs.append(("S1L", N, m, L, run, st))
    if "2" in STAGE_SEL:
        have = done_keys(CSV_2, 4)
        for N in S6_SIZES:
            st = converged_depth(N)
            arms = [("S6p", None)] + [("S6p_S5", L) for L in ladder_for(N)]
            for m, L in arms:
                for run in range(N_ENS):
                    key = (str(N), m, "NA" if L is None else str(int(L)), str(run))
                    if key not in have:
                        jobs.append(("S6C", N, m, L, run, st))
    return jobs


# ================================= analysis ==================================
def _logistic(logL, f0, lxi, w):
    return f0 / (1.0 + np.exp((logL - lxi) / w))

def retention_and_xi(rho_true, rho_S1, s5, n_boot=4000, rng=None):
    """s5: dict L -> array of rho. Returns (f(L) table, xi, 68% CI)."""
    rng = rng or np.random.default_rng(7)
    Ls = np.array(sorted(s5))
    def _xi(t, s, d):
        den = s.mean() - t.mean()
        if abs(den) < 1e-9: return np.nan
        f = np.array([(s.mean() - d[L].mean()) / den for L in Ls])
        try:
            p, _ = curve_fit(_logistic, np.log(Ls), f, p0=[1.0, np.log(Ls.mean()), 0.5],
                             maxfev=20000)
            g = lambda x: _logistic(x, *p) - 0.5 * p[0]
            lo, hi = np.log(Ls[0]) - 2.0, np.log(Ls[-1]) + 2.0
            if g(lo) * g(hi) > 0: return np.nan
            return float(np.exp(brentq(g, lo, hi)))
        except Exception:
            # fall back to linear interpolation of the half crossing
            for a, b in zip(range(len(f) - 1), range(1, len(f))):
                if (f[a] - 0.5) * (f[b] - 0.5) <= 0:
                    w = (0.5 - f[a]) / (f[b] - f[a] + 1e-12)
                    return float(np.exp(np.log(Ls[a]) + w * (np.log(Ls[b]) - np.log(Ls[a]))))
            return np.nan
    xi = _xi(rho_true, rho_S1, s5)
    boots = []
    # (if the point estimate fails, the bootstrap median is substituted below)
    for _ in range(n_boot):
        t = rng.choice(rho_true, len(rho_true), replace=True)
        s = rng.choice(rho_S1, len(rho_S1), replace=True)
        d = {L: rng.choice(v, len(v), replace=True) for L, v in s5.items()}
        b = _xi(t, s, d)
        if np.isfinite(b): boots.append(b)
    boots = np.array(boots)
    ci = (np.nan, np.nan) if len(boots) < 50 else (np.percentile(boots, 16),
                                                   np.percentile(boots, 84))
    if not np.isfinite(xi) and len(boots) >= 50:
        xi = float(np.median(boots))   # point estimate failed; bootstrap median stands in
    den = rho_S1.mean() - rho_true.mean()
    table = [(L, (rho_S1.mean() - s5[L].mean()) / den) for L in Ls]
    return table, xi, ci

def _load_ladder(path, s5name, refname):
    rows = load_rows(path)
    d = {}
    for r in rows:
        N = int(r[0]); m = r[1]; L = r[2]; rho = float(r[5])
        d.setdefault(N, {}).setdefault((m, L), []).append(rho)
    return {N: {k: np.array(v) for k, v in v2.items()} for N, v2 in d.items()}

def analyze_ladders(path, ref_true, ref_s5, label):
    data = _load_ladder(path, ref_s5, ref_true)
    out = {}
    print("\n" + "=" * 78)
    print(f"{label}")
    print("=" * 78)
    for N in sorted(data):
        arms = data[N]
        try:
            t = arms[(ref_true, "NA")]; s = arms[("S1", "NA")] if ("S1", "NA") in arms else None
        except KeyError:
            print(f"  N={N}: reference arms missing"); continue
        if s is None:
            s1 = _load_ladder(CSV_1, "S5", "True").get(N, {}).get(("S1", "NA"))
            if s1 is None: print(f"  N={N}: S1 reference missing"); continue
            s = s1
        s5 = {int(L): v for (m, L), v in arms.items() if m == ref_s5 and L != "NA"}
        if len(s5) < 3: print(f"  N={N}: ladder too short"); continue
        table, xi, ci = retention_and_xi(t, s, s5)
        ng = n_gap(N)
        print(f"\n  N = {N}   n_gap = {ng}   rho_ref = {t.mean():+.4f}  rho_S1 = {s.mean():+.4f}"
              f"  gap = {t.mean()-s.mean():+.4f}")
        print(f"    {'L':>7}{'f(L)':>9}{'rho':>10}{'n':>5}")
        for L, f in table:
            print(f"    {L:>7}{f:>9.3f}{s5[L].mean():>10.4f}{len(s5[L]):>5}")
        print(f"    xi = {xi:.0f} gaps   68% CI [{ci[0]:.0f}, {ci[1]:.0f}]   "
              f"xi/n_gap = {xi/ng:.3f}")
        out[N] = (ng, xi, ci)
    return out

def scaling_test(xis, label):
    Ns = sorted(xis)
    if len(Ns) < 3:
        print(f"\n  [{label}] need >=3 sizes for a scaling test."); return
    ng = np.array([xis[N][0] for N in Ns], float)
    xi = np.array([xis[N][1] for N in Ns], float)
    lo = np.array([xis[N][2][0] for N in Ns], float)
    hi = np.array([xis[N][2][1] for N in Ns], float)
    sig = 0.5 * (hi - lo)
    print("\n" + "-" * 78)
    print(f"  SCALING TEST -- {label}")
    print("-" * 78)
    # free exponent, in log space
    lw = sig / xi
    p, cov = np.polyfit(np.log(ng), np.log(xi), 1, w=1 / lw, cov=True)
    a, sa = p[0], np.sqrt(cov[0, 0])
    print(f"    free exponent   : xi ~ n_gap^({a:.2f} +/- {sa:.2f})   "
          f"[{abs(a-1)/sa:.1f} sigma from 1]")
    # strictly proportional, one parameter
    c = np.sum(xi * ng / sig ** 2) / np.sum(ng ** 2 / sig ** 2)
    chi_p = np.sum(((xi - c * ng) / sig) ** 2)
    # fixed length, one parameter
    k = np.sum(xi / sig ** 2) / np.sum(1 / sig ** 2)
    chi_f = np.sum(((xi - k) / sig) ** 2)
    # logarithmic correction xi ~ A n/ln n, one parameter
    z = ng / np.log(ng)
    A = np.sum(xi * z / sig ** 2) / np.sum(z ** 2 / sig ** 2)
    chi_l = np.sum(((xi - A * z) / sig) ** 2)
    dof = len(Ns) - 1
    for nm, ch, par in (("proportional  xi = c n_gap", chi_p, f"c = {c:.3f}"),
                        ("fixed length  xi = const  ", chi_f, f"xi = {k:.0f}"),
                        ("log-corrected xi = A n/ln n", chi_l, f"A = {A:.2f}")):
        print(f"    {nm}: chi2 = {ch:5.2f}/{dof}  p = {1-stats.chi2.cdf(ch,dof):.3f}   {par}")
    print(f"    xi/n_gap across sizes: {np.round(xi/ng,3)}")


def analyze_all():
    xi_main = analyze_ladders(CSV_1, "True", "S5",
                              "STAGE 1 -- S5 LADDERS AT CONVERGED DEPTH")
    if xi_main: scaling_test(xi_main, "true primes, converged")
    if os.path.exists(CSV_2):
        xi_ctrl = analyze_ladders(CSV_2, "S6p", "S6p_S5",
                                  "STAGE 2 -- S6' CONTROL (PNT gradient only)")
        if xi_ctrl: scaling_test(xi_ctrl, "S6' gradient-only control")
        print("\n" + "=" * 78)
        print("  VERDICT ON THE SCALE-FREE CLAIM")
        print("=" * 78)
        print("""    Compare xi/n_gap for the true primes against S6'.
    If S6' also returns xi ~ 0.33 n_gap, the proportional law is a property of
    the smooth 1/ln i envelope and carries NO arithmetic content: the headline
    must be withdrawn. Only a true-primes ratio that is significantly LARGER
    than the S6' ratio establishes scale-free ARITHMETIC order.""")

def analyze_cal():
    rows = load_rows(CSV_0)
    if not rows: print("\n[Stage 0] no data yet."); return
    d = {}
    for r in rows:
        d.setdefault((int(r[0]), r[1], int(r[2])), []).append(
            (int(r[3]), float(r[4]), float(r[6]), float(r[7])))
    print("\n" + "=" * 78)
    print("STAGE 0 -- tau_relax CALIBRATION")
    print("=" * 78)
    prot = {}
    f2 = lambda x, a, A1, T1, A2, T2: a - A1*np.exp(-x/T1) - A2*np.exp(-x/T2)
    f1 = lambda x, a, A, T: a - A*np.exp(-x/T)
    for N in sorted({k[0] for k in d}):
        taus, phis, rgs = [], [], []
        for arm in ("True", "S1"):
            tr = [v for k, v in d.items() if k[0] == N and k[1] == arm]
            if not tr: continue
            t = np.array([p[0] for p in tr[0]], float)
            y = np.mean([[p[1] for p in s] for s in tr], axis=0)
            phis += [s[-1][3] for s in tr]; rgs += [s[-1][2] for s in tr]
            try:
                p0 = [float(np.clip(y[-1], -0.99, 0.99)),
                      float(np.clip(y[-1] - y[0], -1.99, 1.99)),
                      float(np.clip(1.7e4, CAL_LOG * 1.01, CAL_HOLD * 0.99))]
                p, _ = curve_fit(f1, t, y, p0=p0, maxfev=80000,
                                 bounds=([-1.0, -2.0, CAL_LOG], [1.0, 2.0, CAL_HOLD]))
                taus.append(abs(p[2]))
            except Exception: pass
        if not taus:
            print(f"  N={N:>7}: relaxation fit did not converge in either arm; "
                  f"no protocol written. Inspect {CSV_0} for this size.")
            continue
        tau = float(np.mean(taus))               # in HOLD steps at dt=HOLD_DT
        tau_lj = tau * HOLD_DT
        phi_end = float(np.mean(phis))
        # GUARD: a tau comparable to the hold length means the hold was too short
        # to constrain the fit. Do not silently turn an unconstrained fit into a
        # protocol -- that is how a budget grows by four orders of magnitude.
        if tau > CAL_HOLD / 3.0:
            print(f"  N={N:>7}: tau fit UNCONSTRAINED (tau={tau:.3g} vs hold {CAL_HOLD}). "
                  f"Re-run Stage 0 with PDTP_CAL_HOLD >= {int(6*tau):d}. "
                  f"No protocol written for this size.")
            continue
        need = int(np.ceil(6 * tau_lj / DT / PROD_STEPS)) * PROD_STEPS
        depth = max(DEFAULT_MULT * PROD_STEPS, need)
        cap = MAX_MULT * PROD_STEPS
        if depth > cap:
            print(f"  N={N:>7}: required depth {depth} exceeds the {MAX_MULT}x cap; "
                  f"clipping. Budget this size explicitly before proceeding.")
            depth = cap
        prot[str(N)] = depth
        flag = "" if phi_end > 0.45 else "   [WARNING: phi < 0.45, not at melt density]"
        print(f"  N={N:>7}: tau_relax = {tau:.3g} steps = {tau_lj:.0f} tau_LJ "
              f"({tau_lj/50.0:.2f} x the confinement mode 1/(mu k_conf)=50)   "
              f"end phi = {phi_end:.3f}  Rg = {np.mean(rgs):.2f}   "
              f"-> protocol {depth} steps ({depth//PROD_STEPS}x){flag}")
    json.dump(prot, open("converged_protocol.json", "w"), indent=1)
    print("\n  written converged_protocol.json; Stage 1/2 will use these depths.")
    print("  If tau_LJ is flat across N, the confinement mode dominates and the")
    print("  Letter's matched-step protocol was NOT confounded by size. If it")
    print("  grows, it was, and the per-size depths above are the fix.")


# ================================ cost model =================================
def sec_per_step(N):
    return SEC_PER_STEP_40K * (N / 40000.0)

def cost_report():
    print("=" * 78)
    print("CORE-HOUR BUDGET   (single core, %.4f s/step at N=4e4)" % SEC_PER_STEP_40K)
    print("=" * 78)
    tot = 0.0
    print("\nSTAGE 0 -- tau_relax calibration  (2 arms x %d runs, %d anneal + %d hold steps)"
          % (CAL_N, PROD_STEPS, CAL_HOLD))
    s0 = 0.0
    for N in SIZES:
        h = 2 * CAL_N * (PROD_STEPS + CAL_HOLD) * sec_per_step(N) / 3600
        s0 += h; print(f"    N={N:>7}: {h:8.1f} core-h")
    print(f"    {'subtotal':>9}: {s0:8.1f} core-h"); tot += s0

    print("\nSTAGE 1 -- S5 ladders  (n=%d per arm)" % N_ENS)
    s1 = 0.0
    for N in SIZES:
        st = converged_depth(N); L = ladder_for(N); arms = 2 + len(L)
        h = arms * N_ENS * (st + POST_HOLD) * sec_per_step(N) / 3600
        s1 += h
        print(f"    N={N:>7}: n_gap={n_gap(N):>6}  depth={st:>6} steps  "
              f"{arms} arms  L={L}  -> {h:8.1f} core-h")
    print(f"    {'subtotal':>9}: {s1:8.1f} core-h"); tot += s1

    print("\nSTAGE 2 -- S6' scale control  (n=%d per arm)" % N_ENS)
    s2 = 0.0
    for N in S6_SIZES:
        st = converged_depth(N); L = ladder_for(N); arms = 1 + len(L)
        h = arms * N_ENS * (st + POST_HOLD) * sec_per_step(N) / 3600
        s2 += h; print(f"    N={N:>7}: {arms} arms -> {h:8.1f} core-h")
    print(f"    {'subtotal':>9}: {s2:8.1f} core-h"); tot += s2

    print("\n" + "-" * 78)
    print(f"  TOTAL: {tot:.0f} core-h   = {tot/N_WORKERS:.1f} h wall clock on "
          f"{N_WORKERS} cores = {tot/N_WORKERS/24:.1f} days")
    print("-" * 78)
    print("""
  RECOMMENDED ORDER OF EXPENDITURE
    1. Stage 0  -- cheap, and decides the protocol depth. Never skip.
    2. Stage 2  -- the S6' control. If the gradient alone reproduces
                   xi ~ 0.33 n_gap, STOP: the scale-free claim is not
                   arithmetic and Stage 1 buys an artefact at full price.
    3. Stage 1  -- the main ladders, only once Stage 2 has cleared.

  ON PRECISION. With n=40 the converged gap has ~19% relative error (vs 16% at
  the production protocol, because the gap shrinks by 41% while the per-arm
  spread shrinks by only ~30%). Propagated through the retention curve this puts
  ~25-30% error bars on each xi, which over the 10k-80k range (0.9 decades)
  gives an exponent uncertainty of roughly +/-0.18 -- the SAME ambiguity the
  Letter already has, where alpha = 0.80 +/- 0.18 is consistent with 1 at 1.1
  sigma and with n/ln n as well. Spending 2000 core-h to reproduce that
  ambiguity at converged depth answers the robustness question but NOT the
  physics question.

  To actually resolve alpha, buy lever arm, not ensemble size: adding
  N = 1.6e5 extends the range to 1.2 decades and cuts the exponent error by
  ~1.35x, for about the cost of doubling n at the existing sizes (which would
  cut it by only 1.41x on four points). Run with
      PDTP_SIZES=10000,20000,40000,80000,160000
  and check the memory headroom on the k-d tree first.
""")


# ================================= driver ====================================
def _run(job):
    return run_cal(job) if job[0] == "CAL" else run_ladder(job)

def main():
    if COSTONLY: cost_report(); return
    if ANALYZE:
        analyze_cal(); analyze_all(); return
    os.makedirs(CKPT, exist_ok=True)
    jobs = build_queue()
    print(f"[{_ts()}] {len(jobs)} jobs queued on {N_WORKERS} workers "
          f"(stages {STAGE_SEL}, sizes {SIZES}, n={N_ENS})")
    if not jobs:
        analyze_cal(); analyze_all(); return
    t0 = time.time(); ndone = 0
    if N_WORKERS == 1:
        for j in jobs:
            try: _run(j)
            except Exception: traceback.print_exc()
            ndone += 1
            print(f"[{_ts()}] {ndone}/{len(jobs)}  elapsed {(time.time()-t0)/3600:.2f} h")
    else:
        with cf.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(_run, j): j for j in jobs}
            for fut in cf.as_completed(futs):
                try: fut.result()
                except Exception: traceback.print_exc()
                ndone += 1
                if ndone % 5 == 0 or ndone == len(jobs):
                    el = time.time() - t0
                    print(f"[{_ts()}] {ndone}/{len(jobs)}  elapsed {el/3600:.2f} h  "
                          f"eta {el/ndone*(len(jobs)-ndone)/3600:.2f} h")
    if "0" in STAGE_SEL: analyze_cal()
    analyze_all()


if __name__ == "__main__":
    main()
