"""
pdtp_langevin.py -- the PDTP force field, integrator and annealing protocol (Secs. 2-3 of the
paper), with the surrogate constructions used by the version-4.0 Langevin tests.

The physics is identical to the production code of versions 1.0-3.0 (`code/*.py`):
harmonic bonds, gap-driven cosine bending, capped WCA self-avoidance, weak confinement,
overdamped Langevin dynamics with geometric annealing and a per-bead displacement cap.

Arms: True (primes), S1 (shuffle of the exact gap multiset), S6p (S6': exact multiset
rank-mapped onto exponential keys of mean ln p_k).  S6'' and IAAFT are built in the drivers.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats

# force field / dynamics -- identical to production (Table 1)
B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003


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


def mask_shuffle(primes, N, rng):
    gaps = np.diff(primes).copy(); rng.shuffle(gaps)
    newpos = primes[0] + np.concatenate(([0], np.cumsum(gaps)))
    mask = np.zeros(N, dtype=bool); mask[newpos] = True; return mask


def mask_s6prime(primes, N, rng):
    """S6': exact gap multiset rank-mapped onto independent exponential keys of mean ln p_k."""
    gaps = np.diff(primes).astype(np.int64)
    m = len(gaps)
    k = np.arange(1, m + 1, dtype=float)
    lk = np.log(np.maximum(k, 2.0)); llk = np.log(np.maximum(lk, 1.0))
    trend = np.log(np.maximum(k * (lk + llk), 2.0))
    key = rng.exponential(scale=trend)
    order = np.argsort(key, kind="stable")
    newgaps = np.empty(m, dtype=np.int64)
    newgaps[order] = np.sort(gaps)
    newpos = primes[0] + np.concatenate(([0], np.cumsum(newgaps)))
    mask = np.zeros(N, dtype=bool); mask[newpos] = True
    return mask


def build_mask(model, primes, sieve_bool, N, rng):
    if model == "True": return sieve_bool.copy()
    if model == "S1":   return mask_shuffle(primes, N, rng)
    if model == "S6p":  return mask_s6prime(primes, N, rng)
    raise ValueError(f"unknown model {model!r}")


# ================================ physics ====================================
def _scatter_add(F, idx, vals):
    n = F.shape[0]
    for c in range(3): F[:, c] += np.bincount(idx, weights=vals[:, c], minlength=n)


def compute_forces(r, kbend):
    N = r.shape[0]; F = np.zeros_like(r)
    d = r[1:] - r[:-1]; L = np.linalg.norm(d, axis=1) + 1e-12
    fb = (K_BOND * (L - B0) / L)[:, None] * d; F[0:N - 1] += fb; F[1:N] -= fb
    a = r[1:-1] - r[:-2]; b = r[2:] - r[1:-1]
    na = np.linalg.norm(a, axis=1) + 1e-12; nb = np.linalg.norm(b, axis=1) + 1e-12
    ah, bh = a / na[:, None], b / nb[:, None]
    cos = np.clip(np.sum(ah * bh, axis=1), -1.0, 1.0); kk = kbend[1:-1][:, None]
    dda = (bh - cos[:, None] * ah) / na[:, None]; ddb = (ah - cos[:, None] * bh) / nb[:, None]
    F[0:N - 2] += -kk * dda; F[2:N] += kk * ddb; F[1:N - 1] += kk * (dda - ddb)
    tree = cKDTree(r); pairs = tree.query_pairs(RC_LJ * SIGMA, output_type="ndarray")
    if len(pairs) > 0:
        i, j = pairs[:, 0], pairs[:, 1]; keep = np.abs(i - j) > 1; i, j = i[keep], j[keep]
        for s in range(0, len(i), 4_000_000):
            ii, jj = i[s:s + 4_000_000], j[s:s + 4_000_000]
            rij = r[ii] - r[jj]; dist = np.linalg.norm(rij, axis=1) + 1e-12
            sr6 = (SIGMA / dist) ** 6
            fmag = np.clip(24 * EPS_LJ * (2 * sr6 ** 2 - sr6) / dist, 0.0, FCAP_LJ)
            fvec = (fmag / dist)[:, None] * rij
            _scatter_add(F, ii, fvec); _scatter_add(F, jj, -fvec)
    if K_CONF > 0.0: F += -K_CONF * (r - r.mean(axis=0))
    return F


def init_chain(N, rng, persistence=0.55):
    dirs = np.zeros((N, 3)); dv = rng.standard_normal(3); dv /= np.linalg.norm(dv); dirs[0] = dv
    for i in range(1, N):
        dv = persistence * dv + (1 - persistence) * rng.standard_normal(3)
        dv /= np.linalg.norm(dv) + 1e-12; dirs[i] = dv
    r = np.cumsum(dirs, axis=0) * B0; return r - r.mean(axis=0)


def anneal(r, kbend, n_steps, rng):
    """geometric cooling KT_HI -> KT_LO over n_steps, dt = 0.008, displacement cap on"""
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1); kT = KT_HI * (KT_LO / KT_HI) ** frac
        F = compute_forces(r, kbend)
        disp = MOBILITY * F * DT + np.sqrt(2.0 * MOBILITY * kT * DT) * rng.standard_normal(r.shape)
        dmag = np.linalg.norm(disp, axis=1); big = dmag > MAX_DISP
        if np.any(big): disp[big] *= (MAX_DISP / dmag[big])[:, None]
        r += disp
    return r


run_langevin = anneal


def observables(r, kbend):
    """Spearman rho(stiffness, radius), Delta R (p80 - p20 stiffness), R_g, packing fraction"""
    rad = np.linalg.norm(r - r.mean(axis=0), axis=1)
    rho, _ = stats.spearmanr(kbend, rad)
    hi = kbend > np.percentile(kbend, 80); lo = kbend < np.percentile(kbend, 20)
    Rg = float(np.sqrt(np.mean(rad ** 2)))
    Reff = Rg * np.sqrt(5.0 / 3.0)
    phi = len(r) * (np.pi / 6.0) / (4.0 / 3.0 * np.pi * Reff ** 3)
    return float(rho), float(rad[hi].mean() - rad[lo].mean()), Rg, float(phi)
