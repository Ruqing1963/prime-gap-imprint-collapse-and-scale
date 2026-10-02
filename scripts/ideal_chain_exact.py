#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ideal_chain_exact.py -- the S5 half-retention length xi in the EXACT ideal-chain readout,
for the true primes and for two surrogates that contain no prime-specific arithmetic:

    True  : the prime-gap sequence itself (one fixed sequence)
    S6'   : exact gap multiset rank-mapped onto independent exponential keys of mean ln p_k
            (smooth PNT envelope, every fine correlation erased)
    S6''  : exact gap multiset rank-mapped onto keys ln p_k * g'_J, with g'_J resampled from
            the global multiset of g/ln p (a pure PNT scale family)

The readout is rho_id = Spearman(kappa_i, P_i), where P_i = <|r_i - r_cm|^2> is the thermally
averaged radial profile of the ideal (phantom) chain with the quenched PDTP bending field,
computed exactly in O(N) (Paper 1 v4.0, Prop. 2; code/v4/core.py).  There is no thermal
noise; the only randomness is the surrogate construction, over which we average NREAL
realisations and bootstrap.

Retention and crossing follow the Letter's definitions:
    f(L) = (rho_S1 - rho_S5(L)) / (rho_S1 - rho_top),
    xi   = last downward crossing of f through 1/2, log-interpolated.
Bootstrap (NBOOT resamples): S1 realisations, top-arm realisations (S6'/S6'' only; True is
a single sequence) and the S5 realisations at every L are resampled with replacement and the
crossing construction is repeated; 68% percentile intervals are reported.

A second, estimator-independent statistic is also printed: the master-curve collapse.  For
each arm, f(L) is interpolated onto a common grid of L/n_gap and the r.m.s. spread of f
across the four sizes is reported, together with the half-point of the SIZE-AVERAGED curve.
Theorem 1 of Paper 1 v4.0 predicts that all arms collapse onto one curve in L/n_gap.

Usage
    python ideal_chain_exact.py                       # N = 1,2,4,8 x 10^4, NREAL=40, NBOOT=2000
    NREAL=80 NBOOT=4000 python ideal_chain_exact.py
    python ideal_chain_exact.py --sizes 10000,20000 --arms True,S6p
Outputs (in ../data/):
    ideal_exact_summary.csv     xi, 68% CI, xi/n_gap, rho values, z, per (N, arm)
    ideal_exact_retention.csv   f(L) with bootstrap s.d. per (N, arm, L)
    ideal_exact_collapse.csv    master-curve statistics per arm
Dependencies: numpy, numba (via core.py of Paper 1 v4.0).
"""
import os, sys, csv, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.environ.get("PDTP_V4_CODE", os.path.join(HERE, "..", "code"))   # core.py from PDTP v4.0
sys.path.insert(0, os.path.abspath(CORE_DIR))
import core  # noqa: E402  (Paper 1 v4.0 exact solver and surrogate library, unmodified)

DATA = os.path.join(HERE, "..", "data")
LS_DEFAULT = [10, 25, 50, 100, 200, 300, 500, 800, 1200, 2000, 3000, 5000]
BETA = 1.0


# ------------------------------------------------------------------ surrogates
def s6pp(gaps, primes, rng):
    """S6'': PNT scale family (Paper 1 v4.0, Sec. 6.2)."""
    lnp = np.log(np.maximum(primes[:-1].astype(float), 2.0))
    gn = gaps / lnp
    key = lnp * rng.choice(gn, size=len(gaps), replace=True) + 1e-6 * rng.random(len(gaps))
    out = np.empty_like(gaps)
    out[np.argsort(key, kind="stable")] = np.sort(gaps)
    return out


def top_generator(arm, gaps, primes):
    if arm == "True":
        return lambda rng: gaps
    if arm == "S6p":
        return lambda rng: core.s6prime(gaps, rng)
    if arm == "S6pp":
        return lambda rng: s6pp(gaps, primes, rng)
    raise ValueError(arm)


# ------------------------------------------------------------------ readout
def rho_ideal(kappa):
    return core.spearman(kappa, core.ideal_profile(kappa, BETA)["P"])


def xi_crossing(Ls, f):
    """Last downward crossing of f through 1/2, log-interpolated (Letter definition)."""
    for j in range(len(Ls) - 1, 0, -1):
        if f[j - 1] >= 0.5 > f[j]:
            t = (f[j - 1] - 0.5) / (f[j - 1] - f[j])
            return math.exp(math.log(Ls[j - 1]) + t * (math.log(Ls[j]) - math.log(Ls[j - 1])))
    return float("nan")


def retention(rS1, rTop, rS5):
    span = rS1.mean() - rTop.mean()
    return np.array([(rS1.mean() - rS5[L].mean()) / span for L in sorted(rS5)])


# ------------------------------------------------------------------ one (N, arm)
def run_case(N, arm, nreal, nboot, Ls, seed):
    rng = np.random.default_rng(seed)
    pr = np.where(core.sieve(N - 1))[0]
    gaps, p0, ng = np.diff(pr), pr[0], len(pr) - 1
    K = lambda gg: core.pdtp_kappa_from_gaps(p0, gg, N)
    gen = top_generator(arm, gaps, pr)
    LL = [L for L in Ls if L < ng]

    rS1 = np.array([rho_ideal(K(core.s1_shuffle(gaps, rng))) for _ in range(nreal)])
    rTop = np.array([rho_ideal(K(gaps))]) if arm == "True" else \
        np.array([rho_ideal(K(gen(rng))) for _ in range(nreal)])
    rS5 = {}
    for L in LL:
        # every S5 realisation scrambles a FRESH top-arm sequence (as in the Langevin control);
        # for True the top sequence is fixed and only the scramble is random
        rS5[L] = np.array([rho_ideal(K(core.s5_local_scramble(gen(rng), L, rng))) for _ in range(nreal)])

    f = retention(rS1, rTop, rS5)
    xi = xi_crossing(LL, f)
    z = (rTop.mean() - rS1.mean()) / rS1.std(ddof=1)

    boot_xi, boot_f = [], []
    for _ in range(nboot):
        b1 = rS1[rng.integers(0, nreal, nreal)]
        bt = rTop if arm == "True" else rTop[rng.integers(0, nreal, nreal)]
        b5 = {L: v[rng.integers(0, nreal, nreal)] for L, v in rS5.items()}
        fb = retention(b1, bt, b5)
        boot_f.append(fb)
        c = xi_crossing(LL, fb)
        if np.isfinite(c):
            boot_xi.append(c)
    boot_f = np.array(boot_f)
    lo, hi = (np.percentile(boot_xi, 16), np.percentile(boot_xi, 84)) if len(boot_xi) > 50 else (np.nan, np.nan)
    frac_defined = len(boot_xi) / nboot
    return dict(N=N, arm=arm, n_gap=ng, rho_top=rTop.mean(), rho_S1=rS1.mean(), sd_S1=rS1.std(ddof=1),
                z_top_vs_S1=z, xi=xi, ci68_lo=lo, ci68_hi=hi, xi_over_ngap=xi / ng,
                boot_defined_fraction=frac_defined, nreal=nreal, nboot=nboot), \
        LL, f, boot_f.std(axis=0)


# ------------------------------------------------------------------ master-curve collapse
def collapse(curves, arm, grid=None):
    """curves: {N: (LL, f, n_gap)} for one arm. Interpolate f onto a common x = L/n_gap grid."""
    if grid is None:
        grid = np.geomspace(0.02, 0.8, 25)
    F = []
    for N, (LL, f, ng) in sorted(curves.items()):
        x = np.array(LL) / ng
        F.append(np.interp(np.log(grid), np.log(x), f, left=np.nan, right=np.nan))
    F = np.array(F)
    ok = np.all(np.isfinite(F), axis=0)
    spread = float(np.sqrt(np.nanmean(F[:, ok].var(axis=0, ddof=1))))
    fbar = F[:, ok].mean(axis=0)
    xbar = grid[ok]
    half = xi_crossing(list(xbar), list(fbar))  # in units of n_gap
    return dict(arm=arm, n_sizes=len(curves), rms_spread=spread, half_point_over_ngap=half,
                grid_lo=xbar[0], grid_hi=xbar[-1])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000,20000,40000,80000")
    ap.add_argument("--arms", default="True,S6p,S6pp")
    ap.add_argument("--nreal", type=int, default=int(os.environ.get("NREAL", 40)))
    ap.add_argument("--nboot", type=int, default=int(os.environ.get("NBOOT", 2000)))
    ap.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    arms = a.arms.split(",")
    os.makedirs(DATA, exist_ok=True)

    rows, ret_rows, curves = [], [], {arm: {} for arm in arms}
    print(f"{'N':>6} {'n_gap':>6} {'arm':>5} {'rho_top':>8} {'rho_S1':>8} {'z':>5} "
          f"{'xi':>6} {'68% CI':>14} {'xi/n_gap':>8}", flush=True)
    for N in sizes:
        for k, arm in enumerate(arms):
            r, LL, f, fsd = run_case(N, arm, a.nreal, a.nboot, LS_DEFAULT, a.seed + 1000 * N + k)
            rows.append(r)
            curves[arm][N] = (LL, f, r["n_gap"])
            for L, fv, s in zip(LL, f, fsd):
                ret_rows.append(dict(N=N, arm=arm, L=L, L_over_ngap=L / r["n_gap"], f=fv, f_boot_sd=s))
            print(f"{N:>6} {r['n_gap']:>6} {arm:>5} {r['rho_top']:+8.4f} {r['rho_S1']:+8.4f} "
                  f"{r['z_top_vs_S1']:+5.1f} {r['xi']:6.0f} [{r['ci68_lo']:5.0f},{r['ci68_hi']:5.0f}] "
                  f"{r['xi_over_ngap']:8.3f}", flush=True)

    with open(os.path.join(DATA, "ideal_exact_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(DATA, "ideal_exact_retention.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ret_rows[0])); w.writeheader(); w.writerows(ret_rows)

    print("\nMaster-curve collapse in x = L/n_gap (Theorem 1 predicts one curve per arm):")
    crows = []
    for arm in arms:
        if len(curves[arm]) >= 2:
            c = collapse(curves[arm], arm); crows.append(c)
            print(f"  {arm:>5}: r.m.s. spread of f across {c['n_sizes']} sizes = {c['rms_spread']:.3f}; "
                  f"half-point of the size-averaged curve = {c['half_point_over_ngap']:.3f} n_gap "
                  f"(grid {c['grid_lo']:.2f}-{c['grid_hi']:.2f})")
    if crows:
        with open(os.path.join(DATA, "ideal_exact_collapse.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(crows[0])); w.writeheader(); w.writerows(crows)

    # proportional fit per arm (weighted by symmetrised CI half-width), for the paper's table
    print("\nProportional law xi = c n_gap per arm (weighted least squares; chi2 on n_sizes-1 dof):")
    from scipy import stats
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm and np.isfinite(r["xi"]) and np.isfinite(r["ci68_lo"])]
        if len(rr) < 3: continue
        g = np.array([r["n_gap"] for r in rr], float); x = np.array([r["xi"] for r in rr])
        e = np.array([max(1.0, (r["ci68_hi"] - r["ci68_lo"]) / 2) for r in rr])
        c = np.sum(x * g / e ** 2) / np.sum((g / e) ** 2); chi = np.sum(((x - c * g) / e) ** 2)
        dof = len(rr) - 1
        print(f"  {arm:>5}: c = {c:.3f}  chi2 = {chi:.2f}/{dof}  p = {1 - stats.chi2.cdf(chi, dof):.2f}")


if __name__ == "__main__":
    main()
