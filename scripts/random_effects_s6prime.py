#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_effects_s6prime.py -- sequence-level (random-effects) statistics for the PNT
decomposition table (Table II of the Letter / Sec. IV of the merged paper).

THE PROBLEM
  The true arm is ONE fixed sequence.  In the Letter every S6' (and S1) trajectory drew a
  fresh surrogate sequence, so the Welch test compares one sequence against a mixture of
  sequences and omits the between-sequence variance sigma^2_seq, which does not shrink
  with n (Paper 1 v4.0, Sec. 7.3).

THE DESIGN
  K fixed surrogate sequences x m trajectories each, at the converged protocol
  (32000-step anneal, the depth of s5_converged_results.csv), N = 2 x 10^4 by default.
  Default K = 10, m = 5 for the S6' class; the S1 class can be added (PDTP_ARMS=S6p,S1).
  The True arm (n = 40, same N and depth) is read from s5_converged_results.csv; S1 at
  the same depth (n = 40, fresh sequences) is read from the same file for the classical
  comparison.

STATISTICS (per surrogate class)
  rho_{k,r} = mu + s_k + eps_{k,r}.   One-way ANOVA on the K x m table gives
      MS_between, MS_within,  sigma^2_th = MS_within,
      sigma^2_seq = max(0, (MS_between - MS_within) / m),     F, p (H0: sigma_seq = 0).
  Sequence-level contrast of the true chain against the class:
      Z_seq = (rho_T - mu_hat) / sqrt(sigma^2_seq + s_T^2/n_T + Var(mu_hat)),
      Var(mu_hat) = (sigma^2_seq + sigma^2_th/m) / K,
  with a t-reference on K-1 d.o.f., plus the rank of rho_T among the K sequence means.
  Retention of the class, f = (rho_S1 - mu_hat)/(rho_S1 - rho_T), is bootstrapped at the
  SEQUENCE level (resample sequences, then trajectories within sequences).

SEEDS
  zlib.crc32 of a label string (NOT Python's hash(), which is randomised per process):
  the sequence for (N, class, k) is drawn from crc32(f"seq|{N}|{class}|{k}"); the chain
  initialisation and thermal noise for trajectory r from crc32(f"traj|{N}|{class}|{k}|{r}").
  Every row is therefore reproducible bit-for-bit.

ENGINE
  Paper 1 v4.0 code/v4/pdtp_langevin.py, unmodified (identical force field, integrator,
  annealing schedule and observables to all production runs).

USAGE
  python random_effects_s6prime.py                       # K=10, m=5, S6' only, 32000 steps
  PDTP_WORKERS=12 python random_effects_s6prime.py
  PDTP_ARMS=S6p,S1 python random_effects_s6prime.py      # add the fixed-shuffle class
  PDTP_K=10 PDTP_M=5 PDTP_N=20000 PDTP_STEPS=32000 ...
  PDTP_ANALYZE_ONLY=1 python random_effects_s6prime.py   # statistics from the CSV only
  SMOKE=1 python random_effects_s6prime.py               # seconds-long check

COST  N = 2e4, 32000 steps: ~25-30 min per trajectory on one core; 50 trajectories on 12
      workers ~ 2-2.5 h.  Resumable: rerun to continue.  Output: ../data/random_effects_s6prime.csv,
      ../data/random_effects_summary.csv
"""
import os
_W = os.environ.get("PDTP_WORKERS")
if _W is None or int(_W) != 1:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

import sys, csv, time, zlib, traceback
from datetime import datetime
import numpy as np
from scipy import stats
import concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.environ.get("PDTP_V4_CODE", os.path.join(HERE, "..", "code"))   # pdtp_langevin.py from PDTP v4.0
sys.path.insert(0, os.path.abspath(ENGINE_DIR))
import pdtp_langevin as m  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
SMOKE = os.environ.get("SMOKE", "0") == "1"
ANALYZE_ONLY = os.environ.get("PDTP_ANALYZE_ONLY", "0") == "1"
N = 1000 if SMOKE else int(os.environ.get("PDTP_N", 20000))
STEPS = 60 if SMOKE else int(os.environ.get("PDTP_STEPS", 32000))
K = 2 if SMOKE else int(os.environ.get("PDTP_K", 10))
M = 2 if SMOKE else int(os.environ.get("PDTP_M", 5))
ARMS = os.environ.get("PDTP_ARMS", "S6p").split(",")
CSV_OUT = os.path.join(DATA, "random_effects_s6prime_smoke.csv" if SMOKE else "random_effects_s6prime.csv")
CSV_SUM = os.path.join(DATA, "random_effects_summary.csv")
REF_CSV = os.environ.get("PDTP_REF_CSV", os.path.join(DATA, "s5_converged_results.csv"))
try:
    N_WORKERS = int(os.environ["PDTP_WORKERS"])
except (KeyError, ValueError):
    N_WORKERS = max(1, (os.cpu_count() or 1) - 1)
HDR = ["N", "Model", "seq_id", "Run_ID", "steps", "spearman_rho", "dR", "Rg", "phi", "seed_seq", "seed_traj"]


def _ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def crc(label): return zlib.crc32(label.encode()) & 0xFFFFFFFF


# ------------------------------------------------------------------ one trajectory
def run_one(job):
    model, k, r = job
    t0 = time.time()
    s_seq, s_traj = crc(f"seq|{N}|{model}|{k}"), crc(f"traj|{N}|{model}|{k}|{r}")
    sb = m.sieve_primes(N - 1); primes = np.where(sb)[0]
    mask = m.build_mask(model, primes, sb, N, np.random.default_rng(s_seq))   # FIXED per k
    kb = m.stiffness_from_gap(m.gap_from_mask(mask, N), mask)
    rng = np.random.default_rng(s_traj)
    x = m.anneal(m.init_chain(N, rng), kb, STEPS, rng)
    rho, dR, Rg, phi = m.observables(x, kb)
    return (model, k, r, rho, dR, Rg, phi, s_seq, s_traj, time.time() - t0)


def load_done(path):
    d = {}
    if not os.path.exists(path): return d
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if int(row["N"]) != N or int(row["steps"]) != STEPS: continue
            d[(row["Model"], int(row["seq_id"]), int(row["Run_ID"]))] = float(row["spearman_rho"])
    return d


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(HDR)
        w.writerow(row); f.flush()
        try: os.fsync(f.fileno())
        except OSError: pass


# ------------------------------------------------------------------ reference arms
def load_reference():
    """True and fresh-sequence S1 at the same N and depth from the converged ladder file."""
    T, S1 = [], []
    if os.path.exists(REF_CSV):
        with open(REF_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["N"]) != N or int(row["steps"]) != STEPS: continue
                if row["Model"] == "True" and row["L_block"] == "NA": T.append(float(row["spearman_rho"]))
                if row["Model"] == "S1" and row["L_block"] == "NA": S1.append(float(row["spearman_rho"]))
    return np.array(T), np.array(S1)


# ------------------------------------------------------------------ statistics
def random_effects(table):
    """table: K x m array. Returns dict of ANOVA / variance components."""
    Kk, mm = table.shape
    seq_means = table.mean(axis=1); grand = table.mean()
    ms_b = mm * np.sum((seq_means - grand) ** 2) / (Kk - 1)
    ms_w = np.sum((table - seq_means[:, None]) ** 2) / (Kk * (mm - 1))
    F = ms_b / ms_w; p = 1 - stats.f.cdf(F, Kk - 1, Kk * (mm - 1))
    s2_seq = max(0.0, (ms_b - ms_w) / mm)
    return dict(K=Kk, m=mm, mu=grand, sigma_seq=np.sqrt(s2_seq), sigma_th=np.sqrt(ms_w),
                anova_F=F, anova_p=p, seq_means=seq_means, var_mu=(s2_seq + ms_w / mm) / Kk)


def summarize():
    done = load_done(CSV_OUT)
    if not done: print("no results yet."); return
    T, S1 = load_reference()
    rows = []
    print("\n" + "=" * 80)
    print(f"SEQUENCE-LEVEL (RANDOM-EFFECTS) STATISTICS   N={N}  steps={STEPS}")
    print("=" * 80)
    if len(T): print(f"  True (fixed sequence): rho = {T.mean():+.4f} +/- {T.std(ddof=1)/np.sqrt(len(T)):.4f}  (n={len(T)})")
    if len(S1): print(f"  S1 fresh-per-trajectory reference: rho = {S1.mean():+.4f} (n={len(S1)})")
    for model in ARMS:
        ks = sorted({k for (mo, k, r) in done if mo == model})
        tab = []
        for k in ks:
            v = [done[(model, k, r)] for r in range(M) if (model, k, r) in done]
            if len(v) == M: tab.append(v)
        if len(tab) < 2:
            print(f"\n  {model}: fewer than 2 complete sequences ({len(tab)}); skipped."); continue
        tab = np.array(tab); re = random_effects(tab)
        print(f"\n  class {model}: K={re['K']} complete sequences x m={re['m']}")
        print(f"    mu_hat = {re['mu']:+.4f}   sigma_seq = {re['sigma_seq']:.4f}   sigma_th = {re['sigma_th']:.4f}"
              f"   ANOVA F = {re['anova_F']:.2f}  p = {re['anova_p']:.3f}")
        print("    per-sequence means: " + " ".join(f"{x:+.3f}" for x in re["seq_means"]))
        out = dict(N=N, steps=STEPS, model=model, K=re["K"], m=re["m"], mu=re["mu"], sigma_seq=re["sigma_seq"],
                   sigma_th=re["sigma_th"], anova_F=re["anova_F"], anova_p=re["anova_p"])
        if len(T):
            sT2 = T.var(ddof=1) / len(T)
            Z = (T.mean() - re["mu"]) / np.sqrt(re["sigma_seq"] ** 2 + sT2 + re["var_mu"])
            p_t = 2 * stats.t.sf(abs(Z), re["K"] - 1)
            rank = int(np.sum(re["seq_means"] < T.mean()))
            # classical (naive) Welch against the pooled trajectories, for comparison
            p_naive = stats.ttest_ind(T, tab.ravel(), equal_var=False).pvalue
            print(f"    True - mu_hat = {T.mean()-re['mu']:+.4f}   Z_seq = {Z:+.2f}  (t, {re['K']-1} dof: p = {p_t:.3g})"
                  f"   rank of True among sequence means: {rank}/{re['K']}   [naive Welch p = {p_naive:.2g}]")
            out.update(rho_true=T.mean(), n_true=len(T), delta=T.mean() - re["mu"], Z_seq=Z, p_seq_t=p_t,
                       true_rank=f"{rank}/{re['K']}", welch_p_naive=p_naive)
        if len(T) and len(S1):
            # retention of the class, bootstrapped at the sequence level
            rng = np.random.default_rng(0); fb = []
            for _ in range(4000):
                kk = rng.integers(0, re["K"], re["K"])
                bt = np.array([tab[k][rng.integers(0, re["m"], re["m"])].mean() for k in kk]).mean()
                s1b = S1[rng.integers(0, len(S1), len(S1))].mean(); tb = T[rng.integers(0, len(T), len(T))].mean()
                fb.append((s1b - bt) / (s1b - tb))
            f0 = (S1.mean() - re["mu"]) / (S1.mean() - T.mean())
            lo, hi = np.percentile(fb, 16), np.percentile(fb, 84)
            print(f"    retention f_{model} = {f0:.2f}   68% sequence-level bootstrap [{lo:.2f}, {hi:.2f}]")
            out.update(f_class=f0, f_lo=lo, f_hi=hi, rho_S1_ref=S1.mean())
        rows.append(out)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda s: s)
        with open(CSV_SUM, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"\n  summary written to {CSV_SUM}")
    print("=" * 80)


# ------------------------------------------------------------------ driver
def main():
    os.makedirs(DATA, exist_ok=True)
    print("=" * 80)
    print(f"random_effects_s6prime | {_ts()} | N={N} steps={STEPS} K={K} m={M} arms={ARMS} workers={N_WORKERS}")
    print(f"  output {CSV_OUT}")
    print("=" * 80, flush=True)
    if ANALYZE_ONLY:
        summarize(); return
    done = load_done(CSV_OUT)
    jobs = [(mo, k, r) for mo in ARMS for k in range(K) for r in range(M) if (mo, k, r) not in done]
    print(f"[{_ts()}] {len(done)} rows present; {len(jobs)} trajectories to run.", flush=True)
    if not jobs:
        summarize(); return
    t0 = time.time(); n = 0; times = []

    def tick(res):
        nonlocal n
        model, k, r, rho, dR, Rg, phi, s_seq, s_traj, secs = res
        append(CSV_OUT, [N, model, k, r, STEPS, f"{rho:.6f}", f"{dR:.6f}", f"{Rg:.4f}", f"{phi:.4f}", s_seq, s_traj])
        n += 1; times.append(secs)
        avg = np.mean(times[-20:]); eta = avg * (len(jobs) - n) / max(1, N_WORKERS)
        print(f"[{_ts()}] {model} seq {k:02d} run {r} rho={rho:+.4f} Rg={Rg:6.2f} phi={phi:.3f} ({secs:6.0f}s)"
              f"  [{n}/{len(jobs)} | ETA {eta/3600:.1f} h]", flush=True)

    try:
        if N_WORKERS <= 1:
            for j in jobs: tick(run_one(j))
        else:
            with cf.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                futs = {ex.submit(run_one, j): j for j in jobs}
                for fu in cf.as_completed(futs):
                    try: tick(fu.result())
                    except Exception:
                        print(f"[{_ts()}] ERROR {futs[fu]}:\n{traceback.format_exc()}", flush=True)
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] interrupted; progress saved, rerun to resume."); sys.exit(130)
    print(f"\n[{_ts()}] done: {n} trajectories in {(time.time()-t0)/3600:.2f} h")
    summarize()


if __name__ == "__main__":
    main()
