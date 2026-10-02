#!/usr/bin/env python3
"""make_fig_collapse.py -- Fig. 5 of the merged paper: exact ideal-chain retention curves
against L/n_gap for True, S6' and S6'' at four sizes (from ideal_exact_retention.csv), and
the Langevin converged ladders (True and S6') on the same axis."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); FIG = os.path.join(HERE, "..", "paper", "figures")


def sieve(n):
    ip = np.ones(n + 1, dtype=bool); ip[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if ip[i]: ip[i * i::i] = False
    return ip
def n_gap(N): return int(sieve(N - 1).sum()) - 1


def load_exact():
    d = {}
    with open(os.path.join(DATA, "ideal_exact_retention.csv"), newline="") as f:
        for r in csv.DictReader(f):
            d.setdefault((r["arm"], int(r["N"])), []).append((float(r["L_over_ngap"]), float(r["f"]), float(r["f_boot_sd"])))
    return d


def load_langevin(path, top, s5):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d.setdefault((int(r["N"]), r["Model"], r["L_block"]), []).append(float(r["spearman_rho"]))
    out = {}
    for N in sorted({k[0] for k in d}):
        if (N, top, "NA") not in d or (N, "S1", "NA") not in d: continue
        T = np.mean(d[(N, top, "NA")]); S1 = np.mean(d[(N, "S1", "NA")]); span = S1 - T
        pts = []
        for (n, mdl, L), v in d.items():
            if n == N and mdl == s5 and len(v) >= 10:
                pts.append((int(L) / n_gap(N), (S1 - np.mean(v)) / span, np.std(v, ddof=1) / np.sqrt(len(v)) / abs(span)))
        out[N] = sorted(pts)
    return out


def main():
    ex = load_exact()
    conv = load_langevin(os.path.join(DATA, "s5_converged_results.csv"), "True", "S5")
    # S6' ladder needs the S1 reference from the converged file
    s6 = {}
    rows = []
    for p in ("s6prime_scale_control.csv", "s5_converged_results.csv"):
        with open(os.path.join(DATA, p), newline="") as f: rows += list(csv.DictReader(f))
    tmp = os.path.join(DATA, "_merged_tmp.csv")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    s6 = load_langevin(tmp, "S6p", "S6p_S5"); os.remove(tmp)

    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.2))
    cols = {10000: "#1f77b4", 20000: "#ff7f0e", 40000: "#2ca02c", 80000: "#d62728"}
    for a, arm, title in zip(ax[:2], ["True", "S6p"], ["(a) exact readout, true primes", "(b) exact readout, S6$'$ (PNT envelope)"]):
        for N in (10000, 20000, 40000, 80000):
            pts = sorted(ex.get((arm, N), []))
            if not pts: continue
            x, y, s = map(np.array, zip(*pts))
            a.errorbar(x, y, yerr=s, fmt="o-", ms=3, lw=1, color=cols[N], label=f"$N={N//1000}$k")
        # S6'' on panel (b) as dashed, size-averaged
        if arm == "S6p":
            for N in (10000, 20000, 40000, 80000):
                pts = sorted(ex.get(("S6pp", N), []))
                if pts:
                    x, y, _ = map(np.array, zip(*pts)); a.plot(x, y, "--", lw=0.8, color=cols[N], alpha=0.7)
        a.axhline(0.5, ls=":", color="k", lw=0.8); a.set_xscale("log"); a.set_ylim(-0.9, 1.5)
        a.set_xlabel(r"$L/n_{\rm gap}$"); a.set_title(title, fontsize=9); a.legend(fontsize=7, frameon=False)
    ax[0].set_ylabel(r"retention $f(L)$")
    for N, pts in conv.items():
        x, y, s = map(np.array, zip(*pts)); ax[2].errorbar(x, y, yerr=s, fmt="s-", ms=3, lw=1, color=cols[N], label=f"True $N={N//1000}$k")
    for N, pts in s6.items():
        x, y, s = map(np.array, zip(*pts)); ax[2].errorbar(x, y, yerr=s, fmt="^--", ms=3, lw=0.8, color=cols[N], alpha=0.6)
    ax[2].axhline(0.5, ls=":", color="k", lw=0.8); ax[2].set_xscale("log"); ax[2].set_ylim(-0.9, 2.2)
    ax[2].set_xlabel(r"$L/n_{\rm gap}$"); ax[2].set_title("(c) Langevin, converged: True (squares), S6$'$ (triangles)", fontsize=9)
    ax[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig5_collapse.pdf")); fig.savefig(os.path.join(FIG, "fig5_collapse.png"), dpi=160)
    print("written fig5_collapse.pdf/.png")


if __name__ == "__main__":
    main()
