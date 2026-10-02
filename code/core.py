"""
core.py -- shared machinery for the PDTP v4 theory programme.

Three layers, all laptop-scale:

  1. Arithmetic: sieves (including a segmented sieve for windows [M, M+N) with M up to
     ~1e12), the PDTP per-bead gap field, and the surrogate family
     (S1 shuffle, S4 block shuffle, S5 local scramble, S6' PNT-keyed, IAAFT, AR(1)
     rank-matched, spectral synthesis).

  2. Exact ideal-chain theory. For a chain of unit bonds with a bead-dependent cosine
     bending potential kappa_i (1 - cos theta_i) and no excluded volume, the bond
     tangents form a Markov chain on S^2 with E[t_{a+1} | t_a] = L(beta kappa_{a+1}) t_a,
     L(x) = coth x - 1/x (Langevin function). Hence, EXACTLY, for bonds a < b,

            <t_a . t_b> = prod_{k=a+1}^{b} ell_k ,     ell_k = L(beta kappa_k).

     From this, every second moment of the conformation is an explicit functional of the
     quenched sequence. `ideal_profile` evaluates the radial profile
     P_i = <|r_i - r_cm|^2>, the radius of gyration and the end-to-end distance in O(N)
     (derivation in theory/pdtp_v4_theory.tex, Prop. 2). `ideal_R2_matrix` gives all
     pair distances <R_ij^2> in O(N^2), for the Gaussian-closure scattering function.

  3. Exact sampling of ideal conformations (no Markov chain Monte Carlo needed):
     cos theta has density ~ exp(x c) on [-1, 1], inverted in closed form.

Conventions follow the PDTP model: beads i = 0..N-1, bond a joins beads a and a+1, the bending
angle at bead i (1 <= i <= N-2) is between bonds i-1 and i and carries kappa_i.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit

# ------------------------------------------------------------------ PDTP constants (Table 1)
K_HINGE, ALPHA, K_MAX = 0.25, 0.15, 500.0


# ======================================================================= arithmetic
def sieve(n: int) -> np.ndarray:
    """Boolean primality mask for 0..n."""
    ip = np.ones(n + 1, dtype=bool)
    ip[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if ip[i]:
            ip[i * i::i] = False
    return ip


def primes_in_window(M: int, N: int) -> np.ndarray:
    """Primes p with M <= p < M + N (segmented sieve)."""
    lo, hi = max(M, 2), M + N
    small = np.where(sieve(int(math.isqrt(hi)) + 1))[0]
    seg = np.ones(hi - lo, dtype=bool)
    for p in small:
        start = max(p * p, ((lo + p - 1) // p) * p)
        seg[start - lo::p] = False
    return lo + np.where(seg)[0]


def gap_field(anchors: np.ndarray, N: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """PDTP per-bead gap field on beads 0..N-1 (bead i <-> integer offset + i).

    `anchors` are absolute integer positions (they may extend beyond the window, which is
    how window runs get real boundary gaps). Returns (g, is_anchor)."""
    a = np.asarray(anchors, dtype=np.int64) - offset
    is_anchor = np.zeros(N, dtype=bool)
    inside = a[(a >= 0) & (a < N)]
    is_anchor[inside] = True
    g = np.zeros(N)
    for x, y in zip(a[:-1], a[1:]):
        if y - x > 1:
            lo, hi = max(x + 1, 0), min(y, N)
            if lo < hi:
                g[lo:hi] = y - x
    if offset == 0:                      # PDTP end conventions
        g[:a[0]] = a[0]
        g[a[-1] + 1:N] = N - a[-1]
    return g, is_anchor


def kappa_pdtp(g: np.ndarray, is_anchor: np.ndarray,
                 k_hinge=K_HINGE, alpha=ALPHA, k_max=K_MAX) -> np.ndarray:
    k = np.full(len(g), k_hinge)
    c = ~is_anchor
    k[c] = np.minimum(k_hinge * np.exp(alpha * g[c]), k_max)
    return k


def anchors_from_gaps(p0: int, gaps: np.ndarray) -> np.ndarray:
    return p0 + np.concatenate(([0], np.cumsum(gaps)))


def pdtp_kappa_from_gaps(p0: int, gaps: np.ndarray, N: int, **kw) -> np.ndarray:
    g, ia = gap_field(anchors_from_gaps(p0, gaps), N)
    return kappa_pdtp(g, ia, **kw)


# ======================================================================= surrogates
def s1_shuffle(gaps, rng):
    return rng.permutation(gaps)


def s4_block_shuffle(gaps, L, rng):
    blocks = [gaps[s:s + L] for s in range(0, len(gaps), L)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[o] for o in order])


def s5_local_scramble(gaps, L, rng):
    out = np.array(gaps, copy=True)
    for s in range(0, len(out), L):
        blk = out[s:s + L]
        rng.shuffle(blk)
    return out


def s6prime(gaps, rng):
    """S6': exact multiset rank-mapped onto exponential keys of mean ln p_k."""
    m = len(gaps)
    k = np.arange(1, m + 1, dtype=float)
    lk = np.log(np.maximum(k, 2.0))
    llk = np.log(np.maximum(lk, 1.0))
    trend = np.log(np.maximum(k * (lk + llk), 2.0))
    key = rng.exponential(scale=trend)
    out = np.empty(m, dtype=np.asarray(gaps).dtype)
    out[np.argsort(key, kind="stable")] = np.sort(gaps)
    return out


def rank_remap(values_sorted: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Give s the exact marginal `values_sorted` (sorted), keeping the rank order of s."""
    out = np.empty_like(values_sorted)
    out[np.argsort(s, kind="stable")] = values_sorted
    return out


def iaaft(x: np.ndarray, rng, max_iter: int = 1000, detrend: np.ndarray | None = None):
    """Iterated amplitude-adjusted Fourier transform (Schreiber & Schmitz 1996).

    Returns a sequence with EXACTLY the multiset of x and (to convergence) its Fourier
    amplitude spectrum. If `detrend` is given, the surrogate is built for x - detrend and
    the trend is added back (the multiset is then that of the residual + trend)."""
    base = np.asarray(x, dtype=float)
    if detrend is not None:
        base = base - detrend
    srt = np.sort(base)
    amp = np.abs(np.fft.rfft(base))
    r = rng.permutation(base)
    prev = None
    for _ in range(max_iter):
        ph = np.angle(np.fft.rfft(r))
        s = np.fft.irfft(amp * np.exp(1j * ph), n=len(base))
        rk = np.argsort(np.argsort(s, kind="stable"), kind="stable")
        if prev is not None and np.array_equal(rk, prev):
            break
        prev = rk
        r = srt[rk]
    if detrend is not None:
        r = r + detrend
    return r


def ar1_gaussian(n, xi, rng):
    r = math.exp(-1.0 / xi) if xi > 0 else 0.0
    z = np.empty(n)
    z[0] = rng.standard_normal()
    e = rng.standard_normal(n) * math.sqrt(1 - r * r)
    for t in range(1, n):
        z[t] = r * z[t - 1] + e[t]
    return z


def spectral_synthesis(n, exponent, rng):
    """Gaussian sequence with S(w) ~ |2 sin(w/2)|^exponent (exponent>0: hyperuniform,
    exponent<0: 1/f^|exponent| long-range correlated). Zero mode removed."""
    w = 2 * np.pi * np.fft.rfftfreq(n)
    filt = np.zeros_like(w)
    filt[1:] = np.abs(2 * np.sin(w[1:] / 2)) ** (exponent / 2.0)
    z = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * filt, n=n)
    return (z - z.mean()) / z.std()


# ================================================================ exact ideal chain
@njit(cache=True)
def langevin_L(x):
    out = np.empty_like(x)
    for i in range(x.size):
        v = x[i]
        if v < 1e-4:
            out[i] = v / 3.0
        elif v > 30.0:
            out[i] = 1.0 - 1.0 / v
        else:
            out[i] = 1.0 / math.tanh(v) - 1.0 / v
    return out


@njit(cache=True)
def _profile_from_ell(ell):
    N = ell.size
    nb = N - 1                       # bonds 0..N-2
    H = np.empty(nb)
    H[nb - 1] = 1.0
    for a in range(nb - 2, -1, -1):
        H[a] = (N - 1 - a) + ell[a + 1] * H[a + 1]
    J = np.empty(nb)
    J[0] = 1.0
    for b in range(1, nb):
        J[b] = (b + 1) + ell[b] * J[b - 1]
    Q = np.zeros(N)
    acc = 0.0                        # A(i) = sum_{a >= i} G_a
    for i in range(N - 1, -1, -1):
        if i <= nb - 1:
            acc += 2.0 * H[i] - (N - 1 - i)
        Q[i] = acc
    acc = 0.0                        # B(i) = sum_{b < i} K_b
    for i in range(N):
        Q[i] += acc
        if i <= nb - 1:
            acc += 2.0 * J[i] - (i + 1)
    # end-to-end: <R_ee^2> = sum_{a,b} w(a,b) = 2 sum_a h_a - nb, h_a = sum_{b>=a} w(a,b)
    h = 1.0
    tot = 1.0
    for a in range(nb - 2, -1, -1):
        h = 1.0 + ell[a + 1] * h
        tot += h
    Ree2 = 2.0 * tot - nb
    return Q, Ree2


def ideal_profile(kappa: np.ndarray, beta: float = 1.0):
    """Exact thermal second moments of the ideal (phantom) chain, O(N).

    Returns dict with P (radial profile <|r_i - r_cm|^2>), Rg2, Ree2, Q (sum_j <R_ij^2>)."""
    ell = langevin_L(beta * np.asarray(kappa, dtype=float))
    Q, Ree2 = _profile_from_ell(ell)
    N = len(kappa)
    Rg2 = Q.sum() / (2.0 * N * N)
    return {"P": Q / N - Rg2, "Rg2": Rg2, "Ree2": Ree2, "Q": Q, "ell": ell}


@njit(cache=True)
def _R2_matrix(ell):
    N = ell.size
    R2 = np.zeros((N, N))
    for i in range(N):
        S = 0.0
        acc = 0.0
        for j in range(i + 1, N):
            b = j - 1                # new bond
            if b > i:
                S = ell[b] * (1.0 + S)
            acc += 1.0 + 2.0 * S
            R2[i, j] = acc
            R2[j, i] = acc
    return R2


def ideal_R2_matrix(kappa, beta=1.0):
    return _R2_matrix(langevin_L(beta * np.asarray(kappa, dtype=float)))


@njit(cache=True)
def _debye_gauss(R2, q):
    N = R2.shape[0]
    out = np.zeros(q.size)
    for iq in range(q.size):
        c = q[iq] * q[iq] / 6.0
        s = float(N)
        for i in range(N):
            for j in range(i + 1, N):
                s += 2.0 * math.exp(-c * R2[i, j])
        out[iq] = s / N
    return out


def debye_gaussian_closure(R2, q):
    """S(q) = (1/N) sum_ij exp(-q^2 <R_ij^2>/6): Gaussian closure of the Debye sum."""
    return _debye_gauss(np.ascontiguousarray(R2), np.asarray(q, dtype=float))


# ============================================================ exact ideal sampling
@njit(cache=True)
def _sample_chain(xk, seed):
    np.random.seed(seed)
    N = xk.size
    r = np.zeros((N, 3))
    # first bond: uniform on sphere
    z = 2 * np.random.random() - 1
    ph = 2 * math.pi * np.random.random()
    s = math.sqrt(1 - z * z)
    t = np.array([s * math.cos(ph), s * math.sin(ph), z])
    r[1] = t
    for i in range(1, N - 1):
        x = xk[i]
        u = np.random.random()
        if x < 1e-8:
            c = 2 * u - 1
        else:
            c = 1.0 + math.log(1.0 - u * (1.0 - math.exp(-2.0 * x))) / x
        c = min(1.0, max(-1.0, c))
        sn = math.sqrt(max(0.0, 1 - c * c))
        ph = 2 * math.pi * np.random.random()
        # orthonormal frame around t
        if abs(t[2]) < 0.9:
            a = np.array([0.0, 0.0, 1.0])
        else:
            a = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(t, a)
        e1 /= math.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
        e2 = np.cross(t, e1)
        t = c * t + sn * (math.cos(ph) * e1 + math.sin(ph) * e2)
        t /= math.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2)
        r[i + 1] = r[i] + t
    return r


def sample_ideal(kappa, beta=1.0, n_conf=1, seed=0):
    """Exact i.i.d. samples of the ideal chain with bending field kappa. (n_conf, N, 3)."""
    xk = beta * np.asarray(kappa, dtype=float)
    return np.stack([_sample_chain(xk, seed * 100003 + c) for c in range(n_conf)])


@njit(cache=True)
def _pair_hist(r, rmax, nb):
    N = r.shape[0]
    h = np.zeros(nb)
    dr = rmax / nb
    for i in range(N):
        for j in range(i + 1, N):
            d = math.sqrt((r[i, 0] - r[j, 0]) ** 2 + (r[i, 1] - r[j, 1]) ** 2 + (r[i, 2] - r[j, 2]) ** 2)
            k = int(d / dr)
            if k < nb:
                h[k] += 1.0
    return h


def debye_exact(confs, q, nb=20000):
    """Exact Debye S(q) = (1/N) sum_jk sin(q r_jk)/(q r_jk), averaged over conformations,
    via a fine pair-distance histogram."""
    N = confs.shape[1]
    rmax = max(np.ptp(c, axis=0).max() for c in confs) * 1.8 + 1.0
    h = np.zeros(nb)
    for c in confs:
        h += _pair_hist(np.ascontiguousarray(c), rmax, nb)
    h /= len(confs)
    rc = (np.arange(nb) + 0.5) * rmax / nb
    qr = np.outer(q, rc)
    return (N + 2.0 * (h[None, :] * np.sinc(qr / np.pi)).sum(1)) / N


# ================================================================ small utilities
def rankdata(x):
    from scipy.stats import rankdata as _r
    return _r(x)


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
