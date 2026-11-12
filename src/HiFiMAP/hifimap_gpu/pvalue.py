"""Mixture-of-chi-square p-value: Davies -> Kuonen -> Liu, matching
pval_caculation_functions.R exactly, including its fallback thresholds
and its quirks (e.g. the n==1 shortcut in .pKuonen is computed but never
returned in R, so this port never takes that shortcut either).

Davies itself is not reimplemented in Python: q_pval calls the vendored C
algorithm directly (davies.py, via ctypes) -- the identical compiled code
CompQuadForm::davies calls in R, not a Python re-derivation of it. Kuonen
and Liu are ported below as fallbacks for cases Davies itself does not
resolve.
"""
import numpy as np
from scipy.stats import ncx2

from . import davies as _davies


def zapsmall(x, digits=7):
    """R's zapsmall(x, digits=7): nd = round(max(0, digits - log10(max(|x|)))),
    then round(x, nd). R rounds the digits argument to the nearest integer
    before rounding x (e.g. round(v, 2.9) rounds to 3 places in R, while
    round(v, 2.1) rounds to 2 places).
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return x
    mx = np.max(np.abs(x[finite]))
    if mx > 0:
        nd = max(0.0, digits - np.log10(mx))
    else:
        nd = float(digits)
    nd = round(nd)  # Python's round() is round-half-to-even, like R's
    # Python's built-in round() is used element-wise (not np.round) because it
    # is correctly-rounded from the true binary value, closer to R >= 4.0's
    # decimal-rounding algorithm than numpy's multiply/divide approach.
    return np.array([round(float(v), nd) for v in x], dtype=np.float64)


def pkuonen(x, lam):
    """Port of .pKuonen from pval_caculation_functions.R, with h=1, delta=0
    (the only case .Q_pval ever calls it with).

    Known, small, explainable difference from R: this root-finds the same
    equation R's uniroot(..., tol=1e-8) solves, but via scipy.optimize.brentq
    (xtol=1e-8). Both use a Brent-type algorithm at matching nominal
    tolerance, but are different implementations -- not bit-identical
    the way the vendored Davies call is, though immaterial at the p-value
    scale this is used at. Unlike Davies, this was not ported from R's
    literal source (no vendored uniroot C code was available to call), so
    this is the one piece of the p-value chain that is a faithful
    re-derivation rather than the identical compiled algorithm.
    """
    lam = np.asarray(lam, dtype=np.float64)
    nz = lam != 0
    lam = lam[nz]
    df = np.ones_like(lam)
    delta = np.zeros_like(lam)
    d = np.max(lam)
    lam = lam / d
    xx = x / d

    def k0(zeta):
        return -np.sum(df * np.log(1 - 2 * zeta * lam)) / 2 + np.sum(
            (delta * lam * zeta) / (1 - 2 * zeta * lam)
        )

    def kprime0(zeta):
        return np.sum(((delta + df) * lam) / (1 - 2 * zeta * lam)) + np.sum(
            2 * (delta * zeta * lam ** 2) / (1 - 2 * zeta * lam) ** 2
        )

    def kpprime0(zeta):
        return np.sum((2 * (2 * delta + df) * lam ** 2) / (1 - 2 * zeta * lam) ** 2) + np.sum(
            8 * delta * zeta * lam ** 3 / (1 - 2 * zeta * lam) ** 3
        )

    if np.any(lam < 0):
        lmin = np.max(1 / (2 * lam[lam < 0])) * 0.99999
    elif xx > np.sum((df + delta) * lam):
        lmin = -0.01
    else:
        lmin = -len(lam) * np.max(df + delta) / (2 * xx)
    lmax = np.min(1 / (2 * lam[lam > 0])) * 0.99999

    from scipy.optimize import brentq

    hatzeta = brentq(lambda z: kprime0(z) - xx, lmin, lmax, xtol=1e-8)
    w = np.sign(hatzeta) * np.sqrt(2 * (hatzeta * xx - k0(hatzeta)))
    v = hatzeta * np.sqrt(kpprime0(hatzeta))
    if abs(hatzeta) < 1e-4:
        return np.nan
    from scipy.stats import norm

    return norm.sf(w + np.log(v / w) / w)


def liu(q, lam):
    """Port of CompQuadForm::liu, with h=1, delta=0 (the only case .Q_pval
    ever calls it with)."""
    lam = np.asarray(lam, dtype=np.float64)
    c1 = np.sum(lam)
    c2 = np.sum(lam ** 2)
    c3 = np.sum(lam ** 3)
    c4 = np.sum(lam ** 4)
    s1 = c3 / (c2 ** 1.5)
    s2 = c4 / c2 ** 2
    muQ = c1
    sigmaQ = np.sqrt(2 * c2)
    tstar = (q - muQ) / sigmaQ
    if s1 ** 2 > s2:
        a = 1 / (s1 - np.sqrt(s1 ** 2 - s2))
        delta = s1 * a ** 3 - a ** 2
        l = a ** 2 - 2 * delta
    else:
        a = 1 / s1
        delta = 0.0
        l = c2 ** 3 / c3 ** 2
    muX = l + delta
    sigmaX = np.sqrt(2) * a
    return ncx2.sf(tstar * sigmaX + muX, l, delta)


def q_pval_no_davies(q, lam):
    """.Q_pval's kuonen -> liu tail, for use once a Davies port/bridge
    decides the davies step has failed (ifault>0 or pval<=1e-5 or pval>=1)."""
    pval = pkuonen(q, lam)
    if np.isnan(pval):
        pval = liu(q, lam)
    return pval


def q_pval(q, lam):
    """Full port of .Q_pval from pval_caculation_functions.R: davies (the
    vendored qfc algorithm, called natively, not bridged to R) -> kuonen ->
    liu, with the exact same fallback thresholds R uses.
    """
    Qq, ifault, _ = _davies.davies(q, lam, acc=1e-6)
    pval = Qq
    if (ifault > 0) or (pval <= 1e-5) or (pval >= 1):
        pval = pkuonen(q, lam)
        if np.isnan(pval):
            pval = liu(q, lam)
    return pval


def calc_pval(q, lam):
    """Mirrors Step2_Run_HiFiMAP.R's calc_pval exactly:
        pv <- try(.Q_pval(0, lambda), silent=T)
        if (inherits(pv, "try-error")) pv <- NA
        if (all(lambda < 0)) pv <- 0
    R's try() catches ANY error raised inside .Q_pval (including ones from
    an uncaught failure deep in .pKuonen's uniroot call) and turns it into
    NA, not a crash: a pathological lambda vector can make R's own
    .pKuonen fail with an uncaught error too, and R's calc_pval silently
    returns NA for it. SciPy's brentq raises an exception on the
    equivalent failure rather than returning NaN, so that exception must
    be caught here the same way.
    """
    try:
        pval = q_pval(q, lam)
    except Exception:
        pval = np.nan
    if np.all(np.asarray(lam) < 0):
        pval = 0.0
    return pval
