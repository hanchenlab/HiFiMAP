"""Python wrapper around the vendored qfc (Davies, 1980) C algorithm.

Mirrors CompQuadForm's R/davies.R exactly: same .C("qfc", ...) argument
order, same res -> Qq inversion (Qq = 1 - res). This calls the identical
compiled algorithm R itself calls, not a reimplementation, so results are
expected to match to floating-point precision, not just approximately.
"""
import ctypes
import os
import platform
import numpy as np

_LIB_EXT = "dylib" if platform.system() == "Darwin" else "so"
_LIB_PATH = os.path.join(os.path.dirname(__file__), "_vendor", "compquadform", f"libqfc.{_LIB_EXT}")
_lib = ctypes.CDLL(_LIB_PATH)
_lib.qfc.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # lambda
    ctypes.POINTER(ctypes.c_double),  # noncentral (delta)
    ctypes.POINTER(ctypes.c_int),     # df (h)
    ctypes.POINTER(ctypes.c_int),     # r
    ctypes.POINTER(ctypes.c_double),  # sigma
    ctypes.POINTER(ctypes.c_double),  # q
    ctypes.POINTER(ctypes.c_int),     # lim
    ctypes.POINTER(ctypes.c_double),  # acc
    ctypes.POINTER(ctypes.c_double),  # trace[7]
    ctypes.POINTER(ctypes.c_int),     # ifault
    ctypes.POINTER(ctypes.c_double),  # res
]
_lib.qfc.restype = None


def davies(q, lam, h=None, delta=None, sigma=0.0, lim=10000, acc=1e-4):
    """Same signature and defaults as CompQuadForm::davies(). Returns
    (Qq, ifault, trace) to mirror davies()'s returned list(trace, ifault, Qq).
    """
    lam = np.asarray(lam, dtype=np.float64)
    r = len(lam)
    h = np.ones(r, dtype=np.int32) if h is None else np.asarray(h, dtype=np.int32)
    delta = np.zeros(r, dtype=np.float64) if delta is None else np.asarray(delta, dtype=np.float64)

    r_c = ctypes.c_int(r)
    sigma_c = ctypes.c_double(sigma)
    q_c = ctypes.c_double(q)
    lim_c = ctypes.c_int(lim)
    acc_c = ctypes.c_double(acc)
    trace = (ctypes.c_double * 7)()
    ifault = ctypes.c_int(0)
    res = ctypes.c_double(0.0)

    _lib.qfc(
        lam.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        delta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(r_c),
        ctypes.byref(sigma_c),
        ctypes.byref(q_c),
        ctypes.byref(lim_c),
        ctypes.byref(acc_c),
        trace,
        ctypes.byref(ifault),
        ctypes.byref(res),
    )
    Qq = 1.0 - res.value
    return Qq, ifault.value, list(trace)
