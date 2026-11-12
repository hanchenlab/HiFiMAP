"""Subject-level kernel construction and incremental Q / E updates.

Mirrors the R fast path in Step2_Run_HiFiMAP.R:
  K_subj <- P_proj %*% K_hap %*% t(P_proj); diag(K_subj) <- diag(K_subj)/2
  Q_curr <- crossprod(residuals, crossprod(K_subj, residuals))
  Eigen_curr <- crossprod(random_vecs, crossprod(K_subj, random_vecs))
  n_ibd_curr <- round(sum(K_subj)/2)
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def build_P_proj(id_include, samples_vcf, n_vcf):
    """R: match_idx <- match(id_include, samples_vcf); rows <- rep(1:N, each=2);
    cols <- c(rbind(2*match_idx-1, 2*match_idx))  (1-based).
    Here: 0-based columns 2*idx and 2*idx+1 for each subject row.
    """
    index = {s: j for j, s in enumerate(samples_vcf)}
    N = len(id_include)
    match_idx = np.array([index[i] for i in id_include], dtype=np.int64)  # KeyError -> NA equivalent, same as R stopping
    rows = np.repeat(np.arange(N), 2)
    cols = np.empty(2 * N, dtype=np.int64)
    cols[0::2] = 2 * match_idx
    cols[1::2] = 2 * match_idx + 1
    data = np.ones(2 * N, dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(N, 2 * n_vcf)), match_idx


def project_and_halve(P_proj, K_hap):
    """K_subj = P_proj @ K_hap @ P_proj.T, with the diagonal halved
    (matches R's diag(tmp) <- diag(tmp)/2, for same-subject haplotype pairs).
    """
    K_subj = (P_proj @ K_hap @ P_proj.T).tocsr()
    K_subj.sort_indices()
    diag = K_subj.diagonal().copy()
    K_subj = K_subj.tolil()
    K_subj.setdiag(diag / 2.0)
    return K_subj.tocsr()


def n_ibd_from(K_subj):
    # R: round(sum(K_subj)/2), using R's round-half-to-even on a scalar.
    total = K_subj.sum() / 2.0
    return r_round(total)


def r_round(x):
    """R's default round() is round-half-to-even, same convention as Python's
    built-in round() for floats (both follow IEEE 754), so this is a thin
    wrapper kept for clarity at call sites that mirror R's round(x)."""
    return round(float(x))


def q_and_eigen(K_subj, residuals, random_vectors):
    Kr = K_subj @ residuals
    Q = float(residuals @ Kr)
    KR = K_subj @ random_vectors
    E = random_vectors.T @ KR
    return Q, E


# ---------------------------------------------------------------------------
# Fast variants: mathematically equivalent reformulations of build_P_proj's
# projection and q_and_eigen, exploiting structure that the generic
# implementations above don't use. The generic sparse-sparse matmul in
# project_and_halve and the LIL-based diagonal halving cost more than the
# actual amount of data involved justifies, especially for a per-site delta
# update (which typically touches only a handful of haplotype pairs);
# q_and_eigen's dense (B,N)@(N,B) GEMM does the same -- it multiplies by
# every one of the N rows of random_vectors even though almost all of them
# are provably zero for a sparse delta.
#
# project_and_halve_fast is bit-identical to project_and_halve (pure
# algebraic reindexing, no reassociated floating point). q_and_eigen_fast
# is not bit-identical to q_and_eigen (summing the same real terms in a
# different order via a smaller GEMM is a valid but different
# floating-point rounding), though the difference is far below the
# pipeline's own zapsmall(digits=7) rounding.
# ---------------------------------------------------------------------------

def build_inverse_map(match_idx, n_vcf):
    """Inverse of build_P_proj's match_idx: inv_map[p] = i such that
    match_idx[i] == p, or -1 if haplotype-pair p is not in id_include
    (an excluded subject). Build once per run, reused for every site.
    """
    inv_map = np.full(n_vcf, -1, dtype=np.int64)
    inv_map[match_idx] = np.arange(len(match_idx))
    return inv_map


def project_and_halve_fast(inv_map, K_hap, N):
    """Algebraically identical to project_and_halve(P_proj, K_hap), for the
    specific P_proj build_P_proj always produces (exactly two 1.0 entries
    per row, at haplotype columns 2*match_idx[i] and 2*match_idx[i]+1):
    K_subj[i,j] is exactly the sum of the four K_hap entries at haplotype
    rows/cols {2*match_idx[i], 2*match_idx[i]+1} x {2*match_idx[j],
    2*match_idx[j]+1}. Computing that directly from K_hap's own nonzero
    entries (reindex by //2, keep pairs where both sides are in
    id_include, let COO->CSR's duplicate-summing do the accumulation) costs
    O(nnz(K_hap)); the generic P_proj @ K_hap @ P_proj.T sparse-sparse
    matmul pays a cost bound by the matrices' shape (5000x10000 here)
    regardless of how little data K_hap actually holds, which dominates
    for a typical per-site delta (nnz often under 30).

    The diagonal halving (R's diag(tmp)<-diag(tmp)/2) is done by
    subtracting a sparse correction built only at the diagonal positions
    that are actually nonzero, instead of routing through tolil()
    (whose setdiag() does one slow Python-level __setitem__ per row,
    N of them, regardless of how many are actually nonzero).
    """
    K_hap = K_hap.tocoo()
    pr = K_hap.row // 2
    pc = K_hap.col // 2
    si = inv_map[pr]
    sj = inv_map[pc]
    keep = (si >= 0) & (sj >= 0)
    K_subj = coo_matrix((K_hap.data[keep], (si[keep], sj[keep])), shape=(N, N)).tocsr()
    K_subj.sort_indices()
    diag = K_subj.diagonal()
    nz = diag.nonzero()[0]
    if nz.size:
        correction = coo_matrix((diag[nz] / 2.0, (nz, nz)), shape=(N, N)).tocsr()
        K_subj = (K_subj - correction).tocsr()
    return K_subj


def q_and_eigen_fast(K_subj, residuals, random_vectors):
    """Same Q and E as q_and_eigen, computed without multiplying by rows of
    random_vectors that K_subj @ random_vectors provably zeroes out. If
    K_subj's only nonzero rows are `nz_rows`, every other row of
    K_subj @ random_vectors is exactly zero, so
    random_vectors.T @ (K_subj @ random_vectors) only needs
    random_vectors[nz_rows].T @ (K_subj[nz_rows] @ random_vectors) -- the
    same sum, restricted to terms that aren't already known to be zero.
    For a typical per-site delta this cuts the dense GEMM's contraction
    dimension from N down to a handful of rows, since N x B GEMM
    throughput on CPU doesn't scale down proportionally with nnz the way
    a sparse-aware computation does.
    """
    Kr = K_subj @ residuals
    Q = float(residuals @ Kr)
    nz_rows = np.unique(K_subj.tocoo().row)
    KR_sub = K_subj[nz_rows, :] @ random_vectors
    E = random_vectors[nz_rows, :].T @ KR_sub
    return Q, E
