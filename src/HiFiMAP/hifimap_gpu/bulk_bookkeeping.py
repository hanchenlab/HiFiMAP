"""Vectorized/batched replacement for the per-site bookkeeping loop
in kernel_update.py (project_and_halve_fast + q_and_eigen_fast + n_ibd_from,
called once per site with a diff).

Why this exists: the per-site bookkeeping loop's real cost isn't the
linear algebra itself (the actual math per site is trivial -- a handful
of nonzero entries) but fixed Python/scipy object-construction overhead
paid once per site, even for sites with no diff (tar lookup), and
separately for every scipy sparse matrix built per fired site.

This module eliminates nearly all of that per-site overhead by:
1. Parsing every diff file's content in ONE pass (bulk_parse_all_diffs),
   instead of one `tarfile.extractfile()` + text-parse call per site.
2. Computing each site's (dQ, dE, n_ibd delta) using small PLAIN NUMPY
   arrays local to just that site's actually-touched subjects (typically
   ~9-21 of N~5000), instead of general-purpose scipy sparse matrix
   objects sized (N, N). scipy's sparse matrix classes carry real
   fixed per-object overhead (shape validation, dtype resolution,
   index-array construction) that dominates when the matrices are this
   tiny.

Each site's own delta here depends on nothing but that site's own diff
data (and the fixed, precomputed inv_map/residuals/random_vectors) --
never on any other site or on the running Q_curr/E_curr totals. The
chromosome-wide Q_curr/E_curr trajectory is a cumulative sum (prefix sum)
of these independent per-site deltas; building it via np.cumsum over the
SAME site order as the original sequential `+=` loop reproduces the
identical summation order, so it is bit-identical to the sequential
accumulation wherever the per-site deltas themselves are bit-identical.
"""
import io

import numpy as np
import pandas as pd

# Above this many touched subjects at one site, build M as a sparse
# matrix instead of the dense (k,k) array a typical site uses.
# Touched-per-site is almost always small, but a rare site (e.g. a
# chromosome's last site, where every segment still active gets
# force-closed at once) can have k in the thousands while the actual
# nonzero update count stays in the same thousands -- i.e. the dense
# (k,k) matrix at that site would be mostly zero. This threshold sits
# above the typical case and below the pathological one, so it only
# engages for the rare case, never for a typical fired site.
SPARSE_FALLBACK_THRESHOLD = 2000


def bulk_parse_all_diffs(tar_path, members):
    """Read every diff file's content directly out of the tar's raw bytes
    (one read of the whole archive, sliced per member via its known
    offset/size -- no per-member tarfile.extractfile() call), then parse
    every line across every site in one pandas C-engine pass.

    members: pre-built list of TarInfo (from ibd_io.open_diff_tar's
    index) -- building that index (one scan of the tar's headers) is a
    shared fixed cost paid once regardless of which reader is used
    afterward, so it's passed in rather than rebuilt here.

    Returns (site_id, op, h1, h2) as four flat numpy arrays, one row per
    diff line across the whole chromosome.
    """
    with open(tar_path, "rb") as f:
        raw = f.read()

    site_ids = []
    chunks = []
    for m in members:
        name = m.name.removeprefix("./")
        site = int(name[len("delta_"):-len(".diff")])
        content = raw[m.offset_data: m.offset_data + m.size]
        site_ids.append(site)
        chunks.append(content)

    line_counts = np.array([c.count(b"\n") for c in chunks], dtype=np.int64)
    site_id_arr = np.array(site_ids, dtype=np.int64)
    per_line_site = np.repeat(site_id_arr, line_counts)

    big_blob = b"".join(chunks)
    df = pd.read_csv(io.BytesIO(big_blob), sep=" ", header=None,
                      names=["op", "h1", "h2"], dtype=np.int64, engine="c")
    assert len(df) == len(per_line_site)
    return per_line_site, df["op"].to_numpy(), df["h1"].to_numpy(), df["h2"].to_numpy()


def r_round(x):
    return round(float(x))


def compute_all_site_deltas(site_id, op, h1, h2, inv_map, N, residuals, random_vectors,
                             start_site=0):
    """Compute every site's independent (dQ, dE, n_ibd delta), from the
    flat per-line arrays bulk_parse_all_diffs returns.

    Mirrors project_and_halve_fast + q_and_eigen_fast + n_ibd_from
    exactly (same symmetrization, same diagonal-halving), but builds a
    small PLAIN NUMPY matrix restricted to each site's own touched
    subjects, instead of a scipy sparse matrix sized (N, N).

    start_site: the chunk/chromosome's starting site is handled
    separately from its own full checkpoint matrix (ibd_mat_<start>.mtx),
    never as a delta -- e.g. delta_0.diff exists in the data and holds
    real content (segments active from the start of the chromosome
    through site 0), but that's already baked into ibd_mat_0.mtx, so
    applying it again as an incremental delta on top would double-count
    it. This must match whatever start site the caller's own initial
    K_hap0/ibd_mat_<start>.mtx read used -- default 0 for a full,
    unchunked chromosome pass.

    Returns four arrays (fired_sites, dQ, dE, n_ibd_delta), ascending by
    site, one entry per site with a nonzero net contribution, excluding
    start_site -- identical set of sites to what the sequential per-site
    loop (project_and_halve_fast -> eliminate_zeros -> nnz > 0 ->
    q_and_eigen_fast), run over range(start_site + 1, ...), would have
    produced. Arrays, not a dict keyed by site: the caller needs this
    data back out in site order anyway (for the cumulative-sum step),
    and building it here avoids a second Python-level pass reconstructing
    arrays from dict lookups.
    """
    not_start = site_id != start_site
    site_id = site_id[not_start]
    op = op[not_start]
    h1 = h1[not_start]
    h2 = h2[not_start]

    p1 = h1 // 2
    p2 = h2 // 2
    si = inv_map[p1]
    sj = inv_map[p2]
    v = np.where(op == 1, 1.0, -1.0)
    keep = (si >= 0) & (sj >= 0)

    site_k = site_id[keep]
    si_k = si[keep]
    sj_k = sj[keep]
    v_k = v[keep]

    order = np.argsort(site_k, kind="stable")
    site_s = site_k[order]
    si_s = si_k[order]
    sj_s = sj_k[order]
    v_s = v_k[order]

    boundaries = np.flatnonzero(np.diff(site_s)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(site_s)]))
    group_site_ids = site_s[starts]

    # group_site_ids is already ascending (site_s was sorted above), so
    # writing into preallocated arrays in this loop's order produces
    # site-sorted arrays directly -- no dict, and no second pass to pull
    # values back out of one afterward (the same per-site-Python-overhead
    # pattern this module exists to eliminate, just one level up).
    # Preallocating dE_arr at its upper-bound size (len(starts), every
    # site has at least one group whether or not it ends up firing) and
    # writing each site's dE directly into it avoids ever holding every
    # individual (B,B) array in a Python list AND a second, separately-
    # allocated stacked copy of the same data at the same time.
    B = random_vectors.shape[1]
    max_fired = len(starts)
    fired_arr = np.empty(max_fired, dtype=np.int64)
    dQ_arr = np.empty(max_fired, dtype=np.float64)
    dE_arr = np.empty((max_fired, B, B), dtype=np.float64)
    n_ibd_arr = np.empty(max_fired, dtype=np.int64)
    count = 0

    for g in range(len(starts)):
        a, b = starts[g], ends[g]
        s = int(group_site_ids[g])
        rows = si_s[a:b]
        cols = sj_s[a:b]
        vals = v_s[a:b]

        touched = np.unique(np.concatenate([rows, cols]))
        local_rows = np.searchsorted(touched, rows)
        local_cols = np.searchsorted(touched, cols)
        k = touched.size

        if k > SPARSE_FALLBACK_THRESHOLD:
            # Rare pathological site (e.g. a chromosome's last site, where
            # every still-active segment gets force-closed at once): a
            # (k,k) array -- dense OR sparse -- is the wrong shape of
            # computation here. This mirrors Bohong's own fallback in
            # Step2_Run_HiFiMAP.R (stream_diff_stats): when R's Matrix
            # package errors out on a too-large matrix, his C++ code
            # streams the raw diff lines one at a time straight into the
            # (B,B) accumulator, never building a (k,k) intermediate at
            # all. Vectorized equivalent here: accumulate each line's
            # rank-1-ish contribution directly via one (B,n_lines)@(n_lines,B)
            # matmul, holding only (n_lines, B) arrays, much smaller than
            # the dense (k,k) array this replaces, or a sparse (k,k)
            # matrix's own index/data arrays.
            rows_sym = np.concatenate([local_rows, local_cols])
            cols_sym = np.concatenate([local_cols, local_rows])
            vals_sym = np.concatenate([vals, vals])
            diag_mask = rows_sym == cols_sym
            vals_sym = np.where(diag_mask, vals_sym / 2.0, vals_sym)  # same diagonal-halving as the dense path

            res_r = residuals[touched][rows_sym]
            res_c = residuals[touched][cols_sym]
            dQ = float(np.sum(vals_sym * res_r * res_c))

            rv_r = random_vectors[touched[rows_sym], :]
            rv_c = random_vectors[touched[cols_sym], :]
            dE = (rv_r * vals_sym[:, None]).T @ rv_c

            n_ibd_delta = r_round(float(np.sum(vals_sym)) / 2.0)
            if dQ == 0.0 and not np.any(dE):
                continue
        else:
            M = np.zeros((k, k), dtype=np.float64)
            np.add.at(M, (local_rows, local_cols), vals)
            np.add.at(M, (local_cols, local_rows), vals)
            d = np.diag(M).copy()
            np.fill_diagonal(M, d / 2.0)

            if not np.any(M):
                continue

            rv_local = random_vectors[touched, :]
            res_local = residuals[touched]
            Mr = M @ res_local
            dQ = float(res_local @ Mr)
            MR = M @ rv_local
            dE = rv_local.T @ MR
            n_ibd_delta = r_round(M.sum() / 2.0)

        fired_arr[count] = s
        dQ_arr[count] = dQ
        dE_arr[count] = dE
        n_ibd_arr[count] = n_ibd_delta
        count += 1

    return fired_arr[:count], dQ_arr[:count], dE_arr[:count], n_ibd_arr[:count]
