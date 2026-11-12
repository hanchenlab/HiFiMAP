"""Read sites, samples, IBD matrix checkpoints, and per-site diffs.

Mirrors Step2_Run_HiFiMAP.R: read_diff_as_dK, and the sites.txt /
samples.txt / ibd_mat_*.mtx conventions written by src/hapIBD_parsing/parsing_hapIBD.py.
"""
import tarfile

import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix, csr_matrix


def _lines_to_dK(lines, n_hap):
    """Shared core of read_diff_as_dK / read_diff_as_dK_from_tar: parse
    "op h1 h2" lines into the symmetrized sparse delta matrix. Identical
    math regardless of where the lines came from (plain file or tar
    member) -- only the I/O differs between the two callers.
    """
    ops, rows, cols = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) != 3:
            continue
        op, h1, h2 = int(parts[0]), int(parts[1]), int(parts[2])
        ops.append(1.0 if op == 1 else -1.0)
        rows.append(h1)
        cols.append(h2)
    if not ops:
        return None
    vals = np.array(ops, dtype=np.float64)
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    upper = coo_matrix((vals, (rows, cols)), shape=(n_hap, n_hap))
    return (upper + upper.T).tocsr()


def read_sites(path):
    """Returns (chrom, bp) arrays, 0-indexed by row to match R's 1-indexed site+1."""
    bp = []
    chrom = None
    with open(path) as f:
        next(f)  # header: CHR\tBP
        for line in f:
            c, p = line.rstrip("\n").split("\t")
            chrom = c
            bp.append(int(p))
    return chrom, np.array(bp, dtype=np.int64)


def read_samples(path):
    with open(path) as f:
        return [line.strip() for line in f]


def read_mtx(path, n_hap):
    """MatrixMarket coordinate file, already written fully symmetric by the parser."""
    m = mmread(path)
    return csr_matrix(m, shape=(n_hap, n_hap), dtype=np.float64)


def read_diff_as_dK(path, n_hap):
    """Mirrors read_diff_as_dK in Step2_Run_HiFiMAP.R.

    Diff lines are "op h1 h2" with 0-based haplotype indices and op in {0, 1}.
    The parser guarantees h1 < h2 (strict upper triangle, no diagonal), so
    symmetrizing is exactly coo + coo.T (matches R's forceSymmetric(dK) for
    this data, which only ever sees upper-triangle, non-diagonal entries).
    """
    import os
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return _lines_to_dK(f, n_hap)


def open_diff_tar(tar_path):
    """Open an uncompressed tar of delta_<site>.diff files for random-access
    reading without extracting them to individual files first.

    Built for the HPC transfer: chr22 has ~55,000 diff files, and
    extracting that many individual small files onto the HPC's home
    filesystem hit its per-file quota ("Disk quota exceeded") well before
    any byte-size limit, even though the total data is only ~170MB.
    Reading straight out of one tar archive sidesteps that entirely.
    Uses a plain (uncompressed) tar, not .tar.gz: gzip doesn't support
    cheap random access, and this needs ~tens of thousands of independent
    lookups, not one sequential pass.

    Returns (TarFile, {member_name: TarInfo}) -- the index is built once
    here (a single pass over the tar's headers) so each later lookup in
    read_diff_as_dK_from_tar is an O(1) dict access, not a re-scan.
    """
    tar = tarfile.open(tar_path, "r:")
    # Normalize names defensively: tar built with a leading "./" (e.g. from
    # `find . -name ... | tar --files-from=-`, as macOS's tar does) stores
    # members as "./delta_N.diff", not "delta_N.diff" -- strip it so lookups
    # work regardless of how the archive was actually built.
    index = {m.name.removeprefix("./"): m for m in tar.getmembers() if m.isfile()}
    return tar, index


def open_diff_tar_fast(tar_path, index):
    """Like open_diff_tar, but skips rebuilding the index -- for worker
    processes that already received an index built once by the parent
    (e.g. passed through a multiprocessing initializer). Opening a tar
    without calling getmembers() is cheap (no header scan); it's the
    header scan itself that's expensive and only needs doing once,
    not once per worker.
    """
    tar = tarfile.open(tar_path, "r:")
    return tar, index


def read_diff_as_dK_from_tar(tar, index, site, n_hap):
    """Same semantics and return value as read_diff_as_dK (None if the site
    has no diff), reading from an already-open tar/index (see
    open_diff_tar) instead of a plain file path.
    """
    member = index.get(f"delta_{site}.diff")
    if member is None:
        return None
    f = tar.extractfile(member)
    lines = f.read().decode().splitlines()
    return _lines_to_dK(lines, n_hap)
