#!/usr/bin/env python3
"""Step 2 (Python/GPU port): batched version of the per-position HiFiMAP
scan, bounding peak memory to one batch's worth of (B,B) matrices instead
of materializing the whole chromosome's fired-site array.

Key design point: batches here are the script's OWN arbitrary, runtime-
configurable boundaries over the fired-site list -- NOT a re-adoption of
Step 0's precomputed ibd_mat_<N>.mtx checkpoints. The diff chain itself
is checkpoint-independent, so batching only needs to carry forward the
tiny running (Q_curr, E_curr) state between batches, never a saved
snapshot file.

Only the (batch_size, B, B) M array is held per batch; it's eigh'd then
discarded. The only thing accumulated across the whole chromosome is the
eigenvalues themselves (fired_sites x B, not fired_sites x B x B, since
eigenvalues are B numbers per site, not a BxB matrix), plus the
n_ibd/fired-site-id bookkeeping, both negligible.

Usage:
    python3 Step2_Run_HiFiMAP_GPU.py <chr> <ibd_out_dir> <step1_export_dir> [outfile] [batch_size]

    <chr>              Chromosome label (matches the "chr" column in the output).
    <ibd_out_dir>       Step 0's output directory for this chromosome (sites.txt,
                        samples.txt, ibd_mat_0.mtx, delta_*.diff files). Must also
                        contain diffs.tar, built from those delta_*.diff files by
                        build_diffs_tar.sh in this same folder.
    <step1_export_dir>  Output directory of export_step1.R (theta.bin, ncovar.txt,
                        id_include.txt, scaled_residuals.bin, random_vectors.bin +
                        dim file, RSigma_R.bin + dim file).
    [outfile]           Output path (default: results/HiFiMAP_GPU_chr<chr>.txt).
    [batch_size]        Sites processed per GPU/CPU eigendecomposition batch
                        (default: 5000).

GPU is used automatically when available (torch + CUDA); set FORCE_CPU=1 to
run on CPU regardless of GPU availability.
"""
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hifimap_gpu import ibd_io, kernel_update, pvalue, bulk_bookkeeping  # noqa: E402
from hifimap_gpu.null_model import load_step1_export  # noqa: E402

if len(sys.argv) < 4:
    sys.exit(
        "usage: Step2_Run_HiFiMAP_GPU.py <chr> <ibd_out_dir> <step1_export_dir> "
        "[outfile] [batch_size]"
    )

CHR = sys.argv[1]
IBD_DIR = sys.argv[2]
STEP1_EXPORT = sys.argv[3]
DIFFS_TAR = os.path.join(IBD_DIR, "diffs.tar")
if not os.path.exists(DIFFS_TAR):
    sys.exit(
        f"{DIFFS_TAR} not found. Build it from {IBD_DIR}'s delta_*.diff files "
        f"with build_diffs_tar.sh first."
    )

try:
    import torch
    USE_GPU = torch.cuda.is_available() and not os.environ.get("FORCE_CPU")
except ImportError:
    USE_GPU = False

if USE_GPU:
    torch.cuda.reset_peak_memory_stats()


def _eigvalsh(M):
    """Single GPU/CPU dispatch point for eigvalsh, used for every batch
    AND for lam0 (the handful of sites before the first fired site) --
    one shared function so the two call sites can't drift apart (an
    earlier version of this script had lam0 skip the `.to("cuda")` move,
    a bug invisible on a CPU-only machine since both paths produce the
    same result there, but would have silently run lam0 on CPU even in
    GPU mode on real hardware).
    """
    if USE_GPU:
        # No explicit synchronize() here: CUDA orders operations on the same
        # stream automatically, so eigvalsh already waits for the h2d copy
        # to finish, and `.to("cpu")` below is itself a blocking call that
        # already waits for eigvalsh to finish -- two prior sync() calls (one
        # before eigvalsh, one after the CPU copy already happened) were both
        # pure no-ops, waiting for work that was already correctly ordered or
        # already done. Also no empty_cache(): every batch allocates a
        # near-identical-sized tensor, which is exactly what PyTorch's own
        # caching allocator exists to reuse cheaply -- forcing a real
        # cudaFree/cudaMalloc cycle every batch defeats that for no memory
        # benefit (peak memory is bounded within one batch's own compute,
        # not by what's cached between batches).
        M_gpu = torch.from_numpy(M).to(torch.float64).to("cuda")
        lam = torch.linalg.eigvalsh(M_gpu).to("cpu").numpy()
        del M_gpu
        return lam
    return np.linalg.eigvalsh(M)

out_file = sys.argv[4] if len(sys.argv) > 4 else f"results/HiFiMAP_GPU_chr{CHR}.txt"
BATCH_SIZE = int(sys.argv[5]) if len(sys.argv) > 5 else 5000
os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
print(f"chromosome: {CHR}  GPU available: {USE_GPU}  batch_size: {BATCH_SIZE}", flush=True)

t_total0 = time.time()

step1 = load_step1_export(STEP1_EXPORT)
residuals = step1["scaled_residuals"]
random_vectors = step1["random_vectors"]
RSigma_R = step1["RSigma_R"]
ncovar = step1["ncovar"]
id_include = step1["id_include"]
N = len(id_include)
B = random_vectors.shape[1]

chrom, bp = ibd_io.read_sites(f"{IBD_DIR}/sites.txt")
samples_vcf = ibd_io.read_samples(f"{IBD_DIR}/samples.txt")
n_vcf = len(samples_vcf)
n_hap = 2 * n_vcf
N_SITES = len(bp)

P_proj, match_idx = kernel_update.build_P_proj(id_include, samples_vcf, n_vcf)
inv_map = kernel_update.build_inverse_map(match_idx, n_vcf)

Q_offset = 2.0 * float(np.sum(residuals ** 2))
RtR2 = 2.0 * (random_vectors.T @ random_vectors)
eigenmat_offset_const = RtR2 - (Q_offset / (N - ncovar)) * RSigma_R

# ---- Phase 0: initial site (same math as the single-pass predecessor) ----
t0 = time.time()
K_hap0 = ibd_io.read_mtx(f"{IBD_DIR}/ibd_mat_0.mtx", n_hap)
K_subj0 = kernel_update.project_and_halve_fast(inv_map, K_hap0, N)
Q0, E0 = kernel_update.q_and_eigen_fast(K_subj0, residuals, random_vectors)
n_ibd0 = kernel_update.n_ibd_from(K_subj0)
t_phase0 = time.time() - t0

# ---- Bulk-parse ALL diffs once. This is cheap: flat per-line arrays,
# not the (fired_sites, B, B) array -- e.g. chr21's ~1.2M diff lines at
# ~25 bytes/line is ~30MB, nowhere near the real memory driver. ----
t0 = time.time()
tar, index = ibd_io.open_diff_tar(DIFFS_TAR)
members = list(index.values())
site_id, op, h1, h2 = bulk_bookkeeping.bulk_parse_all_diffs(DIFFS_TAR, members)

# np.unique + the batch count were previously computed after the parse timer
# had already stopped, so their real (if modest) cost was falling into an
# unmeasured, unlabeled gap -- same category as the write-loop finding below.
# Stopping the timer here instead means every phase's printed time now
# actually accounts for 100% of the run, nothing hidden between the lines.
fired_sites_all = np.unique(site_id[site_id != 0])
n_fired_total = len(fired_sites_all)
n_batches = int(np.ceil(n_fired_total / BATCH_SIZE))
t_phase1_parse = time.time() - t0
print(f"bulk parse={t_phase1_parse:.2f}s, {n_fired_total} fired sites, {n_batches} batches of up to {BATCH_SIZE}", flush=True)

# ---- Batch loop: build and eigh one batch's (batch_size, B, B) array at
# a time, keep only its eigenvalues (B numbers/site, not B^2), discard
# the M array before the next batch. Running (Q_curr, E_curr) state
# carries across the batch boundary -- the same quantity the unbatched
# predecessor carries across every ordinary site, just checked less often. ----
t0 = time.time()
Q_running = Q0
E_running = E0.copy()
n_ibd_running = n_ibd0

all_fired = []
all_lam = []
all_n_ibd_cum = []
peak_batch_rss_bytes = 0

for b in range(n_batches):
    batch_fired_sites = fired_sites_all[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
    lo, hi = int(batch_fired_sites[0]), int(batch_fired_sites[-1])
    mask = (site_id >= lo) & (site_id <= hi)

    fired_b, dQ_b, dE_b, dn_b = bulk_bookkeeping.compute_all_site_deltas(
        site_id[mask], op[mask], h1[mask], h2[mask], inv_map, N, residuals, random_vectors,
        start_site=-1,  # nothing to exclude mid-chromosome (only true site 0 is excluded, handled by phase0)
    )

    Q_cum = Q_running + np.cumsum(dQ_b)
    n_ibd_cum = n_ibd_running + np.cumsum(dn_b)

    np.cumsum(dE_b, axis=0, out=dE_b)
    dE_b += E_running[None, :, :]
    E_running_next = dE_b[-1].copy()  # true E_curr trajectory, before the M-only offset/correction below

    # Reuse dE_b's own buffer for M (no extra allocation): M = E_curr + offset - (Q/(N-ncovar))*RSigma_R
    dE_b += eigenmat_offset_const[None, :, :]
    dE_b -= (Q_cum / (N - ncovar))[:, None, None] * RSigma_R[None, :, :]
    M_batch = dE_b

    lam_batch = _eigvalsh(M_batch)

    all_fired.append(fired_b)
    all_lam.append(lam_batch)
    all_n_ibd_cum.append(n_ibd_cum)

    Q_running = float(Q_cum[-1])
    E_running = E_running_next
    n_ibd_running = int(n_ibd_cum[-1])

    batch_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_batch_rss_bytes = max(peak_batch_rss_bytes, batch_rss)
    del M_batch, dE_b, dQ_b, dn_b

    print(f"batch {b + 1}/{n_batches}: sites {lo}-{hi}, {len(fired_b)} fired", flush=True)

t_phase_batches = time.time() - t0

# ---- Final assembly: concatenate the compact per-batch results (B
# numbers/site, not B^2) and forward-fill every site, from the GLOBAL
# fired-site list built across all batches. ----
t0 = time.time()
fired_arr = np.concatenate(all_fired)
lam_all = np.concatenate(all_lam, axis=0)
n_ibd_full_fired = np.concatenate(all_n_ibd_cum)

n_ibd_full = np.concatenate(([n_ibd0], n_ibd_full_fired))
m_idx_per_site = np.searchsorted(fired_arr, np.arange(N_SITES), side="right")
n_ibd_per_site = n_ibd_full[m_idx_per_site]

# phase0's own eigenvalues (sites before the first fired site use this)
M0 = E0 + eigenmat_offset_const - (Q0 / (N - ncovar)) * RSigma_R
lam0 = _eigvalsh(M0)

# Compute each DISTINCT p-value exactly once -- one per unique state (lam0,
# then each fired site's own eigenvalues) -- instead of once per position.
# Most positions between two fired sites share the identical eigenvalues as
# their neighbors (nothing changed there), the same reason n_ibd_per_site
# above is already a simple lookup rather than recomputed per position.
n_unique = len(lam_all) + 1
pv_unique = [None] * n_unique
lam0_z = pvalue.zapsmall(lam0 / B, digits=7)
pv_unique[0] = pvalue.calc_pval(0.0, lam0_z)
for k in range(len(lam_all)):
    lam_z = pvalue.zapsmall(lam_all[k] / B, digits=7)
    pv_unique[k + 1] = pvalue.calc_pval(0.0, lam_z)

rows = [(chrom, bp[i], int(n_ibd_per_site[i]), pv_unique[m_idx_per_site[i]]) for i in range(N_SITES)]
t_phase_finalize = time.time() - t0

# Builds one string and issues one write() instead of N_SITES individual
# write() calls (same data, same formatting, just not done one
# syscall-worth-of-Python-overhead at a time).
t0 = time.time()
out_lines = ["chr\tpos\tn.ibd.segs\tp.value\n"]
out_lines.extend(f"{c}\t{p}\t{n}\t{pv}\n" for c, p, n, pv in rows)
with open(out_file, "w") as f:
    f.write("".join(out_lines))
t_phase_write = time.time() - t0

t_total = time.time() - t_total0
print(f"wrote {len(rows)} rows to {out_file}", flush=True)
print(f"TOTAL = {t_total:.2f}s (phase0={t_phase0:.2f} parse={t_phase1_parse:.2f} "
      f"batches={t_phase_batches:.2f} finalize={t_phase_finalize:.2f} write={t_phase_write:.2f}), "
      f"{n_batches} batches of {BATCH_SIZE}", flush=True)

_rss_divisor = 1e6 if sys.platform.startswith("linux") else 1e9  # ru_maxrss: KB on Linux, bytes on macOS
peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _rss_divisor
print(f"peak host RSS (whole run) = {peak_rss_gb:.3f} GB", flush=True)
print(f"peak host RSS during batch loop only = {peak_batch_rss_bytes/_rss_divisor:.3f} GB", flush=True)
if USE_GPU:
    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    print(f"peak GPU memory: allocated={peak_alloc_gb:.3f} GB reserved={peak_reserved_gb:.3f} GB", flush=True)
