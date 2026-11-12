"""Load Bohong's Step 1 output (exported by export_step1.R, in this
same src/HiFiMAP/ folder).

Step 1 itself stays in R, per the project's governing rule. This module only
reads R's own output, so both sides use the exact same random vectors.
"""
import numpy as np


def load_step1_export(export_dir):
    theta = np.fromfile(f"{export_dir}/theta.bin", dtype=np.float64)
    with open(f"{export_dir}/ncovar.txt") as f:
        ncovar = int(f.read().strip())
    with open(f"{export_dir}/id_include.txt") as f:
        id_include = [line.strip() for line in f]

    scaled_residuals = np.fromfile(f"{export_dir}/scaled_residuals.bin", dtype=np.float64)

    with open(f"{export_dir}/random_vectors_dim.txt") as f:
        n, b = (int(line.strip()) for line in f)
    random_vectors = np.fromfile(f"{export_dir}/random_vectors.bin", dtype=np.float64).reshape(
        (n, b), order="F"  # R matrices are column-major
    )

    with open(f"{export_dir}/RSigma_R_dim.txt") as f:
        rb, rb2 = (int(line.strip()) for line in f)
    RSigma_R = np.fromfile(f"{export_dir}/RSigma_R.bin", dtype=np.float64).reshape(
        (rb, rb2), order="F"
    )

    return {
        "theta": theta,
        "ncovar": ncovar,
        "id_include": id_include,
        "scaled_residuals": scaled_residuals,
        "random_vectors": random_vectors,
        "RSigma_R": RSigma_R,
    }
