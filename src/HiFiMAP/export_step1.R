#!/usr/bin/env Rscript
# Exports a glmmkin2randomvec .rds object (Bohong's Step 1 output) to plain
# binary files a non-R program can read. This does not change any of
# Bohong's logic; it only re-serializes his Step 1 output so Python can
# consume it, per the project rule that both sides read the same random
# vectors from the same R object.
args <- commandArgs(trailingOnly = TRUE)
rds_path <- args[1]
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(rds_path)

writeBin(as.numeric(obj$theta), file.path(out_dir, "theta.bin"))
writeLines(as.character(obj$ncovar), file.path(out_dir, "ncovar.txt"))
writeLines(as.character(obj$id_include), file.path(out_dir, "id_include.txt"))
writeBin(as.numeric(obj$scaled.residuals), file.path(out_dir, "scaled_residuals.bin"))

rv <- as.matrix(obj$random.vectors)
writeLines(as.character(dim(rv)), file.path(out_dir, "random_vectors_dim.txt"))
writeBin(as.numeric(rv), file.path(out_dir, "random_vectors.bin"))  # column-major (R default)

rs <- as.matrix(obj$RSigma_R)
writeLines(as.character(dim(rs)), file.path(out_dir, "RSigma_R_dim.txt"))
writeBin(as.numeric(rs), file.path(out_dir, "RSigma_R.bin"))  # column-major

cat("Exported", rds_path, "to", out_dir, "\n")
cat("theta:", obj$theta, "\n")
cat("ncovar:", obj$ncovar, "\n")
cat("N:", length(obj$id_include), " random.vectors:", paste(dim(rv), collapse="x"), "\n")
