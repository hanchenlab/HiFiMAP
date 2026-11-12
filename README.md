# HiFiMAP
High resolution fast identity-by-descent (IBD) mapping test
<br />
Current version: 2.1.0 (adds a Python/GPU implementation of Step 2; see [version history](#version-history))

## Overview
The new HiFiMAP pipeline has been rebuilt for scalability and efficiency. It uses a highly optimized Python preparser to convert IBD segments into chunked sparse matrices (`.mtx`) and differential updates (`.diff`). The main R program utilizes C++ (`Rcpp`) to stream these updates, allowing for fast, parallelized genome-wide association testing without frequently loading massive IBD matrices into RAM.

## Workflow

<img src="docs/workflow.svg" alt="HiFiMAP pipeline: Step 0 parses IBD segments in Python, Step 1 fits the null model in R, then Step 2 runs as either the R scan or the Python/GPU scan, producing the same output format." width="100%" />

## Quick Installation 

`HiFiMAP` can be downloaded via:
```bash
git clone https://github.com/baihongguo/HiFiMAP
cd HiFiMAP
```

## Preparations
The software is developed and tested in Linux environments. 

**Python Dependencies (Python >= 3.6):**
* `numpy` (Step 0)
* `scipy`, `pandas` (Step 2, Python/GPU implementation only)
* `torch` (optional; enables GPU acceleration for Step 2's Python implementation, which otherwise runs on CPU)

**R Dependencies (R >= 4.0.0):**
* `data.table`
* `Matrix`
* `GMMAT`
* `Rcpp`
* `CompQuadForm`

---

## Usage & Example Pipeline

Based on the repository structure, all test files are located in the `example/` directory, and all source codes are in the `src/` directory. Suppose you want to run a test analysis for chromosome 21 using the provided toy data:

* `example/chr21_toy.ibd.gz`: IBD segments generated from [hapIBD](https://github.com/browning-lab/hap-ibd).
* `example/chr21_toy.vcf.gz`: The corresponding VCF file used to generate the IBD segments.
* `example/toy_phenotype.txt`: Contains individual IDs, phenotype (currently continuous only), and covariates (e.g., age, sex).
* `example/toy_global_kinship.RData`: The global IBD (kinship) matrix.

### Step 0: Pre-parse the IBD Segments (Python)
This step transforms the raw IBD segments into high-performance sparse matrices and delta differences. It automatically chunks the chromosome to allow for parallel processing in Step 2.

Run the parser from the root of the repository:

```bash
python3 src/hapIBD_parsing/parsing_hapIBD.py \
    --ibd example/chr21_toy.ibd.gz \
    --vcf example/chr21_toy.vcf.gz \
    --output example/ibd_prep/chr21 \
    --n-checkpoints 20
```

| Argument | Description |
| ------------- | ------------- |
| `--ibd` | The IBD segment file output from hapIBD. Its sample IDs must be the VCF sample IDs; the parser reports how many IBD records match and stops with an error if none do. |
| `--vcf` | The reference VCF file to extract exact SNP positions and sample IDs (normally the VCF used to run hapIBD). |
| `--output` | The directory where the chunked matrices and diffs will be saved. |
| `--n-checkpoints` | Number of chunks to divide the chromosome into (e.g., `20`). This controls parallelization scaling in Step 2. Set this number to be less than the number of CPUs you have for better performance, recommended is 20-30. |

*(This will generate a folder `example/ibd_prep/chr21` containing `sites.txt` (bp of each testing site), `samples.txt` (individual IDs for matching with the null model), and the chunked `.mtx` / `.diff` files).*

<br />

### Step 1: Fit the Null GLMM Model (R)
Next, fit the null model using your phenotypes and global kinship matrix through the glmmkin function from the GMMAT package. **This step only needs to be run once per phenotype**, regardless of how many chromosomes you analyze. Please modify the glmmkin formula in Step1_Save_null_model_HiFiMAP.R based on your phenotype (currently continuous only) and covariate names. ID column should be named id.

```bash
Rscript src/HiFiMAP/Step1_Save_null_model_HiFiMAP.R \
    example/toy_phenotype.txt \
    example/toy_global_kinship.RData \
    toy_pheno \
    12345
```

| Argument | Description |
| ------------- | ------------- |
| `phenotype.txt` | The file containing ID, Phenotype, and covariates. |
| `global_kinship.RData` | The kinship matrix saved in `.RData` format. Row names and col names should be the corresponding subject IDs |
| `toy_pheno` | Prefix for the output file. |
| `12345` | Random seed for reproducibility. |

*(This generates the null model object `toy_pheno_glmmkin2randomvec.rds` in the working directory.)*

**Subjects:** the subjects in the phenotype file can be any subset of the samples in the VCF used for hapIBD and Step 0, in any order. For example, you can call and parse the IBD segments once for the whole cohort, then fit null models for different phenotypes on different subsets of it.
- **Step 1** keeps the subjects that are in both the phenotype file and the kinship matrix and have no missing values.
- **Step 2** keeps the IBD sharing among those subjects only. It stops with an error if a subject is not a VCF sample (listed in `samples.txt` after Step 0).

<br />

### Step 2: Run Parallelized HiFiMAP Scan (Bash wrapper)
To run the actual association scan efficiently across all chunks, you can use the `Run_HiFiMAP_parallel.sh` wrapper script. 

Make sure the configuration inside `Run_HiFiMAP_parallel.sh` points to the correct `src` paths and `example` outputs:

```bash
#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
# Set the number of threads for matrix operations in R via Sys.setenv(MKL_NUM_THREADS = threads).
# NOTE: This is only effective if R is compiled with the Intel Math Kernel Library (MKL).
threads=2
target_chromosomes="21"

OUT_BASE="./results"
IBD_PREP_BASE="./example/ibd_prep"
GLMM_RDS="toy_pheno_glmmkin2randomvec.rds" 
script_name="src/HiFiMAP/Step2_Run_HiFiMAP.R"

mkdir -p "$OUT_BASE"

# ==========================================
# EXECUTION LOOP
# ==========================================
for chr in $target_chromosomes; do
    ibd_out_dir="${IBD_PREP_BASE}/chr${chr}"
    
    echo "=================================================="
    echo "Processing Chromosome $chr..."
    echo "=================================================="

    if [ ! -d "$ibd_out_dir" ]; then
        echo "  [ERROR] Input dir not found for Chr $chr: $ibd_out_dir"
        continue
    fi

    # Read the ACTUAL site numbers from the freshly generated Python files
    all_checkpoints=($(ls "${ibd_out_dir}"/ibd_mat_*.mtx 2>/dev/null | xargs -n 1 basename | sed 's/ibd_mat_//;s/\.mtx//' | sort -n))
    num_avail_checkpoints=${#all_checkpoints[@]}

    if [ "$num_avail_checkpoints" -eq 0 ]; then
        echo "  [ERROR] No .mtx checkpoints found in $ibd_out_dir. Skipping."
        continue
    fi

    total_sites=$(tail -n +2 "${ibd_out_dir}/sites.txt" | wc -l)
    
    echo "  [LAUNCH] Chr $chr | Spawning exactly $num_avail_checkpoints parallel chunks..."

    # --- LAUNCH CHUNKS ---
    for (( j=0; j<num_avail_checkpoints; j++ )); do
        
        start_idx=${all_checkpoints[$j]}
        
        # Calculate End Index
        next_job_index=$((j + 1))
        if [ "$next_job_index" -lt "$num_avail_checkpoints" ]; then
            end_idx=$(( ${all_checkpoints[$next_job_index]} - 1 ))
        else
            end_idx=$((total_sites - 1))
        fi

        chunk=$((j+1))
        outfile_prefix="${OUT_BASE}/HiFiMAP_toy_pheno_chr${chr}_chunk${chunk}"

        Rscript "$script_name" \
            "$chr" \
            "$outfile_prefix" \
            "$threads" \
            "$GLMM_RDS" \
            "$ibd_out_dir" \
            "$start_idx" \
            "$end_idx" > "${outfile_prefix}.log" 2>&1 &
            
    done

    echo "  [MONITOR] All chunks dispatched. Waiting for completion..."
    wait 

    echo "  [MERGE] Merging results for Chromosome $chr..."
    final_out="${OUT_BASE}/HiFiMAP_toy_pheno_chr${chr}.txt"
    chunk1="${OUT_BASE}/HiFiMAP_toy_pheno_chr${chr}_chunk1.txt"
    
    if [ -f "$chunk1" ]; then
        head -n 1 "$chunk1" > "$final_out"
        count=0
        for (( j=1; j<=num_avail_checkpoints; j++ )); do
            f="${OUT_BASE}/HiFiMAP_toy_pheno_chr${chr}_chunk${j}.txt"
            if [ -f "$f" ]; then
                tail -n +2 -q "$f" >> "$final_out"
                rm "$f" 
                ((count++))
            fi
        done
        echo "  [SUCCESS] Chr $chr: Merged $count chunks -> $final_out"
    fi
done

echo "Pipeline Complete!"
```

### Final Output
The final merged result file (`results/HiFiMAP_toy_pheno_chr21.txt`) will contain the association statistics for every tested site. For example, with the toy data and seed `12345`:

```text 
chr     pos       n.ibd.segs    p.value
21	14240779	1052	0.6283424	
21	14242245	1054	0.632808	
21	14250602	1056	0.6316593	
21	14256336	1057	0.6555438
```
The p-values are computed with simulated random vectors from Step 1. They therefore differ slightly with the seed, and can differ slightly between systems; with seeds 1, 2 and 3 these four p-values ranged from 0.61 to 0.65. `n.ibd.segs` does not depend on the seed.

### Step 2 (alternative): Run the HiFiMAP Scan on Python/GPU

`Step2_Run_HiFiMAP_GPU.py` is a Python implementation of the same test as
`Step2_Run_HiFiMAP.R`: it batches the whole chromosome into one (or a few)
eigendecomposition calls instead of chunking across CPU processes, and runs
on a GPU automatically when one is available. It reads the same Step 0 and
Step 1 outputs and writes the same `chr  pos  n.ibd.segs  p.value` format.

**One-time setup, per machine:**
```bash
bash src/HiFiMAP/build_qfc.sh
```
Builds the Davies' algorithm C++ extension (`libqfc.dylib` on macOS,
`libqfc.so` on Linux) that `hifimap_gpu/davies.py` calls into. A prebuilt
`libqfc.dylib` is checked in for macOS; Linux/HPC machines must run this
once to build their own `libqfc.so`.

**Per-chromosome preparation, after Step 0:**
```bash
bash src/HiFiMAP/build_diffs_tar.sh example/ibd_prep/chr21
```
Packages Step 0's `delta_*.diff` files into one `diffs.tar` inside that
same directory, which the Python implementation reads in bulk instead of
opening each file individually. (`Step2_Run_HiFiMAP.R` reads the loose
`.diff` files directly and does not need this.)

**Export Step 1's null model, once per phenotype:**
```bash
Rscript src/HiFiMAP/export_step1.R \
    toy_pheno_glmmkin2randomvec.rds \
    example/step1_export
```
Re-serializes Step 1's `.rds` output — the same object `Step2_Run_HiFiMAP.R`
reads — to flat binary files Python can read, without changing any of its
values.

**Run the scan:**
```bash
python3 src/HiFiMAP/Step2_Run_HiFiMAP_GPU.py \
    21 \
    example/ibd_prep/chr21 \
    example/step1_export
```

| Argument | Description |
| ------------- | ------------- |
| `21` | Chromosome label (matches the `chr` column in the output). |
| `example/ibd_prep/chr21` | Step 0's output directory for this chromosome, plus the `diffs.tar` built above. |
| `example/step1_export` | Step 1's exported flat files, from `export_step1.R` above. |
| *(optional, positional)* output path | Default: `results/HiFiMAP_GPU_chr<chr>.txt`. |
| *(optional, positional)* batch size | Positions processed per eigendecomposition call. Default: `5000`. |

Set `FORCE_CPU=1` to run on CPU even when a GPU is available.

<br />

## Version history
**2.1.0**
- Adds `Step2_Run_HiFiMAP_GPU.py`, a Python implementation of Step 2 that batches the whole chromosome into GPU (or CPU) eigendecomposition calls instead of chunking across CPU processes, producing the same output as `Step2_Run_HiFiMAP.R`.
- Adds `build_qfc.sh`, `build_diffs_tar.sh`, and `export_step1.R` as setup scripts for the Python implementation, and the `hifimap_gpu` module it runs on.
- Adds a workflow diagram (`docs/workflow.svg`) summarizing all steps, both implementations of Step 2, and the file names involved in each.

**2.0.1**
- `parsing_hapIBD.py` reports how many IBD records match the VCF sample IDs and stops with an error if none match. Previously it silently wrote empty checkpoint matrices and no `.diff` files. It also skips a header line in the IBD file.
- Step 2 stops with a clear error if a subject of the null model is not a VCF sample, and compares IDs as text.
- Steps 1 and 2 load their helper scripts from their own folder, so they run from any working directory.
- Removed the unused `fastmatch` dependency.
- Updated the example output to the current toy data.

**2.0.0**: optimized streaming pipeline.

## References
<p>If you use HiFiMAP, please cite
<li>Guo, B. et al. HiFiMAP: High-resolution fast identity-by-descent mapping test. <em>medRxiv</em> 2026.05.06.26352570
[<b>Preprint</b>]. May 17, 2026. DOI: <a href="https://www.medrxiv.org/content/10.64898/2026.05.06.26352570v1">(https://doi.org/10.64898/2026.05.06.26352570)</a>.</li></p>
