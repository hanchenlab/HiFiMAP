#!/bin/bash
# Packages a Step 0 output directory's delta_*.diff files into diffs.tar,
# which Step2_Run_HiFiMAP_GPU.py's bulk reader (hifimap_gpu/bulk_bookkeeping.py)
# reads directly instead of opening tens of thousands of small files one at a
# time. Run once per chromosome, after Step 0 and before Step 2 (Python/GPU).
#
# The R path (Step2_Run_HiFiMAP.R) reads the loose .diff files directly and
# does not need this.
set -euo pipefail
if [ -z "${1:-}" ]; then
    echo "usage: build_diffs_tar.sh <ibd_out_dir>" >&2
    exit 1
fi
ibd_out_dir="$1"
cd "$ibd_out_dir"
n=$(ls delta_*.diff 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -eq 0 ]; then
    echo "no delta_*.diff files found in $ibd_out_dir" >&2
    exit 1
fi
# COPYFILE_DISABLE=1 stops macOS's tar from adding a "._delta_N.diff"
# AppleDouble metadata entry per file; those extra entries break the
# Python reader's filename parsing (it expects exactly one entry per
# delta_N.diff). Linux tar ignores this variable, so it's set unconditionally.
COPYFILE_DISABLE=1 tar -cf diffs.tar delta_*.diff
echo "Built $ibd_out_dir/diffs.tar from $n diff files"
