#!/bin/bash
# Builds the Davies (1980) p-value C++ extension used by hifimap_gpu/davies.py,
# from the vendored, unmodified qfc_standalone.cpp. See
# hifimap_gpu/_vendor/compquadform/README.md for provenance.
#
# Produces libqfc.dylib on macOS, libqfc.so on Linux -- davies.py picks the
# right one for the current platform automatically. Run this once per
# platform before using Step2_Run_HiFiMAP_GPU.py (a prebuilt libqfc.dylib
# for macOS is checked in; Linux/HPC must build its own libqfc.so with this
# script, since it depends on the local compiler/glibc).
set -euo pipefail
cd "$(dirname "$0")/hifimap_gpu/_vendor/compquadform"

if [ "$(uname)" = "Darwin" ]; then
    clang++ -O2 -shared -fPIC -o libqfc.dylib qfc_standalone.cpp
    echo "Built libqfc.dylib"
    nm -g libqfc.dylib | grep qfc
else
    g++ -O2 -shared -fPIC -o libqfc.so qfc_standalone.cpp
    echo "Built libqfc.so"
    nm -g libqfc.so | grep qfc
fi
