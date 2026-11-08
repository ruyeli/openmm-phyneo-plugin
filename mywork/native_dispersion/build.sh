#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
env_dir="${PHYNEO_ENV_PREFIX:-${CONDA_PREFIX:-}}"
if [[ -z "$env_dir" ]]; then
  echo 'Activate phyneo_env or set PHYNEO_ENV_PREFIX before building.' >&2
  exit 1
fi
cuda_compiler="${CUDACXX:-$(command -v nvcc)}"
cxx_compiler="${CXX:-$env_dir/bin/x86_64-conda-linux-gnu-g++}"
export TMPDIR="$here/build/tmp"
mkdir -p "$TMPDIR"
"$env_dir/bin/cmake" -S "$here" -B "$here/build" \
  -DOPENMM_DIR="$env_dir" \
  -DPOCKETFFT_INCLUDE_DIR="$here/vendor" \
  -DCMAKE_CXX_COMPILER="$cxx_compiler" \
  -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
  -DCMAKE_CUDA_HOST_COMPILER="$cxx_compiler" \
  -DCMAKE_CUDA_ARCHITECTURES="${PHYNEO_CUDA_ARCHITECTURES:-native}"
"$env_dir/bin/cmake" --build "$here/build" -j2
