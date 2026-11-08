#!/usr/bin/env bash
# Build from source on the destination cluster; never reuse old binary hashes.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
env_dir="${CONDA_PREFIX:?Activate the phyneo_env conda environment first}"
cuda_compiler="${CUDACXX:-$(command -v nvcc)}"
cuda_root="${CUDA_TOOLKIT_ROOT_DIR:-$(cd "$(dirname "$cuda_compiler")/.." && pwd)}"
export CXX="${CXX:-$env_dir/bin/x86_64-conda-linux-gnu-g++}"
export CC="${CC:-$env_dir/bin/x86_64-conda-linux-gnu-gcc}"
export TMPDIR="$repo_dir/build-portable/tmp"
mkdir -p "$TMPDIR"
cmake -S "$repo_dir" -B "$repo_dir/build-portable" \
    -DCMAKE_BUILD_TYPE=Release -DOPENMM_DIR="$env_dir" \
    -DCMAKE_INSTALL_PREFIX="$env_dir" -DCMAKE_PREFIX_PATH="$env_dir" \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_INSTALL_RPATH="$env_dir/lib;$env_dir/lib/plugins" \
    -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" -DPYTHON_EXECUTABLE="$env_dir/bin/python" \
    -DPhyNEO_BUILD_CUDA_LIB=ON -DPhyNEO_BUILD_PYTHON_WRAPPERS=ON
cmake --build "$repo_dir/build-portable" --parallel "${BUILD_JOBS:-2}"
cmake --install "$repo_dir/build-portable"
cmake --build "$repo_dir/build-portable" --target PythonInstall --parallel "${BUILD_JOBS:-2}"
export PHYNEO_ENV_PREFIX="$env_dir" CUDACXX="$cuda_compiler"
bash "$repo_dir/mywork/native_dispersion/build.sh"
"$env_dir/bin/python" "$repo_dir/mywork/portable_runtime.py"
